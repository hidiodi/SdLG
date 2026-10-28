import json
import numpy as np
import optax
import orbax.checkpoint
import wandb
import os
import hashlib
import base64

import gymnasium as gym
import time
import logging

from src.trainers.base_trainer import BaseTrainer
from src.tasks.tmaze import create_tmazev2
from src.tasks.memory_gym_env import  create_memorygym_env
from src.agents.a2c import A2CAgent
from src.agents.ppo import PPOAgent
from src.model_fns import *
from src.tasks.wrappers import *
from src.trainers.utils import *
from gymnasium.wrappers import AutoResetWrapper as Autoreset
from omegaconf import  OmegaConf


logger = logging.getLogger(__name__)


def get_env_initializers(env_config):
    env_config=OmegaConf.to_container(env_config)
    if env_config['task']=='tmazev2':
        env_fn=lambda: create_tmazev2(**env_config)
        repr_fn=flatten_repr_model()
        return env_fn,env_fn,repr_fn
    elif env_config['task']=='memory_gym':
        env_fn=lambda: create_memorygym_env(env_config['env_name'])
        repr_fn=atari_conv_repr_model()
        eval_env_fn=lambda: create_memorygym_env(env_config['env_name'])
        return env_fn,eval_env_fn,repr_fn


class ControlTrainer(BaseTrainer):

    def __init__(self,**kwargs):
        env_fn,eval_env_fn,repr_fn=get_env_initializers(kwargs['env_config'])
        self.wandb_run=kwargs['wandb_run']
        self.trainer_config=kwargs['trainer_config']
        self.env_config=kwargs['env_config']
        self.global_config=kwargs['global_args']
        self.checkpoint_dir=self.global_config.get('checkpoint_dir',None)
        self.checkpoint_interval=self.global_config.get('checkpoint_interval',None)
        self.rollout_len=self.trainer_config['rollout_len']
        self.num_envs=self.trainer_config['num_envs']
        self.gamma=self.trainer_config['gamma']
        #We will create two environments, one for training (vectorized) and one for evaluation
        train_seeds=np.random.randint(0,9999,size=self.num_envs,dtype=int).tolist()
        eval_seeds=int(np.random.randint(0,9999,size=1,dtype=int)[0])
        env_type=kwargs['trainer_config'].get('env_pool','async')
        if env_type=='async':
            env_type=gym.vector.AsyncVectorEnv
            logger.info('Using async env')
            train_envs=env_type([lambda: EpisodeStatisticsWrapper(Autoreset((env_fn())))for seed in train_seeds],shared_memory=True)
        elif env_type=='sync':
            logger.info('Using sync env')
            env_type=gym.vector.SyncVectorEnv
            train_envs=env_type([lambda: EpisodeStatisticsWrapper(Autoreset((env_fn())))for seed in train_seeds])
        elif env_type=='jax':
            logger.info('Using JAX-native vectorized env')
            from src.tasks.jax_tmaze import JaxVectorTMaze
            train_envs = JaxVectorTMaze(
                num_envs=self.num_envs,
                corridor_len=self.env_config.get('corridor_len', 10),
                num_distractors=self.env_config.get('num_distractors', 6),
                max_episode_steps=self.env_config.get('max_episode_steps', 5000),
                seed=kwargs['seed']
            )
        eval_env=RecordRollout(Autoreset(eval_env_fn()))
        train_envs.reset(seed=train_seeds)
        eval_env.reset(seed=eval_seeds)
        logger.info("Observation space: "+str(eval_env.observation_space))
        logger.info("Action space: "+str(eval_env.action_space))

        params_key,self.random_key=jax.random.split(kwargs['key'])

        if self.trainer_config.seq_model.name=='lstm':
            model_fn=seq_model_lstm(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='gru':
            model_fn=seq_model_gru(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='gtrxl':
            model_fn=seq_model_gtrxl(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='feedforward':
            model_fn=seq_model_feedforward(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='galite':
            model_fn=seq_model_galite(**self.trainer_config['seq_model'])
        elif self.trainer_config.seq_model.name=='agalite':
            model_fn=seq_model_agalite(**self.trainer_config['seq_model'],seed=kwargs['seed'])
        elif self.trainer_config.seq_model.name=='hybrid':
            model_fn=seq_model_hybrid(**self.trainer_config['seq_model'],seed=kwargs['seed'])
        elif self.trainer_config.seq_model.name=='sequential_hybrid':
            model_fn=seq_model_sequential_hybrid(**self.trainer_config['seq_model'],seed=kwargs['seed'])
        elif self.trainer_config.seq_model.name=='sdlg_hybrid':
            model_fn=seq_model_sdlg_hybrid(**self.trainer_config['seq_model'])

        actor_fn=actor_model_discete(self.trainer_config['d_actor'],eval_env.action_space.n)
        critic_fn=critic_model(self.trainer_config['d_critic'])
        #Setup optimizer

        if self.trainer_config['agent']=='a2c':
            self.optimizer=optax.chain(optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),  # Clip by the gradient by the global norm.
                                    optax.adam(**self.trainer_config.optimizer))  # Use Adam optimizer with learning rate.
            self.agent=A2CAgent(train_envs=train_envs,eval_env=eval_env,optimizer=self.optimizer, repr_model_fn=repr_fn,
                                seq_model_fn=model_fn,actor_fn=actor_fn,critic_fn=critic_fn,
                                rollout_len=self.rollout_len,
                                gamma=self.trainer_config['gamma'],lamb=self.trainer_config['lamb'],
                                value_loss_coef=self.trainer_config['value_coef'],
                                entropy_coef=self.trainer_config['entropy_coef'],
                                arg_max=self.trainer_config['arg_max'])
        elif self.trainer_config['agent']=='ppo':
            #Used from CleanRL PPO implementation
            batch_size = self.trainer_config['num_envs']*self.trainer_config['rollout_len']
            num_updates = self.global_config.steps // batch_size
            optimizer_config = dict(self.trainer_config.optimizer)
            learning_rate=optimizer_config.pop("learning_rate")
            if learning_rate['final'] is None:
                learning_rate['final']=learning_rate['initial'] #Set to none if you don't want decay
            if self.trainer_config['ent_coef']['final'] is None:
                self.trainer_config['ent_coef']['final']=self.trainer_config['ent_coef']['initial']
            lr_schedule=optax.polynomial_schedule(learning_rate['initial'],learning_rate['final'],learning_rate['power'],learning_rate['max_decay_steps'])
            ent_schedule=optax.polynomial_schedule(self.trainer_config['ent_coef']['initial'],self.trainer_config['ent_coef']['final'],
                                                   self.trainer_config['ent_coef']['power'],self.trainer_config['ent_coef']['max_decay_steps'])

            self.optimizer=optax.chain(
                                optax.clip_by_global_norm(self.trainer_config['max_grad_norm']),
                                optax.inject_hyperparams(optax.adam)(
                                    learning_rate=learning_rate['initial'], **optimizer_config
                                ),
                            )
            self.agent=PPOAgent(train_envs=train_envs,eval_env=eval_env,optimizer=self.optimizer, repr_model_fn=repr_fn,
                                seq_model_fn=model_fn,actor_fn=actor_fn,critic_fn=critic_fn,
                                num_steps=self.rollout_len,
                                anneal_lr=self.trainer_config.get('anneal_lr', True),
                                gamma=self.trainer_config.get('gamma', 0.99),
                                gae_lambda=self.trainer_config.get('gae_lambda', 0.95),
                                num_minibatches=self.trainer_config.get('num_minibatches', 4),
                                update_epochs=self.trainer_config.get('update_epochs', 4),
                                norm_adv=self.trainer_config.get('norm_adv', True),
                                clip_coef=self.trainer_config.get('clip_coef', 0.1),
                                lr_schedule=lr_schedule,
                                ent_schedule=ent_schedule,
                                vf_coef=self.trainer_config.get('vf_coef', 0.5),
                                max_grad_norm=self.trainer_config.get('max_grad_norm', 0.5),
                                target_kl=self.trainer_config.get('target_kl', None),
                                sequence_length=self.trainer_config.get('sequence_length', None),
                                arg_max=self.trainer_config.get('arg_max', False))


        self.agent.reset(params_key,self.random_key)
        self.step_count=0
        self.episode_lengths=[]
        self.average_reward_per_episode=[]
        self.average_return_per_episode=[]
        self.losses=[]
        self.critic_losses=[]
        self.actor_losses=[]
        self.entropy_losses=[]
        self.sps=[]
        self.approx_kls=[]
        self.result_data=[]
        self.reward_sum=0
        self.statistic_data=dict()
        self.B=self.num_envs*self.rollout_len
        self.log_interval=self.global_config.log_interval
        self.next_log_step=self.log_interval
        self.average_return_per_episode=[]
        if 'eval_interval' in self.global_config:
            self.eval_interval=self.global_config['eval_interval']
            self.next_eval_step=self.eval_interval
        else:
            self.eval_interval=None
        if 'save_interval' in self.global_config:
            self.save_interval=self.global_config['save_interval']
            self.next_save_step=self.save_interval
        else:
            self.save_interval=None

        #If path does not exist, create it
        if 'save_tag' in self.global_config:
            self.save_tag=self.global_config['save_tag']
        else:
            self.save_tag='auto'
        if self.save_tag=='auto':
            self.save_tag=str(compress_string_to_id(json.dumps({'env_config':OmegaConf.to_container(self.env_config),
                                                                'trainer_config':OmegaConf.to_container(self.trainer_config),
                                                                'seed':kwargs['seed']})))

        #Checkpointing code
        self.checkpoint_dir=os.path.abspath(os.path.join(self.global_config.get('save_dir','./'),self.save_tag))
        ckpt_exists=os.path.exists(self.checkpoint_dir)
        if (self.save_interval is not None) and (not os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)
            options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
            self.manager = orbax.checkpoint.CheckpointManager(self.checkpoint_dir, options=options)

        #Restore from checkpoint if true)
        if ckpt_exists and self.global_config.get('restore',False):
            self.restore_checkpoint()

    def step(self, **kwargs):
        self.random_key=jax.random.split(self.random_key)[0]

        #Measure steps per second
        start_time=time.time()

        (loss,(value_loss,entropy_loss,actor_loss,rewards),infos)=self.agent.step(self.random_key)
        if hasattr(self.agent, 'last_approx_kl'):
            self.approx_kls.append(self.agent.last_approx_kl)
        #Extract info data across all actors and steps
        #Get the leaves of the infos tree where the final_info key is present

        leaves=[info for info in infos if '_final_info' in info]
        # Increase the step counter
        self.step_count+=(self.B)
        #Iterate over the leaves and extract the final_info data
        for leaf in leaves:
             for env_info in leaf['final_info'][leaf['_final_info']]:
                 if 'final_info' in env_info:
                     for key,value in env_info['final_info'].items(): #Autoreset adds everything in info to final_info after reset along with info from first timestep
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if isinstance(v, (int, float, np.float32)):
                                    if k not in self.statistic_data:
                                        self.statistic_data[k]=[]
                                    self.statistic_data[k].append(v)
                        elif isinstance(value, (int, float,np.float32)):
                            if key not in self.statistic_data:
                                self.statistic_data[key]=[]
                            self.statistic_data[key].append(value)
                 for key,value in env_info.items():
                     if isinstance(value, dict):
                         for k, v in value.items():
                             if isinstance(v, (int, float, np.float32)):
                                 if k not in self.statistic_data:
                                     self.statistic_data[k]=[]
                                 self.statistic_data[k].append(v)
                     elif isinstance(value, (int, float,)):
                          if key not in self.statistic_data:
                              self.statistic_data[key]=[]
                          self.statistic_data[key].append(value)

                 ep_rewards=jnp.array(env_info['rewards'],dtype=jnp.float32)
                 _,average_return_per_episode=average_reward_and_return_in_episode(ep_rewards,self.gamma)
                 self.average_return_per_episode.append(average_return_per_episode)


        # Log the data
        end_time=time.time()
        self.sps.append(self.B/(end_time-start_time))
        self.losses.append(loss)
        self.critic_losses.append(value_loss)
        self.actor_losses.append(actor_loss)
        self.entropy_losses.append(entropy_loss)
        self.reward_sum+=rewards.sum()
        if self.step_count>=self.next_log_step:
            #Calculate the mean of the elements in statistic_data and log them, finally clear the statistic_data
            metrics={}
            for key in list(self.statistic_data.keys()):
                if len(self.statistic_data[key]) > 0:
                    arr = np.array(self.statistic_data[key])
                    metrics[key] = float(np.mean(arr))
                    metrics[f"{key}_std"] = float(np.std(arr))
                    metrics[f"{key}_min"] = float(np.min(arr))
                    metrics[f"{key}_max"] = float(np.max(arr))
                else:
                    metrics[key] = np.nan
                self.statistic_data[key]=[]
            self.next_log_step+=self.log_interval
            # Coerce JAX arrays to standard Python floats to free GPU VRAM and prevent system hangs
            critic_loss=float(np.mean(self.critic_losses))
            actor_loss=float(np.mean(self.actor_losses))
            entropy_loss=float(np.mean(self.entropy_losses))
            loss=float(np.mean(self.losses))
            reward_mean=float(self.reward_sum/self.log_interval)
            
            if len(self.average_return_per_episode) > 0:
                ret_arr = np.array(self.average_return_per_episode)
                return_mean = float(np.mean(ret_arr))
                return_std = float(np.std(ret_arr))
                return_min = float(np.min(ret_arr))
                return_max = float(np.max(ret_arr))
            else:
                return_mean = np.nan
                return_std = np.nan
                return_min = np.nan
                return_max = np.nan
                
            try:
                if self.trainer_config['agent'] == 'ppo':
                    lr = float(self.agent.optimizer_state[1].hyperparams['learning_rate'])
                else:
                    lr = float(self.trainer_config.optimizer.learning_rate)
            except Exception:
                lr = float(self.trainer_config.optimizer.get('learning_rate', 0.0001))

            if len(self.approx_kls) > 0:
                mean_kl = float(np.mean(self.approx_kls))
                self.approx_kls = []
            else:
                mean_kl = np.nan
                
            mean_sps=np.mean(self.sps)
            self.reward_sum=0
            self.critic_losses=[]
            self.actor_losses=[]
            self.entropy_losses=[]
            self.losses=[]
            self.sps=[]
            self.average_return_per_episode=[]
            metrics={'step':self.step_count,'sps':mean_sps,'loss':loss,'critic_loss':critic_loss,
                                    'actor_loss':actor_loss,'entropy_loss':entropy_loss,'mean_reward':reward_mean,
                                    'return_per_episode':return_mean,
                                    'return_per_episode_std':return_std,
                                    'return_per_episode_min':return_min,
                                    'return_per_episode_max':return_max,
                                    'learning_rate':lr,
                                    'approx_kl':mean_kl,
                                    **metrics
                                    }
            self.result_data.append(metrics)
        else:
            metrics=None

        if self.eval_interval is not None and self.step_count>=self.next_eval_step:
            self.next_eval_step+=self.eval_interval
            avg_episode_len,avg_episode_return,rollouts=self.agent.evaluate(self.random_key,self.global_config['eval_episodes'])
            rollouts=np.concatenate(rollouts,axis=0)
            eval_metrics = {
                'eval_avg_episode_len': float(avg_episode_len),
                'eval_avg_episode_return': float(avg_episode_return),
            }
            if metrics is None:
                metrics={}
            metrics['step']=self.step_count
            metrics.update(eval_metrics)
            metrics['rollouts']=wandb.Video(rollouts, fps=self.global_config.get('record_fps',5), format="gif")
            
            # Save evaluation metrics to result_data
            if len(self.result_data) > 0 and self.result_data[-1]['step'] == self.step_count:
                self.result_data[-1].update(eval_metrics)
            else:
                self.result_data.append({'step': self.step_count, **eval_metrics})
                
        if self.save_interval is not None and self.step_count>=self.next_save_step:
            self.next_save_step+=self.save_interval
            logger.info('Saving checkpoint at step '+str(self.step_count)+' to '+self.checkpoint_dir)
            self.save_checkpoint()
        return loss,metrics,self.step_count

    def save_checkpoint(self):
        ckpt={'step':self.step_count,'agent':self.agent.params,'optimizer':self.agent.optimizer_state}
        self.manager.save(self.step_count, args=orbax.checkpoint.args.StandardSave(ckpt))

    def restore_checkpoint(self):
        step=self.manager.latest_step()
        self.next_save_step=step+self.save_interval
        ckpt={'step':self.step_count,'agent':self.agent.params,'optimizer':self.agent.optimizer_state}
        ckpt=self.manager.restore(step=step, args=orbax.checkpoint.args.StandardRestore(ckpt))
        self.agent.params=ckpt['agent']
        self.agent.optimizer_state=ckpt['optimizer']
        self.step_count=step
        logger.info('Restored checkpoint from step '+str(step)+' from '+self.checkpoint_dir)



def compress_string_to_id(input_string):
    # Calculate the SHA-256 hash of the input string
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))
    hash_value = sha256.digest()

    # Encode the hash value to base64 and take the first 12 characters
    encoded_id = base64.b64encode(hash_value, altchars=b'-_')[:12]

    # Convert bytes to a string and remove padding characters
    compressed_id = encoded_id.decode('utf-8').rstrip('=')

    return compressed_id
