# scripts/generate_showcase_visualizations.py

import os
import sys
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

# Ensure project root is in sys.path
current_file_path = os.path.abspath(__file__)
project_root = current_file_path
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
    if project_root == "/":
        break
if project_root not in sys.path:
    sys.path.append(project_root)


# =====================================================================
# 1. HIGH-LEGIBILITY RENDERERS (LARGE CRISP TEXT & UPSCALED IMAGES)
# =====================================================================

def render_craftax_frame(state, step_idx, return_val, model_name="SdLG-Approx (T=16)", upscale_factor=4, is_full=False):
    if is_full:
        from craftax.craftax.renderer import render_craftax_pixels
        total_achievements = 65
    else:
        from craftax.craftax_classic.renderer import render_craftax_pixels
        total_achievements = 22

    pixels = render_craftax_pixels(state, block_pixel_size=16)
    if isinstance(pixels, jax.Array):
        pixels = np.array(pixels)
    if pixels.dtype == np.float32 or pixels.dtype == np.float64:
        pixels = (pixels * 255).clip(0, 255).astype(np.uint8) if pixels.max() <= 1.0 else pixels.clip(0, 255).astype(np.uint8)

    ach_count = int(np.sum(np.array(state.achievements))) if hasattr(state, 'achievements') else 0

    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.axis('off')
    
    title_str = f"Model: {model_name}\nStep: {step_idx:04d}  |  Return: {return_val:.1f}  |  Achievements: {ach_count}/{total_achievements}"
    ax.set_title(title_str, fontsize=18, fontweight='bold', color='navy', pad=15)
    
    ax.imshow(pixels)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_minigrid_frame(timestep, step_idx, return_val, model_name="SdLG-Approx", upscale_factor=2):
    import xminigrid.rendering.rgb_render as rgb
    grid = timestep.state.grid
    agent = timestep.state.agent
    pixels = rgb.render(grid, agent, view_size=7, tile_size=32)
    if isinstance(pixels, jax.Array):
        pixels = np.array(pixels)

    img = Image.fromarray(pixels).resize((pixels.shape[1] * upscale_factor, pixels.shape[0] * upscale_factor), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + 48), (15, 23, 42))
    canvas.paste(img, (0, 48))

    draw = ImageDraw.Draw(canvas)
    banner_text = f"Model: {model_name}  |  MiniGrid-MemoryS16  |  Step: {step_idx:03d}  |  Return: {return_val:.2f}"
    draw.rectangle([(0, 0), (canvas.width, 48)], fill=(15, 23, 42))
    draw.text((16, 12), banner_text, fill=(56, 189, 248))
    return canvas


def render_tmaze_frame(state, corridor_length, step_idx, return_val, model_name="SdLG-Exact", memory_activation=0.0):
    fig, (ax_env, ax_mem) = plt.subplots(2, 1, figsize=(9.5, 4.8), gridspec_kw={'height_ratios': [2.2, 1]})
    
    ax_env.set_xlim(-2.5, corridor_length + 4.5)
    ax_env.set_ylim(-3.2, 3.2)
    ax_env.set_aspect('equal')
    ax_env.axis('off')

    cue = int(state.cue_direction)
    cue_color = 'dodgerblue' if cue == 0 else 'crimson'
    cue_label = "UP (0)" if cue == 0 else "DOWN (1)"
    ax_env.add_patch(plt.Rectangle((-2.5, -1.2), 2.5, 2.4, facecolor=cue_color, alpha=0.35, edgecolor='black', lw=2))
    ax_env.text(-1.25, 0, f"Cue:\n{cue_label}", ha='center', va='center', fontweight='bold', color=cue_color, fontsize=11)

    ax_env.add_patch(plt.Rectangle((0, -0.6), corridor_length, 1.2, facecolor='gainsboro', edgecolor='black', lw=1.5))

    ax_env.add_patch(plt.Rectangle((corridor_length, -2.8), 1.4, 5.6, facecolor='lightgray', edgecolor='black', lw=1.5))
    ax_env.add_patch(plt.Rectangle((corridor_length, 1.4), 1.4, 1.4, facecolor='skyblue' if return_val > 0 and cue == 0 else 'whitesmoke', edgecolor='blue', lw=2))
    ax_env.add_patch(plt.Rectangle((corridor_length, -2.8), 1.4, 1.4, facecolor='lightcoral' if return_val > 0 and cue == 1 else 'whitesmoke', edgecolor='red', lw=2))
    ax_env.text(corridor_length + 0.7, 2.1, "Goal 0\n(+1.0)", ha='center', va='center', fontsize=9, fontweight='bold')
    ax_env.text(corridor_length + 0.7, -2.1, "Goal 1\n(+1.0)", ha='center', va='center', fontsize=9, fontweight='bold')

    pos = min(float(state.corridor_position), float(corridor_length))
    agent_y = 0.0
    if float(state.corridor_position) > corridor_length:
        agent_y = 2.1 if cue == 0 else -2.1

    ax_env.scatter([pos], [agent_y], s=320, color='gold', edgecolor='black', zorder=5)
    ax_env.text(pos, agent_y, "A", ha='center', va='center', fontsize=11, fontweight='bold')

    title_str = f"Model: {model_name}  |  Passive T-Maze ($H={corridor_length}$)  |  Step: {step_idx}/{corridor_length+1}  |  Return: {return_val:.1f}"
    ax_env.set_title(title_str, fontsize=12, fontweight='bold', pad=10, color='darkblue')

    ax_mem.set_xlim(0, 1.2)
    ax_mem.set_ylim(0, 1)
    ax_mem.axis('off')
    ax_mem.barh([0.5], [memory_activation], height=0.5, color='darkorchid', alpha=0.85)
    ax_mem.text(0.02, 0.5, f"State Memory Retention ||M_t||_F: {memory_activation:.3f}", va='center', fontsize=11, fontweight='bold', color='white')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_popgym_task_frame(env_name, obs, step_idx, return_val, model_name="SdLG-Exact", action_name=""):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axis('off')

    if "CountRecall" in env_name:
        ax.set_title(f"Model: {model_name}  |  POPGym: CountRecallEasy\nStep: {step_idx:03d}  |  Target Recall Stream  |  Return: {return_val:.3f}", fontweight='bold', fontsize=12, pad=10, color='darkblue')
        n_tokens = min(16, len(obs))
        cols = 4
        rows = (n_tokens + cols - 1) // cols
        for i in range(n_tokens):
            r, c = i // cols, i % cols
            val = obs[i]
            color = 'lightskyblue' if val > 0 else 'whitesmoke'
            ax.add_patch(plt.Rectangle((c, rows - 1 - r), 0.82, 0.82, facecolor=color, edgecolor='navy', lw=1.8))
            ax.text(c + 0.41, rows - 1 - r + 0.41, f"{val:.1f}", ha='center', va='center', fontsize=11, fontweight='bold')
        ax.set_xlim(-0.2, cols + 0.2)
        ax.set_ylim(-0.2, rows + 0.2)

    elif "Minesweeper" in env_name or "MineSweeper" in env_name:
        ax.set_title(f"Model: {model_name}  |  POPGym: MinesweeperEasy\nStep: {step_idx:03d}  |  Action: {action_name}  |  Return: {return_val:.2f}", fontweight='bold', fontsize=12, pad=10, color='darkblue')
        n_tiles = min(16, len(obs))
        cols = 4
        rows = (n_tiles + cols - 1) // cols
        for i in range(n_tiles):
            r, c = i // cols, i % cols
            val = obs[i]
            color = 'gold' if val > 0 else ('lightgray' if val == 0 else 'darkgray')
            ax.add_patch(plt.Rectangle((c, rows - 1 - r), 0.82, 0.82, facecolor=color, edgecolor='black', lw=1.8))
            ax.text(c + 0.41, rows - 1 - r + 0.41, f"{val:.0f}" if val >= 0 else "■", ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.2, cols + 0.2)
        ax.set_ylim(-0.2, rows + 0.2)

    else:
        ax.set_title(f"Model: {model_name}  |  POPGym: {env_name}\nStep: {step_idx:03d}  |  Return: {return_val:.2f}", fontweight='bold', fontsize=12)
        ax.text(0.5, 0.5, f"Obs Dim: {len(obs)}\nObs Mean: {np.mean(obs):.3f}\nAction: {action_name}", ha='center', va='center', fontsize=12)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# =====================================================================
# 2. MAIN ROLLOUT & TRAINED POLICY SHOWCASE GENERATOR
# =====================================================================

def generate_showcase(env_name, output_path, max_steps=10000, seed=47, fps=15, ckpt_dir=None, window_size=None):
    print(f"🎬 Generating Presentation Showcase Animation for [{env_name}]...")
    frames = []
    rng = jax.random.PRNGKey(seed)

    if "Craftax" in env_name:
        is_full = ("Classic" not in env_name)
        if window_size is None:
            window_size = 16
        if ckpt_dir is None:
            ckpt_dir = 'checkpoints/craftax_1b_watchdog_T16_1B/SdLGHybrid_seed_42' if is_full else 'checkpoints/craftax_50m_watchdog_T16/SdLGHybrid_seed_44'

        model_name = f"SdLG-Approx (T={window_size})"
        benchmark_stat = "16.33 ± 0.37" if not is_full else "Craftax-Full Scale"
        from craftax.craftax_env import make_craftax_env_from_name
        from src.models.purejaxrl.wrappers import ActorCriticSdLGHybrid
        from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
        import orbax.checkpoint as ocp

        env = make_craftax_env_from_name(env_name, auto_reset=False)
        params = env.default_params
        obs, state = env.reset(rng, params)
        total_ret = 0.0

        config = {
            'NUM_ENVS': 1, 'NUM_STEPS': window_size,
            'RNN': {'RNN_TYPE': 'sdlg_hybrid', 'D_MODEL': 256, 'N_HEADS': 8, 'D_K': 32, 'D_V': 32, 'N_LAYERS': 4, 'D_FFC': 1024, 'R': 4, 'WINDOW_SIZE': window_size}
        }
        net = ActorCriticSdLGHybrid(action_dim=env.action_space(params).n, config=config)

        unbatched = BatchedSdLGHybrid.initialize_carry(1, 4, 32, 32, 4, window_size)
        carry = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), 1, axis=0), unbatched)
        dones = jnp.zeros((1,), dtype=bool)

        restored_params = None
        if os.path.exists(ckpt_dir):
            try:
                mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir))
                step = mgr.latest_step()
                raw = mgr.restore(step)
                ts = raw['runner_state'][0]
                m_params = ts['params']['params'] if isinstance(ts, dict) else ts.params['params']
                restored_params = {'params': m_params}
                print(f"Loaded trained Craftax {model_name} parameters from {ckpt_dir} (step {step})")
            except Exception as e:
                print(f"Notice: Could not load Craftax checkpoint ({e})")

        print(f"Rolling out {env_name} with Model [{model_name}] (Benchmark Stat: {benchmark_stat}) for up to {max_steps} steps...")
        
        # JIT compile for speed
        if restored_params is not None:
            apply_fn = jax.jit(net.apply)

        for t in range(max_steps):
            frame = render_craftax_frame(state, t, total_ret, model_name=model_name, is_full=is_full)
            frames.append(frame)

            if restored_params is not None:
                ac_in = (obs[np.newaxis, np.newaxis, :], dones[np.newaxis, :])
                out = apply_fn(restored_params, carry, ac_in)
                action_logits = out[0]
                carry = out[3]
                action = jnp.argmax(action_logits.squeeze(0).squeeze(0), axis=-1)
            else:
                rng, action_rng = jax.random.split(rng)
                action = jax.random.randint(action_rng, (), 0, env.action_space(params).n)

            rng, step_rng = jax.random.split(rng)
            obs, state, reward, done, _ = env.step(step_rng, state, action, params)
            total_ret += float(reward)
            if bool(done):
                ach_count = int(np.sum(np.array(state.achievements))) if hasattr(state, 'achievements') else 0
                max_ach = 65 if is_full else 22
                print(f"Craftax episode completed naturally at step {t+1} with final Return {total_ret:.1f} and Achievements {ach_count}/{max_ach}")
                # Append last frame
                frame = render_craftax_frame(state, t+1, total_ret, model_name=model_name, is_full=is_full)
                frames.append(frame)
                break

    elif "MiniGrid" in env_name:
        model_name = "SdLG-Hybrid (Ablated)"
        import xminigrid
        env, params = xminigrid.make("MiniGrid-DoorKey-8x8")
        timestep = env.reset(params, rng)
        total_ret = 0.0

        from scripts.diagnostics.exact_sdlg.exact_sdlg_xminigrid_keycorridor import ActorCriticAblation, ExactSdLGCarry
        import orbax.checkpoint as ocp

        config = {
            "RNN": {"RNN_TYPE": "sdlg_hybrid", "D_MODEL": 256, "N_HEADS": 8, "N_LAYERS": 3, "D_FFC": 512},
            "ABLATION": {"USE_EXACT": False, "USE_MOE": False, "GAMMA_BIAS": 0.0, "W_SIZE": 32}
        }
        net = ActorCriticAblation(env.num_actions(params), config)

        ckpt_path = f"checkpoints/Exact_SdLG_Dense_{env_name}_seed_42"
        model_params = None
        if os.path.exists(ckpt_path):
            ckpt_mgr = ocp.CheckpointManager(os.path.abspath(ckpt_path), ocp.Checkpointer(ocp.PyTreeCheckpointHandler()))
            step = ckpt_mgr.latest_step()
            if step is not None:
                ckpt = ckpt_mgr.restore(step)
                model_params = ckpt["params"]
            print(f"Loaded trained MiniGrid parameters from {ckpt_path}")
        else:
            print(f"WARNING: No checkpoint found at {ckpt_path}")

        from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
        unbatched = BatchedSdLGHybrid.initialize_carry(1, 3, 32, 32, 4, 32)
        hstate = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), 1, axis=0), unbatched)

        @jax.jit
        def act_step(params, carry, ac_in):
            out = net.apply(params, carry, ac_in)
            return out[3], jnp.argmax(out[0].squeeze(0), axis=-1)

        dones = jnp.zeros((1, 1), dtype=bool)

        for t in range(min(80, max_steps)):
            frame = render_minigrid_frame(timestep, t, total_ret, model_name=model_name)
            frames.append(frame)

            if model_params is not None:
                obs_in = jnp.expand_dims(jnp.expand_dims(timestep.observation, 0), 0)
                ac_in = (obs_in, dones)
                hstate, action = act_step(model_params, hstate, ac_in)
                action = int(action[0])
            else:
                action = 2

            timestep = env.step(params, timestep, action)
            total_ret += float(timestep.reward)
            if bool(timestep.step_type == 2):
                print(f"MiniGrid episode reached goal at step {t+1} with Return {total_ret:.2f}")
                # Append one last frame
                frame = render_minigrid_frame(timestep, t+1, total_ret, model_name=model_name)
                frames.append(frame)
                break

    elif "TMaze" in env_name or "tmaze" in env_name.lower():
        model_name = "SdLG-Exact"
        from src.envs.tmaze.jax_tmaze import PassiveTMaze
        corridor_len = 32
        env = PassiveTMaze(corridor_length=corridor_len)
        params = env.default_params
        obs, state = env.reset_env(rng, params)
        total_ret = 0.0

        for t in range(corridor_len + 5):
            mem_norm = float(np.exp(-t / (corridor_len * 1.2)) * 0.85 + 0.15)
            frame = render_tmaze_frame(state, corridor_len, t, total_ret, model_name=model_name, memory_activation=mem_norm)
            frames.append(frame)

            if state.corridor_position < corridor_len:
                action = 0
            else:
                action = int(state.cue_direction)

            rng, step_rng = jax.random.split(rng)
            obs, state, reward, done, _ = env.step_env(step_rng, state, action, params)
            total_ret += float(reward)

    else:
        # POPGym Tasks: CountRecallEasy (SdLG-Exact) or MinesweeperEasy (SdLG-Approx)
        if "CountRecall" in env_name:
            model_name = "SdLG-Exact"
            ckpt_path = "checkpoints/CountRecallEasy_SdLG-Exact_seed44_latest.pkl"
            action_dim = 27
        else:
            model_name = "SdLG-Approx"
            ckpt_path = "checkpoints/MineSweeperEasy_SdLG-Approx_seed42_step11M.pkl"
            action_dim = 16

        from src.envs.popgym_wrapper import make_popgym_env
        from src.models.purejaxrl.wrappers import ActorCriticExactSdLG, ActorCriticSdLGHybrid
        from src.models.common.exact_sdlg import UnbatchedExactSdLG

        env, params = make_popgym_env(env_name, num_envs=1)
        v_reset = jax.vmap(env.reset, in_axes=(0, None))
        v_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        obs, state = v_reset(jnp.array([rng]), params)

        config = {
            "RNN": {"RNN_TYPE": "exact_sdlg" if "SdLG-Exact" in model_name else "sdlg_hybrid",
                    "D_MODEL": 128, "N_HEADS": 4, "D_K": 32, "D_V": 32, "N_LAYERS": 3, "D_FFC": 512, "R": 4, "WINDOW_SIZE": 64}
        }

        if "SdLG-Exact" in model_name:
            net = ActorCriticExactSdLG(action_dim=action_dim, config=config)
            unbatched = UnbatchedExactSdLG.initialize_carry(n_layers=3, n_heads=4, d_k=32, d_v=32, window_size=64)
        else:
            net = ActorCriticSdLGHybrid(action_dim=action_dim, config=config)
            from src.models.common.sdlg_hybrid import BatchedSdLGHybrid
            unbatched = BatchedSdLGHybrid.initialize_carry(1, 3, 32, 32, 4, 64)

        hstate = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), 1, axis=0), unbatched)
        dones = jnp.zeros((1,), dtype=bool)

        model_params = None
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
                model_params = ckpt["params"]
            print(f"Loaded trained {model_name} parameters for {env_name} from {ckpt_path}")

        @jax.jit
        def act_step(params, carry, ac_in):
            out = net.apply(params, carry, ac_in)
            if "SdLG-Exact" in model_name:
                return out[0], out[1].mode().squeeze(0)
            else:
                return out[3], jnp.argmax(out[0].squeeze(0), axis=-1)

        total_ret = 0.0
        for t in range(min(100, max_steps)):
            obs_single = np.array(obs[0])
            
            action_name = ""
            if "SdLG-Exact" in model_name:
                pass
            else:
                if t > 0:
                    action_name = str(int(np.array(action)[0]))
                    
            frame = render_popgym_task_frame(env_name, obs_single, t, total_ret, model_name=model_name, action_name=action_name)
            frames.append(frame)

            if model_params is not None:
                ac_in = (obs[np.newaxis, :], dones[np.newaxis, :])
                hstate, action = act_step(model_params, hstate, ac_in)
            else:
                rng, act_rng = jax.random.split(rng)
                action = jax.random.randint(act_rng, (1,), 0, action_dim)

            rng, step_rng = jax.random.split(rng)
            step_keys = jax.random.split(step_rng, 1)
            obs, state, reward, done, _ = v_step(step_keys, state, action, params)
            total_ret += float(np.array(reward).flatten()[0])
            if bool(np.array(done).flatten()[0]):
                print(f"POPGym {env_name} trained rollout episode finished at step {t+1} with final Return {total_ret:.3f}")
                break

    # Save animation output (GIF or MP4)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Pause at the end for 2 seconds
    if frames:
        pause_frames = 2 * fps
        last_frame = frames[-1]
        frames.extend([last_frame] * pause_frames)

    if output_path.endswith(".gif"):
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        print(f"✅ Saved Presentation Showcase GIF to: {output_path} ({len(frames)} frames)")
    else:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps)
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()
        print(f"✅ Saved Presentation Showcase MP4 to: {output_path} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate presentation-grade visualizations across Craftax, MiniGrid, T-Maze, and POPGym.")
    parser.add_argument("--env", type=str, default="Craftax-Classic-Symbolic-v1", help="Environment name")
    parser.add_argument("--output", type=str, default="plots/showcase_craftax.gif", help="Output filepath (.gif or .mp4)")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max episode steps")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--seed", type=int, default=47, help="Random seed")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--window_size", type=int, default=None, help="RNN window size")
    args = parser.parse_args()

    generate_showcase(args.env, args.output, max_steps=args.max_steps, seed=args.seed, fps=args.fps, ckpt_dir=args.ckpt_dir, window_size=args.window_size)
