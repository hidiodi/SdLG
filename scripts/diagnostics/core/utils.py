import os
import wandb
import logging
import sys

def setup_logger(name: str, log_dir: str, log_file: str) -> logging.Logger:
    """Sets up a standardized logger that writes to both console and a local file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the logger is requested multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # File handler
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

def init_wandb(project_override: str = None, group: str = None, name: str = None, config: dict = None, model_name: str = None, resume: str = "allow", project: str = None, tags: list = None, notes: str = None, **kwargs):
    """
    Initializes Weights & Biases robustly.
    Merges tags from the environment variable (set by run_all.py) with the specific model tag.
    Fallback to offline mode if API key is missing or connection fails.
    """
    actual_project = project or project_override
    base_project = os.environ.get("WANDB_PROJECT", actual_project)
    
    # Read environment tags
    env_tags = os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else []
    
    # Merge env_tags with passed tags (if any) and model_name
    final_tags = list(env_tags)
    if tags:
        for t in tags:
            if t and t not in final_tags:
                final_tags.append(t)
    if model_name and model_name not in final_tags:
        final_tags.append(model_name)
        
    base_notes = os.environ.get("WANDB_NOTES", notes or "")

    try:
        run_id = os.environ.get("WANDB_RUN_ID")
        run = wandb.init(
            project=base_project,
            group=group,
            name=name,
            config=config,
            tags=final_tags,
            notes=base_notes,
            resume=resume,
            id=run_id
        )
        return run
    except Exception as e:
        print(f"Failed to initialize wandb online: {e}. Falling back to offline mode.")
        os.environ["WANDB_MODE"] = "offline"
        run_id = os.environ.get("WANDB_RUN_ID")
        run = wandb.init(
            project=base_project,
            group=group,
            name=name,
            config=config,
            tags=final_tags,
            notes=base_notes,
            resume=resume,
            id=run_id
        )
        return run

import jax
import jax.numpy as jnp

def compute_gradient_cosine_similarity(grads_actor, grads_critic):
    """
    Computes the cosine similarity between flattened actor and critic gradients
    to measure gradient harmony (or interference) on a shared trunk.
    """
    # Flatten the PyTrees into single arrays
    flat_actor, _ = jax.tree_util.tree_flatten(grads_actor)
    flat_critic, _ = jax.tree_util.tree_flatten(grads_critic)

    vec_actor = jnp.concatenate([jnp.ravel(x) for x in flat_actor])
    vec_critic = jnp.concatenate([jnp.ravel(x) for x in flat_critic])

    dot_product = jnp.dot(vec_actor, vec_critic)
    norm_actor = jnp.linalg.norm(vec_actor)
    norm_critic = jnp.linalg.norm(vec_critic)

    # Avoid div by zero
    similarity = dot_product / (norm_actor * norm_critic + 1e-8)
    return similarity
