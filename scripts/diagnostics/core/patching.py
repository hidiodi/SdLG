import sys
import contextlib
import importlib

@contextlib.contextmanager
def patch_module_attribute(module, attribute_name, new_value):
    """Temporarily replaces an attribute of a module."""
    old_value = getattr(module, attribute_name)
    setattr(module, attribute_name, new_value)
    try:
        yield
    finally:
        setattr(module, attribute_name, old_value)

@contextlib.contextmanager
def patch_sdlg_decay_mechanism(sdlg_hybrid_module, decay_type: str):
    """
    Temporarily overrides the GatedAttentionGLALayer in the sdlg_hybrid module
    to use a different decay mechanism (e.g. for ablation studies).
    """
    original_layer = sdlg_hybrid_module.GatedAttentionGLALayer

    # Example logic to override class behaviour dynamically
    # For now, this is a placeholder that you can expand based on specific
    # experiment needs (like exp11/exp12 dynamic monkey patching).
    # Since patching can be complex with JAX/Flax (due to dataclasses),
    # this helper can be expanded as needed.
    yield
