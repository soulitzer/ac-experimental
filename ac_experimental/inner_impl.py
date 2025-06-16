import torch

# Import _UNSAFE_allow_side_effects with fallback to shim
try:
    from torch._dynamo.utils import _UNSAFE_allow_side_effects
except (ImportError, AttributeError):
    # Fallback to no-op shim
    def _UNSAFE_allow_side_effects(fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _tag_with_policy_impl(t, policy):
    """Implementation of tag_with_policy"""
    # Import utilities from ac.py to avoid duplication
    from .ac import is_save_policy, _is_compiling, set_policy_for_partitioner, save_tensor, is_checkpoint_enabled, flatten_nested_subclasses
    
    if not isinstance(t, torch.Tensor):
        raise ValueError(
            f"tag_with_policy: Expected a tensor, but got: {t}"
        )

    if not is_save_policy(policy):
        raise ValueError(
            f"tag_with_policy: Only 'saving' policies are currently supported. but got: {policy}"
        )

    def pack(t):
        if _is_compiling(None, (t,), None):
            set_policy_for_partitioner(t, policy)

        if is_checkpoint_enabled():
            save_tensor(t, policy)

    flatten_nested_subclasses(t, pack, lambda x: x)


# Apply the allow_in_graph decorator at module level
@torch._dynamo.allow_in_graph
def tag_with_policy_decorated(t, policy):
    """Tag a single tensor with a policy - decorated version"""
    return _tag_with_policy_impl(t, policy)
