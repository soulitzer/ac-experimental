import contextlib
import torch
from torch.utils.weak import WeakTensorKeyDictionary
from _impl.tracer import (
    TracerHooks,
    TracerMode,
    flatten,
    _is_compiling,
)
from _impl.sac import (
    push_policy,
    SAVE_WITH_HOOKS,
    is_save_policy,
)

# We're doing side effects that dynamo doesn't know about
_global_node_outputs = None


class apply_ac_policy:
    """Apply a policy to all tensors produced in the context"""
    # Recomputing can only be done at the op level, but saving can be done at
    # the tensor level. When does this distiction matter?
    def __init__(self, policy_fn="recompute_all"):
        self.policy_fn = policy_fn

    @torch._dynamo.disable
    def __enter__(self):
        global _global_node_outputs

        self.outer_most = _global_node_outputs is None

        if self.outer_most:
            self.node_outputs = WeakTensorKeyDictionary()
            self.tracer_mode_ctx = TracerMode(self.node_outputs)
            self.tracer_hooks_ctx = TracerHooks(self.node_outputs)
            self.tracer_mode_ctx.__enter__()
            self.tracer_hooks_ctx.__enter__()
            _global_node_outputs = self.node_outputs

        self.push_policy_ctx = push_policy(self.policy_fn)
        self.push_policy_ctx.__enter__()

        return self

    @torch._dynamo.disable
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_node_outputs

        if self.outer_most:
            self.tracer_hooks_ctx.__exit__(exc_type, exc_val, exc_tb)
            self.tracer_mode_ctx.__exit__(exc_type, exc_val, exc_tb)
            self.node_outputs.clear()
            _global_node_outputs = None

        self.push_policy_ctx.__exit__(exc_type, exc_val, exc_tb)

        return False  # Don't suppress exceptions


@torch._dynamo.allow_in_graph
def tag_with_policy(t, policy):
    """Tag a single tensor with a policy"""
    # We might want want to improve the interaction when we're already in a
    # context manager, e.g. respect MUST policies. For now, we override all
    # current policies in the global policy stack.
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    def pack(t):
        if _is_compiling(None, (t,), None):
            # If we are compiling:
            # 1. always save the tensor, so we don't trace out the recompute
            # 2. set the policy on the fx node
            from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

            mode = get_proxy_mode()
            meta = None

            if mode is not None:
                # Notes:
                # - We don't have a proxy mode during the first time we
                #   trace through in AOTAutograd.
                # - ProxyMode sees tensors AFTER FunctionalTensor is unwrapped
                proxy_tensor = mode.tracer.tensor_tracker[mb_unwrap_functional_tensor(t)]
                meta = proxy_tensor.proxy.node.meta

            if _global_node_outputs is not None:
                ac_node, idx = _global_node_outputs[t]
                ac_node.out[idx] = t.detach()
                if meta is None:
                    meta = ac_node.current_fx_meta

            if meta is not None:
                meta["recompute"] = policy
            return
        elif _global_node_outputs is None:
            # If we're in eager and not in an ac context, I am only allowed to
            # "save" the tensor, which would be a no-op in eager.
            # Allowing this is useful if the user sometimes runs under eager
            # and compile.
            if is_save_policy(policy):
                # no-op
                return
            else:
                raise RuntimeError(
                    "Cannot use tag_with_policy with non-save policy outside of"
                    " a context manager"
                )
        else:
            # If we're in eager and we are in a context
            node, idx = _global_node_outputs[t]
            if is_save_policy(policy):
                if _global_node_outputs[t].node.out is None:
                    _global_node_outputs[t].node.out = dict()
                _global_node_outputs[t].node.out[idx] = t.detach()
            else:
                _global_node_outputs[t].node.out[idx] = None
            return t

    _unused_unflatten_fn = flatten(t, pack, lambda x: x)