import torch

from .ac import (
    _global_node_outputs,
    _is_compiling,
    flatten,
    is_checkpoint_enabled,
    is_save_policy,
)


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
                proxy_tensor = mode.tracer.tensor_tracker[
                    mb_unwrap_functional_tensor(t)
                ]
                meta = proxy_tensor.proxy.node.meta

            if is_checkpoint_enabled():
                ac_node, idx = _global_node_outputs[t]
                ac_node.out[idx] = t.detach()
                if meta is None:
                    meta = ac_node.current_fx_meta

            if meta is not None:
                meta["recompute"] = policy
            return
        elif not is_checkpoint_enabled():
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
