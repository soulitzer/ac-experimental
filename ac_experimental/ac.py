import contextlib
import functools
from typing import Callable, NamedTuple, Union

import torch
import torch._subclasses.functional_tensor
import torch.fx.traceback as fx_traceback
from torch._functorch._aot_autograd.functional_utils import is_fun
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map_only

from torch.utils.checkpoint import _policy_from_bool, CheckpointPolicy
from torch.utils.weak import WeakTensorKeyDictionary


_policy_stack = []


def recompute_all(ctx, op, *args, **kwargs):
    return CheckpointPolicy.PREFER_RECOMPUTE


def must_save_all(ctx, op, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE


def try_handle_policy_str(policy_str):
    if policy_str == "recompute_all":
        return recompute_all
    elif policy_str == "must_save_all":
        return must_save_all
    else:
        raise ValueError(f"Unknown policy string: {policy_str}")


@contextlib.contextmanager
def push_policy(policy_fn):
    if isinstance(policy_fn, str):
        policy_fn = try_handle_policy_str(policy_fn)

    try:
        _policy_stack.append(policy_fn)
        yield
    finally:
        _policy_stack.pop()


def is_must_policy(policy):
    return policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.MUST_RECOMPUTE)


def is_save_policy(policy):
    return policy in (
        CheckpointPolicy.MUST_SAVE,
        CheckpointPolicy.PREFER_SAVE,
    ) or isinstance(policy, SAVE_WITH_HOOKS)


class SAVE_WITH_HOOKS:
    def __init__(self, pack, unpack):
        self.pack = pack
        self.unpack = unpack

    def is_must_policy(self):
        # Allow the user to override this?
        return True


def current_global_policy(ctx, out, op, *args, **kwargs):
    # Applies policy functions in _policy_stack from outermost to innermost.
    # - Default policy is PREFER_SAVE.
    # - A MUST_SAVE policy overrides any PREFER_SAVE policies seen so far.
    # - If a policy function returns a bool, convert it to a policy via _policy_from_bool.
    # - If two MUST_SAVE policies conflict, raise an error.
    # - Return the final resolved policy.
    current_policy = CheckpointPolicy.PREFER_SAVE
    for policy_fn in _policy_stack:
        policy = policy_fn(ctx, out, op, *args, **kwargs)
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)
        elif current_policy != policy:
            if is_must_policy(policy) and is_must_policy(current_policy):
                raise RuntimeError(
                    "Conflicting policies found in the policy stack. "
                    "Please ensure that the policy stack is consistent."
                )
            if is_must_policy(current_policy):
                continue
            current_policy = policy
    return current_policy



# Subclass instead of using namedtuple directly and remove `_asdict` method
# so that it's a pytree leaf
class _NodeOutput(NamedTuple):
    node: "Node"
    idx: int


del _NodeOutput._asdict


class NodeOutput(_NodeOutput):
    pass


def realize_and_decref(node_output):
    return node_output.node.realize_and_decref(node_output.idx)


def incref(node_output):
    node_output.node.incref(node_output.idx)


class Node:
    def __init__(
        self,
        func=None,
        args=None,
        outs: tuple = None,
        custom_pack=None,
        custom_unpack=None,
        current_fx_meta=None,
    ):
        self.custom_unpack = custom_unpack if custom_unpack is not None else lambda x: x
        custom_pack = custom_pack if custom_pack is not None else lambda x: x
        self.func = func
        self.args = args
        self.out = None
        self.current_fx_meta = current_fx_meta
        self.nb_users = dict()  # out_idx -> nb_users
        if outs is not None:
            self.out = [
                custom_pack(x) if isinstance(x, torch.Tensor) else x for x in outs
            ]

    def realize_and_decref(self, idx):
        if (
            self.out is None
            or (isinstance(self.out, list) and self.out[idx] is None)
            or (isinstance(self.out, dict) and self.out.get(idx) is None)
        ):
            new_args = tree_map_only(NodeOutput, realize_and_decref, self.args)
            raw_out = self.func(*new_args)
            self.out = list(raw_out) if isinstance(raw_out, tuple) else [raw_out]
        out = self.out[idx]
        self.nb_users[idx] -= 1
        if self.nb_users[idx] == 0:
            self.out[idx] = None
        x_out = self.custom_unpack(out)
        return x_out

    def incref(self, idx):
        if self.out is None and len(self.nb_users) == 0:
            # _ = [incref(arg) if isinstance(arg, NodeOutput) else arg for arg in self.args]
            tree_map_only(NodeOutput, incref, self.args)
        self.nb_users[idx] = self.nb_users.get(idx, 0) + 1


def get_node_output(node_outputs, t):
    if t not in node_outputs:
        # If the tensor was created in the checkpoint region, then it would've
        # been saved to node_outputs. If it's not there, then it's an input
        node_outputs[t] = NodeOutput(Node(None, None, (t,)), 0)
    return node_outputs[t]


class Context:
    def __init__(self, nodes):
        self.nodes = nodes


def _is_compiling(func, args, kwargs):
    # Check if we are under AOTAutograd tracing
    # There should probably be a better way to do this...
    # TODO: unify _is_compiling across all compile stacks
    for arg in args:
        if isinstance(arg, torch.Tensor) and is_fun(arg):
            return True
    return False


# Allow modes to interpose below AC mode for testing
_tracer_is_infra_mode = True


class TracerMode(TorchDispatchMode):
    def __init__(self, node_outputs):
        self.node_outputs = node_outputs
        if (
            hasattr(torch._C._TorchDispatchModeKey, "AC_TRACER")
            and _tracer_is_infra_mode
        ):
            self._mode_key = torch._C._TorchDispatchModeKey.AC_TRACER

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if any(
            t
            not in [
                torch._subclasses.FakeTensor,
                torch.Tensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ]
            for t in types
        ):
            return NotImplemented
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)

        # Non-tensor are always kept alive by the node
        wrapped_args = tree_map_only(
            torch.Tensor, functools.partial(get_node_output, self.node_outputs), args
        )
        # TODO: nodes shouldn't be public API right?
        ctx = Context(self.node_outputs)
        global_policy = current_global_policy(ctx, out, func, *args, **kwargs)
        should_save = is_save_policy(global_policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if is_compiling:
            fx_traceback.current_meta["recompute"] = global_policy

        custom_pack, custom_unpack = None, None
        if isinstance(global_policy, SAVE_WITH_HOOKS):
            custom_pack, custom_unpack = global_policy.pack, global_policy.unpack

        out_tuple = tuple(out) if isinstance(out, (list, tuple)) else (out,)
        node = (
            Node(
                func,
                None,
                out_tuple,
                custom_pack,
                custom_unpack,
                fx_traceback.current_meta,
            )
            if (should_save or is_compiling)
            else Node(func, wrapped_args, None)
        )
        for idx, t in enumerate(out_tuple):
            if isinstance(t, torch.Tensor):
                self.node_outputs[t] = NodeOutput(node, idx)
        return out

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True


def flatten(t, pack, unpack):
    if not is_traceable_wrapper_subclass(t):
        packed = pack(t)
        return lambda: unpack(packed)

    cls = t.__class__
    outer_size = t.shape
    outer_stride = t.stride()
    attrs, ctx = t.__tensor_flatten__()
    unflatten_fns = {attr: flatten(getattr(t, attr), pack, unpack) for attr in attrs}

    def unflatten():
        attrs_ = {attr: unflatten_fn() for attr, unflatten_fn in unflatten_fns.items()}
        return cls.__tensor_unflatten__(attrs_, ctx, outer_size, outer_stride)

    return unflatten


class TracerHooks(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, node_outputs):
        def _pack(raw_tensor):
            node_output = get_node_output(node_outputs, raw_tensor)
            incref(node_output)
            return node_output

        def _unpack(node_output):
            out = realize_and_decref(node_output)
            return out

        def pack_hook(raw_tensor):
            return flatten(raw_tensor, _pack, _unpack)

        def unpack_hook(unflatten_fn):
            x = unflatten_fn()
            return x

        super().__init__(pack_hook, unpack_hook)


# We're doing side effects that dynamo doesn't know about
_global_node_outputs = WeakTensorKeyDictionary()
_is_checkpoint_enabled = False


def is_checkpoint_enabled():
    return _is_checkpoint_enabled


class apply_ac_policy:
    """Apply a policy to all tensors produced in the context"""

    # Recomputing can only be done at the op level, but saving can be done at
    # the tensor level. When does this distiction matter?
    def __init__(self, policy_fn: Union[str, Callable] = "recompute_all"):
        self.policy_fn = policy_fn

    @torch._dynamo.disable
    def __enter__(self):
        global _is_checkpoint_enabled
        self.outer_most = not _is_checkpoint_enabled

        if self.outer_most:
            self.tracer_mode_ctx = TracerMode(_global_node_outputs)
            self.tracer_hooks_ctx = TracerHooks(_global_node_outputs)
            self.tracer_mode_ctx.__enter__()
            self.tracer_hooks_ctx.__enter__()
            _is_checkpoint_enabled = True

        self.push_policy_ctx = push_policy(self.policy_fn)
        self.push_policy_ctx.__enter__()

        return self

    @torch._dynamo.disable
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _is_checkpoint_enabled

        if self.outer_most:
            self.tracer_hooks_ctx.__exit__(exc_type, exc_val, exc_tb)
            self.tracer_mode_ctx.__exit__(exc_type, exc_val, exc_tb)
            _global_node_outputs.clear()
            _is_checkpoint_enabled = False

        self.push_policy_ctx.__exit__(exc_type, exc_val, exc_tb)

        return False  # Don't suppress exceptions


def tag_with_policy(t, policy):
    # Avoid circular imports with torch._dynamo.allow_in_graph
    from ._tag_with_policy import tag_with_policy as _tag_with_policy

    _tag_with_policy(t, policy)
