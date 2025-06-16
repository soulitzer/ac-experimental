import contextlib
import functools
from typing import Any, Callable, NamedTuple, Union

import torch
import torch._subclasses.functional_tensor
import torch.fx.traceback as fx_traceback
from torch._functorch._aot_autograd.functional_utils import is_fun
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map_only

from torch.utils.checkpoint import _policy_from_bool, CheckpointPolicy, _maybe_detach
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


def set_policy_for_partitioner(t: torch.Tensor, policy):
    assert isinstance(t, torch.Tensor)
    assert _is_compiling(None, (t,), None)

    from torch.fx.experimental.proxy_tensor import get_proxy_mode
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    mode = get_proxy_mode()
    meta = None

    if mode is None:
        # We won't have meta during the first trace through AOTAutograd.
        # That is fine though because it is the second trace where
        # we actually care about annotating.
        return
    # ProxyMode sees tensors after FunctionalTensor is unwrapped
    proxy_tensor = mode.tracer.tensor_tracker[
        mb_unwrap_functional_tensor(t)
    ]
    meta = proxy_tensor.proxy.node.meta
    meta["recompute"] = policy
    # Note [ dummy ac_graph_id ]
    # ac_graph_id is used by the old AC to be able to insert MUST_SAVE
    # between adjacent AC regions. This is not needed for the new AC,
    # so give everything the same id to bypass the logic.
    meta["ac_graph_id"] = 0


def save_tensor(t: torch.Tensor, policy):
    # Precondition: t is being tracked
    ac_node, idx = _global_node_outputs[t]

    if t.is_inference():
        raise RuntimeError(
            "Saving inference tensors is not supported. "
        )
    if isinstance(policy, SAVE_WITH_HOOKS):
        # Another option is to grab the ambient saved tensor hooks?
        # Why are we doing it this way?
        t = policy.pack(t)
        ac_node.custom_pack = policy.unpack
    else:
        # What was the issue with the old SAC wrt inference mode + detach?
        # Is there any reason to not always detach? If we care about tensor identity.
        t = _maybe_detach(t, True)

    ac_node.out[idx] = t
    # Note [ Version counter checking ]
    #
    # Each NodeOutput corresponds to a particular version of a tensor.
    # If an operator is in-place, we create a new Node just like we do
    # for out-of-place operators. The tracker dict is also updated for
    # the mutated tensor to now point to the newer NodeOutput.
    #
    # IMPORTANT: It is required for any operator that mutates its inputs to
    # return that mutated tensor as output. This is currently not enforced!
    #
    # If an operator is in-place, we currently save its all of its outputs.
    # This is not always optimal. If the mutated tensor wasn't otherwise
    # saved e.g. it is not an input or saved via SAC. OR if the mutated
    # tensor is saved but this op has multiple outputs and now we need to
    # save additional tensors.
    ac_node.out_versions[idx] = t._version



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
        kwargs=None,
        custom_pack=None,
        custom_unpack=None,
    ):
        self.custom_unpack = custom_unpack
        custom_pack = custom_pack if custom_pack is not None else lambda x: x
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.out: dict[int, Any] = dict()
        self.nb_users = dict()  # out_idx -> nb_users
        self.out_versions = dict()

    def realize_and_decref(self, idx):
        if self.out.get(idx) is None:
            new_args, new_kwargs = tree_map_only(
                NodeOutput, realize_and_decref, (self.args, self.kwargs)
            )
            raw_out = self.func(*new_args, **new_kwargs)
            out_tuple = tuple(raw_out) if isinstance(raw_out, (list, tuple)) else (raw_out,)
            for t in out_tuple:
                if isinstance(t, torch.Tensor):
                    self.out_versions[idx] = t._version
                self.out[idx] = t

        out = self.out[idx]
        self.nb_users[idx] -= 1
        if self.nb_users[idx] == 0:
            self.out[idx] = None

        if self.custom_unpack is not None:
            out = self.custom_unpack(out)

        # See Note [ Version counter checking ]
        if (
            self.custom_unpack is None
            and self.out_versions.get(idx) is not None
        ):
            assert (
                self.out_versions[idx] == out._version
            ), f"{self.func}, expected version {self.out_versions[idx]}, but got version {out._version}"
        return out

    # See Note [AC Node use-count tracking to clear cached tensors sooner]
    def incref(self, idx):
        # TODO: have some test to check for leaks?
        if all(v is None for v in self.out.values()):
            tree_map_only(NodeOutput, incref, self.args)
        self.nb_users[idx] = self.nb_users.get(idx, 0) + 1


def get_node_output(node_outputs, t):
    if t not in node_outputs:
        # If the tensor was created in the checkpoint region, then it would've
        # been saved to node_outputs. If it's not there, then it's an input
        node_outputs[t] = NodeOutput(Node(None, None, None), 0)

        if _is_compiling(None, (t,), None):
            set_policy_for_partitioner(t, CheckpointPolicy.MUST_SAVE)
        save_tensor(t, CheckpointPolicy.MUST_SAVE)

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

        # Non-tensor are always kept alive by the node
        wrapped_args, wrapped_kwargs = tree_map_only(
            torch.Tensor,
            functools.partial(get_node_output, self.node_outputs),
            (args, kwargs),
        )

        any_is_inplace = False
        for idx, arg in enumerate(func._schema.arguments):
            if arg.alias_info is not None and arg.alias_info.is_write:
                any_is_inplace = True
                break

        out = func(*args, **kwargs)

        # TODO: nodes shouldn't be public API right?
        ctx = Context(self.node_outputs)
        global_policy = current_global_policy(ctx, out, func, *args, **kwargs)
        out_tuple = tuple(out) if isinstance(out, (list, tuple)) else (out,)

        if torch.Tag.nondeterministic_seeded in func.tags and not is_save_policy(global_policy):
            raise RuntimeError(
                f"apply_ac_policy: recomputing RNG ops are not supported, but got {func}. "
                "Please save the outputs of this op."
            )

        do_save_logic = (
            is_save_policy(global_policy)
            or any_is_inplace
            or _is_compiling(func, args, kwargs)
        )
        if do_save_logic:
            # We don't want to store the args in the case where we save!
            wrapped_args, wrapped_kwargs = None, None

        node = Node(
            func,
            wrapped_args,
            wrapped_kwargs)

        # Register the nodes before fully setting them up since save_tensors requires it
        for idx, t in enumerate(out_tuple):
            if isinstance(t, torch.Tensor):
                self.node_outputs[t] = NodeOutput(node, idx)

        for idx, t in enumerate(out_tuple):
            if not isinstance(t, torch.Tensor):
                node.out[idx] = t
                continue

            if do_save_logic:
                save_tensor(t, global_policy)

            if _is_compiling(None, (t,), None):
                set_policy_for_partitioner(t, global_policy)

        # HACK: For in-place, we need to manually increment the counter since the
        # version won't be updated until we return to the ADInplaceOrView kernel :(
        # For now assume that the first output is the one that was mutated, and
        # that result's version is always incremented by 1.
        # This should be true for most built-in inplace ops, but not always true,
        # e.g. for BatchNorm.
        if any_is_inplace:
            assert node.out_versions.get(0) is not None, f"{func}"
            node.out_versions[0] += 1

        return out

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True


def flatten_nested_subclasses(t, pack, unpack):
    if not is_traceable_wrapper_subclass(t):
        packed = pack(t)
        return lambda: unpack(packed)

    cls = t.__class__
    outer_size = t.shape
    outer_stride = t.stride()
    attrs, ctx = t.__tensor_flatten__()
    unflatten_fns = {attr: flatten_nested_subclasses(getattr(t, attr), pack, unpack) for attr in attrs}

    def unflatten():
        attrs_ = {attr: unflatten_fn() for attr, unflatten_fn in unflatten_fns.items()}
        return cls.__tensor_unflatten__(attrs_, ctx, outer_size, outer_stride)

    return unflatten


class TracerHooks(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, node_outputs):
        def _pack(raw_tensor):
            node_output = get_node_output(node_outputs, raw_tensor)
            # The assumption here is that every pack has a corresponding
            # unpack. Otherwise the refcounting is wrong and more tensors
            # will be kept alive.
            # We also don't support retain_graph=True.
            incref(node_output)
            return node_output

        def _unpack(node_output):
            out = realize_and_decref(node_output)
            return out

        def pack_hook(raw_tensor):
            return flatten_nested_subclasses(raw_tensor, _pack, _unpack)

        def unpack_hook(unflatten_fn):
            x = unflatten_fn()
            return x

        super().__init__(pack_hook, unpack_hook)


# We're doing side effects that dynamo doesn't know about
_global_node_outputs = WeakTensorKeyDictionary()
_trace_mode_active = False
_is_checkpoint_enabled = False


def is_checkpoint_enabled():
    return _is_checkpoint_enabled


class _apply_ac_policy:
    """Apply a policy to all tensors produced in the context"""

    # Recomputing can only be done at the op level, but saving can be done at
    # the tensor level. When does this distiction matter?
    def __init__(self, policy_fn: Union[str, Callable] = "recompute_all"):
        self.policy_fn = policy_fn

    @torch._dynamo.disable
    def __enter__(self):
        global _trace_mode_active
        self.outer_most = not _trace_mode_active

        if self.outer_most:
            self.tracer_mode_ctx = TracerMode(_global_node_outputs)
            self.tracer_hooks_ctx = TracerHooks(_global_node_outputs)
            self.tracer_mode_ctx.__enter__()
            self.tracer_hooks_ctx.__enter__()
            _trace_mode_active = True

        self.push_policy_ctx = push_policy(self.policy_fn)
        self.push_policy_ctx.__enter__()

        return self

    @torch._dynamo.disable
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _trace_mode_active

        if self.outer_most:
            self.tracer_hooks_ctx.__exit__(exc_type, exc_val, exc_tb)
            self.tracer_mode_ctx.__exit__(exc_type, exc_val, exc_tb)
            _global_node_outputs.clear()
            _trace_mode_active = False

        self.push_policy_ctx.__exit__(exc_type, exc_val, exc_tb)

        return False  # Don't suppress exceptions


@contextlib.contextmanager
def apply_ac_policy(policy_fn="recompute_all"):
    global _is_checkpoint_enabled

    prev_is_checkpoint_enabled = _is_checkpoint_enabled
    _is_checkpoint_enabled = True
    try:
        with _apply_ac_policy(policy_fn=policy_fn):
            yield
    finally:
        _is_checkpoint_enabled = prev_is_checkpoint_enabled


def _apply_ac_policy_wrapper_impl(fn, *args, policy_fn="recompute_all", **kwargs):
    def wrapper(*args, **kwargs):
        with _apply_ac_policy(policy_fn):
            return fn(*args, **kwargs)
    return wrapper

def _apply_ac_policy_wrapper_factory_impl(fn, *args, make_policy_fn=None, **kwargs):
    def wrapper(*args, **kwargs):
        assert make_policy_fn is not None
        policy_fn = make_policy_fn()
        with _apply_ac_policy(policy_fn):
            return fn(*args, **kwargs)
    return wrapper


# I don't know if I necessarily like having apply_ac_policy and apply_ac_policy_fn as the API names
def apply_ac_policy_fn(fn, *args, policy_fn: Union[str, Callable[[Any], CheckpointPolicy]]="recompute_all", is_factory=False, **kwargs):
    global _is_checkpoint_enabled

    from torch._higher_order_ops.wrap import dynamo_bypassing_wrapper
    try:
        # Need https://github.com/pytorch/pytorch/pull/155715 or something else
        from torch._dynamo.utils import _UNSAFE_allow_side_effects
    except ImportError:
        # Fallback to no-op shim
        def _UNSAFE_allow_side_effects(fn: Callable, *args, **kwargs):
            return fn(*args, **kwargs)

    def wrapped_fn(*args, **kwargs):
        return _UNSAFE_allow_side_effects(fn, *args, **kwargs)

    prev_is_checkpoint_enabled = _is_checkpoint_enabled
    _is_checkpoint_enabled = True
    try:
        if is_factory:
            return dynamo_bypassing_wrapper(
                functools.partial(_apply_ac_policy_wrapper_factory_impl, make_policy_fn=policy_fn),
                wrapped_fn, *args, **kwargs
            )

        return dynamo_bypassing_wrapper(
            functools.partial(_apply_ac_policy_wrapper_impl, policy_fn=policy_fn),
            wrapped_fn, *args, **kwargs
        )
    finally:
        _is_checkpoint_enabled = prev_is_checkpoint_enabled


# Lazy initialization to avoid reference cycles on import
_tag_with_policy_impl = None


def _tag_with_policy(t, policy):
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

        if _is_checkpoint_enabled:
            save_tensor(t, policy)

    flatten_nested_subclasses(t, pack, lambda x: x)


def tag_with_policy(t, policy):
    """Tag a single tensor with a policy"""
    global _tag_with_policy_impl

    # Lazy initialization to avoid reference cycles on import
    if _tag_with_policy_impl is None:
        _tag_with_policy_impl = torch._dynamo.allow_in_graph(_tag_with_policy)

    return _tag_with_policy_impl(t, policy)
