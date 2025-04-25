import contextlib

from torch.utils.checkpoint import CheckpointPolicy, _policy_from_bool


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
    return (
        policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)
        or isinstance(policy, SAVE_WITH_HOOKS)
    )


class SAVE_WITH_HOOKS():
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
