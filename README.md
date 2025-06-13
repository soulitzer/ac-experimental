# Activation Checkpoint v2 (AC2)

AC2 is a new tracer-based activation checkpoint implemented using TorchDispatchMode. Unlike `torch.utils.checkpoint`, AC2 traces out a graph of forward ops and replays pieces of that graph as necessary. Although it improves on the current AC in many aspects, is not a full replacement in all cases. See comparison section below.

## Installation
```bash
git clone https://github.com/soulitzer/ac-experimental.git && cd ac-experimental && pip install -e .
```
Or just copy paste the single self-contained file `ac_experimental/ac.py` for quick experimentation.

## Usage

<details>
<summary>
Click to see setup logic
</summary>

```python
import torch
from torch.utils.checkpoint import CheckpointPolicy
from ac_experimental import apply_ac_policy, apply_ac_policy_fn, tag_with_policy, SAVE_WITH_HOOKS

x = torch.rand((2, 2), requires_grad=True)
```
</details>

```python
with apply_ac_policy(policy1):  # context manager syntax
    with apply_ac_policy(policy2):  # nest policies
        z = y @ y
        tag_with_policy(z, CheckpointPolicy.MUST_SAVE)  # tag tensor to save
        ...
# OR
def fn(x):
    ...
apply_ac_policy_fn(fn, *args, **kwargs)  # function call syntax
```
Note: Today compile is ONLY supported for the second syntax. (Nightly version of torch required). Context manager syntax does not yet support compile (planned).

## 📊 Comparing with `torch.utils.checkpoint.Checkpoint`
(`torch.utils.checkpoint` is using the recommended `use_reentrant=False` setting)


| **Features**      | **`torch.utils.checkpoint`** | **AC2**      |
|--------|-----------------------------------|----------------------------|
| Selective activation checkpoint (SAC) support | ❌ Limited support (cannot specific tensor to save, SAC policies cannot be nested, sub-optimal eager performance) | ✅ First-class support (nesting, tensor tagging are supported) |
| Eager performance | ❌ Possibly suboptimal: Always recompute from the beginning (incurring higher peak memory since all recomputed buffers must be materialized all at once); Unnecessarily do extra recompute; Save tensors that are not needed | ✅ More optimal: Recompute can start from the middle of the graph. Only tensors needed for recompute are saved. |
| Semantics preserving | ❌ AC can change eager semantics (e.g. mutating a global or printing is performed twice)  | ✅ Side effects are not executed twice |
| Robustness to global state | ❌ Brittle; you must ensure that the provided function runs in the same exact way during the original forward and recompute (e.g. TorchFunctionMode/TorchDispatchMode, spurious logging, first iteration initialization). | ✅ Post-dispatch graph is captured and replayed |
| Recursive checkpointing | ✅ Supported | ❌ Not supported |
| Higher order gradients support | ✅ Supported | ❌ Not supported |
| RNG operators |  ✅ Supported | ❌ Today the output of RNG ops are required to be saved (alternatively, we could stash RNG state for every op)
| non-torch/non-aten ops, e.g. NumPy ops | ✅ non-torch/non-aten can be recomputed | ❌ non-ATen ops cannot be recomputed (you should wrap them in custom ops) |
| In-place | ... | ... |
| Side Effects in compile | ❌ Not supported | 🚧 Planned |
| Retain_graph | ... | ... |



## Advanced features

### Selective offloading and activation quantization
```python
save_with_hooks = SAVE_WITH_HOOKS(
    lambda x: x.detach().cpu(),
    lambda x: x.cuda()
)

def policy(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.sin.default:
        return save_with_hooks
    return CheckpointPolicy.PREFER_RECOMPUTE

with apply_ac_policy(policy):
    ...

# or

with apply_ac_policy("recompute_all"):
    ...
    tag_with_policy(save_with_hooks)
    ...
```
