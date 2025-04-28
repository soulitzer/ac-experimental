# ac-experimental


## Installation
```bash
git clone https://github.com/soulitzer/ac-experimental.git
cd ac-experimental
pip install -e .
```

## Usage


<details>
<summary>
Click to see setup logic
</summary>

```python
import torch
from torch.utils.checkpoint import CheckpointPolicy
from ac-experimental import apply_ac_policy, tag_with_policy, SAVE_WITH_HOOKS

# For torch.compile support
torch._dynamo.config._enable_hopify_generic_context_manager.add(apply_ac_policy)

x = torch.rand((2, 2), requires_grad=True)
```
</details>

### Nest selective activation checkpoint (SAC) policies
```python
with apply_ac_policy("recompute_all"):
    ...
    with apply_ac_policy("must_save_all"):
        ...
```

### Tag a specific tensor for saving
```python
with apply_ac_policy("recompute_all"):
    y = ...
    z = y @ y
    # Save the output of this matmul instead of recomputing it
    tag_with_policy(z, CheckpointPolicy.MUST_SAVE)
    ...
```

### Influence compiler (min-cut partitioner) decisions
```python
y = x.sin().cos()
# Calling tag_with_policy without being in the body of an ac_policy context
# Hints at the compiler.
tag_with_policy(y, CheckpointPolicy.MUST_SAVE)
```


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
