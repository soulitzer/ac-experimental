import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.checkpoint import CheckpointPolicy
from torch.overrides import TorchFunctionMode
from torch.testing._internal.two_tensor import TwoTensor

from ac_experimental import (
    apply_ac_policy,
    SAVE_WITH_HOOKS,
    tag_with_policy
)


class SaveRecomputePolicyTest(TestCase):
    def test_save_everything(self):
        def fn(x):
            return (x.sin().cos() * 10 * 10 * 10 * 10).sin().cos()

        a = torch.tensor(1., requires_grad=True)

        with apply_ac_policy(policy_fn="recompute_all"):
            with apply_ac_policy(policy_fn="must_save_all"):
                out = fn(a)

        grad, = torch.autograd.grad(out, a)
        grad_ref, = torch.autograd.grad(fn(a), a)

        self.assertEqual(grad, grad_ref)

    def test_two_parallel_chains(self):
        def fn(a, b):
            out = a.sin().sin().sin()
            out2 = b.cos().cos().cos()
            out3 = out + out2
            return out3

        a = torch.tensor(1., requires_grad=True)
        b = torch.tensor(2., requires_grad=True)

        with apply_ac_policy("recompute_all"):
            out = fn(a, b)

        grad_a, grad_b = torch.autograd.grad(out, (a, b))
        grad_a_ref, grad_b_ref = torch.autograd.grad(fn(a, b), (a, b))

        self.assertEqual(grad_a, grad_a_ref)
        self.assertEqual(grad_b, grad_b_ref)

    def test_traceable_wrapper_subclass(self):
        count = [0]

        a = torch.rand(2, 2)
        b = TwoTensor(a, a.clone()).requires_grad_()

        def policy_fn(ctx, out, op, *args, **kwargs):
            if op == torch.ops.aten.sin.default:
                count[0] += 1
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            return x.sin().sum()

        with apply_ac_policy(policy_fn):
            out = fn(b)
        grad, = torch.autograd.grad(out, b)

        out_ref = fn(b)
        grad_ref, = torch.autograd.grad(out_ref, b)
        self.assertEqual(grad, grad_ref)
        self.assertEqual(count[0], 2)

    def test_torch_function_mode(self):
        def func(x, y) -> None:
            return torch.matmul(x, y)

        class DistributeFunction(TorchFunctionMode):
            def __torch_function__(self, func, types, args, kwargs=None):
                if kwargs is None:
                    kwargs = {}

                if func != torch.matmul:
                    return func(*args, **kwargs)

                a0 = args[0].reshape((-1, 128))
                a1 = args[1].reshape((128, -1))
                return func(a0, a1)

        with DistributeFunction():
            a = torch.randn(64, 64, requires_grad=True)
            with apply_ac_policy("recompute_all"):
                out = func(a, a)
            out.sum().backward()

    def test_torch_dispatch_mode(self):
        count = [0]

        class Mul2Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                return func(*args, **kwargs).sin()

        def policy_fn(ctx, out, op, *args, **kwargs):
            if op == torch.ops.aten.sin.default:
                count[0] += 1
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            return x.cos().cos().cos()

        a = torch.tensor(1., requires_grad=True)

        with Mul2Mode():
            with apply_ac_policy(policy_fn):
                out = fn(a)

        grad, = torch.autograd.grad(out, a)

        with Mul2Mode():
            out_ref = fn(a)

        grad_ref, = torch.autograd.grad(out_ref, a)

        self.assertEqual(grad, grad_ref)
        self.assertEqual(count[0], 3)

    def test_offloading(self):
        def save_with_hooks(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.sin.default:
                return SAVE_WITH_HOOKS(
                    lambda x: x.detach().cpu(),
                    lambda x: x.cuda()
                )
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            return x.sin().cos() * 10 * 10 * 10 * 10

        a = torch.tensor(1., requires_grad=True)

        with apply_ac_policy(save_with_hooks):
            out = fn(a)

        grad, = torch.autograd.grad(out, a)
        grad_ref, = torch.autograd.grad(fn(a), a)

        self.assertEqual(grad, grad_ref)

    def test_tag_with_policy(self):
        def fn(x):
            cos = x.sin().cos()
            tag_with_policy(cos, CheckpointPolicy.MUST_SAVE)
            return cos * 10 * 10 * 10 * 10

        a = torch.tensor(1., requires_grad=True)

        with apply_ac_policy("recompute_all"):
            out = fn(a)

        grad, = torch.autograd.grad(out, a)
        grad_ref, = torch.autograd.grad(fn(a), a)

        self.assertEqual(grad, grad_ref)

    def test_compile(self):
        torch._dynamo.config._enable_hopify_generic_context_manager.add(apply_ac_policy)
        # This tests:
        # 1. nesting
        # 2. multiple child ctxs
        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def fn(x):
            b = x.clone()

            with apply_ac_policy("recompute_all"):
                x = x.sin().cos().sin()
                with apply_ac_policy("must_save_all"):
                    d = []
                    y = x.sin().cos()
                    with apply_ac_policy("recompute_all"):
                        z = y @ y
                        z_1 = y.sin().cos()

                    with apply_ac_policy("recompute_all"):
                        z_2 = z * 10 * 10 * 10 * 10
                    d.append(z_1)
                    d.append(z_2)
            x = y
            c = y + d[0]
            return c

        a = torch.rand((4, 4), requires_grad=True)
        out = fn(a)
        out.sum().backward()

    def test_tag_with_policy(self):
        torch._dynamo.config._enable_hopify_generic_context_manager.add(apply_ac_policy)

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def fn(x):
            with apply_ac_policy("recompute_all"):
                cos = x.sin().sin().cos()
                z = cos @ cos
                tag_with_policy(z, CheckpointPolicy.MUST_SAVE)
                out = z.sin().sin() * 10 * 10 * 10 * 10
            return out

        a = torch.rand((2, 2), requires_grad=True)
        out = fn(a)
        out.sum().backward()
        grad_a = a.grad

        grad_a_ref = torch.autograd.grad(fn(a).sum(), a)[0]
        self.assertEqual(grad_a, grad_a_ref)

    def test_user_allow_in_graph(self):
        # Be able to handle arbitrary allow-in-graph functions
        torch._dynamo.config._enable_hopify_generic_context_manager.add(apply_ac_policy)

        @torch._dynamo.allow_in_graph
        def test_fn(x):
            return x.clone(), x.clone()

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def fn(x):
            with apply_ac_policy("recompute_all"):
                a, b = test_fn(x)
            c = a + b
            return c.sin()

        a = torch.rand((2, 2), requires_grad=True)
        out = fn(a)
        out.sum().backward()

    def test_compile_inside_context_manager(self):
        # To handle context managers, stash the ambient context manager while
        # dynamo is running. Restore it when we run aot autograd.
        pass

    # Test peak `memory
    # Test multi-output non-optimality
    # Test Nested subclasses



if __name__ == "__main__":
    run_tests()
