import contextlib
import unittest

import torch

from ac_experimental import apply_ac_policy, SAVE_WITH_HOOKS, tag_with_policy
from torch.overrides import TorchFunctionMode
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.checkpoint import CheckpointPolicy
from torch.utils.flop_counter import FlopCounterMode


@contextlib.contextmanager
def allow_ambient_mode_to_run_first():
    # Temporarily allow user modes to interpose below AC mode for testing purposes
    import ac_experimental

    try:
        ac_experimental._tracer_is_infra_mode = False
        yield
    finally:
        ac_experimental._tracer_is_infra_mode = True


class SaveRecomputePolicyTest(TestCase):
    def test_save_everything(self):
        def fn(x):
            return (x.sin().cos() * 10 * 10 * 10 * 10).sin().cos()

        a = torch.tensor(1.0, requires_grad=True)

        with apply_ac_policy(policy_fn="recompute_all"):
            with apply_ac_policy(policy_fn="must_save_all"):
                out = fn(a)

        (grad,) = torch.autograd.grad(out, a)
        (grad_ref,) = torch.autograd.grad(fn(a), a)

        self.assertEqual(grad, grad_ref)

    def test_two_parallel_chains(self):
        def fn(a, b):
            out = a.sin().sin().sin()
            out2 = b.cos().cos().cos()
            out3 = out + out2
            return out3

        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)

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
        (grad,) = torch.autograd.grad(out, b)

        out_ref = fn(b)
        (grad_ref,) = torch.autograd.grad(out_ref, b)
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

        a = torch.tensor(1.0, requires_grad=True)

        with Mul2Mode():
            with apply_ac_policy(policy_fn):
                out = fn(a)

        (grad,) = torch.autograd.grad(out, a)

        with Mul2Mode():
            out_ref = fn(a)

        (grad_ref,) = torch.autograd.grad(out_ref, a)

        self.assertEqual(grad, grad_ref)
        self.assertEqual(count[0], 3)

    def test_offloading(self):
        def save_with_hooks(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.sin.default:
                return SAVE_WITH_HOOKS(lambda x: x.detach().cpu(), lambda x: x.cuda())
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            return x.sin().cos() * 10 * 10 * 10 * 10

        a = torch.tensor(1.0, requires_grad=True)

        with apply_ac_policy(save_with_hooks):
            out = fn(a)

        (grad,) = torch.autograd.grad(out, a)
        (grad_ref,) = torch.autograd.grad(fn(a), a)

        self.assertEqual(grad, grad_ref)

    def test_tag_with_policy_eager(self):
        def fn(x):
            cos = x.sin().cos()
            tag_with_policy(cos, CheckpointPolicy.MUST_SAVE)
            return cos * 10 * 10 * 10 * 10

        a = torch.tensor(1.0, requires_grad=True)

        with apply_ac_policy("recompute_all"):
            out = fn(a)

        (grad,) = torch.autograd.grad(out, a)
        (grad_ref,) = torch.autograd.grad(fn(a), a)

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

    def test_tag_with_policy_compile(self):
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

    def test_compile_simple_policy_fn(self):
        torch._dynamo.config._enable_hopify_generic_context_manager.add(apply_ac_policy)

        def policy_fn(ctx, out, op, *args, **kwargs):
            return CheckpointPolicy.PREFER_RECOMPUTE

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def fn(x):
            with apply_ac_policy(policy_fn):
                return x.sin().cos() * 10

        a = torch.rand((4, 4), requires_grad=True)
        out = fn(a)
        out.sum().backward()

    def test_flops_and_mem(self):
        with allow_ambient_mode_to_run_first():
            if not torch.cuda.is_available():
                raise unittest.SkipTest("CUDA is unavailable")

            # From https://github.com/pytorch/pytorch/pull/126320
            def get_act_mem(f):
                out = f()
                out.backward()
                start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
                out = f()
                cur_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
                act_mem = (cur_mem - start_mem) / (1024 * 1024)
                out.backward()
                return act_mem

            def get_bw_flops(f):
                # Normalized so that a 512 square matmul returns 1
                f().backward()
                out = f()
                # NB: FlopCounterMode is pushed onto the mode stack before entering
                # AC's context so we'll be able to observe what is cached.
                with FlopCounterMode(display=False) as mode:
                    out.backward()
                return mode.get_total_flops() / (512**3 * 2)

            def policy_fn(ctx, out, op, *args, **kwargs):
                if op == torch.ops.aten.mm.default:
                    return CheckpointPolicy.MUST_SAVE
                else:
                    return CheckpointPolicy.PREFER_RECOMPUTE

            x = torch.randn(512, 512, requires_grad=True, device="cuda")
            y = torch.randn(512, 512, requires_grad=True, device="cuda")

            def fn(x, y):
                return torch.mm(x.cos(), y).sin().sum()

            def fn_ac(x, y):
                with apply_ac_policy("recompute_all"):
                    return fn(x, y)

            def fn_sac(x, y):
                with apply_ac_policy(policy_fn):
                    return fn(x, y)

            def fn_sac2(x, y):
                with apply_ac_policy("recompute_all"):
                    z = torch.mm(x.cos(), y)
                    tag_with_policy(z, CheckpointPolicy.MUST_SAVE)
                    return z.sin().sum()

            act_mem_noac = get_act_mem(lambda: fn(x, y))
            bw_flops_noac = get_bw_flops(lambda: fn(x, y))

            self.assertEqual(act_mem_noac, 2.0)
            self.assertEqual(bw_flops_noac, 2.0)

            act_mem_ac = get_act_mem(lambda: fn_ac(x, y))
            bw_flops_ac = get_bw_flops(lambda: fn_ac(x, y))

            self.assertEqual(act_mem_ac, 0.0)
            self.assertEqual(bw_flops_ac, 3.0)

            act_mem_sac = get_act_mem(lambda: fn_sac(x, y))
            bw_flops_sac = get_bw_flops(lambda: fn_sac(x, y))

            self.assertEqual(act_mem_sac, 1.0)
            self.assertEqual(bw_flops_sac, 2.0)

            act_mem_sac2 = get_act_mem(lambda: fn_sac2(x, y))
            bw_flops_sac2 = get_bw_flops(lambda: fn_sac2(x, y))

            self.assertEqual(act_mem_sac2, 1.0)
            self.assertEqual(bw_flops_sac2, 2.0)

    # Test peak `memory
    # Test multi-output non-optimality
    # Test Nested subclasses


if __name__ == "__main__":
    run_tests()
