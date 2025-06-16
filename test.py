import contextlib
import functools
import unittest

from ac_experimental.ac import is_checkpoint_enabled, push_policy
import torch

from ac_experimental import apply_ac_policy, SAVE_WITH_HOOKS, tag_with_policy, apply_ac_policy_fn
from torch.overrides import TorchFunctionMode
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.checkpoint import CheckpointPolicy
from torch.utils.flop_counter import FlopCounterMode
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm


class EagerRecordGraphAndInputs:
    def __init__(self) -> None:
        self.graphs = []
        self.example_inputs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        self.example_inputs.append(example_inputs)
        return gm


@contextlib.contextmanager
def allow_ambient_mode_to_run_first():
    # Temporarily allow user modes to interpose below AC mode for testing purposes
    import ac_experimental

    try:
        ac_experimental._tracer_is_infra_mode = False
        yield
    finally:
        ac_experimental._tracer_is_infra_mode = True


# This might take a while to land in cores
def skip_if_no_hopify_generic_context_manager(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(torch._dynamo.config, "_enable_hopify_generic_context_manager"):
            raise unittest.SkipTest("HopifyGenericContextManager not enabled")
        return func(*args, **kwargs)
    return wrapper


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

    def test_with_kwargs(self):
        def fn(a):
            # If the kwarg for to_copy is not preserved, we get the error:
            # view_as_complex is only supported for half, float and double tensors
            return torch.view_as_complex(a.float()).sin()

        a = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16, requires_grad=True)

        with apply_ac_policy("recompute_all"):
            out = fn(a)

        out.sum().real.backward()

    def test_multi_output_with_tensor_tagging(self):
        def fn(x):
            a, b = x.split_with_sizes([2, 3], dim=0)
            x = x * a.sum() * b.sum()
            var, mean = torch.var_mean(x, dim=0)
            tag_with_policy(var, policy=CheckpointPolicy.MUST_SAVE)
            return var + mean

        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)

        with apply_ac_policy(policy_fn="recompute_all"):
            out = fn(a)

        (grad,) = torch.autograd.grad(out, a)
        (grad_ref,) = torch.autograd.grad(fn(a), a)

        torch.equal(grad, grad_ref)

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
        # This is testing that the dispatcher mode happens above, so that policy_fn
        if not hasattr(torch._C._TorchDispatchModeKey, "AC_TRACER"):
            raise unittest.SkipTest("AC_TRACER Infra mode has not landed")
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

    @skip_if_no_hopify_generic_context_manager
    def test_compile_with_context_manager(self):
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

    @skip_if_no_hopify_generic_context_manager
    def test_tag_with_policy_compile_with_context_manager(self):
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

    @skip_if_no_hopify_generic_context_manager
    def test_user_allow_in_graph_with_context_manager(self):
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

    def test_compile_simple_policy_fn_with_context_manager(self):
        if not hasattr(torch._dynamo.config, "_enable_hopify_generic_context_manager"):
            raise unittest.SkipTest("HopifyGenericContextManager not enabled")
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

    def test_nn_module_buffer_mutation(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.zeros(2, 2))

            def forward(self, x):
                self.buffer += x
                return self.buffer.sin().cos().sum()

        a = torch.ones((2, 2), requires_grad=True)

        m = MyModule()

        with apply_ac_policy("recompute_all"):
            out = m(a)

        (grad,) = torch.autograd.grad(out, a)

        m = MyModule()
        out_ref = m(a)
        (grad_ref,) = torch.autograd.grad(out_ref, a)

        self.assertEqual(grad, grad_ref)

    def test_nn_module_buffer_mutation_compile(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.zeros(2, 2))

            def forward(self, x):
                self.buffer += x
                return self.buffer.sin().cos().sum()

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def compiled_fn(m, x):
            return apply_ac_policy_fn(m, x, policy_fn="recompute_all")

        a = torch.ones((2, 2), requires_grad=True)

        # Test compiled version
        m = MyModule()
        out = compiled_fn(m, a)
        (grad,) = torch.autograd.grad(out, a)

        # Reference (eager version)
        m_ref = MyModule()
        with apply_ac_policy("recompute_all"):
            out_ref = m_ref(a)
        (grad_ref,) = torch.autograd.grad(out_ref, a)

        self.assertEqual(grad, grad_ref)

    # The same tests as above, but with the wrapper version of the API
    def test_compile_nesting(self):
        def g(x):
            return x.sin().cos().sin()

        def g_ac(x):
            return apply_ac_policy_fn(g, x, policy_fn="recompute_all")

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def f(x):
            return g(x.exp().exp()) * 10 * 10 * 10 * 10

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def f_ac(x):
            # Is PREFER_SAVE the neutral policy that we are looking for?
            return apply_ac_policy_fn(g_ac, x.exp().exp(), policy_fn="must_save_all") * 10 * 10 * 10 * 10

        a = torch.rand((2, 2), requires_grad=True)
        f_ac(a).sum().backward()
        b = a.detach().clone().requires_grad_(True)
        f(b).sum().backward()
        self.assertEqual(a.grad, b.grad)

    def test_tag_with_policy_compile(self):
        def g(x):
            cos = x.sin().sin().cos()
            z = cos @ cos
            tag_with_policy(z, CheckpointPolicy.MUST_SAVE)
            out = z.sin().sin() * 10 * 10 * 10 * 10
            return out

        def g2(x):
            # Same function but without tagging
            cos = x.sin().sin().cos()
            z = cos @ cos
            out = z.sin().sin() * 10 * 10 * 10 * 10
            return out

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def f(x):
            return apply_ac_policy_fn(g, x, policy_fn="recompute_all")

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def f2(x):
            return apply_ac_policy_fn(g2, x, policy_fn="recompute_all")

        # Test the case where there is no outer AC context
        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def f3(x):
            cos = x.sin().sin().cos()
            z = cos @ cos
            tag_with_policy(z, CheckpointPolicy.MUST_SAVE)
            out = z.sin().sin() * 10 * 10 * 10 * 10
            return out

        a = torch.rand((2, 2), requires_grad=True)
        f(a).sum().backward()

        b = a.detach().clone().requires_grad_(True)
        f2(b).sum().backward()

        c = a.detach().clone().requires_grad_(True)
        f3(c).sum().backward()

        self.assertEqual(a.grad, b.grad)
        self.assertEqual(b.grad, c.grad)

    def test_adjacent_ac(self):
        # tests the case where two recompute_all regions are next to one another
        # it tests to see that we properly save the inputs to both of the regions
        def f(x):
            return x.sin().cos() * 2

        def g(x):
            return x.exp().exp() * 3

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def h(x):
            out = apply_ac_policy_fn(f, x, policy_fn="recompute_all")
            out = apply_ac_policy_fn(g, out, policy_fn="recompute_all")
            return out

        a = torch.rand((2, 2), requires_grad=True)
        out = h(a)
        out.sum().backward()

    def test_user_allow_in_graph(self):
        @torch._dynamo.allow_in_graph
        def test_fn(x):
            return x.clone(), x.clone()

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def fn(x):
            a, b = apply_ac_policy_fn(test_fn, x, policy_fn="recompute_all")
            c = a + b
            return c.sin()

        a = torch.rand((2, 2), requires_grad=True)
        out = fn(a)
        out.sum().backward()

    def test_compile_simple_policy_fn(self):
        def policy_fn(ctx, out, op, *args, **kwargs) -> CheckpointPolicy:
            if op == torch.ops.aten.cos.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def g(x):
            return x.sin().cos().sin().exp() * 10

        @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
        def f(x):
            return apply_ac_policy_fn(g, x, policy_fn=policy_fn)

        a = torch.rand((4, 4), requires_grad=True)
        out = f(a)
        out.sum().backward()

    def test_saving_inference_mode_tensor_errors(self):
        b = torch.tensor(2., requires_grad=True)

        with apply_ac_policy("recompute_all"):
            c = b.sin()
            with torch.inference_mode():
                d = c.exp()
                with self.assertRaisesRegex(
                    RuntimeError, "Saving inference tensors is not supported"
                ):
                    tag_with_policy(d, CheckpointPolicy.MUST_SAVE)

    def test_batch_norm(self):

        pass

    def test_custom_policy_with_state(self):
        # This doesn't test the exact setup, mainly the 
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        _save_list = {
            torch.ops.aten.cos.default,
            torch.ops.aten.sin.default,
        }
        # If you want your policy to have state, pass a class. Make sure to
        # create it in global scope to avoid new instances triggering recompiles.
        class CustomPolicy:
            def __init__(self):
                super().__init__()
                self.meta = dict()

            def __call__(self, ctx, out, func, *args, **kwargs):
                cos_count_key = f"cos_count"
                if func == torch.ops.aten.cos.default:
                    self.meta[cos_count_key] = self.meta.get(cos_count_key, 0) + 1

                to_save = func in _save_list and not (
                    func == torch.ops.aten.cos.default and self.meta[cos_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

        def f(x):
            return x.sin().cos().sin().cos().exp() * 10

        def f_ac(x):
            return apply_ac_policy_fn(f, x, policy_fn=CustomPolicy, is_factory=True)

        a = torch.rand((2, 2), requires_grad=True)
        out = f_ac(a).sum()
        out.backward()

        def context_fn():
            policy = CustomPolicy()
            return create_selective_checkpoint_contexts(policy)

        b = a.detach().clone().requires_grad_(True)

        def f_ac_2(x):
            return torch.utils.checkpoint.checkpoint(
                f, b,
                use_reentrant=False,
                context_fn=context_fn,
            )
        out_ref = f_ac_2(a).sum()
        self.assertEqual(out, out_ref)
        out_ref.backward()
        self.assertEqual(a.grad, b.grad)

        compiled_f = torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)(f_ac)
        compiled_f_2 = torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)(f_ac_2)
        out = compiled_f(a).sum()
        out.backward()
        out_ref = compiled_f_2(a).sum()
        self.assertEqual(out, out_ref)
        out_ref.backward()
        self.assertEqual(a.grad, b.grad)

    # We want to support detecting whether we are in a checkpoint region.
    # Just seeing if we've entered into checkpoint is not enough because
    # the policy could be to SAVE_ALL. The more accurate thing to do is
    # to actually refer to the global policy stack and see if the current
    # policy is to save for the particular op in question.
    #
    # This is hard to do. If the policy stack were updated inside the HOP wrapper,
    # Since that logic is not run during dynamo, if I'm observing in the
    # HOP body I wouldn't see the update.
    # We could fix that by letting dynamo trace the policy stack push/pop.
    # But this also doens't  work though because the code observing
    # the side effect (the torch dispatch mode) because side effects traced by
    # dynamo are not observable in aot autograd tracing.
    #
    # Today what we do is update in parallel a "is in checkpoint" flag that
    # the actual implementation in aot autograd doesn't rely on.
    def test_branch_on_is_in_ac_region(self):
        # What happens in the AC outside graph case?
        # Note: Dynamo is tracing through the policy_fn now
        # from ac_experimental import is_save_policy, current_global_policy
        from ac_experimental import is_checkpoint_enabled

        def test_fn(x):
            # if not is_save_policy(current_global_policy(None, None, torch.ops.aten.sin.default, (None,), {})):
            if is_checkpoint_enabled():
                return x.sin().sin()
            else:
                return x.cos(), x.cos()

        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            a, b = apply_ac_policy_fn(test_fn, x, policy_fn="recompute_all")
            c = a + b
            return c.exp()

        # Next: actually inspect the graph to see if there are sin
        a = torch.rand((2, 2), requires_grad=True)
        out = fn(a)
        out.sum().backward()

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.example_inputs), 1)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 2]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        dynamo_bypassing_wrapper = torch.ops.higher_order.dynamo_bypassing_wrapper('_dynamo_bypassing_wrapper_fn', wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[2, 2]" = dynamo_bypassing_wrapper[0];  dynamo_bypassing_wrapper = None

        a: "f32[2]" = getitem[0]
        b: "f32[2]" = getitem[1];  getitem = None

        c: "f32[2]" = a + b;  a = b = None

        exp: "f32[2]" = c.exp();  c = None
        return (exp,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[2, 2]"):
            sin: "f32[2, 2]" = l_x_.sin();  l_x_ = None
            sin_1: "f32[2, 2]" = sin.sin();  sin = None
            return (sin_1,)
""",
        )

        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def fn_no_ac(x):
            a, b = test_fn(x)
            c = a + b
            return c.exp()

        # Next: actually inspect the graph to see if there are sin
        a = torch.rand((2, 2), requires_grad=True)
        out = fn_no_ac(a)
        out.sum().backward()

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 4)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.example_inputs), 1)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 2]"):
        l_x_ = L_x_

        a: "f32[2, 2]" = l_x_.cos()
        b: "f32[2, 2]" = l_x_.cos();  l_x_ = None

        c: "f32[2, 2]" = a + b;  a = b = None

        exp: "f32[2, 2]" = c.exp();  c = None
        return (exp,)
""",
        )

    def test_randomness_errors(self):
        with self.assertRaisesRegex(
            RuntimeError, "recomputing RNG ops are not supported"
        ):
            with apply_ac_policy("recompute_all"):
               torch.rand(10, 10)

        # Saving the RNG op is OK
        with apply_ac_policy("must_save_all"):
            torch.rand(10, 10)


    # Test multi-output non-optimality
    # Test Nested subclasses


if __name__ == "__main__":
    run_tests()
