import operator

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch.utils.checkpoint import CheckpointPolicy

aten = torch.ops.aten

# All-zero constructors // 전부 0인 텐서를 만드는 op
ZERO_SRC = {aten.zeros.default, aten.zeros_like.default, aten.new_zeros.default, aten._efficientzerotensor.default}
ZERO_FULL = {aten.full.default, aten.full_like.default}          # zero only when the fill value is 0 // 채움값 0일 때만
# zero in → zero out (first argument) // 첫 인자가 0이면 결과도 0
ZERO_PASS = {aten.t.default, aten.permute.default, aten.transpose.int, aten.view.default, aten.reshape.default,
             aten._unsafe_view.default, aten.expand.default, aten.clone.default, aten.neg.default, aten.squeeze.dim,
             aten.squeeze.dims, aten.squeeze.default, aten.unsqueeze.default, aten.slice.Tensor, aten.select.int,
             aten.alias.default, aten.detach.default, aten._to_copy.default, aten.sum.dim_IntList, aten.sum.default,
             aten.mul.Scalar}
ZERO_ANY = {aten.mm.default, aten.matmul.default, aten.bmm.default, aten.mul.Tensor}   # any zero operand → zero // 하나라도 0이면 0
ZERO_ALL = {aten.add.Tensor, aten.sub.Tensor}                    # x ± 0 → x, 0 + x → x, 0 ± 0 → 0
# Pointwise ops; with one tensor operand (the rest scalars / constant fills) the result is recomputed, not saved // 텐서 피연산자가 하나면 저장 대신 재계산
UNARY_PW = {aten.sigmoid.default, aten.neg.default, aten.exp.default, aten.tanh.default, aten.sin.default, aten.cos.default,
            aten.reciprocal.default, aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor, aten.add.Scalar,
            aten.sub.Scalar, aten.mul.Scalar, aten.div.Scalar, aten.rsub.Scalar, aten.pow.Tensor_Scalar}
CONST_FILL = ZERO_SRC | ZERO_FULL | {aten.ones.default, aten.ones_like.default}
SAME_VALUE = {aten.alias.default, aten.clone.default}            # value unchanged // 값이 그대로인 op
RESHAPE = {aten.view.default, aten.reshape.default, aten._unsafe_view.default}


def ValOf(node):
    return node.meta.get("val") if isinstance(node, torch.fx.Node) else None


def SameMeta(a, b):
    va, vb = ValOf(a), ValOf(b)
    return (isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor)
            and va.shape == vb.shape and va.dtype == vb.dtype and va.device == vb.device)


def IsCall(node, targets):
    return isinstance(node, torch.fx.Node) and node.op == "call_function" and node.target in targets


def SinkScalarAdd(graph):
    # unbind(x + c)[k] → unbind(x)[k] + c // 스택 전체에 더하던 상수를 조각마다 더한다
    #   FiLM γ = 1 + gb[:, :, 0] goes to the blocks through unbind. The sum over the whole (N, n_blocks, width)
    #   stack has n_blocks users, so Inductor materialized it (one read + one write of the stack) and the block
    #   kernels read it again. Per slice the add fuses into the block kernel, which reads γ straight from the
    #   cond GEMM output. // γ 스택을 따로 만들지 않고 블록 커널이 cond GEMM 출력에서 바로 읽는다
    for n in list(graph.nodes):
        if not IsCall(n, {aten.unbind.int}):
            continue
        src = n.args[0]
        while IsCall(src, SAME_VALUE) and len(src.users) == 1:
            src = src.args[0]
        if not (IsCall(src, {aten.add.Tensor}) and not src.kwargs and len(src.users) == 1
                and isinstance(src.args[1], (int, float)) and SameMeta(src, src.args[0])
                and all(IsCall(u, {operator.getitem}) for u in n.users)):
            continue
        base, c = src.args
        n.args = (base,) + tuple(n.args[1:])
        for g in list(n.users):
            with graph.inserting_after(g):
                a = graph.call_function(aten.add.Tensor, (g, c))
            a.meta.update(g.meta)
            g.replace_all_uses_with(a, delete_user_cb=lambda u, a=a: u is not a)


def PiecesOf(node):
    # view(cat([...], 1)) with no other user → the cat inputs // 다른 사용자가 없는 view(cat)이면 cat 입력 목록
    while IsCall(node, RESHAPE) and len(node.users) == 1:
        node = node.args[0]
    if IsCall(node, {aten.cat.default}) and len(node.users) == 1 and len(node.args) == 2 and node.args[1] == 1:
        return list(node.args[0])
    return None


def InterleaveScatter(graph):
    # select_scatter(0, A, 2, i) + select_scatter(0, B, 2, 1 - i), A = view(cat(A_k, 1)), B = view(cat(B_k, 1))
    #   → view(cat([A_0, B_0, A_1, B_1, ...], 1)) // 두 스택을 0 텐서에 끼워 더하는 대신 조각을 한 번에 잇는다
    #   This is the gradient of gb.view(N, n_blocks, 2, width)[:, :, i].unbind(1). Inductor wrote the γ and β
    #   pieces into two (N, n_blocks, width) stacks and one more kernel copied both into the (N, 2·n_blocks·width)
    #   tensor the cond GEMMs read (201 MB per 4096 rows). A cat of more than 8 inputs becomes a ConcatKernel:
    #   the producers write into its output, so that copy is gone. x + 0 → x, bit-identical.
    #   // 조각 생산 커널이 넓은 gradient 텐서에 바로 쓴다
    for n in list(graph.nodes):
        if not IsCall(n, {aten.add.Tensor}) or n.kwargs:
            continue
        v = ValOf(n)
        if not (isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[2] == 2
                and all(IsCall(x, {aten.select_scatter.default}) and len(x.users) == 1 and len(x.args) == 4
                        and IsCall(x.args[0], ZERO_SRC | ZERO_FULL) and x.args[2] == 2 for x in n.args)
                and {n.args[0].args[3], n.args[1].args[3]} == {0, 1}):
            continue
        if any(x.args[0].target in ZERO_FULL and x.args[0].args[1] != 0 for x in n.args):
            continue
        lo, hi = sorted(n.args, key=lambda x: x.args[3])
        xs, ys = PiecesOf(lo.args[1]), PiecesOf(hi.args[1])
        if xs is None or ys is None or len(xs) != v.shape[1] or len(ys) != v.shape[1]:
            continue
        if not all(isinstance(ValOf(t), torch.Tensor) and tuple(ValOf(t).shape) == (v.shape[0], v.shape[3]) for t in xs + ys):
            continue
        with graph.inserting_before(n):
            c = graph.call_function(aten.cat.default, ([t for pair in zip(xs, ys) for t in pair], 1))
            r = graph.call_function(aten.view.default, (c, list(v.shape)))
        c.meta["val"] = v.reshape(v.shape[0], -1)
        r.meta.update(n.meta)
        n.replace_all_uses_with(r)


def TagUnaryRecompute(graph):
    # Pointwise function of one tensor → must-recompute: the partitioner saves that tensor, not the result
    # // 텐서 하나의 원소별 함수는 저장하지 않고 backward에서 입력으로부터 다시 계산하게 표시
    #   The elementwise kernels are memory-bound (VRAM ≈ 610 GB/s, a 4 MB stream ≈ 7 us; SiLU + derivative costs
    #   the same as a plain add). Saving f(x) is one store in the forward and one load in the backward, and x is
    #   saved anyway (GEMM outputs, cond GEMM output), so recomputing f removes both. film 256×12 physics graph:
    #   1 − σ(z) per block and the per-block γ are no longer saved; the forward block kernel goes from 7 loads
    #   + 3 stores to 6 + 2. The saved set changes the fused kernels, so gradients differ by rounding.
    #   // 원소별 커널은 메모리 한계라 재계산이 공짜다
    for n in graph.nodes:
        if not IsCall(n, UNARY_PW) or n.kwargs or "recompute" in n.meta:
            continue
        ts = {a for a in n.args if isinstance(a, torch.fx.Node) and not IsCall(a, CONST_FILL)}
        if len(ts) == 1 and SameMeta(n, next(iter(ts))):
            n.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE


class ZeroFold(CustomGraphPass):
    # Inductor joint-graph pass: fold matmul / add whose operand is an all-zero constant // 0 상수 피연산자를 가진 matmul·add 접기
    # tf.jvp gives closure-captured weights a zero tangent (trap 27), and the traced tangent rule
    #   dz = dh·Wᵀ + h·dWᵀ keeps the second term as a real zeros(out, in) GEMM in the compiled forward
    #   and once more in the backward. Eager skips it through ZeroTensor dispatch; Inductor runs it.
    #   film 256×12: 30 mm per jvp graph (12 fwd + 12 bwd trunk, 6 small) = 120 of ~1070 GEMMs per step.
    #   Measured 2026-10-12 (R9700, config 2026_10_11_21_56_36): step 81.3 → 73.7 ms; first-step
    #   gradients 30 / 38 tensors bit-identical, 1.05e-8 relative overall.
    # The explicit zero tangent of the FiLM cond (StateDerivs) reaches the block kernels through
    #   unbind → getitem. Folding it drops the (N, n_blocks, width) zero fill and one read stream per
    #   block kernel: 73.7 → 72.2 ms. Every fused kernel changes, so gradients differ by rounding:
    #   1.7e-6 relative to the unfolded graph (eager vs compiled is 1.0e-4).
    #   // FiLM cond의 명시적 0 tangent도 unbind를 지나 접는다 — 반올림 수준 차이
    # 0·x is folded to 0 for every x, so a non-finite x no longer makes these terms NaN — the primal
    #   path still carries it. // 비유한 값에서만 결과가 달라진다 (primal 경로는 그대로 NaN 전파)
    # uuid() hashes this file so Inductor's graph cache stays on and any edit here invalidates it.
    #   // 이 파일 해시가 캐시 키에 들어간다
    def __call__(self, graph):
        zero = set()        # nodes known to be all zero // 전부 0으로 확정된 노드
        todo = []           # (node, replacement) ; None = replace with a zero constant // None이면 0 상수로 교체

        def IsZero(a):
            return isinstance(a, torch.fx.Node) and a in zero

        for n in graph.nodes:
            if n.op != "call_function":
                continue
            t, a = n.target, n.args
            if t in ZERO_SRC or (t in ZERO_FULL and len(a) > 1 and isinstance(a[1], (int, float)) and a[1] == 0):
                zero.add(n)
            elif t is aten.unbind.int and IsZero(a[0]):
                zero.add(n)     # tuple of zeros: its getitem users are folded below // 0 튜플 — getitem에서 접힌다
            elif (t in ZERO_PASS and IsZero(a[0])) or (t in ZERO_ANY and (IsZero(a[0]) or IsZero(a[1]))) \
                    or (t is operator.getitem and IsZero(a[0])) \
                    or (t is aten.cat.default and all(IsZero(x) for x in a[0])):
                zero.add(n)
                todo.append((n, None))
            elif t in ZERO_ALL and not n.kwargs:
                if IsZero(a[0]) and IsZero(a[1]):
                    zero.add(n)
                    todo.append((n, None))
                elif IsZero(a[1]) and SameMeta(n, a[0]):
                    todo.append((n, a[0]))
                elif IsZero(a[0]) and SameMeta(n, a[1]) and t is aten.add.Tensor:
                    todo.append((n, a[1]))
            elif t is aten.addmm.default and (IsZero(a[1]) or IsZero(a[2])) and n.kwargs.get("beta", 1) == 1:
                if IsZero(a[0]):
                    zero.add(n)
                    todo.append((n, None))
                elif SameMeta(n, a[0]):
                    todo.append((n, a[0]))

        for n, rep in todo:
            if rep is None:
                v = ValOf(n)
                if not isinstance(v, torch.Tensor) or not n.users:
                    continue
                with graph.inserting_before(n):
                    rep = graph.call_function(aten.full.default, (list(v.shape), 0), {"dtype": v.dtype, "device": v.device})
                rep.meta.update(n.meta)
            n.replace_all_uses_with(rep)
        graph.eliminate_dead_code()

    def uuid(self):
        return get_hash_for_files((__file__,))


class JointPass(ZeroFold):
    # ZeroFold, then the rewrites that cut the bytes the elementwise kernels move // ZeroFold + 원소별 커널의 데이터 이동량을 줄이는 재작성
    #   Measured 2026-10-12 (R9700, config 2026_10_11_21_56_36, 30 steps × 4, two interleaved rounds): step
    #   71.8 → 66.9 ms; Triton kernels 13.98 → 10.14 GB per step, 19.6 → 14.5 ms of GPU time. First-step
    #   gradients against ZeroFold alone: scatter + sink bit-identical (with the sunk add recomputed),
    #   recompute tags 1.8e-6 relative (eager vs compiled is 1.0e-4 either way). Finetune physics graph
    #   (FlipAdapter), same state: 4.3e-7 against ZeroFold, 2.8e-6 against eager (ZeroFold vs eager 3.2e-6).
    #   Tagged nodes make the partitioner reorder the backward like the autograd engine (recompute right before
    #   use). // 표시된 노드가 있으면 backward 순서가 autograd 엔진 순서로 재배열된다
    def __call__(self, graph):
        super().__call__(graph)
        SinkScalarAdd(graph)
        InterleaveScatter(graph)
        graph.eliminate_dead_code()
        TagUnaryRecompute(graph)
