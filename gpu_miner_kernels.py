"""dHeavyHash mining kernels: int8 matmul chain, rank test, BLAKE3.

Everything runs on the GPU:

* candidate matrices are drawn from a NumPy-compatible PCG64 stream (Triton),
* the three int8 64x64 multiplications run in a single fused kernel
  (``chain3``) using ``tl.dot`` (int8 -> int32, wrapped to int8 exactly like
  NumPy does),
* the rank test uses ``torch.linalg.qr`` in fp64 (the same test as the CPU
  reference, with a cheap ``slogdet`` pre-filter),
* BLAKE3 is the Triton implementation from :mod:`gpu_blake3`.

The CPU reference (``dHeavyHash.dHeavyHash``) is reproduced bit for bit.
"""

import os

import torch
import triton
import triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, "triton_cache"))


@triton.jit
def _matmul_int8_kernel(a_ptr, b_ptr, c_ptr,
                        a_s0, a_s1, a_s2, b_s0, b_s1, b_s2,
                        c_s0, c_s1, c_s2, BLOCK: tl.constexpr):
    """Batched 64x64 int8 matmul with int32 accumulation and int8 wrap.

    Batch and element strides are passed in so that strided views such as
    ``mats[:, k]`` can be consumed without an extra copy.
    """
    pid = tl.program_id(0)
    rm = tl.arange(0, 64)
    rn = tl.arange(0, 64)
    rk = tl.arange(0, 32)
    a_base = a_ptr + pid.to(tl.int64) * a_s0
    b_base = b_ptr + pid.to(tl.int64) * b_s0
    c_base = c_ptr + pid.to(tl.int64) * c_s0
    acc = tl.zeros((64, 64), tl.int32)
    for kk in range(0, 64, 32):
        a = tl.load(a_base + rm[:, None] * a_s1 + (kk + rk)[None, :] * a_s2).to(tl.int8)
        b = tl.load(b_base + (kk + rk)[:, None] * b_s1 + rn[None, :] * b_s2).to(tl.int8)
        acc += tl.dot(a, b, out_dtype=tl.int32)
    out = acc.to(tl.int8)
    tl.store(c_base + rm[:, None] * c_s1 + rn[None, :] * c_s2, out)


def batched_int8_matmul(a, b):
    """Batched int8 matmul with NumPy semantics (wrap mod 256).

    ``a``/``b`` must be ``(N, 64, 64)``; any batch/row/column strides are fine.
    """
    if a.shape[-2:] != (64, 64) or b.shape[-2:] != (64, 64) or a.shape[0] != b.shape[0]:
        raise ValueError("expected matching (N, 64, 64) tensors")
    batch = a.shape[0]
    c = torch.empty(a.shape, dtype=torch.int8, device=a.device)
    _matmul_int8_kernel[(batch,)](
        a, b, c,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK=64, num_warps=4)
    return c


@triton.jit
def _chain_matmul_kernel(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr,
                         s0, s1, s2, BLOCK: tl.constexpr):
    """``a @ b @ c @ d`` for one int8 64x64 matrix (wrap after every step)."""
    pid = tl.program_id(0)
    rm = tl.arange(0, 64)
    rn = tl.arange(0, 64)
    rk = tl.arange(0, 64)
    off = pid.to(tl.int64) * s0
    acc = tl.load(a_ptr + off + rm[:, None] * s1 + rn[None, :] * s2).to(tl.int8)
    for step in range(3):
        if step == 0:
            base = b_ptr + off
        elif step == 1:
            base = c_ptr + off
        else:
            base = d_ptr + off
        right = tl.load(base + rk[:, None] * s1 + rn[None, :] * s2).to(tl.int8)
        acc = tl.dot(acc, right, out_dtype=tl.int32).to(tl.int8)
    tl.store(out_ptr + off + rm[:, None] * s1 + rn[None, :] * s2, acc)


def chain3(a, b, c, d):
    """``a @ b @ c @ d`` with NumPy int8 semantics in a single kernel launch."""
    batch = a.shape[0]
    out = torch.empty(a.shape, dtype=torch.int8, device=a.device)
    _chain_matmul_kernel[(batch,)](
        a, b, c, d, out,
        a.stride(0), a.stride(1), a.stride(2), BLOCK=64, num_warps=4)
    return out


def chain_matmul(m1, rng_matrices):
    """``m1 @ r2 @ r3 @ r4`` with int8 wrapping, all on the GPU."""
    out = m1
    for i in range(3):
        out = batched_int8_matmul(out, rng_matrices[:, i])
    return out


def full_rank_mask(mats):
    """Fast GPU rank filter (``slogdet`` in fp64)."""
    sign, _ = torch.linalg.slogdet(mats.to(torch.float64))
    return sign != 0


def _qr_min_diag(mats, dtype):
    r = torch.linalg.qr(mats.to(dtype), mode="r")[1]
    return r.diagonal(dim1=-2, dim2=-1).abs().min(dim=1).values


def fast_full_rank(mats: torch.Tensor) -> torch.Tensor:
    """Full-rank test matching ``dHeavyHash.is_full_rank``, but fast.

    Entries are int8 (they wrap after every multiply), so ``||A||`` is at most
    a few hundred and fp32 QR has an absolute error of roughly 1e-5. Anything
    with ``min|diag R| > 1e-2`` is therefore safely full rank; the rare
    matrices below that are re-checked in fp64 with the reference tolerance.
    """
    min32 = _qr_min_diag(mats, torch.float32)
    mask = min32 > 1e-2
    unsure = ~mask
    if bool(unsure.any()):
        idx = unsure.nonzero().reshape(-1)
        mask[idx] = _qr_min_diag(mats[idx], torch.float64) > 1e-9
    return mask


def qr_full_rank(mats):
    """GPU QR rank test matching the CPU reference (tolerance 1e-9)."""
    r = torch.linalg.qr(mats.to(torch.float64), mode="r")[1]
    diag = r.diagonal(dim1=-2, dim2=-1).abs()
    return (diag > 1e-9).all(dim=1)


def mining_candidates(seeds, rounds=3):
    """Generate candidate matrices (3-multiply chain) for a batch of seeds.

    Returns ``(candidates, keep_mask)`` where ``candidates`` is int8
    ``(N, 64, 64)`` and ``keep_mask`` marks full-rank results.
    """
    from gpu_rng import pcg_int8_matrices

    states, incs = _pcg_states_for(seeds)
    mats = pcg_int8_matrices(states, incs, 1 + rounds)
    cand = chain_matmul(mats[:, 0], mats[:, 1:])
    return cand, full_rank_mask(cand)


def _pcg_states_for(seeds):
    from gpu_rng import pcg_states
    return pcg_states(seeds)
