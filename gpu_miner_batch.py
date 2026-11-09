"""Batched GPU dHeavyHash: many headers mined simultaneously.

Single-hash GPU latency is dominated by kernel launch overhead, so the fair
comparison with the CPU implementation is throughput on a batch of headers.
This module vectorises the whole pipeline over a batch:

  headers (B, L) -> BLAKE3e (B,32) -> BLAKE3 -> seeds (B,32)
  -> SeedSequence + PCG64 seeding (B lanes, Triton)
  -> 4 random int8 matrices per lane (Triton PCG64)
  -> m0 @ m1 @ m2 @ m3 (Triton, int8 -> int32 -> wrap)
  -> full-rank check (PyTorch QR fp64)
  -> BLAKE3e of each 4096-byte matrix (Triton)

Each lane reproduces the CPU candidate stream of its own header, so the
digests are the same as ``dHeavyHash.dHeavyHash``.
"""

import os
import time

import torch
import triton
import triton.language as tl

from gpu_blake3 import blake3_gpu, blake3e_gpu
from gpu_miner_kernels import chain3, fast_full_rank
from gpu_rng import (PCG_MULT_HI, PCG_MULT_LO, SS_INIT_A_C, SS_INIT_B_C,
                     SS_MIX_L_C, SS_MIX_R_C, SS_MULT_A_C, SS_MULT_B_C,
                     SS_XSHIFT_C, _mul128, pcg_jump_coefficients)

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, "triton_cache"))

PERSON = b"dHeavyHash"


@triton.jit
def _bswap32v(x):
    return (((x & 0xFF) << 24) | ((x & 0xFF00) << 8)
            | ((x >> 8) & 0xFF00) | (x >> 24)).to(tl.uint32)


@triton.jit
def _hashmixv(value, hc):
    v = value ^ hc
    hc = hc * SS_MULT_A_C
    v = v * hc
    return v ^ (v >> SS_XSHIFT_C), hc


@triton.jit
def _mix32v(x, y):
    r = SS_MIX_L_C * x - SS_MIX_R_C * y
    return r ^ (r >> SS_XSHIFT_C)


@triton.jit
def _seed_sequence_batch_kernel(digest_ptr, words_ptr, B, BLOCK: tl.constexpr):
    """Vectorised ``SeedSequence(big_endian(digest)).generate_state(8)``."""
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < B
    base = lane.to(tl.int64) * 8
    u0 = tl.load(digest_ptr + base + 0, mask=mask, other=0)
    u1 = tl.load(digest_ptr + base + 1, mask=mask, other=0)
    u2 = tl.load(digest_ptr + base + 2, mask=mask, other=0)
    u3 = tl.load(digest_ptr + base + 3, mask=mask, other=0)
    u4 = tl.load(digest_ptr + base + 4, mask=mask, other=0)
    u5 = tl.load(digest_ptr + base + 5, mask=mask, other=0)
    u6 = tl.load(digest_ptr + base + 6, mask=mask, other=0)
    u7 = tl.load(digest_ptr + base + 7, mask=mask, other=0)

    e0 = _bswap32v(u7)
    e1 = _bswap32v(u6)
    e2 = _bswap32v(u5)
    e3 = _bswap32v(u4)
    e4 = _bswap32v(u3)
    e5 = _bswap32v(u2)
    e6 = _bswap32v(u1)
    e7 = _bswap32v(u0)

    zero = tl.zeros((BLOCK,), tl.uint32)
    nw = zero + 1
    nw = tl.where(e1 != 0, zero + 2, nw)
    nw = tl.where(e2 != 0, zero + 3, nw)
    nw = tl.where(e3 != 0, zero + 4, nw)
    nw = tl.where(e4 != 0, zero + 5, nw)
    nw = tl.where(e5 != 0, zero + 6, nw)
    nw = tl.where(e6 != 0, zero + 7, nw)
    nw = tl.where(e7 != 0, zero + 8, nw)

    hc = zero + SS_INIT_A_C
    m0, hc = _hashmixv(e0, hc)
    m1, hc = _hashmixv(e1, hc)
    m2, hc = _hashmixv(e2, hc)
    m3, hc = _hashmixv(e3, hc)

    t, hc = _hashmixv(m0, hc)
    m1 = _mix32v(m1, t)
    t, hc = _hashmixv(m0, hc)
    m2 = _mix32v(m2, t)
    t, hc = _hashmixv(m0, hc)
    m3 = _mix32v(m3, t)
    t, hc = _hashmixv(m1, hc)
    m0 = _mix32v(m0, t)
    t, hc = _hashmixv(m1, hc)
    m2 = _mix32v(m2, t)
    t, hc = _hashmixv(m1, hc)
    m3 = _mix32v(m3, t)
    t, hc = _hashmixv(m2, hc)
    m0 = _mix32v(m0, t)
    t, hc = _hashmixv(m2, hc)
    m1 = _mix32v(m1, t)
    t, hc = _hashmixv(m2, hc)
    m3 = _mix32v(m3, t)
    t, hc = _hashmixv(m3, hc)
    m0 = _mix32v(m0, t)
    t, hc = _hashmixv(m3, hc)
    m1 = _mix32v(m1, t)
    t, hc = _hashmixv(m3, hc)
    m2 = _mix32v(m2, t)

    for src in tl.static_range(4, 8):
        e = tl.where(src == 4, e4, e0)
        e = tl.where(src == 5, e5, e)
        e = tl.where(src == 6, e6, e)
        e = tl.where(src == 7, e7, e)
        active = nw > src
        t, hnew = _hashmixv(e, hc)
        hc = tl.where(active, hnew, hc)
        m0 = tl.where(active, _mix32v(m0, t), m0)
        t, hnew = _hashmixv(e, hc)
        hc = tl.where(active, hnew, hc)
        m1 = tl.where(active, _mix32v(m1, t), m1)
        t, hnew = _hashmixv(e, hc)
        hc = tl.where(active, hnew, hc)
        m2 = tl.where(active, _mix32v(m2, t), m2)
        t, hnew = _hashmixv(e, hc)
        hc = tl.where(active, hnew, hc)
        m3 = tl.where(active, _mix32v(m3, t), m3)

    hb = zero + SS_INIT_B_C
    for j in tl.static_range(8):
        m = m0
        m = tl.where(j % 4 == 0, m0, m)
        m = tl.where(j % 4 == 1, m1, m)
        m = tl.where(j % 4 == 2, m2, m)
        m = tl.where(j % 4 == 3, m3, m)
        v = m ^ hb
        hb = hb * SS_MULT_B_C
        v = v * hb
        tl.store(words_ptr + lane * 8 + j, v ^ (v >> SS_XSHIFT_C), mask=mask)


@triton.jit
def _pcg_init_batch_kernel(words_ptr, state_ptr, inc_ptr, B,
                           BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < B
    w0 = tl.load(words_ptr + lane * 8 + 0, mask=mask, other=0)
    w1 = tl.load(words_ptr + lane * 8 + 1, mask=mask, other=0)
    w2 = tl.load(words_ptr + lane * 8 + 2, mask=mask, other=0)
    w3 = tl.load(words_ptr + lane * 8 + 3, mask=mask, other=0)
    w4 = tl.load(words_ptr + lane * 8 + 4, mask=mask, other=0)
    w5 = tl.load(words_ptr + lane * 8 + 5, mask=mask, other=0)
    w6 = tl.load(words_ptr + lane * 8 + 6, mask=mask, other=0)
    w7 = tl.load(words_ptr + lane * 8 + 7, mask=mask, other=0)

    is0 = w2.to(tl.uint64) | (w3.to(tl.uint64) << 32)
    is1 = w0.to(tl.uint64) | (w1.to(tl.uint64) << 32)
    q0 = w6.to(tl.uint64) | (w7.to(tl.uint64) << 32)
    q1 = w4.to(tl.uint64) | (w5.to(tl.uint64) << 32)

    i0 = (q0 << 1) | 1
    i1 = (q1 << 1) | (q0 >> 63)

    t0 = i0 + is0
    cy = tl.where(t0 < i0, tl.full((BLOCK,), 1, tl.uint64),
                  tl.full((BLOCK,), 0, tl.uint64))
    t1 = i1 + is1 + cy

    mlo = tl.full((BLOCK,), PCG_MULT_LO.value, tl.uint64)
    mhi = tl.full((BLOCK,), PCG_MULT_HI.value, tl.uint64)
    p0, p1 = _mul128(t0, t1, mlo, mhi)
    s0 = p0 + i0
    cy2 = tl.where(s0 < p0, tl.full((BLOCK,), 1, tl.uint64),
                   tl.full((BLOCK,), 0, tl.uint64))
    s1 = p1 + i1 + cy2

    tl.store(state_ptr + lane * 2 + 0, s0, mask=mask)
    tl.store(state_ptr + lane * 2 + 1, s1, mask=mask)
    tl.store(inc_ptr + lane * 2 + 0, i0, mask=mask)
    tl.store(inc_ptr + lane * 2 + 1, i1, mask=mask)


def seed_digests(headers: torch.Tensor) -> torch.Tensor:
    """``BLAKE3(BLAKE3e(header) + b"dHeavyHash")`` for a batch of headers."""
    inner = blake3e_gpu(headers)
    person = torch.tensor(list(PERSON), dtype=torch.uint8, device=headers.device)
    person = person.unsqueeze(0).expand(headers.shape[0], -1)
    return blake3_gpu(torch.cat([inner, person], dim=1)).contiguous()


def pcg_states_batch(digests: torch.Tensor, words_per_lane: int = 2048):
    """Per-lane PCG64 states, seeded fully on the GPU."""
    b = digests.shape[0]
    dev = digests.device
    words = torch.empty((b, 8), dtype=torch.uint32, device=dev)
    _seed_sequence_batch_kernel[(triton.cdiv(b, 128),)](
        digests.view(torch.uint32), words, b, BLOCK=128, num_warps=4)
    state = torch.empty((b, 2), dtype=torch.uint64, device=dev)
    inc = torch.empty((b, 2), dtype=torch.uint64, device=dev)
    _pcg_init_batch_kernel[(triton.cdiv(b, 128),)](
        words, state, inc, b, BLOCK=128, num_warps=4)
    return state, inc


def pcg_states_after(state, inc, words: int = 2048):
    """Jump every lane's PCG64 state forward by ``words`` draws."""
    from gpu_rng import pcg_jump_kernel
    b = state.shape[0]
    a, c = pcg_jump_coefficients(words)
    out = torch.empty_like(state)
    pcg_jump_kernel[(triton.cdiv(b, 64),)](
        state.reshape(-1), inc.reshape(-1), out.reshape(-1), b,
        a & ((1 << 64) - 1), a >> 64, c & ((1 << 64) - 1), c >> 64,
        b, 64, num_warps=4)
    return out


@torch.no_grad()
def mine_batch_torch(headers: torch.Tensor):
    """Same batched pipeline, but matmul and hashing run on PyTorch ops."""
    from gpu_blake3 import blake3e_torch
    from gpu_rng import pcg_int8_matrices

    digests = seed_digests(headers)
    state, inc = pcg_states_batch(digests)
    b = headers.shape[0]
    mats = pcg_int8_matrices(state, inc, 4).view(b, 4, 64, 64)
    mats = mats.permute(1, 0, 2, 3).contiguous()
    x = torch.bmm(mats[0].float(), mats[1].float()).remainder(256.0)
    for k in (2, 3):
        x = torch.bmm(x, mats[k].float()).remainder(256.0)
    cand = torch.where(x > 127, x - 256, x).to(torch.int8)
    ok = fast_full_rank(cand)
    flat = cand.view(torch.uint8).reshape(b, 4096)
    return blake3e_torch(flat), ok


@torch.no_grad()
def mine_batch(headers: torch.Tensor, attempt: int = 0):
    """Mine one digest per header, fully batched on the GPU."""
    digests = seed_digests(headers)
    state, inc = pcg_states_batch(digests)
    if attempt:
        state = pcg_states_after(state, inc, 2048 * attempt)

    from gpu_rng import pcg_int8_matrices
    b = headers.shape[0]
    mats = pcg_int8_matrices(state, inc, 4).view(b, 4, 64, 64)
    mats = mats.permute(1, 0, 2, 3).contiguous()
    cand = chain3(mats[0], mats[1], mats[2], mats[3])
    ok = fast_full_rank(cand)
    flat = cand.view(torch.uint8).reshape(b, 4096)
    return blake3e_gpu(flat), ok


def _headers(payloads):
    import numpy as np
    width = max(len(p) for p in payloads)
    buf = np.zeros((len(payloads), width), dtype=np.uint8)
    for i, p in enumerate(payloads):
        buf[i, :len(p)] = np.frombuffer(p, dtype=np.uint8)
    return torch.from_numpy(buf).cuda()


def main():
    from dHeavyHash import dHeavyHash

    if not torch.cuda.is_available():
        print("CUDA is required")
        return
    base = b"test block header data for mining"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"{'batch':>7s} {'ms':>9s} {'hashes/s':>11s} {'tried':>6s}  digest")
    for b in (1, 64, 512, 4096, 16384):
        payloads = [base + i.to_bytes(4, "little") for i in range(b)]
        headers = _headers(payloads)

        for _ in range(2):
            mine_batch(headers)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        digests, ok = mine_batch(headers)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        got = digests.cpu().numpy().tobytes()
        sample = min(4 if b <= 512 else 2, b)
        match = all(got[i * 32:(i + 1) * 32] == dHeavyHash(payloads[i])
                    for i in range(sample))
        print(f"{b:7d} {dt*1e3:9.2f} {b/dt:11.0f} "
              f"{int(ok.sum()):4d}/{b:<4d} {match}")


if __name__ == "__main__":
    main()
