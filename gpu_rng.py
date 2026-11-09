"""NumPy-compatible PCG64 random int8 matrices, generated on the GPU.

``numpy``'s ``Generator.integers(-128, 128, size=(64, 64), dtype=np.int8)``
draws 4096 bytes from the PCG64 stream: every little-endian byte of a 64-bit
output word becomes one int8 sample (value = byte - 128).

This module reproduces that stream exactly -- including ``SeedSequence`` and
the PCG64 seeding procedure -- inside Triton, and can also emulate the
CPU-side RNG on the GPU with PyTorch tensors.

Reference algorithm (NumPy ``pcg64_set_seed``):

    initstate = words[0] << 64 | words[1]
    initseq   = words[2] << 64 | words[3]
    inc       = (initseq << 1 | 1) mod 2**128
    state     = 0
    state     = state * MULT + inc
    state     = state + initstate
    state     = state * MULT + inc

where ``words = SeedSequence(seed).generate_state(8, dtype=uint32)`` and
``MULT = 0x2360ed051fc65da44385df649fccf645``.
"""

import os

import torch
import triton
import triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, "triton_cache"))

PCG_MULT = 47026247687942121848144207491837523525  # 0x2360ed051fc65da44385df649fccf645
PCG_MULT_LO = tl.constexpr(PCG_MULT & ((1 << 64) - 1))
PCG_MULT_HI = tl.constexpr(PCG_MULT >> 64)

# SeedSequence constants (PCG64 uses these via SeedSequence)
SS_INIT_A = 0x43B0D7E5
SS_MULT_A = 0x931E8875
SS_INIT_B = 0x8B51F9DD
SS_MULT_B = 0x58F38DED
SS_MIX_L = 0xCA01F9DD
SS_MIX_R = 0x4973F715
SS_XSHIFT = 16

SS_INIT_A_C = tl.constexpr(SS_INIT_A)
SS_MULT_A_C = tl.constexpr(SS_MULT_A)
SS_INIT_B_C = tl.constexpr(SS_INIT_B)
SS_MULT_B_C = tl.constexpr(SS_MULT_B)
SS_MIX_L_C = tl.constexpr(SS_MIX_L)
SS_MIX_R_C = tl.constexpr(SS_MIX_R)
SS_XSHIFT_C = tl.constexpr(SS_XSHIFT)

MASK32 = 0xFFFFFFFF
MASK128 = (1 << 128) - 1


def seed_sequence_words(seed: int, n: int = 8):
    """Exact reimplementation of ``SeedSequence(seed).generate_state(n)``."""
    entropy = [seed & MASK32]
    rest = seed >> 32
    while rest > 0:
        entropy.append(rest & MASK32)
        rest >>= 32

    mix = [0, 0, 0, 0]
    hc = SS_INIT_A

    def hashmix(value):
        nonlocal hc
        value = (value ^ hc) & MASK32
        hc = (hc * SS_MULT_A) & MASK32
        value = (value * hc) & MASK32
        value ^= value >> SS_XSHIFT
        return value & MASK32

    def mix32(x, y):
        r = (SS_MIX_L * x - SS_MIX_R * y) & MASK32
        r ^= r >> SS_XSHIFT
        return r & MASK32

    for i in range(4):
        mix[i] = hashmix(entropy[i] if i < len(entropy) else 0)
    for src in range(4):
        for dst in range(4):
            if src != dst:
                mix[dst] = mix32(mix[dst], hashmix(mix[src]))
    for src in range(4, len(entropy)):
        for dst in range(4):
            mix[dst] = mix32(mix[dst], hashmix(entropy[src]))

    hb = SS_INIT_B
    out = []
    for i in range(n):
        v = (mix[i % 4] ^ hb) & MASK32
        hb = (hb * SS_MULT_B) & MASK32
        v = (v * hb) & MASK32
        v ^= v >> SS_XSHIFT
        out.append(v & MASK32)
    return out


def pcg64_seed_to_state(seed: int):
    """Return ``(state, inc)`` exactly as ``np.random.PCG64(seed)`` builds them."""
    w = seed_sequence_words(seed, 8)
    pairs = [(w[2 * i] | (w[2 * i + 1] << 32)) for i in range(4)]
    initstate = (pairs[0] << 64) | pairs[1]
    initseq = (pairs[2] << 64) | pairs[3]
    inc = ((initseq << 1) | 1) & MASK128
    state = 0
    state = (state * PCG_MULT + inc) & MASK128
    state = (state + initstate) & MASK128
    state = (state * PCG_MULT + inc) & MASK128
    return state, inc


@triton.jit
def _mul128(a0, a1, b0, b1):
    """(a0 | a1<<64) * (b0 | b1<<64) mod 2**128.

    Triton lowers a 64x64-bit multiply to a hardware 64-bit multiply, which
    silently drops the high half on this backend, so the product is assembled
    from 32-bit limbs whose products always fit into 64 bits.
    """
    M: tl.constexpr = 0xFFFFFFFF
    a0 = a0.to(tl.uint64)
    a1 = a1.to(tl.uint64)
    b0 = b0.to(tl.uint64)
    b1 = b1.to(tl.uint64)
    al0 = (a0 & M).to(tl.uint32)
    al1 = (a0 >> 32).to(tl.uint32)
    al2 = (a1 & M).to(tl.uint32)
    al3 = (a1 >> 32).to(tl.uint32)
    bl0 = (b0 & M).to(tl.uint32)
    bl1 = (b0 >> 32).to(tl.uint32)
    bl2 = (b1 & M).to(tl.uint32)
    bl3 = (b1 >> 32).to(tl.uint32)

    p0 = al0.to(tl.uint64) * bl0.to(tl.uint64)
    p1 = al0.to(tl.uint64) * bl1.to(tl.uint64)
    p2 = al1.to(tl.uint64) * bl0.to(tl.uint64)
    p3 = al1.to(tl.uint64) * bl1.to(tl.uint64)
    p4 = al0.to(tl.uint64) * bl2.to(tl.uint64)
    p5 = al2.to(tl.uint64) * bl0.to(tl.uint64)
    p6 = al1.to(tl.uint64) * bl2.to(tl.uint64)
    p7 = al2.to(tl.uint64) * bl1.to(tl.uint64)
    p8 = al0.to(tl.uint64) * bl3.to(tl.uint64)
    p9 = al3.to(tl.uint64) * bl0.to(tl.uint64)

    s1 = (p0 >> 32) + (p1 & M) + (p2 & M)
    s2 = (p1 >> 32) + (p2 >> 32) + (p3 & M) + (p4 & M) + (p5 & M) + (s1 >> 32)
    s3 = ((p3 >> 32) + (p4 >> 32) + (p5 >> 32) + (p6 & M) + (p7 & M)
          + (p8 & M) + (p9 & M) + (s2 >> 32))
    lo = (p0 & M) | ((s1 & M) << 32)
    hi = (s2 & M) | ((s3 & M) << 32)
    return lo, hi


@triton.jit
def _pcg_out(s0, s1):
    x = s1 ^ s0
    r = (s1 >> 58) & 63
    return (x >> r) | (x << ((64 - r) & 63))


@triton.jit
def _pcg_advance(s0, s1, i0, i1):
    """state = state * PCG_MULT + inc (mod 2**128)."""
    mlo = tl.full((), PCG_MULT_LO.value, tl.uint64)
    mhi = tl.full((), PCG_MULT_HI.value, tl.uint64)
    p0, p1 = _mul128(s0, s1, mlo, mhi)
    lo = p0 + i0
    carry = tl.where(lo < p0, tl.full((), 1, tl.uint64),
                     tl.full((), 0, tl.uint64))
    hi = p1 + i1 + carry
    return lo, hi


@triton.jit
def _pcg_jump(s0, s1, m0, m1, c0, c1):
    """state = state * A + B (mod 2**128), where A/B encode a jump."""
    p0, p1 = _mul128(s0, s1, m0, m1)
    lo = p0 + c0
    carry = tl.where(lo < p0, tl.full((), 1, tl.uint64),
                     tl.full((), 0, tl.uint64))
    hi = p1 + c1 + carry
    return lo, hi


@triton.jit
def pcg_jump_kernel(state_ptr, inc_ptr, out_ptr, n,
                    A0: tl.constexpr, A1: tl.constexpr,
                    B0: tl.constexpr, B1: tl.constexpr,
                    COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """state_k = A**k * state_0 + (A**k - 1)/(A - 1) * inc for k = index."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    idx = pid * BLOCK + lane
    mask = idx < n

    s0 = tl.load(state_ptr + 0) + tl.zeros((BLOCK,), tl.uint64)
    s1 = tl.load(state_ptr + 1) + tl.zeros((BLOCK,), tl.uint64)
    i0 = tl.load(inc_ptr + 0) + tl.zeros((BLOCK,), tl.uint64)
    i1 = tl.load(inc_ptr + 1) + tl.zeros((BLOCK,), tl.uint64)

    # affine map accumulated for this lane: (acc_a, acc_b) with
    # state = acc_a * s + acc_b * inc
    aa0 = tl.full((BLOCK,), 1, tl.uint64)
    aa1 = tl.zeros((BLOCK,), tl.uint64)
    ab0 = tl.zeros((BLOCK,), tl.uint64)
    ab1 = tl.zeros((BLOCK,), tl.uint64)

    pa0 = tl.full((BLOCK,), A0, tl.uint64)
    pa1 = tl.full((BLOCK,), A1, tl.uint64)
    pb0 = tl.full((BLOCK,), B0, tl.uint64)
    pb1 = tl.full((BLOCK,), B1, tl.uint64)

    for bit in range(31):
        take = ((idx >> bit) & 1) == 1
        # acc <- jump o acc : a = pa*aa, b = pa*ab + pb
        t0, t1 = _mul128(pa0, pa1, aa0, aa1)
        u0, u1 = _mul128(pa0, pa1, ab0, ab1)
        v0 = u0 + pb0
        carry = tl.where(v0 < u0, tl.full((BLOCK,), 1, tl.uint64),
                         tl.full((BLOCK,), 0, tl.uint64))
        v1 = u1 + pb1 + carry
        aa0 = tl.where(take, t0, aa0)
        aa1 = tl.where(take, t1, aa1)
        ab0 = tl.where(take, v0, ab0)
        ab1 = tl.where(take, v1, ab1)
        # square the jump: p <- p o p
        q0, q1 = _mul128(pa0, pa1, pa0, pa1)
        r0, r1 = _mul128(pa0, pa1, pb0, pb1)
        w0 = r0 + pb0
        c2 = tl.where(w0 < r0, tl.full((BLOCK,), 1, tl.uint64),
                      tl.full((BLOCK,), 0, tl.uint64))
        w1 = r1 + pb1 + c2
        pa0 = q0
        pa1 = q1
        pb0 = w0
        pb1 = w1

    st0, st1 = _mul128(aa0, aa1, s0, s1)
    bt0, bt1 = _mul128(ab0, ab1, i0, i1)
    f0 = st0 + bt0
    cy = tl.where(f0 < st0, tl.full((BLOCK,), 1, tl.uint64),
                  tl.full((BLOCK,), 0, tl.uint64))
    f1 = st1 + bt1 + cy

    tl.store(out_ptr + idx * 2 + 0, f0, mask=mask)
    tl.store(out_ptr + idx * 2 + 1, f1, mask=mask)


@triton.jit
def pcg_int8_kernel(state_ptr, inc_ptr, out_ptr, n, words_per_matrix,
                    MATRICES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    idx = pid * BLOCK + lane
    mask = idx < n

    s0 = tl.load(state_ptr + idx * 2 + 0, mask=mask, other=0)
    s1 = tl.load(state_ptr + idx * 2 + 1, mask=mask, other=0)
    i0 = tl.load(inc_ptr + idx * 2 + 0, mask=mask, other=0)
    i1 = tl.load(inc_ptr + idx * 2 + 1, mask=mask, other=0)

    total = MATRICES * words_per_matrix
    for w in range(total):
        s0, s1 = _pcg_advance(s0, s1, i0, i1)
        y = _pcg_out(s0, s1)
        # one uint64 word = eight int8 samples (little-endian byte order); pack
        # them into two int32 stores instead of eight byte stores
        lo = (((y >> 0) & 0xFF) - 128) & 0xFF
        lo |= ((((y >> 8) & 0xFF) - 128) & 0xFF) << 8
        lo |= ((((y >> 16) & 0xFF) - 128) & 0xFF) << 16
        lo |= ((((y >> 24) & 0xFF) - 128) & 0xFF) << 24
        hi = (((y >> 32) & 0xFF) - 128) & 0xFF
        hi |= ((((y >> 40) & 0xFF) - 128) & 0xFF) << 8
        hi |= ((((y >> 48) & 0xFF) - 128) & 0xFF) << 16
        hi |= ((((y >> 56) & 0xFF) - 128) & 0xFF) << 24
        base = idx.to(tl.int64) * (MATRICES * 1024) + w * 2
        tl.store(out_ptr + base + 0, lo.to(tl.uint32), mask=mask)
        tl.store(out_ptr + base + 1, hi.to(tl.uint32), mask=mask)


# ---------------------------------------------------------------------------
# SeedSequence / PCG64 seeding on the GPU
# ---------------------------------------------------------------------------

BLAKE3_SEED_PERSON = b"dHeavyHash"


@triton.jit
def _bswap32(x):
    return (((x & 0xFF) << 24) | ((x & 0xFF00) << 8)
            | ((x >> 8) & 0xFF00) | (x >> 24)).to(tl.uint32)


@triton.jit
def _hashmix(value, hc):
    """SeedSequence ``hashmix`` step; returns ``(word, next_hc)``."""
    v = value ^ hc
    hc = hc * SS_MULT_A_C
    v = v * hc
    v = v ^ (v >> SS_XSHIFT_C)
    return v, hc


@triton.jit
def _mix32(x, y):
    r = SS_MIX_L_C * x - SS_MIX_R_C * y
    r = r ^ (r >> SS_XSHIFT_C)
    return r


@triton.jit
def _seed_sequence_kernel(digest_ptr, out_ptr):
    """``SeedSequence(big_endian(digest)).generate_state(8)`` on one wavefront."""
    u0 = tl.load(digest_ptr + 0)
    u1 = tl.load(digest_ptr + 1)
    u2 = tl.load(digest_ptr + 2)
    u3 = tl.load(digest_ptr + 3)
    u4 = tl.load(digest_ptr + 4)
    u5 = tl.load(digest_ptr + 5)
    u6 = tl.load(digest_ptr + 6)
    u7 = tl.load(digest_ptr + 7)

    # big-endian integer -> little-endian 32-bit entropy words
    e0 = _bswap32(u7)
    e1 = _bswap32(u6)
    e2 = _bswap32(u5)
    e3 = _bswap32(u4)
    e4 = _bswap32(u3)
    e5 = _bswap32(u2)
    e6 = _bswap32(u1)
    e7 = _bswap32(u0)

    nw = tl.full((), 1, tl.int32)
    nw = tl.where(e1 != 0, 2, nw)
    nw = tl.where(e2 != 0, 3, nw)
    nw = tl.where(e3 != 0, 4, nw)
    nw = tl.where(e4 != 0, 5, nw)
    nw = tl.where(e5 != 0, 6, nw)
    nw = tl.where(e6 != 0, 7, nw)
    nw = tl.where(e7 != 0, 8, nw)

    hc = tl.full((), SS_INIT_A_C, tl.uint32)
    m0, hc = _hashmix(e0, hc)
    m1, hc = _hashmix(e1, hc)
    m2, hc = _hashmix(e2, hc)
    m3, hc = _hashmix(e3, hc)

    # src = 0: update m1, m2, m3 with hashmix(m0)
    t, hc = _hashmix(m0, hc)
    m1 = _mix32(m1, t)
    t, hc = _hashmix(m0, hc)
    m2 = _mix32(m2, t)
    t, hc = _hashmix(m0, hc)
    m3 = _mix32(m3, t)
    # src = 1: update m0, m2, m3 with hashmix(m1)
    t, hc = _hashmix(m1, hc)
    m0 = _mix32(m0, t)
    t, hc = _hashmix(m1, hc)
    m2 = _mix32(m2, t)
    t, hc = _hashmix(m1, hc)
    m3 = _mix32(m3, t)
    # src = 2: update m0, m1, m3 with hashmix(m2)
    t, hc = _hashmix(m2, hc)
    m0 = _mix32(m0, t)
    t, hc = _hashmix(m2, hc)
    m1 = _mix32(m1, t)
    t, hc = _hashmix(m2, hc)
    m3 = _mix32(m3, t)
    # src = 3: update m0, m1, m2 with hashmix(m3)
    t, hc = _hashmix(m3, hc)
    m0 = _mix32(m0, t)
    t, hc = _hashmix(m3, hc)
    m1 = _mix32(m1, t)
    t, hc = _hashmix(m3, hc)
    m2 = _mix32(m2, t)

    # Additional entropy words: each one is hashed four times (once per
    # destination lane), and ``hc`` advances between those calls.
    for src in tl.range(4, nw):
        e = tl.where(src == 4, e4, e0)
        e = tl.where(src == 5, e5, e)
        e = tl.where(src == 6, e6, e)
        e = tl.where(src == 7, e7, e)
        t, hc = _hashmix(e, hc)
        m0 = _mix32(m0, t)
        t, hc = _hashmix(e, hc)
        m1 = _mix32(m1, t)
        t, hc = _hashmix(e, hc)
        m2 = _mix32(m2, t)
        t, hc = _hashmix(e, hc)
        m3 = _mix32(m3, t)

    hb = tl.full((), SS_INIT_B_C, tl.uint32)
    v = m0 ^ hb
    hb = hb * SS_MULT_B_C
    v = v * hb
    tl.store(out_ptr + 0, (v ^ (v >> SS_XSHIFT_C)).to(tl.uint32))
    v = m1 ^ hb
    hb = hb * SS_MULT_B_C
    v = v * hb
    tl.store(out_ptr + 1, (v ^ (v >> SS_XSHIFT_C)).to(tl.uint32))
    v = m2 ^ hb
    hb = hb * SS_MULT_B_C
    v = v * hb
    tl.store(out_ptr + 2, (v ^ (v >> SS_XSHIFT_C)).to(tl.uint32))
    v = m3 ^ hb
    hb = hb * SS_MULT_B_C
    v = v * hb
    tl.store(out_ptr + 3, (v ^ (v >> SS_XSHIFT_C)).to(tl.uint32))
    v = m0 ^ hb
    hb = hb * SS_MULT_B_C
    v = v * hb
    tl.store(out_ptr + 4, (v ^ (v >> SS_XSHIFT_C)).to(tl.uint32))
    v = m1 ^ hb
    hb = hb * SS_MULT_B_C
    v = v * hb
    tl.store(out_ptr + 5, (v ^ (v >> SS_XSHIFT_C)).to(tl.uint32))
    v = m2 ^ hb
    hb = hb * SS_MULT_B_C
    v = v * hb
    tl.store(out_ptr + 6, (v ^ (v >> SS_XSHIFT_C)).to(tl.uint32))
    v = m3 ^ hb
    hb = hb * SS_MULT_B_C
    v = v * hb
    tl.store(out_ptr + 7, (v ^ (v >> SS_XSHIFT_C)).to(tl.uint32))


@triton.jit
def _pcg_init_kernel(words_ptr, state_ptr, inc_ptr):
    """``pcg64_set_seed`` for the words produced by ``_seed_sequence_kernel``."""
    w0 = tl.load(words_ptr + 0)
    w1 = tl.load(words_ptr + 1)
    w2 = tl.load(words_ptr + 2)
    w3 = tl.load(words_ptr + 3)
    w4 = tl.load(words_ptr + 4)
    w5 = tl.load(words_ptr + 5)
    w6 = tl.load(words_ptr + 6)
    w7 = tl.load(words_ptr + 7)

    # ``pcg64_seed_to_state``: pairs[i] = w[2i] | w[2i+1] << 32, then
    # initstate = pairs[0] << 64 | pairs[1].
    is0 = w2.to(tl.uint64) | (w3.to(tl.uint64) << 32)
    is1 = w0.to(tl.uint64) | (w1.to(tl.uint64) << 32)
    q0 = w6.to(tl.uint64) | (w7.to(tl.uint64) << 32)
    q1 = w4.to(tl.uint64) | (w5.to(tl.uint64) << 32)

    i0 = q0 << 1
    i1 = (q1 << 1) | (q0 >> 63)
    i0 = i0 | 1

    t0 = i0 + is0
    cy = tl.where(t0 < i0, tl.full((), 1, tl.uint64),
                  tl.full((), 0, tl.uint64))
    t1 = i1 + is1 + cy

    mlo = tl.full((), PCG_MULT_LO.value, tl.uint64)
    mhi = tl.full((), PCG_MULT_HI.value, tl.uint64)
    p0, p1 = _mul128(t0, t1, mlo, mhi)
    s0 = p0 + i0
    cy2 = tl.where(s0 < p0, tl.full((), 1, tl.uint64),
                   tl.full((), 0, tl.uint64))
    s1 = p1 + i1 + cy2

    tl.store(state_ptr + 0, s0)
    tl.store(state_ptr + 1, s1)
    tl.store(inc_ptr + 0, i0)
    tl.store(inc_ptr + 1, i1)


def pcg_states_from_digest(digest, count: int, words_per_matrix: int = 512,
                           device: str = "cuda"):
    """PCG64 states for ``count`` matrices, seeded by a 32-byte digest.

    ``digest`` -- CUDA ``uint8`` tensor ``(32,)`` (a BLAKE3 hash). The whole
    SeedSequence + PCG64 seeding path runs in Triton kernels; only the jump
    coefficients (constant for a given ``words_per_matrix``) come from Python.
    """
    if digest.dim() != 1 or digest.numel() != 32:
        raise ValueError("expected a 32-byte digest")
    dev = digest.device
    words = torch.empty(8, dtype=torch.uint32, device=dev)
    _seed_sequence_kernel[(1,)](digest.contiguous().view(torch.uint32), words,
                               num_warps=1)
    state = torch.empty(2, dtype=torch.uint64, device=dev)
    inc = torch.empty(2, dtype=torch.uint64, device=dev)
    _pcg_init_kernel[(1,)](words, state, inc)

    a, b = pcg_jump_coefficients(words_per_matrix)
    out = torch.empty((count, 2), dtype=torch.uint64, device=dev)
    BLOCK = 64
    pcg_jump_kernel[(triton.cdiv(count, BLOCK),)](
        state, inc, out.reshape(-1), count,
        a & ((1 << 64) - 1), a >> 64, b & ((1 << 64) - 1), b >> 64,
        count, BLOCK, num_warps=4)
    return out, inc.expand(count, 2).contiguous()


_JUMP_CACHE = {}


def pcg_jump_coefficients(words: int):
    """Return ``(A, B)`` such that jumping ``words`` draws maps state->A*state+B.

    ``B = 1 + M + ... + M**(words-1)`` computed by binary exponentiation so the
    cost is ``O(log words)`` big-int multiplications.
    """
    if words in _JUMP_CACHE:
        return _JUMP_CACHE[words]
    modulus = 1 << 128

    # Affine map composition: (a1, b1) o (a0, b0) = (a1*a0, a1*b0 + b1)
    def compose(a1, b1, a0, b0):
        return (a1 * a0) % modulus, (a1 * b0 + b1) % modulus

    acc = (1, 0)          # identity
    base = (PCG_MULT, 1)  # one step: x -> M*x + 1
    n = words
    while n > 0:
        if n & 1:
            acc = compose(*base, *acc)
        base = compose(*base, *base)
        n >>= 1
    _JUMP_CACHE[words] = acc
    return acc


def pcg_states(seeds):
    """GPU tensors with PCG64 ``(state, inc)`` for every seed in ``seeds``."""
    import numpy as np

    if isinstance(seeds, torch.Tensor):
        seed_list = [int(s) for s in seeds.reshape(-1).tolist()]
        dev = seeds.device
    else:
        seed_list = [int(s) for s in seeds]
        dev = "cuda"
    cpu = [pcg64_seed_to_state(s) for s in seed_list]
    arr_st = np.array([[s & ((1 << 64) - 1), s >> 64] for s, _ in cpu],
                      dtype=np.uint64)
    arr_inc = np.array([[i & ((1 << 64) - 1), i >> 64] for _, i in cpu],
                       dtype=np.uint64)
    st = torch.from_numpy(arr_st).to(dev)
    inc = torch.from_numpy(arr_inc).to(dev)
    return st, inc


def pcg_states_parallel(seed: int, count: int, words_per_matrix: int = 512,
                        device: str = "cuda"):
    """PCG64 states for ``count`` consecutive matrices of one stream.

    Matrix ``k`` starts after ``k * words_per_matrix`` draws, which is exactly
    the layout NumPy uses when sampling matrices from a single Generator. The
    jump-ahead is evaluated in parallel on the GPU.
    """
    import numpy as np

    state, inc = pcg64_seed_to_state(seed)
    a, b = pcg_jump_coefficients(words_per_matrix)
    st = torch.tensor([[state & ((1 << 64) - 1), state >> 64]],
                      dtype=torch.uint64, device=device)
    ic = torch.tensor([[inc & ((1 << 64) - 1), inc >> 64]],
                      dtype=torch.uint64, device=device)
    out = torch.empty((count, 2), dtype=torch.uint64, device=device)
    BLOCK = 64
    pcg_jump_kernel[(triton.cdiv(count, BLOCK),)](
        st.reshape(-1), ic.reshape(-1), out.reshape(-1), count,
        a & ((1 << 64) - 1), a >> 64, b & ((1 << 64) - 1), b >> 64,
        count, BLOCK, num_warps=4)
    return out, ic.expand(count, 2).contiguous()


def pcg_int8_matrices(states, incs, matrices_per_seed=1):
    """Generate ``matrices_per_seed`` 64x64 int8 matrices per PCG64 state.

    ``states``/``incs`` -- ``uint64`` tensors of shape ``(N, 2)`` (lo, hi).
    Returns an ``int8`` tensor of shape ``(N * matrices_per_seed, 64, 64)``.
    """
    n = states.shape[0]
    st = states.reshape(-1).contiguous()
    inc = incs.reshape(-1).contiguous()
    out = torch.empty((n, matrices_per_seed, 1024), dtype=torch.int32,
                      device=states.device)
    grid = (triton.cdiv(n, 64),)
    pcg_int8_kernel[grid](st, inc, out, n, 512, matrices_per_seed, 64,
                          num_warps=4)
    return out.view(torch.int8).view(n * matrices_per_seed, 64, 64)


if __name__ == "__main__":
    import numpy as np
    import time

    # Verify bit-exactness against NumPy for a handful of seeds.
    ok = True
    for seed in (0, 1, 123, 999, int.from_bytes(b"dHeavyHash-seed", "big") & ((1 << 63) - 1)):
        ref = np.random.default_rng(seed=seed).integers(
            -128, 128, size=(2, 64, 64), dtype=np.int8)
        st, inc = pcg_states(torch.tensor([seed], dtype=torch.int64, device="cuda"))
        got = pcg_int8_matrices(st, inc, 2).cpu().numpy()
        same = np.array_equal(got, ref)
        ok &= same
        print(f"seed {seed}: match={same}")

    # GPU SeedSequence + PCG64 seeding, compared with the CPU implementation.
    import numpy as np
    from gpu_rng import pcg_states_from_digest, pcg64_seed_to_state

    for _ in range(4):
        digest = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")
        seed = int.from_bytes(digest.cpu().numpy().tobytes(), "big")
        got = pcg_states_from_digest(digest, 1)[0][0].cpu().numpy()
        st, inc = pcg64_seed_to_state(seed)
        want = np.array([st & ((1 << 64) - 1), st >> 64], dtype=np.uint64)
        same = np.array_equal(got, want)
        ok &= same
        print(f"gpu seed 0x{seed & 0xFFFF:04x}...: match={same}")

    n = 4096
    seeds = torch.randint(0, 2**62, (n,), dtype=torch.int64, device="cuda")
    st, inc = pcg_states(seeds)
    pcg_int8_matrices(st, inc, 1)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(5):
        pcg_int8_matrices(st, inc, 1)
    torch.cuda.synchronize()
    dt = (time.time() - t) / 5
    print(f"PCG64 int8: {n} matrices in {dt*1e3:.2f} ms "
          f"({n/dt/1e3:.1f}k matrices/s)")
    print("ALL OK" if ok else "FAILED")
