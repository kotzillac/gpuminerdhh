"""BLAKE3 on the GPU (Triton kernels + PyTorch/CUDA baseline).

``blake3_gpu`` / ``blake3e_gpu`` run the whole compression function inside a
Triton kernel; ``blake3_torch`` / ``blake3e_torch`` are equivalent pure
PyTorch/CUDA tensor implementations used as the benchmark baseline.

All four match the ``blake3`` reference library bit for bit.
Supported input length: 0..4096 bytes.
"""

import os

import torch
import triton
import triton.language as tl

# ~/.triton is frequently read-only; keep the cache inside the project.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, "triton_cache"))

IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)

CHUNK_START, CHUNK_END, PARENT, ROOT = 1, 2, 4, 8
BLOCK_LEN, CHUNK_LEN, MAX_INPUT = 64, 1024, 4096


# --------------------------------------------------------------------------
# Triton backend
# --------------------------------------------------------------------------

@triton.jit
def _rotr(x, n):
    return (x >> n) | (x << (32 - n))


@triton.jit
def _g(v0, v1, v2, v3, mx, my):
    v0 = v0 + v1 + mx
    v3 = _rotr(v3 ^ v0, 16)
    v2 = v2 + v3
    v1 = _rotr(v1 ^ v2, 12)
    v0 = v0 + v1 + my
    v3 = _rotr(v3 ^ v0, 8)
    v2 = v2 + v3
    v1 = _rotr(v1 ^ v2, 7)
    return v0, v1, v2, v3


@triton.jit
def _compress(v0, v1, v2, v3, v4, v5, v6, v7,
              m0, m1, m2, m3, m4, m5, m6, m7,
              m8, m9, m10, m11, m12, m13, m14, m15,
              ctr, blen, flags, iv0, iv1, iv2, iv3):
    v8, v9, v10, v11 = iv0, iv1, iv2, iv3
    v12 = ctr
    v13 = tl.zeros_like(ctr)
    v14 = blen
    v15 = flags
    for r in tl.static_range(7):
        v0, v4, v8, v12 = _g(v0, v4, v8, v12, m0, m1)
        v1, v5, v9, v13 = _g(v1, v5, v9, v13, m2, m3)
        v2, v6, v10, v14 = _g(v2, v6, v10, v14, m4, m5)
        v3, v7, v11, v15 = _g(v3, v7, v11, v15, m6, m7)
        v0, v5, v10, v15 = _g(v0, v5, v10, v15, m8, m9)
        v1, v6, v11, v12 = _g(v1, v6, v11, v12, m10, m11)
        v2, v7, v8, v13 = _g(v2, v7, v8, v13, m12, m13)
        v3, v4, v9, v14 = _g(v3, v4, v9, v14, m14, m15)
        if r < 6:
            t0 = m2
            t1 = m6
            t2 = m3
            t3 = m10
            t4 = m7
            t5 = m0
            t6 = m4
            t7 = m13
            t8 = m1
            t9 = m11
            t10 = m12
            t11 = m5
            t12 = m9
            t13 = m14
            t14 = m15
            t15 = m8
            m0 = t0
            m1 = t1
            m2 = t2
            m3 = t3
            m4 = t4
            m5 = t5
            m6 = t6
            m7 = t7
            m8 = t8
            m9 = t9
            m10 = t10
            m11 = t11
            m12 = t12
            m13 = t13
            m14 = t14
            m15 = t15
    return (v0 ^ v8, v1 ^ v9, v2 ^ v10, v3 ^ v11,
            v4 ^ v12, v5 ^ v13, v6 ^ v14, v7 ^ v15)


@triton.jit
def _chunk_kernel(msg_ptr, out_ptr, n, stride_words,
                  NCHUNK: tl.constexpr, CHUNK_BASE: tl.constexpr,
                  NBLK: tl.constexpr, NWORDS: tl.constexpr, CLEN: tl.constexpr,
                  ROOT_HASH: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    base = idx.to(tl.int64) * stride_words + (slot + CHUNK_BASE) * 256
    zero = tl.zeros((BLOCK,), tl.uint32)
    one = zero + 1

    a0 = zero + 0x6A09E667
    a1 = zero + 0xBB67AE85
    a2 = zero + 0x3C6EF372
    a3 = zero + 0xA54FF53A
    a4 = zero + 0x510E527F
    a5 = zero + 0x9B05688C
    a6 = zero + 0x1F83D9AB
    a7 = zero + 0x5BE0CD19
    iv0, iv1, iv2, iv3 = a0, a1, a2, a3
    ctr = zero + (slot + CHUNK_BASE)

    for i in range(NBLK):
        off = base + i * 16
        lim = NWORDS - i * 16
        keep = mask & (lim > 0)
        m0 = tl.load(msg_ptr + off + 0, mask=keep & (lim > 0), other=0).to(tl.uint32)
        m1 = tl.load(msg_ptr + off + 1, mask=keep & (lim > 1), other=0).to(tl.uint32)
        m2 = tl.load(msg_ptr + off + 2, mask=keep & (lim > 2), other=0).to(tl.uint32)
        m3 = tl.load(msg_ptr + off + 3, mask=keep & (lim > 3), other=0).to(tl.uint32)
        m4 = tl.load(msg_ptr + off + 4, mask=keep & (lim > 4), other=0).to(tl.uint32)
        m5 = tl.load(msg_ptr + off + 5, mask=keep & (lim > 5), other=0).to(tl.uint32)
        m6 = tl.load(msg_ptr + off + 6, mask=keep & (lim > 6), other=0).to(tl.uint32)
        m7 = tl.load(msg_ptr + off + 7, mask=keep & (lim > 7), other=0).to(tl.uint32)
        m8 = tl.load(msg_ptr + off + 8, mask=keep & (lim > 8), other=0).to(tl.uint32)
        m9 = tl.load(msg_ptr + off + 9, mask=keep & (lim > 9), other=0).to(tl.uint32)
        m10 = tl.load(msg_ptr + off + 10, mask=keep & (lim > 10), other=0).to(tl.uint32)
        m11 = tl.load(msg_ptr + off + 11, mask=keep & (lim > 11), other=0).to(tl.uint32)
        m12 = tl.load(msg_ptr + off + 12, mask=keep & (lim > 12), other=0).to(tl.uint32)
        m13 = tl.load(msg_ptr + off + 13, mask=keep & (lim > 13), other=0).to(tl.uint32)
        m14 = tl.load(msg_ptr + off + 14, mask=keep & (lim > 14), other=0).to(tl.uint32)
        m15 = tl.load(msg_ptr + off + 15, mask=keep & (lim > 15), other=0).to(tl.uint32)

        first = tl.where(i == 0, one, zero)
        last = tl.where(i == NBLK - 1, one, zero)
        flags = first * 1 + last * 2
        if ROOT_HASH:
            flags = flags + last * 8
        blen = zero + (CLEN - (NBLK - 1) * 64) * last + 64 * (one - last)

        a0, a1, a2, a3, a4, a5, a6, a7 = _compress(
            a0, a1, a2, a3, a4, a5, a6, a7,
            m0, m1, m2, m3, m4, m5, m6, m7,
            m8, m9, m10, m11, m12, m13, m14, m15,
            ctr, blen, flags, iv0, iv1, iv2, iv3)

    dst = out_ptr + (idx.to(tl.int64) * NCHUNK + slot) * 8
    tl.store(dst + 0, a0, mask=mask)
    tl.store(dst + 1, a1, mask=mask)
    tl.store(dst + 2, a2, mask=mask)
    tl.store(dst + 3, a3, mask=mask)
    tl.store(dst + 4, a4, mask=mask)
    tl.store(dst + 5, a5, mask=mask)
    tl.store(dst + 6, a6, mask=mask)
    tl.store(dst + 7, a7, mask=mask)


@triton.jit
def _parent_kernel(cv_ptr, out_ptr, n,
                   NIN: tl.constexpr, NOUT: tl.constexpr,
                   ROOT_HASH: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pair = tl.program_id(1)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    ia = idx.to(tl.int64) * NIN * 8 + pair * 16
    has_b = 2 * pair + 1 < NIN
    kb = mask & has_b

    a0 = tl.load(cv_ptr + ia + 0, mask=mask, other=0)
    a1 = tl.load(cv_ptr + ia + 1, mask=mask, other=0)
    a2 = tl.load(cv_ptr + ia + 2, mask=mask, other=0)
    a3 = tl.load(cv_ptr + ia + 3, mask=mask, other=0)
    a4 = tl.load(cv_ptr + ia + 4, mask=mask, other=0)
    a5 = tl.load(cv_ptr + ia + 5, mask=mask, other=0)
    a6 = tl.load(cv_ptr + ia + 6, mask=mask, other=0)
    a7 = tl.load(cv_ptr + ia + 7, mask=mask, other=0)
    b0 = tl.load(cv_ptr + ia + 8, mask=kb, other=0)
    b1 = tl.load(cv_ptr + ia + 9, mask=kb, other=0)
    b2 = tl.load(cv_ptr + ia + 10, mask=kb, other=0)
    b3 = tl.load(cv_ptr + ia + 11, mask=kb, other=0)
    b4 = tl.load(cv_ptr + ia + 12, mask=kb, other=0)
    b5 = tl.load(cv_ptr + ia + 13, mask=kb, other=0)
    b6 = tl.load(cv_ptr + ia + 14, mask=kb, other=0)
    b7 = tl.load(cv_ptr + ia + 15, mask=kb, other=0)

    zero = tl.zeros((BLOCK,), tl.uint32)
    flags = zero + 4
    if ROOT_HASH:
        flags = flags + 8
    blen = zero + 64
    iv0, iv1, iv2, iv3 = zero + 0x6A09E667, zero + 0xBB67AE85, zero + 0x3C6EF372, zero + 0xA54FF53A

    d0, d1, d2, d3, d4, d5, d6, d7 = _compress(
        iv0, iv1, iv2, iv3,
        zero + 0x510E527F, zero + 0x9B05688C, zero + 0x1F83D9AB, zero + 0x5BE0CD19,
        a0, a1, a2, a3, a4, a5, a6, a7,
        b0, b1, b2, b3, b4, b5, b6, b7,
        zero, blen, flags, iv0, iv1, iv2, iv3)

    dst = out_ptr + (idx.to(tl.int64) * NOUT + pair) * 8
    tl.store(dst + 0, tl.where(has_b, d0, a0), mask=mask)
    tl.store(dst + 1, tl.where(has_b, d1, a1), mask=mask)
    tl.store(dst + 2, tl.where(has_b, d2, a2), mask=mask)
    tl.store(dst + 3, tl.where(has_b, d3, a3), mask=mask)
    tl.store(dst + 4, tl.where(has_b, d4, a4), mask=mask)
    tl.store(dst + 5, tl.where(has_b, d5, a5), mask=mask)
    tl.store(dst + 6, tl.where(has_b, d6, a6), mask=mask)
    tl.store(dst + 7, tl.where(has_b, d7, a7), mask=mask)


def _as_batch(data):
    return data.unsqueeze(0) if data.dim() == 1 else data


def _prepare(data):
    data = _as_batch(data)
    if data.dtype != torch.uint8:
        data = data.to(torch.uint8)
    if not data.is_cuda:
        raise ValueError("expected a CUDA tensor")
    if not data.is_contiguous():
        data = data.contiguous()
    n, length = data.shape
    if length > MAX_INPUT:
        raise ValueError("supported input length is 0..4096 bytes")
    return data, n, length


def blake3_gpu(data):
    """BLAKE3 of a batch of byte strings, computed on the GPU.

    ``data`` -- CUDA ``uint8`` tensor ``(N, L)`` or ``(L,)``, ``L <= 4096``.
    Returns ``uint8`` tensor ``(N, 32)``.
    """
    data, n, length = _prepare(data)
    padded = max(BLOCK_LEN, ((length + BLOCK_LEN - 1) // BLOCK_LEN) * BLOCK_LEN)
    if padded != length:
        data = torch.cat([data, torch.zeros((n, padded - length), dtype=torch.uint8,
                                            device=data.device)], dim=1)
    words = data.view(torch.int32)
    stride_words = words.shape[1]

    nchunks = max(1, (length + CHUNK_LEN - 1) // CHUNK_LEN)
    BLOCK = 64
    grid0 = triton.cdiv(n, BLOCK)

    if nchunks == 1:
        nblk = max(1, (length + BLOCK_LEN - 1) // BLOCK_LEN)
        nwords = (length + 3) // 4
        out = torch.empty((n, 8), dtype=torch.uint32, device=data.device)
        _chunk_kernel[(grid0, 1)](
            words, out, n, stride_words,
            1, 0, nblk, nwords, length, True, BLOCK, num_warps=4)
        return out.view(torch.uint8)

    cv = torch.empty((n, nchunks, 8), dtype=torch.uint32, device=data.device)

    # Chunks 0..nchunks-2 are always full 1024-byte chunks.
    if nchunks > 1:
        _chunk_kernel[(grid0, nchunks - 1)](
            words, cv, n, stride_words,
            nchunks, 0, 16, 256, 1024, False, BLOCK, num_warps=4)

    last_len = length - (nchunks - 1) * CHUNK_LEN
    last_len = max(1, last_len)
    last_nblk = max(1, (last_len + BLOCK_LEN - 1) // BLOCK_LEN)
    last_nwords = (last_len + 3) // 4
    last_cv = torch.empty((n, 1, 8), dtype=torch.uint32, device=data.device)
    _chunk_kernel[(grid0, 1)](
        words, last_cv, n, stride_words,
        1, nchunks - 1, last_nblk, last_nwords, last_len, False, BLOCK,
        num_warps=4)
    cv[:, nchunks - 1:nchunks] = last_cv

    n_in = nchunks
    cur = cv
    while n_in > 1:
        n_out = (n_in + 1) // 2
        is_root = n_out == 1
        dst = torch.empty((n, n_out, 8), dtype=torch.uint32, device=data.device)
        _parent_kernel[(grid0, n_out)](
            cur, dst, n, n_in, n_out, is_root, BLOCK, num_warps=4)
        cur = dst
        n_in = n_out
    return cur[:, 0, :].contiguous().view(torch.uint8)


def blake3e_gpu(data, rounds=8):
    """BLAKE3 iterated ``rounds`` times, fully on the GPU."""
    out = blake3_gpu(data)
    for _ in range(rounds - 1):
        out = blake3_gpu(out)
    return out


# --------------------------------------------------------------------------
# Pure PyTorch/CUDA backend (benchmark baseline)
# --------------------------------------------------------------------------

_M32 = 0xFFFFFFFF
_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)


def _rotr_t(x, n):
    return ((x >> n) | (x << (32 - n))) & _M32


_P1_A = (0, 1, 2, 3)
_P1_B = (4, 5, 6, 7)
_P1_C = (8, 9, 10, 11)
_P1_D = (12, 13, 14, 15)
_P1_X = (0, 2, 4, 6)
_P1_Y = (1, 3, 5, 7)

_P2_A = (0, 1, 2, 3)
_P2_B = (5, 6, 7, 4)
_P2_C = (10, 11, 8, 9)
_P2_D = (15, 12, 13, 14)
_P2_X = (8, 10, 12, 14)
_P2_Y = (9, 11, 13, 15)


_IDX_CACHE = {}


def _idx(device, key):
    cached = _IDX_CACHE.get((key, str(device)))
    if cached is None:
        idx = {
            "p1a": _P1_A, "p1b": _P1_B, "p1c": _P1_C, "p1d": _P1_D,
            "p1x": _P1_X, "p1y": _P1_Y,
            "p2a": _P2_A, "p2b": _P2_B, "p2c": _P2_C, "p2d": _P2_D,
            "p2x": _P2_X, "p2y": _P2_Y,
            "perm": _PERM,
        }[key]
        cached = torch.tensor(idx, device=device)
        _IDX_CACHE[(key, str(device))] = cached
    return cached


def _round_t(state, msg, a_idx, b_idx, c_idx, d_idx, x_idx, y_idx):
    """Four parallel BLAKE3 quarter rounds on ``(N, 16)`` state."""
    dev = state.device
    ia, ib = _idx(dev, a_idx), _idx(dev, b_idx)
    ic, idd = _idx(dev, c_idx), _idx(dev, d_idx)
    ix, iy = _idx(dev, x_idx), _idx(dev, y_idx)

    a = state.index_select(1, ia)
    b = state.index_select(1, ib)
    c = state.index_select(1, ic)
    d = state.index_select(1, idd)
    mx = msg.index_select(1, ix)
    my = msg.index_select(1, iy)

    a = (a + b + mx) & _M32
    d = _rotr_t(d ^ a, 16)
    c = (c + d) & _M32
    b = _rotr_t(b ^ c, 12)
    a = (a + b + my) & _M32
    d = _rotr_t(d ^ a, 8)
    c = (c + d) & _M32
    b = _rotr_t(b ^ c, 7)

    n = state.shape[0]
    state = state.scatter(1, ia.unsqueeze(0).expand(n, 4), a)
    state = state.scatter(1, ib.unsqueeze(0).expand(n, 4), b)
    state = state.scatter(1, ic.unsqueeze(0).expand(n, 4), c)
    state = state.scatter(1, idd.unsqueeze(0).expand(n, 4), d)
    return state


def _compress_t(cv, w, ctr, blen, flags):
    """BLAKE3 compression for a whole batch (cv: list of 8, w: list of 16)."""
    n, dev = cv[0].shape[0], cv[0].device
    iv = torch.tensor(IV, dtype=torch.int64, device=dev)
    iv_exp = iv[0:4].unsqueeze(1).expand(4, n).transpose(0, 1)
    extra = torch.stack([ctr.reshape(n),
                         torch.zeros(n, dtype=torch.int64, device=dev),
                         blen.reshape(n), flags.reshape(n)], dim=1)
    state = torch.cat([torch.stack(list(cv), dim=1), iv_exp, extra], dim=1)
    msg = torch.stack(list(w), dim=1)

    for r in range(7):
        state = _round_t(state, msg, "p1a", "p1b", "p1c", "p1d", "p1x", "p1y")
        state = _round_t(state, msg, "p2a", "p2b", "p2c", "p2d", "p2x", "p2y")
        if r < 6:
            msg = msg.index_select(1, _idx(dev, "perm"))

    return [(state[:, i] ^ state[:, i + 8]) & _M32 for i in range(8)]


def blake3_torch(data):
    """Reference PyTorch/CUDA BLAKE3, same semantics as :func:`blake3_gpu`."""
    data, n, length = _prepare(data)
    dev = data.device
    iv = torch.tensor(IV, dtype=torch.int64, device=dev).expand(n, 8)
    padded = max(BLOCK_LEN, ((length + BLOCK_LEN - 1) // BLOCK_LEN) * BLOCK_LEN)
    if padded != length:
        data = torch.cat([data, torch.zeros((n, padded - length), dtype=torch.uint8,
                                            device=dev)], dim=1)
    nchunks = max(1, (length + CHUNK_LEN - 1) // CHUNK_LEN)

    def chunk_cv(c):
        clen = max(0, min(CHUNK_LEN, length - c * CHUNK_LEN))
        raw = data[:, c * CHUNK_LEN:c * CHUNK_LEN + clen]
        nblk = max(1, (clen + BLOCK_LEN - 1) // BLOCK_LEN)
        cv = [iv[:, i] for i in range(8)]
        ctr = torch.full((n,), c, dtype=torch.int64, device=dev)
        for i in range(nblk):
            blk = raw[:, i * BLOCK_LEN:(i + 1) * BLOCK_LEN]
            blen = blk.shape[1]
            bp = (-blen) % 4
            if bp:
                blk = torch.cat([blk, torch.zeros((n, bp), dtype=torch.uint8,
                                                  device=dev)], dim=1)
            words = blk.view(torch.int32).to(torch.int64) & _M32
            if words.shape[1] < 16:
                words = torch.cat([words, torch.zeros(
                    (n, 16 - words.shape[1]), dtype=torch.int64, device=dev)], 1)
            wl = [words[:, i] for i in range(16)]
            flags = (CHUNK_START if i == 0 else 0) | (CHUNK_END if i == nblk - 1 else 0)
            if nchunks == 1 and i == nblk - 1:
                flags |= ROOT
            cv = _compress_t(cv, wl, ctr,
                             torch.full((n,), blen, dtype=torch.int64, device=dev),
                             torch.full((n,), flags, dtype=torch.int64, device=dev))
        return cv

    level = [chunk_cv(c) for c in range(nchunks)]
    while len(level) > 1:
        nxt = []
        pairs = (len(level) + 1) // 2
        top = pairs == 1
        for p in range(pairs):
            left = level[2 * p]
            if 2 * p + 1 < len(level):
                msg = left + level[2 * p + 1]
                flags = PARENT | (ROOT if top else 0)
                nxt.append(_compress_t(
                    [iv[:, i] for i in range(8)], msg,
                    torch.zeros(n, dtype=torch.int64, device=dev),
                    torch.full((n,), BLOCK_LEN, dtype=torch.int64, device=dev),
                    torch.full((n,), flags, dtype=torch.int64, device=dev)))
            else:
                nxt.append(left)
        level = nxt
    return torch.stack(level[0], dim=1).to(torch.int32).view(torch.uint8)


def blake3e_torch(data, rounds=8):
    out = blake3_torch(data)
    for _ in range(rounds - 1):
        out = blake3_torch(out)
    return out


if __name__ == "__main__":
    import blake3

    torch.manual_seed(0)
    payloads = [b"", b"h", b"hello", b"a" * 63, b"a" * 64, b"a" * 65,
                b"test block header data for mining",
                bytes(range(256)) * 4, bytes(range(256)) * 16,
                b"x" * 1023, b"x" * 1024, b"x" * 1025, b"x" * 1500]
    ok = True
    for payload in payloads:
        t = torch.tensor(list(payload), dtype=torch.uint8, device="cuda")
        ref = blake3.blake3(payload).digest(32)
        g = blake3_gpu(t).view(-1).cpu().numpy().tobytes()
        p = blake3_torch(t).view(-1).cpu().numpy().tobytes()
        good = g == ref and p == ref
        ok &= good
        print(f"{len(payload):5d} bytes  triton={g == ref}  torch={p == ref}")

    batch = torch.randint(0, 256, (16, 4096), dtype=torch.uint8, device="cuda")
    ref = torch.stack([torch.tensor(list(blake3.blake3(
        batch[i].cpu().numpy().tobytes()).digest(32)), dtype=torch.uint8)
        for i in range(batch.shape[0])]).cuda()
    print("batched triton:", torch.equal(blake3_gpu(batch), ref))
    print("batched torch :", torch.equal(blake3_torch(batch), ref))
    print("ALL OK" if ok else "FAILED")
