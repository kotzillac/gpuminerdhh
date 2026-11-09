"""Tests for the CPU dHeavyHash reference implementation.

Run with:  python test_dheavyhash.py
"""

import os
import time

import numpy as np
import torch
from blake3 import blake3

from dHeavyHash import BLAKE3, BLAKE3e, is_full_rank, dHeavyHash

REFERENCE_PAYLOAD = b"test block header data for mining"
REFERENCE_DIGEST = bytes.fromhex(
    "8f11d1da98cabd054266aeea8b866e40a67a39b28334ac8049ce54e06225f3b7")

results = []


def check(name, cond, detail=""):
    cond = bool(cond)
    results.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def test_blake3_matches_library():
    print("\nBLAKE3 / BLAKE3e vs the blake3 library")
    payloads = [b"", b"h", b"hello", b"a" * 63, b"a" * 64, b"a" * 65,
                REFERENCE_PAYLOAD, bytes(range(256)) * 4, b"x" * 1024,
                b"x" * 1025, b"z" * 4096, b"q" * 10000]
    ok = True
    for payload in payloads:
        want = blake3(payload).digest(32)
        ok &= BLAKE3(payload) == want
        ok &= BLAKE3e(payload) == blake3_iter(payload, 8)
    check(f"{len(payloads)} payloads, 0..10000 bytes", ok)


def blake3_iter(data, rounds):
    for _ in range(rounds):
        data = blake3(data).digest(32)
    return data


def test_blake3e_chain():
    print("\nBLAKE3e iterates exactly 8 rounds")
    payload = REFERENCE_PAYLOAD
    manual = payload
    for _ in range(8):
        manual = BLAKE3(manual)
    check("8 rounds, byte-identical", BLAKE3e(payload) == manual)


def test_known_vector():
    print("\nKnown test vector")
    got = dHeavyHash(REFERENCE_PAYLOAD)
    check("reference payload digest", got == REFERENCE_DIGEST, got.hex())
    check("digest is 32 bytes", len(got) == 32)


def test_determinism():
    print("\nDeterminism")
    got = [dHeavyHash(REFERENCE_PAYLOAD) for _ in range(10)]
    check("10 repeated calls agree", all(x == got[0] for x in got))
    other = dHeavyHash(b"test block header data for mining!")
    check("different input -> different digest", other != got[0])
    check("digest differs from plain BLAKE3e(data)",
          got[0] != BLAKE3e(REFERENCE_PAYLOAD))


def test_avalanche():
    print("\nAvalanche on 1-bit input change")
    base = bytearray(REFERENCE_PAYLOAD)
    digests = [dHeavyHash(bytes(base))]
    for bit in range(8):
        mutated = bytearray(base)
        mutated[0] ^= 1 << bit
        digests.append(dHeavyHash(bytes(mutated)))
    diffs = []
    for i in range(1, len(digests)):
        x = np.frombuffer(digests[0], dtype=np.uint8) ^ np.frombuffer(digests[i], dtype=np.uint8)
        diffs.append(int(np.unpackbits(x).sum()))
    mean = sum(diffs) / len(diffs)
    check("all 8 flips change the digest", len(set(digests)) == len(digests))
    check("about half the bits differ", 80 <= mean <= 176,
          f"mean={mean:.1f}/256 bits, min={min(diffs)}, max={max(diffs)}")


def test_candidate_semantics():
    print("\nCandidate loop semantics (same stream as NumPy)")
    data = BLAKE3e(REFERENCE_PAYLOAD)
    seed = int.from_bytes(BLAKE3(data + b"dHeavyHash"), "big")
    rng = np.random.default_rng(seed)

    candidates = []
    while True:
        mat1 = rng.integers(-128, 128, size=(64, 64), dtype=np.int8)
        for _ in range(3):
            mat2 = rng.integers(-128, 128, size=(64, 64), dtype=np.int8)
            mat1 = mat1 @ mat2
        candidates.append(mat1)
        if is_full_rank(mat1):
            break

    digest = dHeavyHash(REFERENCE_PAYLOAD)
    check("first full-rank candidate produces the digest",
          digest == BLAKE3e(candidates[-1].tobytes()))
    check("no earlier candidate was full rank",
          not any(is_full_rank(c) for c in candidates[:-1]))
    check("matrix is 64x64 int8",
          candidates[-1].shape == (64, 64) and candidates[-1].dtype == np.int8,
          f"tried {len(candidates)} candidate(s)")


def test_rank_detection():
    print("\nFull-rank test")
    rng = np.random.default_rng(2024)
    random_mats = rng.integers(-128, 128, (12, 64, 64), dtype=np.int8)
    ok_random = all(is_full_rank(m) for m in random_mats)
    check("12 random int8 matrices are full rank", ok_random)

    singular = []
    a = rng.integers(-128, 128, (64, 64), dtype=np.int8)
    b = a.copy()
    b[10] = b[11]                      # duplicate row -> rank 63
    singular.append(b)
    c = a.copy()
    c[0] = 0                           # zero row -> rank 63
    singular.append(c)
    d = a.copy()
    d[:, 5] = d[:, 6]                  # duplicate column -> rank 63
    singular.append(d)

    ok_singular = all(not is_full_rank(m) for m in singular)
    check("duplicate row/column and zero row rejected", ok_singular)
    ranked = [np.linalg.matrix_rank(m.astype(np.float64)) for m in singular]
    check("numpy agrees these are rank 63", ranked == [63, 63, 63], str(ranked))


def test_rank_stress():
    print("\nFull-rank test stress (400 full-rank + 400 rank-63 matrices)")
    rng = np.random.default_rng(2026)
    false_neg = 0
    min_diag = float("inf")
    for _ in range(400):
        m = rng.integers(-128, 128, (64, 64), dtype=np.int8)
        for _ in range(3):
            m = m @ rng.integers(-128, 128, (64, 64), dtype=np.int8)
        min_diag = min(min_diag, float(np.abs(np.diag(np.linalg.qr(
            m.astype(np.float64), mode="r"))).min()))
        if np.linalg.matrix_rank(m.astype(np.float64)) == 64 and not is_full_rank(m):
            false_neg += 1
    check("no false negatives on 400 full-rank candidates", false_neg == 0)
    check("QR diagonal safely above the 1e-9 tolerance", min_diag > 1e-3,
          f"min |diag R| = {min_diag:.3f}")

    false_pos = 0
    for _ in range(400):
        m = rng.integers(-128, 128, (64, 64), dtype=np.int8)
        k = int(rng.integers(0, 64))
        m[k] = m[(k + 1) % 64]
        if np.linalg.matrix_rank(m.astype(np.float64)) < 64 and is_full_rank(m):
            false_pos += 1
    check("no false positives on 400 rank-63 matrices", false_pos == 0)


def test_int8_wrap():
    print("\nint8 matmul wraps mod 256, as dHeavyHash assumes")
    rng = np.random.default_rng(7)
    a = rng.integers(-128, 128, (64, 64), dtype=np.int8)
    b = rng.integers(-128, 128, (64, 64), dtype=np.int8)
    got = a @ b
    wide = (a.astype(np.int64) @ b.astype(np.int64))
    want = ((wide + 128) % 256 - 128).astype(np.int8)
    check("dtype int8 with mod-256 wrap", got.dtype == np.int8 and np.array_equal(got, want))
    check("values stay in [-128, 127]",
          int(got.min()) >= -128 and int(got.max()) <= 127)


def test_full_rank_rate():
    print("\nFull-rank rate of the candidate stream")
    data = BLAKE3e(REFERENCE_PAYLOAD)
    seed = int.from_bytes(BLAKE3(data + b"dHeavyHash"), "big")
    rng = np.random.default_rng(seed)
    total, full = 200, 0
    for _ in range(total):
        m = rng.integers(-128, 128, (64, 64), dtype=np.int8)
        for _ in range(3):
            m = m @ rng.integers(-128, 128, (64, 64), dtype=np.int8)
        full += bool(is_full_rank(m))
    check("random candidates are almost always full rank",
          full >= total * 0.95, f"{full}/{total}")


def test_bytes_like_inputs():
    print("\nInput types")
    ok = True
    for payload in (bytearray(REFERENCE_PAYLOAD), memoryview(REFERENCE_PAYLOAD)):
        ok &= dHeavyHash(payload) == REFERENCE_DIGEST
    check("bytearray and memoryview accepted", ok)


def test_gpu_agreement():
    print("\nCPU reference vs GPU Triton miner")
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return
    import triton_miner as tm

    payloads = [b"", b"hello", REFERENCE_PAYLOAD, os.urandom(37), b"x" * 300]
    ok = True
    for payload in payloads:
        gpu, _ = tm.mine_triton(payload)
        ok &= gpu.view(-1).cpu().numpy().tobytes() == dHeavyHash(payload)
    check(f"{len(payloads)} payloads match Triton", ok)


def test_timing():
    print("\nTiming (CPU, single op)")
    dHeavyHash(REFERENCE_PAYLOAD)
    start = time.perf_counter()
    for _ in range(20):
        dHeavyHash(REFERENCE_PAYLOAD)
    dt = (time.perf_counter() - start) / 20
    print(f"  {dt*1e3:.2f} ms/op -> {1/dt:.0f} ops/s")
    check("completes in a sane time", dt < 0.1)


if __name__ == "__main__":
    print(f"Testing dHeavyHash from {os.path.abspath('dHeavyHash.py')}")
    test_blake3_matches_library()
    test_blake3e_chain()
    test_known_vector()
    test_determinism()
    test_avalanche()
    test_candidate_semantics()
    test_rank_detection()
    test_rank_stress()
    test_int8_wrap()
    test_full_rank_rate()
    test_bytes_like_inputs()
    test_gpu_agreement()
    test_timing()

    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"\n{passed} passed, {failed} failed")
    print("RESULT:", "OK" if failed == 0 else "FAILED")
