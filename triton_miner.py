"""GPU dHeavyHash miner + PyTorch CUDA vs Triton comparison.

Everything after reading the input bytes runs on the GPU:

1. ``BLAKE3e`` of the header              -> Triton BLAKE3 kernels
2. ``BLAKE3(BLAKE3e(data) + b"dHeavyHash")`` -> Triton BLAKE3 kernel
3. ``SeedSequence`` + PCG64 seeding      -> Triton kernels
4. candidate matrix generation           -> Triton PCG64 kernel (NumPy-exact)
5. three int8 64x64 multiplications      -> Triton ``tl.dot`` (int8 -> int32)
6. full-rank test                        -> ``torch.linalg.qr`` (fp64)
7. final ``BLAKE3e`` over 4096 bytes     -> Triton BLAKE3 kernels

No tensor is copied back to the CPU while mining; only the final 32-byte digest
is read.  ``mine_torch`` uses the same GPU-generated seed and matrices but does
the multiplications with ``torch.bmm`` and hashing with the pure PyTorch
implementation, so the comparison isolates the Triton kernels.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dHeavyHash import dHeavyHash as cpu_dHeavyHash
from gpu_blake3 import blake3_gpu, blake3e_gpu, blake3_torch, blake3e_torch
from gpu_miner_kernels import chain3, fast_full_rank
from gpu_rng import pcg_states_from_digest, pcg_int8_matrices

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda"
PERSON = b"dHeavyHash"


def _bytes_tensor(payload: bytes) -> torch.Tensor:
    data = np.frombuffer(payload, dtype=np.uint8)
    return torch.from_numpy(data.copy()).to(DEVICE)


@torch.no_grad()
def seed_digest(payload: bytes) -> torch.Tensor:
    """32-byte ``BLAKE3(BLAKE3e(payload) + b"dHeavyHash")``, computed on GPU."""
    data = _bytes_tensor(payload).unsqueeze(0)
    inner = blake3e_gpu(data)
    person = _bytes_tensor(PERSON).unsqueeze(0)
    return blake3_gpu(torch.cat([inner, person], dim=1)).view(32)


@torch.no_grad()
def candidate_matrices(digest: torch.Tensor, count: int, offset: int = 0):
    """The 4 random int8 matrices of candidates ``offset .. offset+count``.

    A candidate consumes exactly four 64x64 matrices (4 * 512 = 2048 PCG64
    words), matching ``numpy.random.Generator.integers`` on one stream.
    """
    states, incs = pcg_states_from_digest(digest, offset + count, words_per_matrix=2048)
    if offset:
        states = states[offset:].contiguous()
        incs = incs[offset:].contiguous()
    mats = pcg_int8_matrices(states, incs, 4).view(count, 4, 64, 64)
    return mats.permute(1, 0, 2, 3).contiguous()  # (4, count, 64, 64)


@torch.no_grad()
def triton_candidates(digest: torch.Tensor, count: int = 4096, offset: int = 0):
    """``count`` Triton candidates and their full-rank mask."""
    mats = candidate_matrices(digest, count, offset)
    cand = chain3(mats[0], mats[1], mats[2], mats[3])
    return cand, exact_full_rank(cand)


@torch.no_grad()
def exact_full_rank(mats: torch.Tensor) -> torch.Tensor:
    """GPU counterpart of ``dHeavyHash.is_full_rank`` (QR, tol 1e-9).

    Uses an fp32 QR fast path and re-checks only the rare ambiguous matrices in
    fp64, which is exact for this input domain (see ``fast_full_rank``).
    """
    return fast_full_rank(mats)


def _torch_int8_chain(mats):
    """``mats[0] @ mats[1] @ mats[2] @ mats[3]`` with wrap after every step."""
    out = torch.bmm(mats[0].float(), mats[1].float())
    out = torch.remainder(out, 256.0)
    for k in range(2, 4):
        out = torch.bmm(out, mats[k].float())
        out = torch.remainder(out, 256.0)
    out = torch.where(out > 127, out - 256, out)
    return out.to(torch.int8)


@torch.no_grad()
def torch_candidates(digest: torch.Tensor, count: int = 4096, offset: int = 0):
    """Same candidates as :func:`triton_candidates`, via ``torch.bmm``."""
    mats = candidate_matrices(digest, count, offset)
    cand = _torch_int8_chain(mats)
    return cand, exact_full_rank(cand)


@torch.no_grad()
def _mine(payload, make_candidates, hash_fn, candidates):
    """Generate growing batches until the first full-rank candidate is found."""
    t0 = time.perf_counter()
    digest = seed_digest(payload)
    tried = 0
    batch = 1
    while tried < candidates:
        count = min(batch, candidates - tried)
        cand, mask = make_candidates(digest, count, tried)
        if bool(mask.any()):
            idx = int(mask.nonzero()[0])
            flat = cand[idx].contiguous().view(torch.uint8).reshape(1, 4096)
            return hash_fn(flat), time.perf_counter() - t0
        tried += count
        batch = min(batch * 4, 4096)
    return None, time.perf_counter() - t0


@torch.no_grad()
def mine_triton(payload: bytes, candidates: int = 4096):
    """Mine dHeavyHash fully on the GPU with Triton kernels."""
    return _mine(payload, triton_candidates, blake3e_gpu, candidates)


@torch.no_grad()
def mine_torch(payload: bytes, candidates: int = 4096):
    """PyTorch CUDA pipeline: same GPU RNG and candidates, ``torch.bmm`` + BLAKE3."""
    return _mine(payload, torch_candidates, blake3e_torch, candidates)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def _timeit(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def benchmark_hash(payload: bytes, count: int = 256, repeat: int = 5):
    data = _bytes_tensor(payload).unsqueeze(0).expand(count, -1).contiguous()
    print(f"\n=== BLAKE3 on GPU: {len(payload)} bytes x {count} ===")
    for name, fn in (("Triton ", blake3_gpu), ("PyTorch", blake3_torch)):
        dt = _timeit(lambda: fn(data), repeat=repeat)
        print(f"  {name:9s} : {dt*1e3:9.2f} ms  {count/dt:10.0f} hashes/s"
              f"  {count*len(payload)/dt/1e6:9.1f} MB/s")
    for name, fn in (("Triton ", blake3e_gpu), ("PyTorch", blake3e_torch)):
        dt = _timeit(lambda: fn(data), repeat=max(2, repeat // 2))
        print(f"  {name} x8: {dt*1e3:9.2f} ms  {count/dt:10.0f} hashes/s"
              f"  {count*len(payload)*8/dt/1e6:9.1f} MB/s")


def benchmark_seed(payload: bytes, repeat: int = 10):
    from dHeavyHash import BLAKE3, BLAKE3e
    print("\n=== Seed derivation ===")
    dt = _timeit(lambda: seed_digest(payload), repeat=repeat)
    print(f"  Triton GPU    : {dt*1e3:9.3f} ms")
    def cpu_seed():
        inner = BLAKE3e(payload)
        return BLAKE3(inner + PERSON)
    cpu_t = _timeit_cpu(cpu_seed, repeat)
    print(f"  CPU blake3    : {cpu_t*1e3:9.3f} ms")
    gpu = seed_digest(payload).cpu().numpy().tobytes()
    print(f"  match CPU     : {gpu == cpu_seed()}")


def benchmark_rng(count: int = 4096, repeat: int = 5):
    print(f"\n=== Random int8 64x64 matrices: {count} ===")
    digest = seed_digest(b"rng benchmark")
    states, incs = pcg_states_from_digest(digest, count)

    def triton_rng():
        return pcg_int8_matrices(states, incs, 1)

    dt = _timeit(triton_rng, repeat=repeat)
    print(f"  Triton PCG64  : {dt*1e3:9.2f} ms  {count/dt/1e3:9.1f}k matrices/s")

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(123456789)

    def torch_rng():
        return torch.randint(-128, 128, (count, 4096), dtype=torch.int8,
                             device=DEVICE, generator=gen)

    dt = _timeit(torch_rng, repeat=repeat)
    print(f"  torch.randint : {dt*1e3:9.2f} ms  {count/dt/1e3:9.1f}k matrices/s"
          "   (different stream, not NumPy-compatible)")


def benchmark_matmul(count: int = 4096, repeat: int = 5):
    digest = seed_digest(b"matmul benchmark")
    states, incs = pcg_states_from_digest(digest, count)
    mats = pcg_int8_matrices(states, incs, 4).view(count, 4, 64, 64)
    mats = mats.permute(1, 0, 2, 3).contiguous()
    a, b = mats[0], mats[1]

    print(f"\n=== Batched 64x64 int8 matmul x3: {count} ===")

    def triton_chain():
        return chain3(mats[0], mats[1], mats[2], mats[3])

    dt = _timeit(triton_chain, repeat=repeat)
    print(f"  Triton int8   : {dt*1e3:9.2f} ms  {count/dt/1e3:9.1f}k chains/s")

    def torch_chain(dtype):
        out = a.to(dtype)
        for k in range(1, 4):
            out = torch.bmm(out, mats[k].to(dtype))
            out = torch.remainder(out, 256.0)
        return torch.where(out > 127, out - 256, out)

    for name, dtype in (("torch fp32", torch.float32), ("torch fp64", torch.float64)):
        dt = _timeit(lambda: torch_chain(dtype), repeat=max(2, repeat // 2))
        print(f"  {name:12s}: {dt*1e3:9.2f} ms  {count/dt/1e3:9.1f}k chains/s")

    ref = triton_chain()
    for dtype in (torch.float32, torch.float64):
        same = torch.equal(ref, torch_chain(dtype).to(torch.int8))
        print(f"  {str(dtype).split('.')[-1]:12s} exact vs Triton: {same}")


def benchmark_end_to_end(payload: bytes, count: int = 4096, repeat: int = 5):
    print("\n=== End-to-end dHeavyHash ===")
    cpu_digest = cpu_dHeavyHash(payload)
    print(f"  CPU reference   : {cpu_digest.hex()}")

    cpu_time = _timeit_cpu(lambda: cpu_dHeavyHash(payload), repeat=5)
    print(f"  CPU NumPy       : {cpu_time*1e3:9.2f} ms/op")

    triton_digest, _ = mine_triton(payload)
    value = triton_digest.view(-1).cpu().numpy().tobytes()
    print(f"  Triton GPU      : {value.hex()}  match={value == cpu_digest}")

    torch_digest, _ = mine_torch(payload, candidates=count)
    torch_value = torch_digest.view(-1).cpu().numpy().tobytes()
    print(f"  PyTorch CUDA    : {torch_value.hex()}  match={torch_value == cpu_digest}")

    triton_t = _timeit(lambda: mine_triton(payload, candidates=count), repeat=repeat)
    torch_t = _timeit(lambda: mine_torch(payload, candidates=count),
                      repeat=max(2, repeat // 2))
    print(f"  Triton latency  : {triton_t*1e3:9.2f} ms (first full-rank candidate)")
    print(f"  PyTorch latency : {torch_t*1e3:9.2f} ms (first full-rank candidate)")
    print(f"  speedup         : {torch_t/triton_t:9.2f}x")

    digest = seed_digest(payload)
    one = candidate_matrices(digest, 1)
    single = chain3(one[0], one[1], one[2], one[3])
    flat = single.contiguous().view(torch.uint8).reshape(1, 4096)
    print("  Triton stages   : "
          f"seed {_timeit(lambda: seed_digest(payload))*1e3:5.2f} ms, "
          f"rng+matmul {_timeit(lambda: chain3(*candidate_matrices(digest, 1)))*1e3:5.2f} ms, "
          f"rank {_timeit(lambda: exact_full_rank(single))*1e3:5.2f} ms, "
          f"blake3 x8 {_timeit(lambda: blake3e_gpu(flat))*1e3:5.2f} ms")

    for name, fn in (("Triton ", triton_candidates), ("PyTorch", torch_candidates)):
        bt = _timeit(lambda: fn(digest, count), repeat=repeat)
        print(f"  {name} batch    : {count} candidates (RNG+3 matmul+QR) in "
              f"{bt*1e3:7.2f} ms -> {count/bt/1e3:6.0f}k candidates/s")


def _timeit_cpu(fn, repeat=10):
    for _ in range(2):
        fn()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) / repeat


def verify(payload: bytes = b"test block header data for mining"):
    print("Verifying Triton GPU miner against the CPU reference")
    cpu_digest = cpu_dHeavyHash(payload)
    gpu_digest, dt = mine_triton(payload)
    value = gpu_digest.view(-1).cpu().numpy().tobytes()
    ok = value == cpu_digest
    print("  CPU   :", cpu_digest.hex())
    print("  GPU   :", value.hex())
    print("  match :", ok, f"({dt*1e3:.1f} ms)")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required")
        sys.exit(1)
    print("Device:", torch.cuda.get_device_name(0))
    payload = b"test block header data for mining"
    ok = verify(payload)
    benchmark_seed(payload)
    benchmark_hash(payload, count=256)
    benchmark_hash(b"x" * 4096, count=32)
    benchmark_rng(4096)
    benchmark_matmul(4096)
    benchmark_end_to_end(payload)
    print("\nRESULT:", "OK" if ok else "MISMATCH")
