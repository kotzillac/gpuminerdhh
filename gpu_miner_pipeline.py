"""GPU dHeavyHash miner: PCG64 -> int8 matmul chain -> rank test -> BLAKE3.

Everything after the initial BLAKE3 seed derivation happens on the GPU. The
candidate stream is identical to the CPU reference sampler, so the miner finds
exactly the same first candidate as ``dHeavyHash.dHeavyHash``.
"""

import os
import time

import numpy as np
import torch

from dHeavyHash import BLAKE3, BLAKE3e, dHeavyHash as cpu_dHeavyHash
from gpu_blake3 import blake3_gpu, blake3e_gpu, blake3_torch, blake3e_torch
from gpu_rng import pcg_states, pcg_int8_matrices

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HERE, "triton_cache"))


@torch.no_grad()
def _matmul_chain(mats):
    """``m0 @ m1 @ m2 @ m3`` for a batch of int8 64x64 matrices."""
    from gpu_miner_kernels import batched_int8_matmul

    out = mats[:, 0]
    out = batched_int8_matmul(out, mats[:, 1])
    out = batched_int8_matmul(out, mats[:, 2])
    out = batched_int8_matmul(out, mats[:, 3])
    return out


@torch.no_grad()
def gpu_dheavyhash(data: bytes, batch: int = 4096, verify_rank: bool = True):
    """Compute dHeavyHash fully on the GPU using the CPU-identical stream.

    The CPU reference draws candidates from one PCG64 stream seeded with
    ``BLAKE3(BLAKE3e(data) + b"dHeavyHash")``. That stream is reproduced on the
    GPU, so this returns the same digest as ``dHeavyHash.dHeavyHash``.
    """
    seed = int.from_bytes(BLAKE3(BLAKE3e(data) + b"dHeavyHash"), "big")

    # The CPU sampler draws from a single PCG64 stream: matrix k uses words
    # [k*512, (k+1)*512). Reproduce that by generating candidates in order.
    states, incs = pcg_states([seed])
    chunk = max(batch, 16384)
    mats = pcg_int8_matrices(states, incs, 4 * chunk)
    mats = mats.view(chunk, 4, 64, 64)

    from gpu_miner_kernels import batched_int8_matmul

    for start in range(0, chunk, batch):
        block = mats[start:start + batch]
        cand = _matmul_chain(block)
        if verify_rank:
            sign, _ = torch.linalg.slogdet(cand.to(torch.float64))
            mask = sign != 0
            if mask.any():
                idx = int(mask.nonzero()[0])
                return _finish(cand[idx])
        else:
            return _finish(cand[0])
    raise RuntimeError("no candidate found in the generated batch")


def _finish(matrix):
    flat = matrix.contiguous().view(torch.uint8).reshape(1, 4096)
    return blake3e_gpu(flat)


@torch.no_grad()
def mine_dheavyhash_gpu(data: bytes, candidates: int = 4096, batch: int = 4096):
    """Mine with a GPU-resident BLAKE3 and int8 matmul pipeline."""
    t0 = time.time()
    seed = int.from_bytes(BLAKE3(BLAKE3e(data) + b"dHeavyHash"), "big")

    states, incs = pcg_states([seed])
    mats = pcg_int8_matrices(states, incs, 4 * candidates)
    mats = mats.view(candidates, 4, 64, 64)

    from gpu_miner_kernels import batched_int8_matmul

    found = None
    for start in range(0, candidates, batch):
        stop = min(start + batch, candidates)
        block = mats[start:stop]
        cand = _matmul_chain(block)
        sign, _ = torch.linalg.slogdet(cand.to(torch.float64))
        mask = sign != 0
        if mask.any():
            idx = int(mask.nonzero()[0])
            found = _finish(cand[idx])
            break
    return found, time.time() - t0


if __name__ == "__main__":
    payload = b"test block header data for mining"
    print("CPU  :", cpu_dHeavyHash(payload).hex())
    h, dt = mine_dheavyhash_gpu(payload)
    print("GPU  :", h.view(-1).cpu().numpy().tobytes().hex())
    print(f"time : {dt*1e3:.2f} ms")
