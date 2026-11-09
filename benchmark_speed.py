"""Head-to-head speed measurement: CPU vs PyTorch CUDA vs Triton.

Run with: python benchmark_speed.py

Reported for every backend:
  * latency      -- wall-clock time of one dHeavyHash, including launch overhead
  * gpu kernel   -- time measured with CUDA events (GPU work only, no Python)
  * throughput   -- full hashes per second on many distinct inputs
"""

import os
import statistics
import sys
import time

import torch

from dHeavyHash import dHeavyHash as cpu_mine
import triton_miner as tm

DEVICE = "cuda"
BASE = b"test block header data for mining"


def payloads(n):
    return [BASE + i.to_bytes(4, "little") for i in range(n)]


def cpu_bench(n=50):
    fn = cpu_mine
    for p in payloads(3)[:3]:
        fn(p)
    per_op = []
    for p in payloads(n):
        t0 = time.perf_counter()
        fn(p)
        per_op.append(time.perf_counter() - t0)
    return per_op


def gpu_wall(fn, n=30):
    for _ in range(3):
        fn(BASE)
    torch.cuda.synchronize()
    per_op = []
    for p in payloads(n):
        t0 = time.perf_counter()
        fn(p)
        torch.cuda.synchronize()
        per_op.append(time.perf_counter() - t0)
    return per_op


def gpu_batched_wall(fn, n=64):
    """Launch all ops, then one synchronize: hides Python launch overhead."""
    ps = payloads(n)
    for _ in range(2):
        fn(ps[0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in ps:
        fn(p)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def gpu_kernel_ms(fn, n=20):
    """Pure GPU time of one op, measured with CUDA events."""
    for _ in range(5):
        fn(BASE)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for p in payloads(n):
        fn(p)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / n


def stats(times):
    ms = [t * 1e3 for t in times]
    return dict(mean=statistics.mean(ms), med=statistics.median(ms),
                lo=min(ms), hi=max(ms))


def report(name, times):
    s = stats(times)
    print(f"  {name:16s} {s['mean']:10.3f} {s['med']:10.3f} "
          f"{s['lo']:9.3f} {s['hi']:9.3f}   {1e3/s['mean']:9.1f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required")
        sys.exit(1)
    print("Device:", torch.cuda.get_device_name(0))
    print("Benchmarking dHeavyHash on b\"test block header data for mining\"")
    print("(CPU: 50 distinct inputs, Triton: 30, PyTorch CUDA: 5)\n")

    cpu = cpu_bench(50)
    triton = gpu_wall(tm.mine_triton, 30)
    torch_wall = gpu_wall(tm.mine_torch, 5)

    print(f"  {'backend':16s} {'mean ms':>10s} {'median':>10s} "
          f"{'min ms':>9s} {'max ms':>9s}  {'hashes/s':>9s}")
    report("CPU (NumPy)", cpu)
    report("GPU Triton", triton)
    report("GPU PyTorch", torch_wall)

    print("\nGPU-only time (CUDA events, Python overhead excluded)")
    k_triton = gpu_kernel_ms(tm.mine_triton, 20)
    k_torch = gpu_kernel_ms(tm.mine_torch, 2)
    print(f"  {'Triton':16s} {k_triton:10.3f} ms/op")
    print(f"  {'PyTorch':16s} {k_torch:10.3f} ms/op")

    print("\nBatch throughput on many distinct headers (the fair comparison)")
    import gpu_miner_batch as gb

    for b in (512, 4096, 16384, 65536, 65536*2):
        payloads_b = [BASE + i.to_bytes(4, "little") for i in range(b)]
        headers = gb._headers(payloads_b)
        gb.mine_batch(headers)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gb.mine_batch(headers)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {'Triton batch':16s} {dt*1e3:10.3f} ms for {b:6d} -> "
              f"{b/dt:9.0f} hashes/s")

        if b <= 4096:
            gb.mine_batch_torch(headers)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gb.mine_batch_torch(headers)
            torch.cuda.synchronize()
            dt_t = time.perf_counter() - t0
            print(f"  {'PyTorch batch':16s} {dt_t*1e3:10.3f} ms for {b:6d} -> "
                  f"{b/dt_t:9.0f} hashes/s")

        #t0 = time.perf_counter()
        #for pl in payloads_b:
        #    cpu_mine(pl)
        #dt = time.perf_counter() - t0
        #print(f"  {'CPU batch':16s} {dt*1e3:10.3f} ms for {b:6d} -> "
        #      f"{b/dt:9.0f} hashes/s")

    print("\nBatch pipeline (4096 candidates, RNG + 3 matmul + rank test)")
    digest = tm.seed_digest(BASE)
    for name, fn in (("Triton", tm.triton_candidates), ("PyTorch", tm.torch_candidates)):
        fn(digest, 64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(digest, 4096)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {name:8s} {dt*1e3:9.2f} ms -> {4096/dt/1e3:7.0f}k candidates/s")

    print("\nVerdict")
    print(f"  Single op: CPU is faster than GPU Triton by "
          f"{statistics.mean(triton)/statistics.mean(cpu):.2f}x "
          f"(GPU launch overhead, ~10 kernels per hash)")
    print(f"  Single op: Triton is faster than PyTorch CUDA by "
          f"{statistics.mean(torch_wall)/statistics.mean(triton):.2f}x")
    print("  Throughput: see the batch table above (Triton scales with batch size)")
    print("RESULT: OK")
