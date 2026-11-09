# GPU Miner for dHeavyHash using Triton
# Copyright (C) 2026

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dHeavyHash import BLAKE3, BLAKE3e, dHeavyHash as cpu_dHeavyHash, is_full_rank

import numpy as np
import time

import os
os.environ["TRITON_CACHE_DIR"] = "C:/Users/Kirill/gpuminerdhh/triton_cache"
os.makedirs("C:/Users/Kirill/gpuminerdhh/triton_cache", exist_ok=True)
try:
    import triton
    import triton.language as tl
    import torch
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not available. Install with: pip install triton")

if HAS_TRITON:
    @triton.jit
    def matmul_int8_kernel_simple(a_ptr, b_ptr, c_ptr, M, N, K):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        if pid_m >= M or pid_n >= N:
            return
        
        acc = 0
        for k in range(K):
            a = tl.load(a_ptr + pid_m * K + k)
            b = tl.load(b_ptr + k * N + pid_n)
            acc += a * b
        
        acc = acc % 256
        acc = tl.where(acc > 127, acc - 256, tl.where(acc < -128, acc + 256, acc))
        
        tl.store(c_ptr + pid_m * N + pid_n, acc)

    def matmul_int8_triton_gpu(a_gpu, b_gpu):
        M, K = a_gpu.shape
        K2, N = b_gpu.shape
        assert K == K2
        
        c_gpu = torch.empty((M, N), dtype=torch.int32, device=a_gpu.device)
        grid = (M, N)
        matmul_int8_kernel_simple[grid](a_gpu, b_gpu, c_gpu, M, N, K)
        return c_gpu.to(torch.int8)

def mine_dheavyhash_triton_gpu_only(data, num_blocks=256, max_iters_per_block=1000):
    initial_hash = BLAKE3e(data)
    seed = int.from_bytes(BLAKE3(initial_hash + b"dHeavyHash"), byteorder="big")
    
    print("Initial seed:", seed)
    print("Launching", num_blocks, "blocks,", max_iters_per_block, "iterations each")
    print("Device:", torch.cuda.get_device_name(0))
    
    start_time = time.time()
    found = False
    final_matrix = None
    final_nonce = 0
    
    for block in range(num_blocks):
        block_seed = seed ^ (block * 0x9e3779b97f4a7c15)
        
        for iteration in range(max_iters_per_block):
            iter_seed = block_seed + iteration
            gen = torch.Generator(device="cuda"); gen.manual_seed(iter_seed & 0xFFFFFFFFFFFFFFFF); mat1_gpu = torch.randint(-128, 128, (64, 64), dtype=torch.int8, device="cuda", generator=gen)
            
            mat1_cpu = mat1_gpu.cpu().numpy()
            if not is_full_rank(mat1_cpu):
                continue
            
            for step in range(3):
                gen.manual_seed((iter_seed + step + 1000) & 0xFFFFFFFFFFFFFFFF); mat2_gpu = torch.randint(-128, 128, (64, 64), dtype=torch.int8, device="cuda", generator=gen)
                mat1_gpu = matmul_int8_triton_gpu(mat1_gpu, mat2_gpu)
            found = True
            final_matrix_gpu = mat1_gpu
            final_nonce = block * max_iters_per_block + iteration
            break
        
        if found:
            break
        
        if block % 20 == 0 and block > 0:
            elapsed = time.time() - start_time
            print("Progress: block {}/{}, elapsed: {:.1f}s".format(block, num_blocks, elapsed))
    
    elapsed = time.time() - start_time
    print("Mining completed in {:.3f}s".format(elapsed))
    
    if found:
        final_hash = BLAKE3e(final_matrix_gpu.cpu().numpy().tobytes(order="C"))
        return final_hash, final_nonce, num_blocks * max_iters_per_block
    
    return None, None, num_blocks * max_iters_per_block


def benchmark_cpu_vs_triton(data, num_iterations=10):
    print("\n=== Benchmarking CPU vs Triton ===")
    
    print("Running CPU benchmark...")
    start = time.time()
    for i in range(num_iterations):
        _ = cpu_dHeavyHash(data + i.to_bytes(4, "little"))
    cpu_time = time.time() - start
    print("CPU: {} iterations in {:.3f}s ({:.2f} iter_per_s)".format(num_iterations, cpu_time, num_iterations/cpu_time))
    
    print("Running Triton benchmark (GPU-only version)...")
    gpu_iters = 100
    start = time.time()
    result, nonce, total_iters = mine_dheavyhash_triton_gpu_only(data, num_blocks=10, max_iters_per_block=gpu_iters//10)
    triton_time = time.time() - start
    print("Triton: {} iterations in {:.3f}s ({:.2f} iter_per_s)".format(gpu_iters, triton_time, gpu_iters/triton_time))


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not installed. Please install with: pip install triton")
        sys.exit(1)
    
    test_data = b"test block header data for mining"
    
    print("Testing dHeavyHash GPU Miner (Triton)")
    print("=" * 50)
    
    print("Verifying CPU implementation...")
    cpu_result = cpu_dHeavyHash(test_data)
    print("CPU result:", cpu_result.hex())
    
    print("\nRunning Triton miner...")
    result, nonce, total_iters = mine_dheavyhash_triton_gpu_only(test_data, num_blocks=64, max_iters_per_block=10000)
    
    if result:
        print("Triton result:", result.hex())
        print("Nonce:", nonce)
        print("Total iterations:", total_iters)
        print("Match CPU:", result == cpu_result)
    else:
        print("No solution found in given iterations")
        print("Try increasing max_iters_per_block or num_blocks")
    
    benchmark_cpu_vs_triton(test_data, num_iterations=20)
