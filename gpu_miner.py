# GPU Miner for dHeavyHash using PyTorch CUDA
# Copyright (C) 2026

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dHeavyHash import BLAKE3, BLAKE3e, dHeavyHash as cpu_dHeavyHash, is_full_rank

import numpy as np
import torch
import time


def mine_dheavyhash_gpu(data, num_blocks=256, max_iters_per_block=1000):
    # Use the same seed as CPU: BLAKE3(BLAKE3e(data) + b'dHeavyHash')
    initial_hash = BLAKE3e(data)
    seed = int.from_bytes(BLAKE3(initial_hash + b'dHeavyHash'), byteorder='big')
    
    print('Initial seed:', seed)
    print('Launching', num_blocks, 'blocks,', max_iters_per_block, 'iterations each')
    print('Device:', torch.cuda.get_device_name(0))
    
    start_time = time.time()
    device = 'cuda'
    
    found = False
    final_matrix = None
    final_nonce = 0
    
    for block in range(num_blocks):
        block_seed = seed ^ (block * 0x9e3779b97f4a7c15)
        rng = np.random.default_rng(seed=block_seed)
        
        for iteration in range(max_iters_per_block):
            mat1_cpu = rng.integers(low=-128, high=128, size=(64, 64), dtype=np.int8)
            
            if not is_full_rank(mat1_cpu):
                continue
            
            mat1_gpu = torch.from_numpy(mat1_cpu.astype(np.float64)).to(device)
            
            for _ in range(3):
                mat2_cpu = rng.integers(low=-128, high=128, size=(64, 64), dtype=np.int8)
                mat2_gpu = torch.from_numpy(mat2_cpu.astype(np.float64)).to(device)
                mat1_gpu = mat1_gpu @ mat2_gpu
                mat1_gpu = torch.fmod(mat1_gpu, 256)
                mat1_gpu = torch.where(mat1_gpu > 127, mat1_gpu - 256, mat1_gpu)
                mat1_gpu = torch.where(mat1_gpu < -128, mat1_gpu + 256, mat1_gpu)
            
            found = True
            final_matrix = np.round(mat1_gpu.cpu().numpy()).astype(np.int8)
            final_nonce = block * max_iters_per_block + iteration
            break
        
        if found:
            break
        
        if block % 20 == 0 and block > 0:
            elapsed = time.time() - start_time
            print('Progress: block {}/{}, elapsed: {:.1f}s'.format(block, num_blocks, elapsed))
    
    elapsed = time.time() - start_time
    print('Mining completed in', elapsed, 's')
    
    if found:
        final_hash = BLAKE3e(final_matrix.tobytes(order='C'))
        return final_hash, final_nonce, num_blocks * max_iters_per_block
    
    return None, None, num_blocks * max_iters_per_block


def benchmark_cpu_vs_gpu(data, num_iterations=10):
    print('\n=== Benchmarking ===')
    
    print('Running CPU benchmark...')
    start = time.time()
    for i in range(num_iterations):
        _ = cpu_dHeavyHash(data + i.to_bytes(4, 'little'))
    cpu_time = time.time() - start
    print('CPU: {} iterations in {:.3f}s ({:.2f} iters/sec)'.format(num_iterations, cpu_time, num_iterations/cpu_time))
    
    print('Running GPU benchmark...')
    gpu_iters = 1000
    start = time.time()
    result, nonce, total_iters = mine_dheavyhash_gpu(data, num_blocks=512, max_iters_per_block=gpu_iters)
    gpu_time = time.time() - start
    print('GPU: {} iterations in {:.3f}s ({:.2f} iters/sec)'.format(gpu_iters, gpu_time, gpu_iters/gpu_time))


if __name__ == '__main__':
    test_data = b'test block header data for mining'
    
    print('Testing dHeavyHash GPU Miner (PyTorch CUDA)')
    print('=' * 50)
    
    print('Verifying CPU implementation...')
    cpu_result = cpu_dHeavyHash(test_data)
    print('CPU result:', cpu_result.hex())
    
    print('\nRunning GPU miner...')
    result, nonce, total_iters = mine_dheavyhash_gpu(test_data, num_blocks=64, max_iters_per_block=10000)
    
    if result:
        print('GPU result:', result.hex())
        print('Nonce:', nonce)
        print('Total iterations:', total_iters)
        print('Match CPU:', result == cpu_result)
    else:
        print('No solution found in given iterations')
        print('Try increasing max_iters_per_block or num_blocks')
    
    benchmark_cpu_vs_gpu(test_data, num_iterations=20)
