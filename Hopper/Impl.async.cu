// --- HOPPER (sm_90) CONCEPTUAL PIPELINE ---

// 1. Host or Device Setup: Create TMA Descriptor for Global Memory Tensor
CudaTmaDesc tma_desc_a = create_2d_tma_descriptor(A_ptr, M, K, tile_m, tile_k);

// 2. In Kernel (Single thread triggers TMA load)
if (threadIdx.x == 0) {
    // Tell barrier to expect X bytes from TMA
    mbarrier_expect_transaction(barrier_ptr, bytes_to_transfer);
    
    // Issue Async Hardware Copy directly Global -> Shared Memory
    tma_load_async(smem_A_ptr, &tma_desc_a, barrier_ptr, tile_row, tile_col);
}

// 3. Warpgroup Waits for Memory via Hardware Transaction Barrier
mbarrier_wait(barrier_ptr, phase);

// 4. Execute WGMMA directly out of Shared Memory (128 threads)
// Reads smem_A_ptr and smem_B_ptr directly into Tensor Cores; updates reg_C
wgmma_mma_async(reg_C, smem_A_ptr, smem_B_ptr);

// 5. Commit & Wait for WGMMA Completion
wgmma_commit_group();
wgmma_wait_group<0>();
