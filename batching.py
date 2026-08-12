import asyncio


async def run_serving_demo():
    print("Initializing PagedAttention Pool & Continuous Batching Engine...")
    
    # Pre-allocate 128 MB equivalent physical KV block pool
    kv_manager = PagedKVCacheManager(
        num_blocks=64,
        block_size=16,
        num_heads=8,
        head_dim=64,
        device="cpu"  # CPU fallback for demonstration
    )
    
    engine = ContinuousBatchingEngine(kv_manager=kv_manager, max_batch_size=2)

    # Submit asynchronous incoming requests
    req1 = GenerationRequest("req_A", "Describe image...", prompt_tokens=[10, 20, 30], max_new_tokens=4)
    req2 = GenerationRequest("req_B", "Analyze text...", prompt_tokens=[40, 50], max_new_tokens=3)
    req3 = GenerationRequest("req_C", "Summarize video...", prompt_tokens=[60, 70, 80, 90], max_new_tokens=2)

    engine.add_request(req1)
    engine.add_request(req2)
    engine.add_request(req3)

    print("\nStarting Continuous Batching Loop:\n" + "-"*40)
    step_count = 0
    while engine.running_batch or engine.waiting_queue:
        step_count += 1
        print(f"--- Engine Iteration Step {step_count} ---")
        await engine.step()
        await asyncio.sleep(0.01)

    print("\nAll Requests Processed Successfully.")

if __name__ == "__main__":
    asyncio.run(run_serving_demo())
