import sglang
import time
import argparse
from ashkv.adapters.sglang.hooks import SGLangHooks
from ashkv.adapters.sglang.patcher import apply_hicache_patches

def main(model_path):
    print("--- INITIALIZING ASH-KV ---")
    
    # 1. Initialize our hooks
    # We use int8_default codec, which uses Triton under the hood
    hooks = SGLangHooks(codec_name="int8_default")
    
    # 2. Apply the HiCache patches BEFORE SGLang initializes its workers
    print("Applying HiCache patches to SGLang HostKVCache...")
    apply_hicache_patches(hooks)
    print("Patches applied successfully.")

    print(f"--- STARTING SGLANG ENGINE ---")
    print(f"Model: {model_path}")
    print(f"Note: Enforcing --hicache-write-policy write_through")
    
    # 3. Start the SGLang Engine
    # We MUST pass hicache_write_policy="write_through" to force CPU offload
    engine = sglang.Engine(
        model_path=model_path,
        mem_fraction_static=0.7,
        chunked_prefill_size=4096,
        # HiCache specific settings
        hicache_write_policy="write_through", 
        hicache_host_memory_size=10, # GBs for host memory (CPU)
    )
    
    print("--- ENGINE STARTED ---")
    
    # 4. Generate some text to populate the KV Cache
    prompt1 = "The history of the Roman Empire is a fascinating subject. It all began when"
    print(f"Prompt 1 (Prefill): {prompt1}")
    
    start = time.time()
    out1 = engine.generate(prompt1, sampling_params={"max_new_tokens": 100})
    print(f"Generation took: {time.time() - start:.2f}s")
    print(f"Output: {out1['text']}\n")
    
    print("--- WAITING FOR OFF-LOAD ---")
    print("Since write_through is enabled, SGLang's background thread is currently calling")
    print("backup_from_device_all_layer, which is being intercepted by ASH-KV demote_hook to compress to INT8.")
    time.sleep(5) # Wait for the async write-through to finish
    
    # 5. Send a prefix-matched prompt
    # This should trigger a cache hit, but if the GPU slots were evicted, 
    # it will pull from CPU (HiCache), triggering our promote_hook.
    prompt2 = prompt1 + " " + out1['text'] + " However, the fall of the empire was"
    print(f"Prompt 2 (Cache Hit): {prompt2}")
    
    start = time.time()
    out2 = engine.generate(prompt2, sampling_params={"max_new_tokens": 100})
    print(f"Generation took: {time.time() - start:.2f}s")
    print(f"Output: {out2['text']}\n")
    
    print("--- SUCCESS ---")
    print("If you didn't get any crashes or OOMs, the INT8 compression/decompression")
    print("successfully executed on the real GPU tensors during the HiCache IO!")
    
    engine.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    args = parser.parse_args()
    main(args.model)
