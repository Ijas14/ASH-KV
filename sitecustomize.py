import sys
import threading
import time

def delayed_patch():
    # Log to a persistent file so we can prove this thread is running in the Ray worker
    with open("stats/ashkv_patch.log", "a") as f:
        f.write("[ASH-KV] Background thread started in process.\n")
        
    while "sglang.srt.mem_cache.memory_pool_host" not in sys.modules:
        time.sleep(0.5)
        
    time.sleep(2.0)
    
    try:
        from ashkv.adapters.sglang.hooks import SGLangHooks
        from ashkv.adapters.sglang.patcher import apply_hicache_patches
        
        hooks = SGLangHooks(codec_name="int8_default")
        apply_hicache_patches(hooks)
        
        with open("stats/ashkv_patch.log", "a") as f:
            f.write("[ASH-KV] Patches APPLIED successfully in background thread.\n")
    except Exception as e:
        with open("stats/ashkv_patch.log", "a") as f:
            f.write(f"[ASH-KV] Failed to auto-patch: {e}\n")

# Ray drops environment variables by default. To defeat this, we run unconditionally.
# If the process never imports SGLang, the thread just sleeps harmlessly forever.
threading.Thread(target=delayed_patch, daemon=True).start()
