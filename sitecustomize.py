import os
import sys
import threading
import time

def delayed_patch():
    # Wait until the target module is naturally imported by SGLang
    while "sglang.srt.mem_cache.memory_pool_host" not in sys.modules:
        time.sleep(0.1)
        
    # Give the SGLang init sequence a moment to finish its massive imports (like Aiter/Triton)
    time.sleep(2.0)
    
    try:
        from ashkv.adapters.sglang.hooks import SGLangHooks
        from ashkv.adapters.sglang.patcher import apply_hicache_patches
        
        hooks = SGLangHooks(codec_name="int8_default")
        apply_hicache_patches(hooks)
    except Exception as e:
        print(f"[ASH-KV] Failed to auto-patch worker process in background thread: {e}")

if os.environ.get("ASHKV_ENABLE_PATCHES") == "1":
    # Run in a daemon thread so it doesn't block PyTorch/SGLang initialization 
    # which causes deadlocks during aiter/triton JIT compilation.
    threading.Thread(target=delayed_patch, daemon=True).start()
