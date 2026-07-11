import os
import sys

if os.environ.get("ASHKV_ENABLE_PATCHES") == "1":
    try:
        from ashkv.adapters.sglang.hooks import SGLangHooks
        from ashkv.adapters.sglang.patcher import apply_hicache_patches
        hooks = SGLangHooks(codec_name="int8_default")
        apply_hicache_patches(hooks)
    except Exception as e:
        print(f"[ASH-KV] Failed to auto-patch worker process: {e}")
