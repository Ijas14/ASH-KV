import sys
sys.path.insert(0, "/mnt/Library/ashkv/sglang_source_codes/extracted/sglang-0.5.14/python")
sys.path.insert(0, "/mnt/Library/ashkv")

import torch
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode, RadixKey
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, EvictParams, MatchPrefixParams
import array

from ashkv.adapters.sglang.patcher import apply_radix_cache_patches
from ashkv.adapters.sglang.hooks import AshKVSGLangHooks

class MockPool:
    def __init__(self):
        self.device = "cuda"
    def allocate(self, n): return torch.arange(n)
    def free(self, idx): pass

pool = MockPool()
sglang_kv_cache = [torch.zeros((8000, 128))]
hooks = AshKVSGLangHooks(tier1_ratio=0.5, tier2_ratio=0.5)

apply_radix_cache_patches(hooks, sglang_kv_cache, pool)
radix_cache = RadixCache.create_simulated(mock_allocator=pool)

idx = torch.arange(4000)
token_ids = array.array("q", range(4000))
key = RadixKey(token_ids)

radix_cache.insert(InsertParams(key=key, value=idx, priority=0, chunked=False))
print("Inserted")

radix_cache.evict(EvictParams(4000))
print("Evicted")

res = radix_cache.match_prefix(MatchPrefixParams(key))
print("Matched:", res)
