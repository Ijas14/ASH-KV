import torch
import time
from typing import List, Optional
import sys

# Attempt to import real SGLang
try:
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
except ImportError:
    print("SGLang not found. Please run this script on the MI300X instance where SGLang 0.5.14 is installed.")
    sys.exit(0)

from ashkv.sglang_integration.allocator import SGLangShadowAllocator
from ashkv.sglang_integration.hooks import SGLangHooks
from ashkv.sglang_integration.patcher import apply_radix_cache_patches

class MockTokenToKVPool:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.allocated_tokens = 0
        self.next_idx = 0
        
    def allocate(self, num_tokens):
        if self.allocated_tokens + num_tokens > self.capacity:
            return None
        idx = torch.arange(self.next_idx, self.next_idx + num_tokens, device="cuda")
        self.allocated_tokens += num_tokens
        self.next_idx += num_tokens
        return idx
        
    def free(self, indices):
        # Simplistic free
        if indices is not None:
            self.allocated_tokens -= len(indices)

class MockModelConfig:
    def __init__(self):
        self.num_hidden_layers = 32

print("--- INITIALIZING ASH-KV ---")
pool = MockTokenToKVPool(capacity=20000)
shadow_alloc = SGLangShadowAllocator(max_bytes=100 * 1024 * 1024)
sglang_kv_cache = torch.zeros((32, 20000, 128), dtype=torch.bfloat16, device="cuda")

hooks = SGLangHooks(page_table=None, shadow_allocator=shadow_alloc, model_config=MockModelConfig(), codec_name="int8_default")

# Apply patches to the real SGLang class!
apply_radix_cache_patches(hooks, sglang_kv_cache, pool)
print("Patches applied to real SGLang RadixCache.")

# Create RadixCache
# 0.5.14 RadixCache takes req_capacity and some other parameters
try:
    radix_cache = RadixCache(req_capacity=8000, triton_backend="default")
except TypeError:
    try:
        # Fallback for some versions
        radix_cache = RadixCache(req_capacity=8000, block_size=1)
    except:
        radix_cache = RadixCache(req_capacity=8000)

print("\n--- TEST: INSERTION ---")
# Simulate system prompt insertion
num_tokens = 4000
idx = pool.allocate(num_tokens)
synthetic_data = torch.randn((num_tokens, 128), dtype=torch.bfloat16, device="cuda")
sglang_kv_cache[0][idx] = synthetic_data

# In SGLang, insert returns a node (or tuple, etc)
# v0.5.x takes a token_ids tuple and returns node or node-like
token_ids = tuple(range(num_tokens))
try:
    node_id = radix_cache.insert(token_ids, idx)
except TypeError:
    # Handle dataclass insert if 0.5.14 changed it again
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    result = radix_cache.insert(InsertParams(key=token_ids, value=idx))
    node_id = result

print(f"Inserted {num_tokens} tokens into RadixCache.")

print("\n--- TEST: EVICTION INTERCEPT ---")
# Force eviction of the tokens
evicted = radix_cache.evict(num_tokens)
print(f"Evicted tokens reported by SGLang: {evicted}")

# Check if the node was compressed and kept alive
matched = radix_cache.match_prefix(token_ids)
matched_node = getattr(matched, "last_device_node", matched[0] if isinstance(matched, tuple) and len(matched)>0 else None)

if matched_node is not None:
    has_shadow = getattr(matched_node, "ashkv_shadow_handle", None) is not None
    is_value_none = matched_node.value is None
    print(f"Node found in tree after evict.")
    print(f"Has shadow handle: {has_shadow}")
    print(f"node.value is None: {is_value_none}")
    assert has_shadow, "Demote hook failed to attach shadow handle!"
    assert is_value_none, "Demote hook failed to clear node.value!"
else:
    print("Node deleted from tree! Evict patch failed.")

print("\n--- TEST: PREFIX MATCH (PROMOTION) ---")
# Clear VRAM to prove promotion restores it
sglang_kv_cache.zero_()

# Re-run match prefix which should trigger promotion
# We already matched above, but let's do it again to test promote logic directly
result = radix_cache.match_prefix(token_ids)

matched_node = getattr(result, "last_device_node", result[0] if isinstance(result, tuple) and len(result)>0 else None)
assert matched_node is not None, "Failed to match node"
has_shadow = getattr(matched_node, "ashkv_shadow_handle", None) is not None
print(f"Has shadow handle after match: {has_shadow}")
assert not has_shadow, "Promote hook failed to remove shadow handle!"
assert matched_node.value is not None, "Promote hook failed to set new node.value indices!"

restored_data = sglang_kv_cache[0][matched_node.value]
is_close = torch.allclose(synthetic_data, restored_data, atol=5e-2)
print(f"Data perfectly reconstructed: {is_close}")
assert is_close, "Data corruption!"

print("\nAll integration tests passed against real SGLang RadixCache!")
