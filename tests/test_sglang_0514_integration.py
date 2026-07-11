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
        self.device = "cuda"
        
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
# Try importing parameter classes for v0.5.14
try:
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, EvictParams, MatchPrefixParams
except ImportError:
    class InsertParams:
        def __init__(self, key, value, priority=0, chunked=False): 
            self.key = key
            self.value = value
            self.priority = priority
            self.chunked = chunked
    class EvictParams:
        def __init__(self, num_tokens): self.num_tokens = num_tokens
    class MatchPrefixParams:
        def __init__(self, key): self.key = key

class MockRadixKey:
    def __init__(self, token_ids):
        self.token_ids = token_ids
    def maybe_to_bigram_view(self, is_eagle): return self, False
    def page_aligned(self, page_size): return self
    def __len__(self): return len(self.token_ids)
    def __hash__(self): return hash(self.token_ids)
    def __eq__(self, other): return self.token_ids == other.token_ids
    def __iter__(self): return iter(self.token_ids)
    def __getitem__(self, item): return self.token_ids[item]

print("Patches applied to real SGLang RadixCache.")

# Create RadixCache using v0.5.14's create_simulated factory method if available
try:
    radix_cache = RadixCache.create_simulated(mock_allocator=pool)
except AttributeError:
    # Fallbacks for older SGLang versions
    try:
        radix_cache = RadixCache(req_capacity=8000, triton_backend="default")
    except TypeError:
        radix_cache = RadixCache(disable=False)

# Mock some internals if RadixCache requires them
if not hasattr(radix_cache, 'page_size'):
    radix_cache.page_size = 1
if not hasattr(radix_cache, 'is_eagle'):
    radix_cache.is_eagle = False
if not hasattr(radix_cache, 'token_to_kv_pool_allocator'):
    radix_cache.token_to_kv_pool_allocator = pool
if not hasattr(radix_cache, 'evictable_leaves'):
    radix_cache.evictable_leaves = set()

print("\n--- TEST: INSERTION ---")
num_tokens = 4000
idx = pool.allocate(num_tokens)
synthetic_data = torch.randn((num_tokens, 128), dtype=torch.bfloat16, device="cuda")
sglang_kv_cache[0][idx] = synthetic_data

token_ids = tuple(range(num_tokens))
radix_key = MockRadixKey(token_ids)

try:
    radix_cache.insert(InsertParams(key=radix_key, value=idx))
except TypeError:
    radix_cache.insert(radix_key, idx)

print(f"Inserted {num_tokens} tokens into RadixCache.")

print("\n--- TEST: EVICTION INTERCEPT ---")
# Force eviction using EvictParams
try:
    evicted = radix_cache.evict(EvictParams(num_tokens))
except TypeError:
    evicted = radix_cache.evict(num_tokens)

print(f"Evicted tokens reported by SGLang: {evicted}")

# Check match_prefix
try:
    matched = radix_cache.match_prefix(MatchPrefixParams(radix_key))
except TypeError:
    matched = radix_cache.match_prefix(radix_key)

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
sglang_kv_cache.zero_()

try:
    result = radix_cache.match_prefix(MatchPrefixParams(radix_key))
except TypeError:
    result = radix_cache.match_prefix(radix_key)

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
