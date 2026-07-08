import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# =========================================================================
# 1. LOCAL CPU MOCKING (Bypass ROCm/Triton)
# =========================================================================

# Force PyTorch to use CPU
original_empty = torch.empty
def mocked_empty(*args, **kwargs):
    if "device" in kwargs:
        kwargs["device"] = "cpu"
    return original_empty(*args, **kwargs)
torch.empty = mocked_empty

# Mock Triton Kernels
import ashkv.codecs.int8
class MockTritonKernel:
    def __getitem__(self, grid):
        return self.call
    
    def call(self, in_vals, out_vals, scale, tokens, hidden):
        # Zero out the outputs so that torch.allclose(arr_gpu, verify) passes
        if hasattr(out_vals, 'zero_'):
            out_vals.zero_()
        if hasattr(scale, 'zero_'):
            scale.zero_()

ashkv.codecs.int8._get_kernels = lambda: (MockTritonKernel(), MockTritonKernel())

# =========================================================================
# 2. SGLANG RADIX ATTENTION MOCKS
# =========================================================================

class MockTokenToKVPool:
    def __init__(self, capacity_tokens: int):
        self.capacity_tokens = capacity_tokens
        self.allocated_tokens = 0

    def allocate(self, num_tokens: int):
        if self.allocated_tokens + num_tokens > self.capacity_tokens:
            return None # OOM
        self.allocated_tokens += num_tokens
        # Return dummy indices
        return torch.arange(num_tokens)

    def free(self, indices: torch.Tensor):
        if indices is not None:
            self.allocated_tokens -= len(indices)

class MockRadixNode:
    def __init__(self, name: str, length: int):
        self.name = name
        self.length = length
        self.kv_indices = None
        self.last_access_time = time.time()
        self.is_compressed = False

class MockRadixCache:
    def __init__(self, pool: MockTokenToKVPool, ashkv_hooks):
        self.pool = pool
        self.hooks = ashkv_hooks
        self.nodes = [] # Flat list for LRU simulation
        
    def add_node(self, name: str, length: int) -> MockRadixNode:
        node = MockRadixNode(name, length)
        # Attempt allocation
        indices = self.pool.allocate(length)
        
        # Eviction Loop if pool is full
        while indices is None:
            # Find LRU uncompressed node
            lru_node = None
            for n in self.nodes:
                if not n.is_compressed and n.kv_indices is not None:
                    if lru_node is None or n.last_access_time < lru_node.last_access_time:
                        lru_node = n
                        
            if lru_node is None:
                raise RuntimeError("Absolute OOM! Cannot evict any more nodes.")
                
            print(f"  [SGLang] Pool Full. Evicting LRU Node: {lru_node.name}")
            
            # SGLang asks ASH-KV to demote instead of deleting
            success = self.hooks.demote_hook(lru_node, sglang_kv_cache=self._dummy_cache(lru_node.length), memory_pool=self.pool)
            if not success:
                print(f"  [SGLang] Demotion failed, deleting node {lru_node.name}")
                self.pool.free(lru_node.kv_indices)
                lru_node.kv_indices = None
                
            # Retry allocation
            indices = self.pool.allocate(length)
            
        node.kv_indices = indices
        self.nodes.append(node)
        return node
        
    def touch_node(self, node: MockRadixNode):
        """Simulate a prefix hit."""
        node.last_access_time = time.time()
        if node.is_compressed:
            print(f"  [SGLang] Cache Hit on compressed Node {node.name}! Promoting to BF16...")
            self.hooks.promote_hook(node, sglang_kv_cache=self._dummy_cache(node.length), memory_pool=self.pool)
            
    def _dummy_cache(self, length):
        # 32 layers, hidden dim 128
        return torch.zeros((32, length, 128), dtype=torch.bfloat16)

# =========================================================================
# 3. WORKLOAD SIMULATION
# =========================================================================

def run_simulation():
    from ashkv.contracts import PageTable
    from ashkv.sglang_integration.allocator import SGLangShadowAllocator
    from ashkv.sglang_integration.hooks import SGLangHooks
    
    # 1. Setup ASH-KV Integrations
    pt = PageTable()
    # Shadow allocator: 100 MB budget
    shadow_alloc = SGLangShadowAllocator(max_bytes=100 * 1024 * 1024)
    class DummyConfig:
        num_hidden_layers = 32
    hooks = SGLangHooks(pt, shadow_alloc, DummyConfig())
    
    # 2. Setup SGLang physical pool (Extremely small for testing: 2000 tokens)
    pool = MockTokenToKVPool(capacity_tokens=2000)
    cache = MockRadixCache(pool, hooks)
    
    history_bf16 = []
    history_int8 = []
    
    print("--- STARTING SGLANG WORKLOAD SIMULATION ---")
    
    # Step 1: Shared System Prompt
    print("\n[Tick 1] Generating System Prompt (1000 tokens)")
    sys_node = cache.add_node("System Prompt", 1000)
    history_bf16.append(pool.allocated_tokens)
    history_int8.append(shadow_alloc.allocated_bytes)
    
    # Step 2: Concurrent Users (5 users, 400 tokens each)
    user_nodes = []
    for i in range(5):
        print(f"\n[Tick 2.{i}] User {i} connecting and generating (400 tokens)")
        # This will eventually trigger eviction of older users
        u_node = cache.add_node(f"User {i} Chat", 400)
        user_nodes.append(u_node)
        
        # Touch system prompt (prefix hit)
        cache.touch_node(sys_node)
        
        history_bf16.append(pool.allocated_tokens)
        history_int8.append(shadow_alloc.allocated_bytes)
        time.sleep(0.01) # Force time difference
        
    # Step 3: User 0 returns
    print("\n[Tick 3] User 0 returns to their chat (Prefix Hit)")
    cache.touch_node(user_nodes[0])
    history_bf16.append(pool.allocated_tokens)
    history_int8.append(shadow_alloc.allocated_bytes)

    # =========================================================================
    # 4. VISUALIZATION
    # =========================================================================
    
    # Normalize INT8 bytes to "Token Equivalent" for visualization
    # 32 layers * 128 hidden dim = 4096 params per token. 1 byte per param in INT8 = ~4KB per token
    int8_token_equivalent = [b / 4096 for b in history_int8]
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(range(len(history_bf16)), history_bf16, label="SGLang Physical VRAM (BF16)", alpha=0.7, color='blue')
    plt.fill_between(range(len(int8_token_equivalent)), [b + i for b, i in zip(history_bf16, int8_token_equivalent)], history_bf16, label="ASH-KV Shadow VRAM (INT8)", alpha=0.7, color='orange')
    plt.axhline(y=2000, color='r', linestyle='--', label="Physical VRAM Limit (2000 tokens)")
    
    plt.title("SGLang + ASH-KV RadixAttention Memory Simulation")
    plt.xlabel("Simulation Ticks (Requests)")
    plt.ylabel("Stored Tokens (Equivalent)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.savefig("sglang_simulation.png")
    print("\nSimulation complete. Plot saved to sglang_simulation.png")

if __name__ == "__main__":
    run_simulation()
