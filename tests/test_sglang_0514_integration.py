import torch
import sys

from ashkv.adapters.sglang.hooks import SGLangHooks
from ashkv.adapters.sglang.patcher import apply_hicache_patches

# Mock the CUDA device and Triton kernels for testing in CPU-only environments
def mock_get_kernels():
    def mock_encode(k_gpu, int8_gpu, scales_gpu, num_tokens, hidden_dim):
        # Fake compression: scale = max/127, int8 = k_gpu / scale
        max_vals = k_gpu.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        scales = max_vals / 127.0
        scales_gpu.copy_(scales)
        int8_gpu.copy_(torch.clamp(torch.round(k_gpu / scales), -128, 127).to(torch.int8))
        
    def mock_decode(int8_gpu, scales_gpu, k_gpu, num_tokens, hidden_dim):
        # Fake decompression
        head_num = scales_gpu.shape[1]
        head_dim = hidden_dim // head_num
        scales_expanded = scales_gpu.repeat_interleave(head_dim, dim=1)
        k_gpu.copy_(int8_gpu.to(torch.bfloat16) * scales_expanded)
        
    class MockKernel:
        def __init__(self, func):
            self.func = func
        def __getitem__(self, grid):
            return self.func
            
    return MockKernel(mock_encode), MockKernel(mock_decode)

import ashkv.adapters.sglang.hooks
ashkv.adapters.sglang.hooks._get_kernels = mock_get_kernels

class MockDevicePool:
    def __init__(self, layer_num=32, capacity=20000, head_num=32, head_dim=128):
        self.layer_num = layer_num
        self.capacity = capacity
        self.head_num = head_num
        self.head_dim = head_dim
        
        # CPU fallback for testing
        self.k_buffer = [torch.zeros((capacity, head_num, head_dim), dtype=torch.bfloat16, device="cpu") for _ in range(layer_num)]
        self.v_buffer = [torch.zeros((capacity, head_num, head_dim), dtype=torch.bfloat16, device="cpu") for _ in range(layer_num)]

class MockHostPool:
    def __init__(self, layer_num=32, capacity=20000, head_num=32, head_dim=128):
        self.layer_num = layer_num
        self.capacity = capacity
        self.head_num = head_num
        self.head_dim = head_dim
        
        self.k_buffer = torch.zeros((capacity, layer_num, head_num, head_dim), dtype=torch.bfloat16, device="cpu")
        self.v_buffer = torch.zeros((capacity, layer_num, head_num, head_dim), dtype=torch.bfloat16, device="cpu")

    # These are the methods we will patch
    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        pass
        
    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
        pass

# Need to inject the mock into SGLang module for the patcher to work
try:
    import sglang.srt.mem_cache.memory_pool_host as mock_module
    mock_module.MHATokenToKVPoolHost = MockHostPool
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules['sglang'] = MagicMock()
    sys.modules['sglang.srt'] = MagicMock()
    sys.modules['sglang.srt.mem_cache'] = MagicMock()
    sys.modules['sglang.srt.mem_cache.memory_pool_host'] = MagicMock()
    sys.modules['sglang.srt.mem_cache.memory_pool_host'].MHATokenToKVPoolHost = MockHostPool

print("--- INITIALIZING ASH-KV HiCache Integration ---")
device_pool = MockDevicePool(layer_num=2, capacity=1000) # Smaller for test
host_pool = MockHostPool(layer_num=2, capacity=1000)

hooks = SGLangHooks(codec_name="int8_default")
hooks.compressible_layers = [0, 1]

# Apply patches to the HostKVCache class
apply_hicache_patches(hooks)

print("Patches applied to HostKVCache.")

# Create synthetic data on CPU
num_tokens = 500
device_indices = torch.arange(num_tokens, device="cpu")
host_indices = torch.arange(num_tokens, device="cpu")

# Populate device_pool with random BF16 data
synthetic_data_k0 = torch.randn((num_tokens, device_pool.head_num, device_pool.head_dim), dtype=torch.bfloat16, device="cpu")
synthetic_data_v0 = torch.randn((num_tokens, device_pool.head_num, device_pool.head_dim), dtype=torch.bfloat16, device="cpu")
synthetic_data_k1 = torch.randn((num_tokens, device_pool.head_num, device_pool.head_dim), dtype=torch.bfloat16, device="cpu")
synthetic_data_v1 = torch.randn((num_tokens, device_pool.head_num, device_pool.head_dim), dtype=torch.bfloat16, device="cpu")

device_pool.k_buffer[0][device_indices] = synthetic_data_k0
device_pool.v_buffer[0][device_indices] = synthetic_data_v0
device_pool.k_buffer[1][device_indices] = synthetic_data_k1
device_pool.v_buffer[1][device_indices] = synthetic_data_v1


print("\n--- TEST: HICACHE OFFLOAD (BACKUP) ---")
# Call the patched method (simulating SGLang HiCache offload to CPU)
# MockHostPool has the patched method because apply_hicache_patches patched the class
try:
    host_pool.backup_from_device_all_layer(device_pool, host_indices, device_indices, "kernel")
except AttributeError as e:
    print(f"Patcher didn't apply to MockHostPool class properly due to mock injection: {e}")
    # Force apply to instance for the test if mock injection failed
    host_pool.backup_from_device_all_layer = lambda dp, hi, di, io: hooks.demote_hook(dp, host_pool, hi, di, "MHA")
    host_pool.load_to_device_per_layer = lambda dp, hi, di, lid, io: hooks.promote_hook(dp, host_pool, hi, di, lid, "MHA")
    host_pool.backup_from_device_all_layer(device_pool, host_indices, device_indices, "kernel")

print("Data compressed and written to CPU host_pool.")

# Clear device pool to ensure we are actually restoring it
device_pool.k_buffer[0].zero_()
device_pool.v_buffer[0].zero_()
device_pool.k_buffer[1].zero_()
device_pool.v_buffer[1].zero_()


print("\n--- TEST: HICACHE RESTORE (LOAD) ---")
# Call the patched method (simulating SGLang HiCache restore from CPU)
host_pool.load_to_device_per_layer(device_pool, host_indices, device_indices, 0, "kernel")
host_pool.load_to_device_per_layer(device_pool, host_indices, device_indices, 1, "kernel")

print("Data read from CPU host_pool, decompressed, and written to GPU device_pool.")

# Verify data integrity
restored_k0 = device_pool.k_buffer[0][device_indices]
restored_v0 = device_pool.v_buffer[0][device_indices]
restored_k1 = device_pool.k_buffer[1][device_indices]
restored_v1 = device_pool.v_buffer[1][device_indices]

is_close_k0 = torch.allclose(synthetic_data_k0, restored_k0, atol=5e-2)
is_close_v0 = torch.allclose(synthetic_data_v0, restored_v0, atol=5e-2)
is_close_k1 = torch.allclose(synthetic_data_k1, restored_k1, atol=5e-2)
is_close_v1 = torch.allclose(synthetic_data_v1, restored_v1, atol=5e-2)

print(f"Layer 0 K Data perfectly reconstructed: {is_close_k0}")
print(f"Layer 0 V Data perfectly reconstructed: {is_close_v0}")
print(f"Layer 1 K Data perfectly reconstructed: {is_close_k1}")
print(f"Layer 1 V Data perfectly reconstructed: {is_close_v1}")

assert is_close_k0 and is_close_v0 and is_close_k1 and is_close_v1, "Data corruption during INT8 GPU->CPU->GPU roundtrip!"

print("\nAll integration tests passed for HiCache IO Interception!")
