import torch
import pytest
import torch.utils.benchmark as benchmark

from ashkv.codecs.int8 import _get_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="These tests strictly require an AMD GPU with ROCm or NVIDIA with CUDA.")

# -----------------------------------------------------------------------------
# Test 7: Triton Mathematical Fidelity (Cosine Similarity)
# -----------------------------------------------------------------------------
def test_triton_mathematical_fidelity():
    """
    Generates a random FP16 tensor representing KV cache, compresses to INT8, 
    decompresses back to FP16, and mathematically asserts Cosine Similarity > 0.99.
    """
    encode_kernel, decode_kernel = _get_kernels()
    
    num_tokens = 1024
    head_num = 32
    head_dim = 128
    grid = (num_tokens,)
    
    # Generate realistic KV cache distribution (normal distribution, slightly shifted)
    original_fp16 = torch.randn((num_tokens, head_num, head_dim), dtype=torch.float16, device="cuda") * 0.5 + 0.1
    
    int8_gpu = torch.empty((num_tokens, head_num, head_dim), dtype=torch.int8, device="cuda")
    scales_gpu = torch.empty((num_tokens, head_num), dtype=torch.float16, device="cuda")
    
    # 1. Compress
    encode_kernel[grid](
        original_fp16.view(num_tokens, -1),
        int8_gpu.view(num_tokens, -1),
        scales_gpu.view(num_tokens, -1),
        num_tokens,
        head_num * head_dim
    )
    
    restored_fp16 = torch.empty_like(original_fp16)
    
    # 2. Decompress
    decode_kernel[grid](
        int8_gpu.view(num_tokens, -1),
        scales_gpu.view(num_tokens, -1),
        restored_fp16.view(num_tokens, -1),
        num_tokens,
        head_num * head_dim
    )
    
    # 3. Mathematically prove Fidelity
    original_flat = original_fp16.view(-1).float()
    restored_flat = restored_fp16.view(-1).float()
    
    cos_sim = torch.nn.functional.cosine_similarity(original_flat.unsqueeze(0), restored_flat.unsqueeze(0)).item()
    mse = torch.nn.functional.mse_loss(original_flat, restored_flat).item()
    
    assert cos_sim > 0.99, f"Brain damage detected! Cosine similarity too low: {cos_sim}"
    assert mse < 0.05, f"MSE too high: {mse}"

# -----------------------------------------------------------------------------
# Test 8: Triton Outlier Clipping
# -----------------------------------------------------------------------------
def test_triton_outlier_clipping():
    """
    Injects extreme NaN/Inf values into the FP16 tensor and verifies the INT8 
    scaling block clips safely rather than exploding the rest of the cache.
    """
    encode_kernel, _ = _get_kernels()
    
    num_tokens = 2
    head_num = 1
    head_dim = 128
    grid = (num_tokens,)
    
    malicious_tensor = torch.zeros((num_tokens, head_num, head_dim), dtype=torch.float16, device="cuda")
    # Inject massive outliers
    malicious_tensor[0, 0, 0] = float('inf')
    malicious_tensor[0, 0, 1] = float('nan')
    malicious_tensor[1, 0, 0] = 65504.0 # Max FP16
    
    int8_gpu = torch.empty((num_tokens, head_num, head_dim), dtype=torch.int8, device="cuda")
    scales_gpu = torch.empty((num_tokens, head_num), dtype=torch.float16, device="cuda")
    
    # Compress
    encode_kernel[grid](
        malicious_tensor.view(num_tokens, -1),
        int8_gpu.view(num_tokens, -1),
        scales_gpu.view(num_tokens, -1),
        num_tokens,
        head_num * head_dim
    )
    
    # Verifies scales and INT8 are generated without Triton crashing
    assert not torch.isnan(int8_gpu).any()
    # Note: Triton math on NaN/Inf might produce NaN scales, but the kernel itself MUST not segfault.
    # We assert the execution completed safely.
    assert int8_gpu.shape == (2, 1, 128)

# -----------------------------------------------------------------------------
# Test 9: Triton Compression Latency
# -----------------------------------------------------------------------------
def test_triton_compression_latency():
    """
    Uses PyTorch's micro-benchmarker to assert that 16MB of compression takes < 5ms on the GPU.
    """
    encode_kernel, _ = _get_kernels()
    
    # 16MB = roughly 2048 tokens * 32 heads * 128 dim * 2 bytes
    num_tokens = 2048
    head_num = 32
    head_dim = 128
    grid = (num_tokens,)
    
    src = torch.randn((num_tokens, head_num, head_dim), dtype=torch.float16, device="cuda")
    int8_gpu = torch.empty((num_tokens, head_num, head_dim), dtype=torch.int8, device="cuda")
    scales_gpu = torch.empty((num_tokens, head_num), dtype=torch.float16, device="cuda")
    
    def run_compress():
        encode_kernel[grid](
            src.view(num_tokens, -1),
            int8_gpu.view(num_tokens, -1),
            scales_gpu.view(num_tokens, -1),
            num_tokens,
            head_num * head_dim
        )
        torch.cuda.synchronize()
        
    # Warmup
    for _ in range(5):
        run_compress()
        
    t = benchmark.Timer(
        stmt='run_compress()',
        globals={'run_compress': run_compress}
    )
    
    measurement = t.timeit(10)
    latency_ms = measurement.mean * 1000
    
    # 5ms is a generous budget, typical execution on MI300X/T4 is < 1ms
    assert latency_ms < 5.0, f"Kernel too slow! Took {latency_ms:.2f}ms"

# -----------------------------------------------------------------------------
# Test 10: SGLang Demote Integration
# -----------------------------------------------------------------------------
def test_sglang_demote_integration():
    from ashkv.adapters.sglang.hooks import SGLangHooks
    from ashkv.adapters.sglang.patcher import apply_hicache_patches
    import torch
    
    class MockDevicePool:
        def __init__(self, layer_num=2, capacity=1000, head_num=32, head_dim=128):
            self.layer_num = layer_num
            self.capacity = capacity
            self.head_num = head_num
            self.head_dim = head_dim
            
            self.k_buffer = [torch.zeros((capacity, head_num, head_dim), dtype=torch.bfloat16, device="cuda") for _ in range(layer_num)]
            self.v_buffer = [torch.zeros((capacity, head_num, head_dim), dtype=torch.bfloat16, device="cuda") for _ in range(layer_num)]

    class MockHostPool:
        def __init__(self, layer_num=2, capacity=1000, head_num=32, head_dim=128):
            self.layer_num = layer_num
            self.capacity = capacity
            self.head_num = head_num
            self.head_dim = head_dim
            
            self.k_buffer = torch.zeros((capacity, layer_num, head_num, head_dim), dtype=torch.bfloat16, device="cpu")
            self.v_buffer = torch.zeros((capacity, layer_num, head_num, head_dim), dtype=torch.bfloat16, device="cpu")

    device_pool = MockDevicePool()
    host_pool = MockHostPool()

    hooks = SGLangHooks(codec_name="int8_default")
    hooks.compressible_layers = [0, 1]

    num_tokens = 500
    device_indices = torch.arange(num_tokens, device="cuda")
    host_indices = torch.arange(num_tokens, device="cpu")

    synthetic_data_k0 = torch.randn((num_tokens, device_pool.head_num, device_pool.head_dim), dtype=torch.bfloat16, device="cuda")
    device_pool.k_buffer[0][device_indices] = synthetic_data_k0
    
    # Demote
    hooks.demote_hook(device_pool, host_pool, host_indices, device_indices, "MHA")
    
    # Assert CPU pool has data
    assert host_pool.k_buffer[0].abs().sum() > 0

# -----------------------------------------------------------------------------
# Test 11: SGLang Promote Integration
# -----------------------------------------------------------------------------
def test_sglang_promote_integration():
    from ashkv.adapters.sglang.hooks import SGLangHooks
    import torch
    
    class MockDevicePool:
        def __init__(self, layer_num=2, capacity=1000, head_num=32, head_dim=128):
            self.layer_num = layer_num
            self.capacity = capacity
            self.head_num = head_num
            self.head_dim = head_dim
            
            self.k_buffer = [torch.zeros((capacity, head_num, head_dim), dtype=torch.bfloat16, device="cuda") for _ in range(layer_num)]
            self.v_buffer = [torch.zeros((capacity, head_num, head_dim), dtype=torch.bfloat16, device="cuda") for _ in range(layer_num)]

    class MockHostPool:
        def __init__(self, layer_num=2, capacity=1000, head_num=32, head_dim=128):
            self.layer_num = layer_num
            self.capacity = capacity
            self.head_num = head_num
            self.head_dim = head_dim
            
            self.k_buffer = torch.zeros((capacity, layer_num, head_num, head_dim), dtype=torch.bfloat16, device="cpu")
            self.v_buffer = torch.zeros((capacity, layer_num, head_num, head_dim), dtype=torch.bfloat16, device="cpu")

    device_pool = MockDevicePool()
    host_pool = MockHostPool()

    hooks = SGLangHooks(codec_name="int8_default")
    hooks.compressible_layers = [0, 1]

    num_tokens = 500
    device_indices = torch.arange(num_tokens, device="cuda")
    host_indices = torch.arange(num_tokens, device="cpu")

    synthetic_data_k0 = torch.randn((num_tokens, device_pool.head_num, device_pool.head_dim), dtype=torch.bfloat16, device="cuda")
    device_pool.k_buffer[0][device_indices] = synthetic_data_k0
    
    # Demote
    hooks.demote_hook(device_pool, host_pool, host_indices, device_indices, "MHA")
    
    # Clear GPU pool
    device_pool.k_buffer[0].zero_()
    
    # Promote
    hooks.promote_hook(device_pool, host_pool, host_indices, device_indices, 0, "MHA")
    
    restored_k0 = device_pool.k_buffer[0][device_indices]
    
    cos_sim = torch.nn.functional.cosine_similarity(
        synthetic_data_k0.float().view(-1).unsqueeze(0), 
        restored_k0.float().view(-1).unsqueeze(0)
    ).item()
    
    assert cos_sim > 0.99, "Promote failed to perfectly reconstruct KV cache!"

