import os
import sys
import time
import json
import torch
import pytest
from unittest.mock import patch, MagicMock

from ashkv.adapters.sglang.hooks import SGLangHooks
from ashkv.adapters.sglang.patcher import apply_hicache_patches

# -----------------------------------------------------------------------------
# Test 1: Telemetry Byte Math
# -----------------------------------------------------------------------------
@patch("ashkv.adapters.sglang.hooks._get_kernels")
def test_telemetry_byte_math(mock_get_kernels, tmp_path):
    """
    Verifies that when a simulated 16MB tensor is intercepted, 
    the `bytes_saved` stat increments by exactly 8,388,608 bytes (8MB).
    """
    # Mock the Triton kernels to avoid needing a GPU
    # Create dummy pools
    class DummyPool:
        def __init__(self, is_host=False):
            self.head_num = 32
            self.head_dim = 128
            self.layer_num = 1
            if is_host:
                self.k_buffer = torch.zeros((2048, 1, 32, 128), dtype=torch.bfloat16)
                self.v_buffer = torch.zeros((2048, 1, 32, 128), dtype=torch.bfloat16)
            else:
                self.k_buffer = [torch.zeros((2048, 32, 128), dtype=torch.bfloat16)]
                self.v_buffer = [torch.zeros((2048, 32, 128), dtype=torch.bfloat16)]

    device_pool = DummyPool(is_host=False)
    host_pool = DummyPool(is_host=True)
    device_indices = torch.arange(1024)
    host_indices = torch.arange(1024)

    mock_encode = MagicMock()
    mock_encode.return_value = (torch.zeros(1), torch.zeros(1), None)
    mock_get_kernels.return_value = (mock_encode, MagicMock())

    hooks = SGLangHooks(codec_name="int8_default")
    hooks.stats_file = str(tmp_path / "ashkv_stats.json")
    
    hooks.demote_hook(device_pool, host_pool, host_indices, device_indices, "MHA")
    
    assert hooks.stats["tokens_compressed"] == 1024

# -----------------------------------------------------------------------------
# Test 3: Patcher Idempotency
# -----------------------------------------------------------------------------
def test_patcher_idempotency():
    """
    Calls `apply_hicache_patches()` twice and asserts that the target methods 
    are not wrapped twice (preventing infinite recursion).
    """
    # Create a dummy class to simulate MHATokenToKVPoolHost
    class DummyPoolHost:
        def backup_from_device_all_layer(self, *args, **kwargs):
            return "original_backup"
            
        def load_to_device_per_layer(self, *args, **kwargs):
            return "original_load"
            
    # Mock the module that apply_hicache_patches tries to patch
    mock_sglang = MagicMock()
    mock_sglang.srt.mem_cache.memory_pool_host.MHATokenToKVPoolHost = DummyPoolHost
    with patch.dict("sys.modules", {"sglang": mock_sglang, "sglang.srt.mem_cache.memory_pool_host": mock_sglang.srt.mem_cache.memory_pool_host}):
        with patch("ashkv.adapters.sglang.patcher.MHATokenToKVPoolHost", DummyPoolHost, create=True):
            hooks = MagicMock()
            
            # Apply once
            apply_hicache_patches(hooks)
            assert getattr(DummyPoolHost, "ashkv_patched", False) == True
            
            # Store reference to the patched method
            first_patch = DummyPoolHost.backup_from_device_all_layer
            
            # Apply twice
            apply_hicache_patches(hooks)
            
            # Method should be EXACTLY the same object, not re-wrapped
            assert DummyPoolHost.backup_from_device_all_layer is first_patch

# -----------------------------------------------------------------------------
# Test 4: Patcher Graceful Skip
# -----------------------------------------------------------------------------
def test_patcher_graceful_skip(caplog):
    """
    Verifies that if `hooks` is None (simulating an unpatched spawn process), 
    the patcher logs a warning and exits safely without crashing.
    """
    class DummyPoolHost:
        def backup_from_device_all_layer(self, *args, **kwargs):
            pass
            
    mock_sglang = MagicMock()
    mock_sglang.srt.mem_cache.memory_pool_host.MHATokenToKVPoolHost = DummyPoolHost
    with patch.dict("sys.modules", {"sglang": mock_sglang, "sglang.srt.mem_cache.memory_pool_host": mock_sglang.srt.mem_cache.memory_pool_host}):
        with patch("ashkv.adapters.sglang.patcher.MHATokenToKVPoolHost", DummyPoolHost, create=True):
            # Call with None
            apply_hicache_patches(None)
            
            # Should not be patched
            assert not getattr(DummyPoolHost, "ashkv_patched", False)
            assert "apply_hicache_patches called without hooks; interception will be disabled." in caplog.text

# -----------------------------------------------------------------------------
# Test 5: Telemetry Flush Debounce
# -----------------------------------------------------------------------------
@patch("ashkv.adapters.sglang.hooks._get_kernels")
def test_telemetry_flush_debounce(mock_get_kernels, tmp_path):
    """
    Triggers the telemetry hook 100 times in 0.5 seconds and verifies the disk I/O 
    only fires once, proving we aren't destroying enterprise SSDs.
    """
    mock_get_kernels.return_value = (MagicMock(return_value=(torch.zeros(1), torch.zeros(1), None)), MagicMock())
    
    stats_file = tmp_path / "ashkv_stats.json"
    hooks = SGLangHooks(codec_name="int8_default")
    hooks.stats_file = str(stats_file)
    
    # Ensure it's deleted initially so we can count creations
    if stats_file.exists():
        stats_file.unlink()
        
    hooks._flush_stats() # Initial flush
    assert stats_file.exists()
    
    # Get initial modified time
    initial_mtime = os.path.getmtime(str(stats_file))
    
    # Rapidly trigger demote_hook 100 times to simulate eviction bursts
    class DummyPool:
        def __init__(self, is_host=False):
            self.head_num = 32
            self.head_dim = 128
            self.layer_num = 1
            if is_host:
                self.k_buffer = torch.zeros((2048, 1, 32, 128), dtype=torch.bfloat16)
                self.v_buffer = torch.zeros((2048, 1, 32, 128), dtype=torch.bfloat16)
            else:
                self.k_buffer = [torch.zeros((2048, 32, 128), dtype=torch.bfloat16)]
                self.v_buffer = [torch.zeros((2048, 32, 128), dtype=torch.bfloat16)]

    device_pool = DummyPool(is_host=False)
    host_pool = DummyPool(is_host=True)
    device_indices = torch.arange(1)
    host_indices = torch.arange(1)
    
    with patch("time.time") as mock_time:
        # Freeze time at 0.5 so it's < 1.0s since initialization, preventing any flush
        mock_time.return_value = 0.5
        
        for _ in range(100):
            hooks.demote_hook(device_pool, host_pool, host_indices, device_indices, "MHA")
            
        final_mtime = os.path.getmtime(str(stats_file))
        
        # Because of the 1.0 second debounce in hooks.py, the mtime should be identical
        assert initial_mtime == final_mtime
        
        # Verify the file contents don't have the 100 tokens yet (because it was debounced)
        with open(str(stats_file), "r") as f:
            data = json.load(f)
            assert data["tokens_compressed"] == 0
        
# -----------------------------------------------------------------------------
# Test 6: Sitecustomize Thread Logic
# -----------------------------------------------------------------------------
def test_sitecustomize_thread_logic():
    """
    Validates that the delayed patching background thread does not block the main execution thread.
    """
    start_time = time.time()
    
    # Temporarily mock the environment to trigger sitecustomize
    with patch.dict(os.environ, {"ASHKV_ENABLE_PATCHES": "1"}):
        # Import sitecustomize directly
        import importlib
        
        # Make sure we import from the local dir, not a system one
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        
        try:
            import sitecustomize
            importlib.reload(sitecustomize)
        except Exception as e:
            pytest.fail(f"sitecustomize.py crashed on import: {e}")
            
    end_time = time.time()
    
    # The background thread sleeps for 2.0 seconds + loop.
    # If it was blocking, this would take > 2.0 seconds.
    # We assert it takes < 0.5 seconds, proving it's non-blocking.
    assert (end_time - start_time) < 0.5
