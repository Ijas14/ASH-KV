"""SGLang HiCache Patcher.

Uses explicit direct imports to hook into SGLang >= 0.5.x HiCache memory pools.
Intercepts the backup_from_device_all_layer and load_to_device_per_layer methods
to perform INT8 compression/decompression during CPU offload.
"""
import logging
import threading

logger = logging.getLogger(__name__)

# Global integration state
_HOOKS = None
_PATCH_LOCK = threading.Lock()


def apply_hicache_patches(hooks=None) -> None:
    """Monkey-patch SGLang's HostKVCache classes for HiCache IO interception.
    
    Args:
        hooks: SGLangHooks instance containing demote_hook and promote_hook.
    """
    global _HOOKS
    _HOOKS = hooks

    if _HOOKS is None:
        logger.warning("apply_hicache_patches called without hooks; interception will be disabled.")
        return

    try:
        from sglang.srt.mem_cache.memory_pool_host import (
            MHATokenToKVPoolHost,
            MLATokenToKVPoolHost,
        )
    except ImportError as e:
        logger.error("Failed to import SGLang HostKVCache. ASH-KV requires SGLang >= 0.5.x for HiCache.")
        raise RuntimeError(f"Incompatible SGLang version or missing dependency: {e}")

    with _PATCH_LOCK:
        if getattr(MHATokenToKVPoolHost, "ashkv_patched", False):
            return

        # 1. Patch MHATokenToKVPoolHost
        original_mha_backup = MHATokenToKVPoolHost.backup_from_device_all_layer
        original_mha_load = MHATokenToKVPoolHost.load_to_device_per_layer

        def mha_backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
            # Intercept GPU -> CPU offload
            # Demote hook will compress `device_indices` from `device_pool` into `host_indices` in `self`
            _HOOKS.demote_hook(
                device_pool=device_pool,
                host_pool=self,
                host_indices=host_indices,
                device_indices=device_indices,
                pool_type="MHA"
            )

        def mha_load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
            # Intercept CPU -> GPU restore
            # Promote hook will decompress `host_indices` from `self` into `device_indices` in `device_pool` for `layer_id`
            _HOOKS.promote_hook(
                device_pool=device_pool,
                host_pool=self,
                host_indices=host_indices,
                device_indices=device_indices,
                layer_id=layer_id,
                pool_type="MHA"
            )

        MHATokenToKVPoolHost.backup_from_device_all_layer = mha_backup_from_device_all_layer
        MHATokenToKVPoolHost.load_to_device_per_layer = mha_load_to_device_per_layer
        MHATokenToKVPoolHost.ashkv_patched = True


        # 2. Patch MLATokenToKVPoolHost (if applicable, using same pattern)
        original_mla_backup = MLATokenToKVPoolHost.backup_from_device_all_layer
        original_mla_load = MLATokenToKVPoolHost.load_to_device_per_layer

        def mla_backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
            _HOOKS.demote_hook(
                device_pool=device_pool,
                host_pool=self,
                host_indices=host_indices,
                device_indices=device_indices,
                pool_type="MLA"
            )

        def mla_load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
            _HOOKS.promote_hook(
                device_pool=device_pool,
                host_pool=self,
                host_indices=host_indices,
                device_indices=device_indices,
                layer_id=layer_id,
                pool_type="MLA"
            )

        MLATokenToKVPoolHost.backup_from_device_all_layer = mla_backup_from_device_all_layer
        MLATokenToKVPoolHost.load_to_device_per_layer = mla_load_to_device_per_layer
        MLATokenToKVPoolHost.ashkv_patched = True

    print("[ASH-KV] Patches applied to SGLang HiCache IO.")
    logger.info("ASH-KV successfully patched SGLang HiCache (HostKVCache).")
