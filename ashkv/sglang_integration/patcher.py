"""SGLang RadixCache Patcher.

Uses explicit direct imports (Path 2) to hook into SGLang >= 0.3.0.
Fails explicitly and safely if the SGLang internal architecture changes.
"""
import logging
import threading

logger = logging.getLogger(__name__)

# Global integration state
_HOOKS = None
_SGLANG_KV_CACHE = None
_MEMORY_POOL = None
_PATCH_LOCK = threading.Lock()


def apply_radix_cache_patches(hooks, sglang_kv_cache, memory_pool) -> None:
    """Monkey-patch SGLang's RadixCache and TreeNode classes.
    
    Args:
        hooks: SGLangHooks instance
        sglang_kv_cache: PyTorch BF16 tensor representing the memory pool
        memory_pool: TokenToKVPool allocator from SGLang
    """
    global _HOOKS, _SGLANG_KV_CACHE, _MEMORY_POOL
    _HOOKS = hooks
    _SGLANG_KV_CACHE = sglang_kv_cache
    _MEMORY_POOL = memory_pool

    try:
        # SGLang >= 0.5.x moved RadixCache to mem_cache
        from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
    except ImportError:
        try:
            # SGLang < 0.5.x used managers
            from sglang.srt.managers.radix_cache import RadixCache, TreeNode
        except ImportError as e:
            logger.error("Failed to import SGLang RadixCache. ASH-KV requires SGLang >= 0.3.0.")
            raise RuntimeError(f"Incompatible SGLang version or missing dependency: {e}")

    # We do NOT patch match_prefix directly anymore.
    # SGLang v0.5.14 crashes if it encounters node.value = None during traversal.
    # Instead of brittle manual traversals, we use JIT Property Auto-Promotion.
    
    if not getattr(TreeNode, "ashkv_patched", False):
        # 1. Patch TreeNode.value to auto-promote on read
        def get_value(self):
            if getattr(self, "ashkv_shadow_handle", None) is not None:
                if not getattr(self, "ashkv_bypass_promote", False):
                    if _HOOKS is not None:
                        _HOOKS.promote_hook(self, _SGLANG_KV_CACHE, _MEMORY_POOL)
            return self.__dict__.get("_value", None)
            
        def set_value(self, val):
            self.__dict__["_value"] = val
            
        TreeNode.value = property(get_value, set_value)
        
        # 2. Patch TreeNode.evicted so it doesn't trigger auto-promotion during status checks
        def get_evicted(self):
            return self.__dict__.get("_value", None) is None and getattr(self, "ashkv_shadow_handle", None) is None
            
        TreeNode.evicted = property(get_evicted)
        TreeNode.ashkv_patched = True

    original_evict = RadixCache.evict

    def ashkv_evict(self, *args, **kwargs):
        """Intercept eviction to trigger demote_hook.
        
        Instead of freeing physical slots to the void, we compress the node
        to INT8, explicitly free the slots to the TokenToKVPool, and keep the
        node alive but marked as `compressed`.
        """
        with _PATCH_LOCK:
            freed_tokens = 0
            
            num_tokens = kwargs.get("num_tokens", 0)
            is_v05x = False
            if len(args) > 0:
                if hasattr(args[0], "num_tokens"):
                    num_tokens = args[0].num_tokens
                    is_v05x = True
                elif isinstance(args[0], int):
                    num_tokens = args[0]
                    
            evict_callback = kwargs.get("evict_callback", None)
            if len(args) > 1 and callable(args[1]):
                evict_callback = args[1]
            
            # Replicate SGLang's LRU eviction loop but inject compression
            while freed_tokens < num_tokens and self.evictable_size_ > 0:
                # SGLang maintains an lru_queue, evictable_queue, or evictable_leaves
                queue = getattr(self, "lru_queue", getattr(self, "evictable_queue", getattr(self, "evictable_leaves", None)))
                if queue is None or len(queue) == 0:
                    break
                    
                # Get the least recently used node
                if isinstance(queue, set):
                    if hasattr(self, "eviction_strategy"):
                        import heapq
                        leaves = list(queue)
                        eviction_heap = [(self.eviction_strategy.get_priority(n), n) for n in leaves]
                        heapq.heapify(eviction_heap)
                        _, node = heapq.heappop(eviction_heap)
                    else:
                        node = next(iter(queue))
                elif isinstance(queue, dict):
                    node = next(iter(queue.values()))
                else:
                    node = queue[0]
                
                # Calculate tokens before compressing (since compression clears node.value)
                node_tokens = getattr(node, "length", 0)
                if node_tokens == 0:
                     val = node.__dict__.get("_value", None)
                     if val is not None:
                         node_tokens = len(val)
                     
                # Attempt to compress it
                success = _HOOKS.demote_hook(node, _SGLANG_KV_CACHE, _MEMORY_POOL)
                     
                if success:
                    # Node is compressed and physical slots freed internally by demote_hook.
                    # We remove it from the evictable queue since it's no longer occupying BF16.
                    freed_tokens += node_tokens
                    if isinstance(queue, set):
                        queue.remove(node)
                        if hasattr(self, "_update_leaf_status"):
                            self._update_leaf_status(node.parent)
                    elif isinstance(queue, dict):
                        del queue[id(node)]
                    else:
                        queue.pop(0)
                        
                    self.evictable_size_ -= node_tokens
                else:
                    # Shadow cache is full (or failed), fallback to actual native eviction
                    kv_indices = getattr(node, "kv_indices", node.__dict__.get("_value", None))
                    if kv_indices is not None:
                        if evict_callback:
                            evict_callback(kv_indices)
                        elif getattr(self, "token_to_kv_pool_allocator", None):
                            self.token_to_kv_pool_allocator.free(kv_indices)
                        freed_tokens += node_tokens
                    
                    if hasattr(self, "_delete_leaf"):
                        self._delete_leaf(node)
                    else:
                        self._remove_node(node)
                    
            if is_v05x:
                try:
                    from sglang.srt.mem_cache.base_prefix_cache import EvictResult
                    return EvictResult(num_tokens_evicted=freed_tokens)
                except ImportError:
                    pass
            return freed_tokens

    # Patch the RadixCache methods
    RadixCache.evict = ashkv_evict
    
    logger.info("ASH-KV successfully patched SGLang RadixCache.")
