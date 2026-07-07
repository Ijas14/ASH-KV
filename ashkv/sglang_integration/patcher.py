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
        from sglang.srt.managers.radix_cache import RadixCache, TreeNode
    except ImportError as e:
        logger.error("Failed to import SGLang RadixCache. ASH-KV requires SGLang >= 0.3.0.")
        raise RuntimeError(f"Incompatible SGLang version or missing dependency: {e}")

    original_match_prefix = RadixCache.match_prefix
    original_evict = RadixCache.evict

    def ashkv_match_prefix(self, key, **kwargs):
        """Intercept prefix matching to trigger promote_hook.
        
        When a request matches a prefix, if that prefix node is compressed
        in the shadow cache, it must be decompressed into fresh physical slots
        before the request can proceed.
        """
        with _PATCH_LOCK:
            # Find the nodes that match the prefix
            nodes_to_promote = []
            
            # Since SGLang's match_prefix is recursive/iterative, we let it run
            # to find the matching node path. SGLang returns the matching nodes 
            # or the matched length. This is highly dependent on SGLang internals.
            # In SGLang >= 0.3.0, it returns (matched_node, matched_length, ...).
            result = original_match_prefix(self, key, **kwargs)
            
            if isinstance(result, tuple) and len(result) > 0:
                matched_node = result[0]
                
                # Traverse up the tree to ensure all parent nodes are promoted
                curr = matched_node
                while curr is not None:
                    if getattr(curr, "is_compressed", False):
                        nodes_to_promote.append(curr)
                    curr = curr.parent
                    
                # Promote from root to leaf
                for node in reversed(nodes_to_promote):
                    _HOOKS.promote_hook(node, _SGLANG_KV_CACHE, _MEMORY_POOL)

            return result

    def ashkv_evict(self, num_tokens: int, evict_callback):
        """Intercept eviction to trigger demote_hook.
        
        Instead of freeing physical slots to the void, we compress the node
        to INT8, explicitly free the slots to the TokenToKVPool, and keep the
        node alive but marked as `compressed`.
        """
        with _PATCH_LOCK:
            freed_tokens = 0
            
            # Replicate SGLang's LRU eviction loop but inject compression
            while freed_tokens < num_tokens and self.evictable_size_ > 0:
                # SGLang maintains an lru_queue (or similar structure)
                if not hasattr(self, "evictable_queue") and not hasattr(self, "lru_queue"):
                    # Fallback if internal names changed
                    break
                    
                queue = getattr(self, "lru_queue", getattr(self, "evictable_queue", None))
                if queue is None or len(queue) == 0:
                    break
                    
                # Get the least recently used node
                # We assume queue is a dict-like or list-like OrderedDict
                node = next(iter(queue.values())) if isinstance(queue, dict) else queue[0]
                
                # Attempt to compress it
                success = _HOOKS.demote_hook(node, _SGLANG_KV_CACHE, _MEMORY_POOL)
                
                node_tokens = getattr(node, "length", 0)
                if node_tokens == 0 and hasattr(node, "value"):
                     node_tokens = len(node.value)
                     
                if success:
                    # Node is compressed and physical slots freed internally by demote_hook.
                    # We remove it from the evictable queue since it's no longer occupying BF16.
                    freed_tokens += node_tokens
                    if isinstance(queue, dict):
                        del queue[id(node)]
                    else:
                        queue.pop(0)
                        
                    self.evictable_size_ -= node_tokens
                else:
                    # Shadow cache is full (or failed), fallback to actual native eviction
                    # SGLang usually calls evict_callback(node.kv_indices) and removes node from tree.
                    if hasattr(node, "kv_indices") and node.kv_indices is not None:
                        evict_callback(node.kv_indices)
                        freed_tokens += node_tokens
                    
                    self._remove_node(node) # Native SGLang cleanup
                    
            return freed_tokens

    # Apply the patches
    RadixCache.match_prefix = ashkv_match_prefix
    RadixCache.evict = ashkv_evict
    
    logger.info("ASH-KV successfully patched SGLang RadixCache.")
