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

    original_match_prefix = RadixCache.match_prefix
    original_evict = RadixCache.evict

    def ashkv_match_prefix(self, *args, **kwargs):
        """Intercept prefix matching to trigger promote_hook.
        
        SGLang v0.5.14 crashes natively if it encounters a node with `value=None`
        during traversal (torch.cat throws TypeError). Thus, we must perform a
        pre-emptive traversal to promote any compressed nodes in the matched path
        before handing execution back to the original function.
        """
        with _PATCH_LOCK:
            # 1. Extract the key
            key = kwargs.get("key", None)
            if key is None and len(args) > 0:
                key = args[0]
                
            # Unwrap MatchPrefixParams if present (v0.5.x)
            if hasattr(key, "key"):
                key = key.key

            # 2. Pre-emptively traverse and promote
            if hasattr(self, "root_node") and key is not None:
                try:
                    curr_key = key
                    page_size = getattr(self, "page_size", 1)
                    
                    # Mimic SGLang's exact key preparation
                    if hasattr(curr_key, "maybe_to_bigram_view"):
                        is_eagle = getattr(self, "is_eagle", False)
                        curr_key, _ = curr_key.maybe_to_bigram_view(is_eagle)
                        
                    if hasattr(curr_key, "page_aligned"):
                        curr_key = curr_key.page_aligned(page_size)
                        
                    curr_node = self.root_node
                    
                    while len(curr_key) > 0:
                        child_key = curr_key.child_key(page_size) if hasattr(curr_key, "child_key") else curr_key[0]
                        if child_key not in curr_node.children:
                            break
                            
                        child = curr_node.children[child_key]
                        
                        # --- ASH-KV HOOK (Promote) ---
                        if getattr(child, "ashkv_shadow_handle", None) is not None:
                            _HOOKS.promote_hook(child, _SGLANG_KV_CACHE, _MEMORY_POOL)
                            
                        # Advance traversal
                        if hasattr(child.key, "match"):
                            prefix_len = child.key.match(curr_key, page_size=page_size)
                        else:
                            break # Fallback if we can't match accurately
                            
                        if prefix_len < len(child.key):
                            break
                        curr_node = child
                        curr_key = curr_key[prefix_len:]
                except Exception as e:
                    # If our traversal fails, let SGLang try it (it might crash, but we don't break early)
                    pass

            # 3. Call original now that all nodes in path have physical values restored
            result = original_match_prefix(self, *args, **kwargs)
            return result

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
                if node_tokens == 0 and hasattr(node, "value") and node.value is not None:
                     node_tokens = len(node.value)
                     
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
                    kv_indices = getattr(node, "kv_indices", getattr(node, "value", None))
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

    # Apply the patches
    RadixCache.match_prefix = ashkv_match_prefix
    RadixCache.evict = ashkv_evict
    
    logger.info("ASH-KV successfully patched SGLang RadixCache.")
