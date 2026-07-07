# ADR-001: vLLM Shadow Cache Architecture

## Status
Accepted

## Date
2026-07-07

## Context
ASH-KV is being integrated into vLLM to provide GPU-resident INT8 compression for KV cache blocks, specifically targeting A100 hardware running Qwen 3.6 27B workloads. We needed a way to interface with vLLM's `BlockSpaceManager` and `kv_cache` tensors without destabilizing vLLM's core execution loop.

Key requirements:
- Cannot interfere with vLLM's expectation that its `kv_cache` is strictly `bfloat16`.
- Must minimize PCIe overhead (CPU <-> GPU byte transfers) on the hot path.
- Preempted sequences must be compressed seamlessly to avoid expensive CPU swapping or recomputation.

## Decision
We implemented a **Shadow Cache Architecture** with the following principles:

1. **Standalone PyTorch Pool**: ASH-KV maintains its own GPU memory pool via `VLLMShadowAllocator` (using native PyTorch caching allocator) rather than hijacking vLLM's internal blocks.
2. **Hook-Based Interception**: We monkey-patch vLLM's `BlockSpaceManager` to intercept `allocate()` and `free()` atomically using a thread lock.
   - **Post-decode (`demote_hook`)**: Triggered prior to `free()`. Cold blocks are compressed directly from vLLM's BF16 cache into the INT8 shadow cache.
   - **Pre-decode (`promote_hook`)**: Triggered upon `allocate()`. If the block belongs to a resuming sequence found in the shadow cache, it is decompressed back into vLLM's newly allocated BF16 block.
3. **Tensor-Direct Fast Path**: We bypass the standard ASH-KV `Codec` byte protocol in the integration layer. The hooks launch the Triton kernels directly on PyTorch GPU tensors (`bfloat16`), eliminating PCIe byte-transfer overhead entirely.

## Alternatives Considered

### In-Place vLLM Block Mutation
- **Pros**: No separate allocator needed.
- **Cons**: Violates vLLM's strict assumption that all cache blocks are identical data types (`bfloat16`). PagedAttention kernels would crash or produce garbage if fed INT8 data unexpectedly.
- **Rejected**: Too brittle and heavily coupled to internal vLLM PagedAttention kernels.

### Standard Python Codec Bytes Protocol (Testing Protocol)
- **Pros**: Perfectly satisfies the standard `Codec` and `Allocator` interfaces.
- **Cons**: Requires moving tensor data to CPU bytes and back 4 times per migration, bottlenecking A100 throughput.
- **Rejected**: Performance overhead was unacceptable for the production fast-path.

## Consequences
- vLLM remains completely unaware of ASH-KV's compression. To vLLM, a preempted sequence simply "disappears" and "reappears" in BF16 instantly upon resumption.
- The `VLLMShadowAllocator` must be carefully tuned via its memory budget to not cause PyTorch OOM alongside vLLM's 90% GPU pre-allocation.
- The Triton kernels must strictly operate on `bfloat16` to match vLLM's internal types and prevent A100 FP16 overflow/NaN issues.
