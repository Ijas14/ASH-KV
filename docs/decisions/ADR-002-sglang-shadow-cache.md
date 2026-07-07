# ADR-002: SGLang Shadow Cache Architecture

## Status
Accepted (Supersedes ADR-001)

## Date
2026-07-07

## Context
ASH-KV is integrating with SGLang to provide GPU-resident INT8 compression for KV cache blocks. SGLang uses a sophisticated `RadixCache` to manage prefix trees and token allocation, leading to a unique block eviction lifecycle compared to other standard `BlockSpaceManager` implementations.

Key requirements:
- We must integrate directly with SGLang (pinned to `>=0.3.0`) without requiring an AST/dynamic module traversal which is brittle and insecure.
- SGLang `RadixCache` evicts memory at the *node* level (a chunk of tokens in the radix tree).
- We must minimize PCIe overhead (CPU <-> GPU byte transfers) on the hot path.

## Decision
We implemented a **Shadow Cache Architecture** tailored specifically to SGLang's Radix tree:

1. **Standalone PyTorch Pool (`SGLangShadowAllocator`)**: ASH-KV maintains its own GPU memory pool using PyTorch rather than polluting SGLang's native `TokenToKVPool`. SGLang's core KV cache remains purely `bfloat16`.
2. **Explicit Patching (Path 2)**: We reject runtime AST traversal and `sys.modules` magic in favor of direct, try/except imports mapping explicitly to SGLang's `sglang.srt.managers.radix_cache`. If SGLang breaks the import path in the future, the system will explicitly fail with a clear version compatibility error.
3. **Node-Level Hooking**: We patch `RadixCache.evict()` and the prefix matching routines.
   - **Post-decode (`demote_hook`)**: When SGLang chooses to evict a RadixNode, the hook intercepts it. It reads the BF16 data from the node's `kv_indices`, compresses it to the INT8 shadow cache, frees the physical SGLang slots, and explicitly marks the node as `compressed`.
   - **Pre-decode (`promote_hook`)**: When a request matches a prefix that points to a `compressed` node, the hook allocates fresh SGLang slots, decompresses the INT8 shadow buffer into those slots, and rewires the node's `kv_indices` to point to the restored BF16 data before the attention kernel runs.

## Alternatives Considered

### Path 1: AST/Dynamic Module Traversal
- **Pros**: Theoretically immune to SGLang renaming their modules across versions.
- **Cons**: Over-engineered, impossible to statically type-check, brittle against fundamentally structural code changes, and represents a security anti-pattern in production codebases.
- **Rejected**: We opted for explicit, standard Python monkey-patching with strict SGLang version constraints (`>=0.3.0`).

### In-Place SGLang Block Mutation
- **Pros**: Avoids maintaining a secondary shadow allocator.
- **Cons**: RadixAttention relies on strict, continuous memory semantics and fixed dtypes. Writing INT8 into a `bfloat16` pool would crash the attention kernels or require rewriting the core CUDA/Triton kernels inside SGLang itself.
- **Rejected**: Too deeply coupled to SGLang's internal compute mechanisms.

## Consequences
- The SGLang `RadixCache` tree now essentially acts as a tiered storage map. Nodes are either "hot" (backed by SGLang `bfloat16` slots) or "shadowed" (backed by ASH-KV INT8 handles).
- SGLang's core attention engine remains completely untouched and runs at native speed.
- The `SGLangShadowAllocator` must be tuned via `--mem-fraction-static` to ensure SGLang leaves enough GPU memory for ASH-KV to operate.
