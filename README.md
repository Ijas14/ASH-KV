# ASH-KV

ASH-KV is a high-performance, GPU-resident memory manager that acts as a shadow KV cache for LLM inference (e.g. vLLM). It intercepts preempted sequences and compresses them natively on the GPU using highly optimized INT8 Triton kernels, preventing expensive CPU swapping or recomputation overhead on hardware like the A100.

## Quick Start
1. Clone the repository
2. Install dependencies (requires PyTorch and Triton):
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment for testing (e.g., in Colab or Linux with an NVIDIA/AMD GPU)
4. Run the test suite:
   ```bash
   pytest tests/
   ```

## Commands
| Command | Description |
|---------|-------------|
| `pytest tests/` | Run the full unit test suite (codecs, telemetry, safety guards) |
| `pytest tests/test_codecs_and_telemetry.py -v` | Run specifically the Triton codec roundtrip tests |

## Architecture
ASH-KV operates under a **Shadow Cache Architecture**. It is strictly separated from the primary vLLM cache:
- **vLLM Cache**: Remains in unmodified `bfloat16`.
- **ASH-KV Pool**: Manages a standalone PyTorch INT8 tensor pool on the GPU (`VLLMShadowAllocator`).
- **Atomic Hooks**: `promote_hook` and `demote_hook` intercept `BlockSpaceManager` operations to seamlessly encode/decode KV blocks directly via GPU tensors, eliminating PCIe overhead.

For full architectural context and why we chose this specific hook-based approach instead of in-place mutation, see [ADR-001: vLLM Shadow Cache Architecture](docs/decisions/ADR-001-vllm-shadow-cache.md).

## Contributing
- **Testing**: All code changes must pass the fault-injection test suite. The system strictly enforces a "never-throw on the hot path" contract.
- **Documentation**: Significant architectural decisions must be accompanied by an ADR in `docs/decisions/`.
- **Codecs**: New codecs should be registered in `ashkv/compiler/registry.py` and must satisfy the stateless bytes-in/bytes-out protocol for testing.
