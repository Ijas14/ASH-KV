# ASH-KV

Stop deleting memory; change its state. ASH-KV is an adaptive, tiered KV cache management system for LLM inference. It prevents concurrency-driven OOMs by compressing old context into cheaper numeric formats (FP8, INT8, INT4) and offloading to CPU, rather than evicting it.

Built on a strict cold/hot boundary: configuration is interpreted at startup, inference is compiled. The entire system is driven by a single score, a single controller, and an 8-number policy surface.

---

## The Problem

When an LLM serves concurrent requests, the KV cache grows until it hits the GPU memory limit. When it hits the limit, serving engines like vLLM crash (OOM) or forcefully preempt sequences (recompute/swap), causing massive latency spikes.

Current solutions are either brute-force (eviction) or static (uniform FP8 quantization). 

ASH-KV treats the KV cache as a **tiered memory system**. Recent, important tokens stay in expensive BF16. Old, cold tokens are compressed to INT8 and kept in a GPU-resident "shadow cache." When memory pressure rises, ASH-KV dynamically demotes cold pages, preventing OOM and pushing the concurrency cliff higher.

---

## The Architecture

ASH-KV is built on a strict **cold/hot boundary** to ensure the inference hot path remains blindingly fast.

**Cold Path (Startup):**
Parses the YAML config, probes the hardware, resolves codecs, and compiles closures. The hot path never reads config or checks flags.

**Hot Path (Every Decode Step):**
1. `score()` — Vectorized numpy computation of page fidelity scores.
2. `desired_tier()` — Controller decides target tier based on score and pressure.
3. `migrate()` — Single-path migration engine moves pages between tiers.

### The 8-Number Policy Surface

The entire tunable surface of ASH-KV is just eight numbers. No heuristic soup, no 50-knob config files.

| # | Knob | Meaning |
|---|---|---|
| 1 | `w_T` | Temporal locality weight (recency) |
| 2 | `w_S` | Saliency weight (attention scores) |
| 3 | `w_N` | Novelty weight (surprise) |
| 4 | `w_P` | Prefix affinity weight (shared prompt prefix) |
| 5 | `theta_high` | Minimum score to promote a page to BF16 |
| 6 | `theta_low` | Maximum score before a page is demoted to INT8 |
| 7 | `delta` | Hysteresis gap to prevent flapping |
| 8 | `p_emergency` | Memory pressure trigger for emergency eviction |

### The Fault Tolerance Ladder

ASH-KV never crashes the inference server.
- **Never throw on the hot path.** Every function returns a typed `MigrationResult`.
- **Shadow Cache.** Compressed INT8 pages live in a separate memory pool (`VLLMShadowAllocator`). vLLM's `kv_cache` tensor is *always* valid `bfloat16`. 
- **Circuit Breakers.** If a Triton codec fails 5 times in 60 seconds, it is disabled. Pages stay on their current tier.
- **BF16 Fallback.** If a page becomes corrupt, it is reconstructed in BF16 from its source checksum. The decode continues.

For full architectural context on the vLLM shadow cache integration, see [ADR-001: vLLM Shadow Cache Architecture](docs/decisions/ADR-001-vllm-shadow-cache.md).

---

## Current Status

ASH-KV is currently integrated with **vLLM** and validated on AMD MI300X and NVIDIA T4.

- **Codecs:** BF16 (identity), INT8 (Triton, per-token scaling, autotuned).
- **Integration:** vLLM `BlockSpaceManager` monkey-patch. Atomic `promote_hook` and `demote_hook` intercept preemptions to seamlessly encode/decode KV blocks directly via GPU tensors, eliminating PCIe overhead.
- **SGLang:** Architecture supports it, integration is on the roadmap.

### Validated on
- **AMD Instinct MI300X** (192GB HBM3)
- **NVIDIA A100** (Target deployment)
- **NVIDIA T4** (Colab/Kaggle testing)

---

## Quick Start

1. Clone the repository
2. Install the package and its dependencies (requires PyTorch and Triton):
```bash
pip install -e .
```

### Run vLLM with ASH-KV

```bash
# Start vLLM with ASH-KV enabled
ASHKV_ENABLED=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-27B \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95
```

### Configuration

Tune the 8 numbers via a simple YAML file:

```yaml
# qwen_2xa100_config.yaml
policy:
  w_T: 0.7
  w_S: 0.1
  w_N: 0.1
  w_P: 0.1
  theta_high: 0.72
  theta_low: 0.33
  delta: 0.04
  p_emergency: 0.95
```

---

## Performance: The Cliff Finder

We benchmarked ASH-KV using [`inferbench`](https://github.com/Ijas14/inferbench) (a benchmark that finds where LLM serving systems break under concurrent load).

**Baseline (vLLM default):** Cliffs at concurrency 8 (Latency Spike >10x baseline).
**With ASH-KV (INT8 Shadow Cache):** The cliff is pushed higher, preventing OOM and keeping preempted sequences on the GPU instead of swapping to CPU.

*(Full MI300X baseline report available in the `results/` directory).*

---

## Repository Structure

```
ashkv/
├── contracts/          # Frozen types, protocols, 8-number config (0 PyTorch deps)
├── runtime/            # Hot path: score, controller, migrate (numpy only)
├── compiler/           # Cold path: config -> closures, hardware probe
├── codecs/             # Triton kernels (INT8, FP8, INT4) + checksums
├── safety/             # Circuit breakers, pressure guard, BF16 fallback
├── vllm_integration/   # Shadow allocator, hooks, block manager patch
├── docs/decisions/     # Architectural Decision Records (ADRs)
└── tests/              # 144 tests (contracts, fault injection, dependency direction)
```

### Dependency Discipline

ASH-KV is built to be lightweight.
- `contracts/` and `runtime/` have **zero** dependencies beyond `numpy` and stdlib.
- `codecs/` lazy-imports `torch` and `triton` only when a codec is actually called.
- The hot path (`score`, `desired_tier`, `migrate`) never touches PyTorch.

---

## Commands

| Command | Description |
|---------|-------------|
| `pytest tests/` | Run the full unit test suite (codecs, telemetry, safety guards) |
| `pytest tests/test_codecs_and_telemetry.py -v` | Run specifically the Triton codec roundtrip tests |

---

## Contributing
- **Testing**: All code changes must pass the fault-injection test suite. The system strictly enforces a "never-throw on the hot path" contract.
- **Documentation**: Significant architectural decisions must be accompanied by an ADR in `docs/decisions/`.
- **Codecs**: New codecs should be registered in `ashkv/compiler/registry.py` and must satisfy the stateless bytes-in/bytes-out protocol for testing.

---

## License

MIT
