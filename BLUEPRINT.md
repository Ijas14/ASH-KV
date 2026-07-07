# ASH-KV Architecture Blueprint

**For:** Split 2 (Codecs) and Split 3 (Integration) teams
**From:** Split 1 (Core) team
**Status:** Phase 0 complete, contracts frozen, ready for implementation

---

## 1. What you are building

You are building **ASH-KV**, an adaptive memory manager for large language model inference. It automatically compresses the model's "memory" (KV cache) into cheaper numeric formats as that memory gets older and less important, so the same GPU can serve longer contexts to more users without the model's output quality collapsing.

The project is split into three teams. **You are two of them.** Split 1 (Core) is already built — that's the foundation you build on. This document explains the whole system, what your team owns, and how your work fits.

### The problem in one paragraph

When an LLM generates text, every token it has ever seen is stored in GPU memory as KV (key/value) tensors. At 128k context on a Qwen-class model, this KV cache is bigger than the model weights. Most servers store all of it in BF16 (16-bit) format until they run out of memory and crash. ASH-KV instead treats KV as a tiered memory system — recent important tokens stay in expensive BF16, older tokens drop to FP8 (8-bit), then INT4 (4-bit), then archive formats, then CPU, then disk. The model still works because the important tokens are still there in full fidelity; the unimportant ones got cheaper.

### Why this matters

- Longer contexts (256k, 1M tokens) on the same GPU
- More concurrent users per GPU
- Lower latency (FP8/INT4 reads are faster than BF16)
- Graceful degradation under memory pressure instead of OOM crashes

The realistic target is 3–5x effective context and 2–3x concurrency on hybrid models like Qwen 3.6, with quality within 5% of the BF16 baseline.

---

## 2. The architecture in one diagram

```
                    ┌─────────────────────────────────────┐
                    │           CONFIG (YAML)             │
                    │  8 numbers: weights + thresholds    │
                    └────────────────┬────────────────────┘
                                     │ (read once at startup)
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │                   COMPILER (cold)                  │
        │   turns config → closures + codec table            │
        │   builds CompiledRuntime object                    │
        └────────────────┬───────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │  RUNTIME (hot)   │      │   SAFETY (cold)  │
  │  score()         │      │  circuit breaker │
  │  desired_tier()  │      │  fallback ladder │
  │  migrate()       │      │  pressure guard  │
  └────────┬─────────┘      └────────┬─────────┘
           │                         │
           ▼                         ▼
  ┌──────────────────────────────────────────────┐
  │              ALLOCATOR + CODECS              │
  │  BF16 / FP8 / INT4 / Archive / CPU / Disk    │
  └──────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   SGLang (host)     │
              │   serves the model  │
              └─────────────────────┘
```

The hot path (runs every decode step, every page) is the three functions in the middle: `score()` → `desired_tier()` → `migrate()`. Everything else is cold — runs once at startup or rarely.

---

## 3. The three splits

| Split | Team | Owns | Deliverable |
|---|---|---|---|
| **1 (Core)** | Already built | Contracts, compiler, runtime logic, fault tolerance contracts | Foundation you build on |
| **2 (Codecs)** | **You** | GPU kernels that compress/decompress KV between numeric formats | Triton kernels + Python wrappers |
| **3 (Integration)** | **You** | SGLang wiring, safety ladder, telemetry, allocator | Running ASH-KV inside SGLang |

**Split 1 is done.** It provides:
- Frozen contracts (types, protocols, config schema)
- The runtime hot-path functions (score, controller, migrate)
- The compiler that turns config into closures
- 77 passing tests, including fault injection

**Your two teams build on top of those contracts.** You do not modify Split 1 — you implement the protocols it defines.

---

## 4. The five architectural principles

These are non-negotiable. If your work violates any of them, the architecture breaks.

### Principle 0 — Cold/Hot separation

```
Cold path (startup):     config parsing, plugin loading, validation,
                         codec resolution, closure compilation
Hot path (every decode): score(), desired_tier(), migrate()
```

Nothing flexible crosses into the hot path. Config is read once at startup and baked into closures. The hot path never reads config, never checks flags, never queries registries. **If your hot-path code reads a config value, you have broken this principle.**

### Principle 1 — Single score, single controller, single migration engine

The entire policy is one score function:
```
R = w_T·T + w_S·S + w_N·N + w_P·P
```

And one controller:
```
target_tier = desired_tier(R, current_tier, pressure)
```

And one migration function:
```
migrate(page, target_tier)
```

There are no separate "BF16 controller," "FP8 controller," "INT4 controller." There is one controller that decides the target tier, and one migration engine that executes the move. Adding a new tier means adding one codec pair, not a new code path.

### Principle 2 — The 8-number surface

The entire tunable surface of ASH-KV is eight numbers:

| # | Name | Meaning |
|---|---|---|
| 1 | `w_T` | temporal decay weight |
| 2 | `w_S` | saliency weight |
| 3 | `w_N` | novelty weight |
| 4 | `w_P` | prefix affinity weight |
| 5 | `theta_high` | score above which → BF16 |
| 6 | `theta_low` | score below which → INT4 |
| 7 | `delta` | hysteresis offset (prevents flapping) |
| 8 | `p_emergency` | pressure that triggers emergency behavior |

If your team needs a 9th knob, that's a coordinated contract change across all three splits. Do not add tunables silently.

### Principle 3 — Never throw on the hot path

Every function in the hot path returns a typed result. It never raises. Codec fails → return FAILURE. Checksum mismatches → return CORRUPT. Allocator OOMs → return FAILURE. The decode loop checks the status code and continues. **If your hot-path code raises an exception that escapes to SGLang, you have broken this principle.**

### Principle 4 — BF16 is the fallback tier for page-level recovery

When a page cannot be served at its current tier (codec broke, checksum mismatched, breaker tripped), that page is reconstructed in BF16. The decode continues. Other pages are unaffected. Recovery is scoped to the page, never to the request or the server.

### Principle 5 — The controller never sees bytes

The controller sees only:
- `R` (the score)
- `pressure` (a single scalar in [0, 1])
- `current_tier` (for hysteresis)

It does not see how many bytes are free. It does not see which GPU. It does not see HBM vs CPU. The allocator computes one scalar — `pressure` — and that's all the controller knows. **If your integration code passes memory details to the controller, you have broken this principle.**

### Principle 6 — The score cannot see the current tier

The score is a function of the page's content and history (age, saliency, novelty, prefix affinity). It does NOT know what tier the page is currently on. Otherwise: score influences tier, tier influences score, feedback loop. The only place current tier enters the decision is the hysteresis offset `delta` in the controller.

### Principle 7 — Hardware is detected, not assumed

The compiler probes the GPU once at startup and picks codecs accordingly. A100 → INT8 (no native FP8). H100/MI300X → FP8. The runtime never probes hardware — it receives a pre-built codec table and uses it. This means ASH-KV runs on any GPU the codec registry supports, without per-deployment configuration.

**If your runtime code calls `torch.cuda.get_device_properties`, you have broken this principle.** Hardware probing is a cold-path concern, owned by `compiler/hardware_probe.py`.

### Principle 8 — Codecs are swappable, tiers are not

The tier enum (BF16 → compressed → cold → archive → CPU → disk) is fixed and frozen. What fills each tier is configurable:
- On A100: the "compressed" tier is filled by an INT8 codec
- On MI300X: the "compressed" tier is filled by an FP8 codec
- Power users can override either with their own registered codec

The tier name is a label. The codec is the implementation. The controller doesn't know which codec backs the "compressed" tier — it just sees "tier 1." This is what makes the system portable across hardware without code changes.

---

## 5. The tier hierarchy

```
BF16  (0)  ← hottest, exact, 16-bit
  ↓
FP8   (1)  ← 2x compression
  ↓
INT4  (2)  ← 4x compression
  ↓
ARCHIVE (3) ← low-rank or vector quantization
  ↓
CPU   (4)  ← offloaded to host RAM
  ↓
DISK  (5)  ← coldest, persistent storage
```

Pages migrate between tiers based on score. Hot pages promote; cold pages demote. The migration engine is single-path: `migrate(page, target)` works for any (from, to) pair, dispatched via a codec table.

### 5.1 Tier names vs. codec implementations

The tier enum is fixed. What fills each tier is hardware-dependent:

| Tier | Tier name | A100 (Ampere) | MI300X / H100+ |
|---|---|---|---|
| 0 | BF16 | BF16 (exact) | BF16 (exact) |
| 1 | "compressed" | INT8 codec | FP8 codec |
| 2 | "cold" | INT4 codec | INT4 codec |
| 3 | "archive" | low-rank or TurboQuant | same |
| 4 | CPU | host RAM | host RAM |
| 5 | disk | persistent | persistent |

The controller sees only "tier 1." Whether tier 1 is INT8 or FP8 is a codec choice, resolved at startup. This is what makes ASH-KV portable across hardware.

---

## 6. The hot path (what runs every decode step)

This is the entire decision loop, conceptually:

```python
# 1. Snapshot all pages as a numpy array
pages = page_table.snapshot()  # one copy, vectorized

# 2. Compute scores for all pages at once
R = runtime.score_fn(pages["T"], pages["S"], pages["N"], pages["P"])

# 3. Decide target tier for all pages at once
targets = runtime.controller_fn(R, pages["tier"], allocator.pressure())

# 4. For each page that needs to move, migrate it
for i in range(len(pages)):
    if targets[i] != pages["tier"][i]:
        result = runtime.migrate_fn(
            page_id=int(pages["page_id"][i]),
            target_tier=Tier(int(targets[i])),
            page_table=page_table,
            allocator=allocator,
            codec_table=runtime.codec_table,
            ...
        )
        runtime.telemetry_fn(result)
```

Steps 1–3 are vectorized numpy — microseconds for thousands of pages. Step 4 is a Python loop but typically processes <10 pages per decode step (most pages don't move). The control-plane overhead target is **under 1% of decode latency**.

---

## 7. The fault tolerance ladder

Every failure degrades gracefully. The ladder, from least to most severe:

```
1. Normal operation
2. Migration fails          → stay on current tier, log, increment breaker
3. Current tier corrupt     → fall back to BF16 for that page
4. BF16 unavailable (OOM)   → evict cold pages, retry
5. Request cannot proceed   → reject request, server stays healthy
6. Server health compromised → drain, shut down
```

Each rung has a defined response. No handler returns "unknown." The unit of recovery is the page, not the request (Principle 4).

**Circuit breakers:** every codec has one. If a codec fails 5 times in 60 seconds, the breaker trips — that codec is not dispatched for the next 60 seconds. Pages that would have used it stay on their current tier. This prevents one bad kernel from crashing the server.

---

## 8. Hardware auto-detection (NEW)

ASH-KV runs on any GPU the codec registry supports, without per-deployment configuration. The compiler probes the hardware once at startup and picks the right codecs.

### 8.1 The startup flow

```
1. Load config.yaml (user intent + overrides)
2. Probe hardware → HardwareProfile
3. Resolve codecs → CodecTable (auto-detect or user override)
4. Freeze codec registry
5. Build runtime → CompiledRuntime
6. Hand to integration layer
```

### 8.2 What the probe detects

```python
@dataclass(frozen=True, slots=True)
class HardwareProfile:
    gpu_name: str                          # "NVIDIA A100-SXM4-80GB"
    compute_capability: tuple[int, int]    # (8, 0) = Ampere
    vram_bytes: int
    has_fp8_native: bool                   # True on Hopper+, MI300X
    has_int8_native: bool                  # True on Turing+
    has_bf16_native: bool                  # True on Ampere+
```

The probe is conservative. If uncertain, it falls back to the safer codec (INT8 over FP8) and logs a warning.

### 8.3 Default codec resolution

| Tier slot | Hopper+ / MI300X | Ampere (A100) | Turing (T4) |
|---|---|---|---|
| BF16 → compressed | FP8 | INT8 | INT8 |
| compressed → cold | INT4 | INT4 | INT4 |
| cold → archive | low-rank or TurboQuant | same | same |

### 8.4 User override

Power users can override any slot in config:

```yaml
codecs:
  bf16_to_compressed: "int8_custom"   # explicit, overrides auto-detect
  compressed_to_cold: "auto"          # let the system pick
```

Resolution precedence:
1. User explicit override (highest)
2. Auto-detect from hardware
3. Built-in default (lowest)

### 8.5 What the runtime sees

The runtime sees only the resolved `CodecTable` — a flat `dict[(from_tier, to_tier), Codec]`. Whether that dict was built by auto-detection or user override is invisible to the hot path. The `HardwareProfile` does NOT cross into `runtime/`.

### 8.6 Startup log

The compiler logs what it picked, so deployments are verifiable:

```
[compiler] hardware probe: NVIDIA A100-SXM4-80GB (cc 8.0)
[compiler]   has_fp8_native: False
[compiler]   has_int8_native: True
[compiler] codec slot 'bf16_to_compressed' = auto → resolved to 'int8_default'
[compiler] codec slot 'compressed_to_cold' = auto → resolved to 'int4_default'
[compiler] codec registry frozen.
```

---

# SPLIT 2: CODECS

## What you own

You write the GPU kernels that convert KV bytes between numeric formats. Your code lives in `ashkv/codecs/`. You implement the `Codec` protocol defined in `ashkv/contracts/protocols.py`.

## Your files

```
codecs/
├── bf16.py          # identity codec (BF16 → BF16, for fallback)
├── fp8.py           # BF16 ↔ FP8   ← for Hopper+/MI300X
├── int8.py          # BF16 ↔ INT8  ← for Ampere (A100), no native FP8
├── int4.py          # BF16 ↔ INT4  ← cold tier, all hardware
├── turboquant.py    # archive tier (vector quantization)
├── lowrank.py       # archive tier (SVD/PCA)
└── checksum.py      # shared checksum utility (xxhash or CRC32)
```

**Note on INT8:** A100 (Ampere) has no native FP8 hardware. The auto-detect resolver picks INT8 for the "compressed" tier on A100. INT8 requires per-token scaling factors to avoid quality collapse (FP8 doesn't). Plan for this — the codec must compute and store scale factors alongside the quantized values.

## What's already implemented

Split 1 has already built the codec framework:

| File | Status | What it does |
|---|---|---|
| `codecs/bf16.py` | ✅ Done | Identity codec (BF16 → BF16). For fallback tier. |
| `codecs/checksum.py` | ✅ Done | Shared checksum (xxhash or SHA-256 fallback). |
| `codecs/mock.py` | ✅ Done | 5 mock codecs for testing. |
| `codecs/int8.py` | ✅ Python fallback | INT8 with per-token scaling. Triton kernel scaffolded. |
| `codecs/fp8.py` | ✅ Python fallback | FP8 (E4M3). Triton kernel scaffolded. |
| `codecs/int4.py` | ✅ Python fallback | INT4 with per-group scaling. Triton kernel scaffolded. |

**Your job:** Replace the Python fallbacks with real Triton kernels. The Python versions are correct but slow (~100x slower than Triton). The kernel interface is documented in comments in each file. Validate the round-trip checksum invariant on real GPU hardware.

## Your interface

```python
class Codec(Protocol):
    def encode(self, source_bytes: bytes) -> bytes: ...
    def decode(self, target_bytes: bytes) -> bytes: ...
    def checksum(self, raw_bytes: bytes) -> int: ...
```

That's the entire interface. You see bytes in, bytes out, and a checksum. You do NOT see:
- Pages, tiers, or pressure
- The controller or score
- SGLang or the model
- Config or thresholds

You are called by `migrate()` in `ashkv/runtime/migrate.py`. The flow:

1. `migrate()` reads source bytes from the allocator
2. `migrate()` calls your `encode(source_bytes)` → compressed bytes
3. `migrate()` allocates a target buffer, writes your compressed bytes
4. `migrate()` calls your `decode(target_bytes)` → reconstructed bytes
5. `migrate()` calls your `checksum(reconstructed)` and compares against the page's `bf16_checksum`
6. On match: commit. On mismatch: return CORRUPT, page unchanged.

## Your correctness invariant

For every codec you ship, this must hold:
```python
encoded = codec.encode(sample_bytes)
decoded = codec.decode(encoded)
assert codec.checksum(decoded) == codec.checksum(sample_bytes)
```

If your codec cannot meet this at a given compression level, that's a quality cliff. Document it; do not silently degrade.

## How you register

At import time (cold path):
```python
from ashkv.compiler.registry import codec_registry

codec_registry.register("fp8_default", FP8Codec())
```

After all imports, the integration team calls `codec_registry.freeze()`. Any later registration raises `RuntimeError`. This prevents dynamic codec loading during inference.

## What you may import

- `ashkv.contracts` (for the `Codec` protocol and `Tier` enum)
- `triton`, `torch`, `numpy` (for your kernels)

You may NOT import: `runtime/`, `compiler/`, `safety/`, `telemetry/`, `sglang_integration/`. The dependency-direction test will catch violations.

## Your priority order

1. **`bf16.py`** — identity codec. Trivial but proves the wiring. Ship first.
2. **`fp8.py`** — BF16 ↔ FP8. Highest ROI. MI300X has native FP8. This alone should give 10–20% latency improvement at long context.
3. **`int4.py`** — BF16 ↔ INT4. The cold tier. More involved; needs careful dequant.
4. **`turboquant.py` or `lowrank.py`** — archive tier. Pick one based on Phase 5/6 benchmarks.

## Your development environment

You need:
- MI300X access (or any modern AMD/NVIDIA GPU with FP8 support)
- Triton installed
- PyTorch installed

You do NOT need SGLang. You do NOT need the full ASH-KV stack running. You can develop and test codecs in isolation against the `Codec` protocol.

## How you test

Write `tests/test_codecs.py`. Each codec gets:
- Roundtrip test (encode → decode → checksum match)
- Performance benchmark (vs naive PyTorch reference)
- Fault test (kernel failure returns empty/garbage, not exception)

The fault injection harness in `tests/test_migrate_fault_injection.py` shows how your codec will be called under failure conditions. Read it.

## The biggest risk you face

**Quality cliffs.** INT4 and archive codecs don't degrade gracefully forever — at some compression level, quality collapses suddenly. Your job is to find where that cliff is, document it, and make sure the controller never pushes pages past it without explicit configuration. The 5% quality budget is the contract; you must stay within it.

---

# SPLIT 3: INTEGRATION + SAFETY + OPS

## What you own

You wire ASH-KV into SGLang. The safety ladder and telemetry are already implemented by Split 1. Your remaining work is the real `Allocator` (wrapping SGLang's block manager) and the SGLang integration hooks.

## What's already implemented

Split 1 has already built:

| Module | Status | What it does |
|---|---|---|
| `safety/circuit_breaker.py` | ✅ Done | Per-codec failure tracking. Trips after 5 failures in 60s. |
| `safety/pressure_guard.py` | ✅ Done | Maps pressure scalar → PressureState. Pure function. |
| `safety/fallback.py` | ✅ Done | BF16 page-level recovery. `handle_migration_failure()` entry point. |
| `safety/health.py` | ✅ Done | System health monitor. HEALTHY/DEGRADED/UNHEALTHY/CRITICAL. |
| `telemetry/counters.py` | ✅ Done | 18 metrics, thread-safe. |
| `telemetry/prometheus.py` | ✅ Done | Prometheus exporter (lazy-imported, optional). |
| `runtime/allocator.py` | ✅ Mock done | `MockAllocator` for testing. Real allocator needs SGLang. |

## What you need to implement

```
runtime/
└── allocator.py              # ADD: real SGLangAllocator (wraps SGLang block manager)

sglang_integration/
├── block_manager_patch.py    # NEW: attach PageTable metadata to SGLang blocks
├── radix_cache_bridge.py     # NEW: bridge to SGLang radix cache for prefix reuse
└── layer_type_filter.py      # NEW: skip DeltaNet/Mamba layers (hybrid Qwen)
```

## Your two main jobs

### Job 1: Implement the real Allocator

```python
class Allocator(Protocol):
    def alloc(self, tier: Tier, size_bytes: int) -> int: ...   # returns handle or -1
    def free(self, handle: int) -> None: ...                    # no-op on invalid
    def read(self, handle: int) -> bytes: ...                   # b"" on invalid
    def write(self, handle: int, data: bytes) -> None: ...      # no-op on invalid
    def pressure(self) -> float: ...                            # scalar in [0, 1]
```

**Contract rules:**
- Never raise. Return -1 / b"" / no-op on failure.
- `pressure()` returns ONE scalar. Not two. If you find yourself wanting HBM pressure AND CPU pressure as separate signals, that's a contract change — talk to Split 1 first.

**Pressure semantics:**
```
0.0  = empty
0.85 = ELEVATED   (start demoting aggressively)
0.95 = CRITICAL   (p_emergency; reject new admissions)
0.99 = SATURATED  (refuse everything, drain)
1.0  = full
```

The `0.85` and `0.99` thresholds are constants in `ashkv.contracts.results`. The `0.95` is `config.p_emergency` and is configurable.

For a single-GPU allocator:
```python
def pressure(self) -> float:
    return self._used_bytes / self._budget_bytes
```

For a multi-tier allocator (HBM + CPU):
```python
def pressure(self) -> float:
    hbm_p = self._hbm_used / self._hbm_budget
    cpu_p = self._cpu_used / self._cpu_budget
    return 0.8 * hbm_p + 0.2 * cpu_p  # one scalar, your weighting
```

### Job 2: Hook into SGLang

You must hook SGLang at these points:

1. **Block allocation** — when SGLang allocates a KV block, call `page_table.add(...)` to register it
2. **Block free** — when SGLang frees a block, call `page_table.remove(page_id)`
3. **Decode step** — before each decode, run the controller on the snapshot, then call `migrate()` for pages that need to move
4. **Prefix cache** — bridge SGLang's radix cache to `page_table.pin()` / `unpin()`. Pinned pages skip migration.
5. **Layer filter** — for hybrid models (Qwen 3.5/3.6), skip DeltaNet/Mamba layers. Only attention layers have KV pages. SGLang's model loader knows which is which; you just ask.

### Job 3: Wire the safety layer (already implemented)

```
safety/circuit_breaker.py
- Per-codec failure tracking
- 5 failures in 60 seconds → trip
- Tripped codec not dispatched for 60 seconds
- Pages that would use it stay on current tier

safety/fallback.py
- BF16 fallback ladder (Principle 4)
- Page-level recovery, not request-level
- When codec fails: stay on current tier
- When current tier corrupt: reconstruct in BF16 from bf16_checksum
- When BF16 OOM: evict cold pages, retry

safety/pressure_guard.py
- Map pressure scalar to PressureState enum
- NORMAL / ELEVATED / CRITICAL / SATURATED
- Each state has defined response

safety/health.py
- Background health monitor
- Export to telemetry
- Detect cascading failures
```

## What you may import

- Everything in `ashkv/` (contracts, runtime, codecs, compiler)
- `sglang`, `torch`, `numpy`
- `prometheus_client` (lazy import)

## Your development environment

You need:
- A running SGLang instance serving Qwen 3.5 or 3.6
- MI300X (or equivalent) GPU access
- The codecs from Split 2 (for end-to-end testing)

You can start before Split 2 finishes — use mock codecs (see `tests/test_migrate_fault_injection.py` for examples) to develop the integration scaffolding.

## Your priority order

1. **`layer_type_filter.py`** — simplest SGLang hook, proves the integration works
2. **`runtime/allocator.py`** — implements the Allocator protocol, wraps SGLang block manager
3. **`safety/pressure_guard.py`** — simplest safety module
4. **`block_manager_patch.py`** — attach PageTable metadata to SGLang blocks
5. **`radix_cache_bridge.py`** — bridge prefix reuse
6. **`safety/circuit_breaker.py`** + **`safety/fallback.py`** — the full safety ladder
7. **`telemetry/`** — metrics export

## The biggest risk you face

**The single-scalar pressure model.** Principle 5 says the controller sees only one scalar. In practice, HBM and CPU memory have different dynamics, and you'll be tempted to leak a second signal. Don't — at least not without coordinating with Split 1. The single-scalar constraint is what keeps the controller simple. If it can't capture a real failure mode, escalate rather than silently violating the principle.

The second risk is **synchronous migrate() on the critical path.** If migration stalls a decode step, you've added latency instead of removing it. You may need a background migration queue — but that's an integration decision, not a contract change. The `migrate()` function itself is synchronous; how you schedule it is your call.

---

## 8. How the three splits coordinate

### Contract changes (rare, coordinated)

If any team needs to:
- Add a field to `Page`
- Add a new `Tier`
- Add a new `MigrationStatus` or `PressureState`
- Change the `Codec` or `Allocator` protocol signatures
- Add a 9th config parameter

**That's a coordinated change.** Open an issue, tag all three teams, do not merge unilaterally. The contracts are in `ashkv/contracts/` and are frozen.

### Codec additions (Split 2, autonomous)

New codecs do NOT require contract changes. Register, use, ship. Split 3 picks them up via the codec table.

### Threshold tuning (Split 3, autonomous)

Per-model YAML files in `config/`. No code changes. The 8 numbers are tuned per model and per workload.

### Dependency direction (enforced by test)

```
contracts/   ← foundation, imports nothing from ashkv
runtime/     ← imports contracts/ only
codecs/      ← imports contracts/ + triton/torch
compiler/    ← imports contracts/, runtime/, codecs/, plugins/, config/
safety/      ← imports contracts/, runtime/, codecs/
telemetry/   ← imports contracts/ only
sglang_integration/ ← imports everything + sglang
```

This is enforced by `tests/test_dependency_direction.py`. It runs on every PR. It greps imports and fails the build if any file violates the rule. We have already verified it works by deliberately breaking it and watching the test catch the violation.

The rule that matters most: **`runtime/` never imports `compiler/`, `config/`, `safety/`, `telemetry/`, `sglang_integration/`, or `codecs/`.** If it does, the cold/hot boundary is broken.

---

## 9. The implementation phases

These are the milestones. Each phase has a clear exit criterion.

| Phase | What | Owner | Exit criterion |
|---|---|---|---|
| 0 | Repo skeleton, contracts, tests | Split 1 ✅ | Done |
| 1 | Instrumentation (metadata + metrics) | Split 3 | Dashboard shows KV occupancy, no behavior change |
| 2 | Prefix plane (pin shared prefixes) | Split 3 | 100 users → 1 prefill + 99 cache hits |
| 3 | FP8 codec + migration wiring | Split 2 + Split 3 | 10–20% latency improvement, negligible quality loss |
| 4 | Score + controller (age only) | Split 1 ✅ | Pages demote by age, no flapping |
| 5 | INT4 codec | Split 2 + Split 3 | INT4 tier active, quality within 5% |
| 6 | Saliency, novelty, prefix affinity | Split 3 | Same or better accuracy at same memory |
| 7 | Hysteresis + emergency pressure | Split 3 ✅ | No flapping, no OOM crashes (safety layer done) |
| 8 | Archive codec | Split 2 | Old context survives cheaply |
| 9 | CPU offload | Split 3 | Long-context tasks that would OOM now run |
| 10 | Hardening | Split 3 ✅ | 24-hour soak test, zero request failures (fault injection done) |
| 11 | MVP | All | Configurable by another team without your help |
| 12 | Production | All | Multi-tenant, SLA-bound |

### Implementation status (current)

| Component | Status | Tests |
|---|---|---|
| `contracts/` | ✅ Frozen, tested | 40 |
| `runtime/` (score, controller, migrate, mock allocator) | ✅ Done | 28 |
| `compiler/` (closures, registry, hardware probe, codec resolver) | ✅ Done | 8 |
| `codecs/` (BF16, mocks, Python fallbacks for INT8/FP8/INT4) | ✅ Framework done, Triton kernels need GPU | 31 |
| `safety/` (circuit breaker, pressure guard, fallback, health) | ✅ Done | 33 |
| `telemetry/` (counters, prometheus exporter) | ✅ Done | included above |
| `sglang_integration/` | ⏳ Needs SGLang runtime | 0 |
| **Total** | **144 passing, 2 skipped** | **144** |

**What's left:**
1. **Triton kernels** for INT8/FP8/INT4 — Python fallbacks work but are slow. Need GPU to write and validate real kernels. The kernel interfaces are scaffolded as comments in each codec file.
2. **SGLang integration** — `block_manager_patch.py`, `radix_cache_bridge.py`, `layer_type_filter.py`. Needs SGLang installed to test.

---

## 10. What to read first

1. **This document** — the architecture and your role
2. **`INTERFACE.md`** — the precise API contracts you must satisfy
3. **`ashkv/contracts/`** — the actual code for those contracts
4. **`tests/test_migrate_fault_injection.py`** — how your code will be called under failure
5. **`tests/test_dependency_direction.py`** — the boundary rules, enforced

Start there. Ask questions before writing code that crosses a boundary.

---

## 11. The one-paragraph summary

ASH-KV is a tiered KV cache manager for LLM inference. It compresses old context into cheaper numeric formats (FP8, INT4, archive) while keeping recent important context exact (BF16), so the same GPU serves longer contexts to more users without quality collapse. The architecture has one score, one controller, one migration engine, eight tunable numbers, and a strict cold/hot boundary enforced by tests. Split 2 writes the compression codecs. Split 3 wires it into SGLang and implements the safety ladder. The contracts are frozen. The foundation is tested. Build on top.
