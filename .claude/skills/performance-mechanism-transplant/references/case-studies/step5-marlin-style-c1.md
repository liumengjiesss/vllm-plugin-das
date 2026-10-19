# Marlin-style fixed-shape specialization → Step5 C1

Status: **CONFIRMED_POSITIVE** for C1; **WHOLE_BACKEND_TRANSPLANT = REJECTED**.
This case is the flagship worked example for
`performance-mechanism-transplant`. It is not a claim that "Marlin is fast" or
that "Marlin was ported".

```text
MARLIN_EXECUTION            = NO on the measured slimquant_marlin arm
SLIMQUANT_REGRESSION        = forced Triton fallback, not Marlin performance
FALLBACK_FIX                = correctness / backend-selection fix (separate)
WHOLE_BACKEND_TRANSPLANT    = REJECTED
C1                          = Marlin-style fixed-shape launch-config
                              specialization on the existing HCU path
C1_LOCAL_KERNEL_GAIN        = 23.0-34.4 % per call (M=16..128)
C1_E2E_BW1100               = +1.11 % median decode
C1_E2E_BW1101               = +1.39 % median decode, 3/3 paired wins
C1_CORRECTNESS              = bit-exact (max_abs 0.0)
C1_CONFIRMED_POSITIVE       = YES
C1_STRICT_NEW_BEST          = NO (below the project's 1.99 % bar)
```

The correct one-line description is:

> C1 transplanted the **Marlin-style mechanism** — an offline shape × M table
> selecting a pre-tuned launch configuration — onto the incumbent HCU kernel. It
> did not transplant Marlin's kernel, weight layout, EP semantics, or dataflow.

---

## 1. The original problem

The requested Step5 configuration:

```text
quantization = slimquant_marlin
```

was expected to improve performance. Instead, decode slowed down.

Measured clean first-gate result for the `slimquant_marlin` arm:

```text
Decode TPS       ~1038.6
ITL              ~86.42 ms
Prefill TPS      ~9261.9
MTP acceptance   ~0.6014
```

Relative to the A0 production baseline:

```text
Decode   ~-28 %
ITL      ~+39 %
Prefill  ~-29 %
```

The naive explanation was:

> "Marlin is slow."

That explanation is wrong.

## 2. Runtime-path proof — Marlin never executed

The `slimquant_marlin` configuration name suggested Marlin. The runtime evidence
showed otherwise.

The LightOp Marlin path is gated by a **data table**, not by shape support. For
the Step5 expert GEMMs it needed:

```text
MOE_BLOCKINT8_CUDA_MARLIN_3072_4096_GFX938_CU64.json   (GEMM1, UP)
MOE_BLOCKINT8_CUDA_MARLIN_4096_1536_GFX938_CU64.json   (GEMM2, DOWN)
```

Neither file existed anywhere in the packaged config set. A directory scan and
an inner-key scan over all 656 packaged JSONs returned **0 hits**, on every arch
and CU count. The nearest siblings — including the existing Step3.x rows
`2560_4096` and `4096_1280` — were:

```text
1536_4096, 3072_1536, 4096_1280, 2560_4096, ...
```

The kernel binaries themselves were shape-generic enough:
`moe_w8a8_fp8_marlin_{32,64,128}x256x128_TN_BF16_{UP,DOWN}.co`, and Step5's
dimensions divide the tile exactly (`3072 = 12*256`, `4096 = 32*128`,
`1536 = 12*128`). The missing piece was a **tuning artifact**, not a capability.

On a config miss, the HCU plugin at
`compressed_tensors_moe_marlin.py:199-221` did this:

```python
triton_moe = copy.copy(layer.moe_config)
triton_moe.moe_backend = "triton"
```

It therefore could not fall back to A0's `DeepGemm` / `ep_fused_moe` path. The
actual runtime path was:

```text
slimquant_marlin config
  -> LightOp Marlin lookup
  -> no matching config for the Step5 shape
  -> forced moe_backend = "triton"
  -> generic vLLM Triton FP8 MoE
  -> HCU production DeepGemm / ep_fused_moe path bypassed
```

Measured Marlin kernel execution count:

```text
MARLIN_EXECUTION_COUNT = 0
```

The server log confirmed the miss per layer:

```text
LightOp Marlin has no config for SlimQuant FP8 MoE layer ...
(experts=352, hidden=4096, intermediate/tp=1536);
using the official Triton FP8 MoE method before weight packing.
```

Therefore:

```text
"slimquant is slow"  !=  "the Marlin kernel is slow"
```

The measured regression was the **generic Triton fallback**, not Marlin.

**Failure class:** `CORRECTNESS_FAILURE / WRONG_FALLBACK` for the backend
selection; `MEASUREMENT_ARTIFACT` for the interpretation "Marlin is slow".

## 3. The fallback fix — separate from C1

The minimal correctness fix is the policy line, not the table: on a Marlin
config miss, fall back to the incumbent's best available path (or the
production `auto` selection) instead of hard-coding `triton`.

After the fallback was corrected, `-q slimquant_marlin` no longer materially
regressed Step5 decode and was **approximately at parity with the default
production path**.

This record is qualitative: the evidence pack does not contain a clean paired
A/B of the post-fix `slimquant_marlin` arm, so no specific percentage is
claimed here. What matters is the classification:

```text
FALLBACK_FIX = CORRECTNESS / BACKEND-SELECTION FIX
FALLBACK_FIX != C1 PERFORMANCE OPTIMIZATION
```

Do not merge these two events. The fallback fix removes a wrong dispatch. C1 is
a later, independent optimization on a different layer.

## 4. Why not simply tune Marlin

After the fallback path was understood, the next question was whether the Marlin
backend could be made to beat A0. The answer was no, for three independent
reasons.

### 4.1 Marlin's FP8 runner is not EP-aware

The shipped FP8 Marlin runner:

- aligns tokens in the **global** expert space
  (`moe_align_block_size(..., global_num_experts, expert_map)`);
- allocates caches with `torch.empty`;
- reduces with `expert_mask=None, num_local_tokens=None`, i.e. sums **all**
  `M*topk` routes.

A0 is the opposite design:

- local-expert-space sort with a bound of
  `min(full, cdiv(routes + min(routes, local_E)*(bm-1), bm)*bm)`
  = **1696** entries at Step5 decode, versus ~6648 for a global-space bound
  (3.9x tighter after the arithmetic correction);
- non-local routes are binned and never laid out;
- activation/quant is bounded by a device-side `nvalid`;
- the router weight is folded into the per-row activation scale;
- the reduce is masked and carries the shared-expert add;
- no zero-fill, no permute/unpermute.

At EP8, the local fraction is exactly 1/8. A non-EP runner pays for routes that
do not exist. A whole-backend swap would have to re-derive A0's local sort,
`nvalid` bound, and masked reduce — at which point the only Marlin contribution
left is the GEMM tile table and weight layout.

### 4.2 The vendor's decode config family is structurally excluded

The vendor's decode geometry and A0's tile are mutually exclusive at the
hardware level:

```text
vendor decode family:  BN=16, BK=512, ns=3, nw=4
A0 G1:                 BN=32, BK=256, ns=1, nw=2
```

Measured on A0's own persistent kernel, all 85 real layers:

```text
candA vendor family BN16/BK512/ns3/nw4
  G1 170.31 -> 665.95 us        +291 % on G1
  G1+G2                         +195.1 %
candB BK=512 alone
  G1+G2                         +3.07 %
candC BK=512 + ns=3
  OutOfResources: Required 73728 B, limit 65536 B
  73728 = (32*512 + 16*512) * 1 byte * 3 stages   (exact)
```

The vendor's `BN=16` form fits LDS (48 KB); the full vendor family measured
3.9x slower on G1, and `BN=16` alone measured +36 % in `EXP-004`. A0's `BN=32`
form is the incumbent but overflows LDS at the vendor's BK/ns. The family is
excluded from both directions. All arms that produced output were bit-exact.

This is the explicit conclusion from scanning the existing gfx938 Marlin config
space: **even selecting a better combination, the complete Marlin MoE still
cannot exceed the current `ep_fused_moe` production path.** The reason is
mechanism-level, not parameter-level — the runner's non-EP dataflow,
global-space alignment, unmasked reduce, and missing local-route specialization
cost more than any tile choice can recover. There is no direct full-Marlin E2E
in the evidence pack because the required tables were absent; the conclusion
rests on source analysis, the measured config-space experiments, and the A0
budget.

A separate sweep of A0's own config space (`EXP-004`, `EXP-009`) found no win:
the persistent program count was exhausted, the decode tile was already at the
vendor's small-M convention, `ep_sort` was at its launch floor, and a cache
modifier on the B stream measured 0.03 % (null).

### 4.3 The remaining artifacts are not reachable

- The `{32,64,128}x256x128` asm tiles are selected only at larger M; at total
  M=128, 5 of 6 shipped sibling tables select the compiled family with an
  alignment block of 16.
- `{32,64,128}` are **tile parameters** (`BLOCK_M`), not token counts. The table
  is keyed by total tokens, not per-expert `M_e`. No Marlin path consults
  per-expert occupancy.
- The `[E, K/64, N*64]` packed layout is mutually unreadable by A0's
  `[E, N, K]` kernel; duplicating it would cost ~70 GB per rank across the 85
  MoE layers.
- The vendor's ASM/CK tiers have no `E=44` row for fp8 channel-wise on gfx938
  (the expert set is `{128,129,160,192,256,257,288,384,512}`).

### 4.4 Verdict

```text
WHOLE_BACKEND_TRANSPLANT = REJECTED
MARLIN_STYLE_EXPLORATION = CLOSED   (stop condition B)
```

Evidence class: source analysis + measured config-space experiments + project
records. There is no direct full-Marlin E2E because Marlin never executed on
this workload.

The useful residue was not a kernel, a layout, or a backend. It was one
**mechanism**:

> an offline, shape × M-keyed table selecting a pre-tuned launch configuration
> for a stable operating point.

That is the mechanism C1 transplanted.

## 5. The remaining gap — MTP draft BF16 MoE

A0's main FP8 MoE was not the place to look. The persistent kernel was already
at ~1.50 TB/s of unique weight bytes against a measured 1.817 TB/s streaming
ceiling (`R/Rmax = 0.824`), and the remaining addressable fraction had been
probed and closed.

The uncovered specialization was elsewhere:

```text
TARGET          Step5 MTP draft-layer MoE, layers 88-90
LAYER TYPE      the only unquantized bf16 MoE in the model
KERNEL          vLLM TritonExperts fused_moe_kernel (not the persistent fp8 kernel)
E_GLOBAL        352
E_LOCAL         44 after EP8
N               1536  (moe_intermediate_size)
K               4096  (hidden_size)
TOPK            8
DTYPE           bf16 / unquantized
CALLS/STEP      6  (GEMM1 + GEMM2 x 3 draft steps)
A0 TIME         2.479 ms/step (EXP-008 differential trace;
                server-term per-life 2.359-2.593 ms/step)
```

vLLM resolves Triton MoE tile configs from a per-shape JSON file. No file
existed for this shape on this device, so every decode step ran the generic
default. The server log said so on each start:

```text
WARNING [fused_moe.py] Using default MoE config. Performance might be sub-optimal!
Config file not found at .../fused_moe/configs/E=44,N=1536,device_name=BW1100.json
```

The default was wrong for the operating point:

- ~1-4 rows per expert, so the kernel is pure weight streaming;
- ~18.9 MB per expert pair in bf16;
- the default 4-warp / 2-stage 16x64 tile leaves bytes in flight on the table.

This is the signature of a good specialization target:

```text
STABLE              yes: fixed model shape
FREQUENT            yes: 6 calls/step, every decode step
BUDGET_RELEVANT     yes: ~2.4 ms/step of a ~62 ms ITL
GENERIC_CONFIG      yes: runtime warning proves the default
CORRECT_KERNEL      yes: no kernel change needed
NARROWLY_SELECTABLE yes: one E/N/dtype/device key
EASY_TO_FALLBACK    yes: a config pointer
CORRECTNESS_SAFE    yes: launch geometry only
```

## 6. C1 action

No backend swap. No new kernel. No math change. No routing, quantization, or
communication change.

One tuned tile:

```text
BLOCK_SIZE_M   16
BLOCK_SIZE_N   64
BLOCK_SIZE_K   128
GROUP_SIZE_M    1
num_warps       2
num_stages      1
```

Delivered as vLLM's own tuned-config file:

```text
vllm_hcu/model_executor/layers/fused_moe/moe_configs/
  E=44,N=1536,device_name=BW1100.json
  E=44,N=1536,device_name=BW1101.json   (byte-identical alias)
```

The plugin's `configure_tuned_moe_config_folder()` sets
`VLLM_TUNED_CONFIG_FOLDER` to the package folder **only when the user has not
set it**; a user value wins.

Selector details:

```text
KEY             E=44, N=1536, device_name, no dtype
OTHER SHAPES    fp8 MoE of the same E/N resolves to a different file name
                (`,dtype=fp8_w8a8`) and is unaffected
M RANGE         M=1..256 tuned; M >= 512 entries are copies of vLLM's default
                so nearest-key lookup cannot drag the decode tile into prefill
SPLIT_K         forced to 1 at the launch site; the table not carrying it is inert
ROLLBACK        point VLLM_TUNED_CONFIG_FOLDER at any folder without the file,
                or remove the feature with git revert
```

Mechanism (INFERENCE, consistent with the measured direction): halving the warps
per 16x64 CTA doubles resident CTAs, and `num_stages=1` drops the LDS
double-buffer that buys nothing when the K loop is memory-latency bound. The
measured direction is the evidence; the microarchitectural explanation is a
labelled inference.

Why this is a mechanism transplant, not a backend transplant:

- the kernel is the incumbent's existing, already-correct kernel;
- the dataflow, layout, routing, quant, and communication are unchanged;
- only the launch geometry and the selector entry are new;
- the change is narrow (one shape, one dtype, one device-name family);
- rollback is configuration.

## 7. Correctness

```text
BIT_EXACT_VS_DEFAULT = PASS
MAX_ABS              = 0.0
BW1100               = EXP-005, 3 sweeps, every M and routing seed
BW1101               = 24/24 measurements
MTP ACCEPTANCE       = unchanged
OTHER SHAPES         = still miss the selector and use vLLM's default
                       (verified: E=352,N=1536; E=44,N=1536,dtype=fp8_w8a8;
                        E=128,N=1024; E=48,N=512)
FALLBACK / CRASH     = none in the formal rounds
```

The tile changes only `BLOCK_SIZE_*`, `GROUP_SIZE_M`, `num_warps`, and
`num_stages`. `SPLIT_K` is forced to 1 at the launch site, so the table not
carrying it has no effect.

Selector proof per life (this is what rules out a silent no-op):

```text
A0 3/3 lives  Using default MoE config ... Config file not found at ...BW1101.json
C1 3/3 lives  Using configuration from .../E=44,N=1536,device_name=BW1101.json for MoE layer.
```

## 8. Local performance

Per `fused_experts` call, CUDA-graph timed, one HCU:

| M | default (µs) | C1 (µs) | BW1100 delta | BW1101 delta |
|---|---|---|---|---|
| 16 | 531.6 / 532.6 | 388.6 / 380.2 | -26.9 % | -28.6 % |
| 32 | 830.2 / 832.4 | 607.3 / 593.4 | -26.9 % | -28.7 % |
| 64 | 1063.2 / 1057.3 | 819.1 / 798.9 | -23.0 % | -24.4 % |
| 128 | 1529.2 / 1519.3 | 1003.7 / 1022.9 | -34.4 % | -32.7 % |

In round terms this is the often-quoted **≈24-33 %** kernel improvement; the
exact per-M deltas span **23.0-34.4 %**.

Server terms:

```text
draft MoE 2.359 -> 1.939 ms/step
draft MoE 2.593 -> 1.862 ms/step
         = -0.42 .. -0.73 ms/decode step (EXP-008)
```

The differential trace used for the stage budget shows `2.479 -> 1.904
ms/step`; the per-life server numbers vary slightly with routing and life.

## 9. E2E performance

### 9.1 BW1100 — EXP-007, interleaved clean restarts

| metric | A0 | C1 |
|---|---|---|
| decode (median, n=10) | 1437.12 tok/s | **1453.10 tok/s** |
| gain | — | **+1.11 % median, +1.28 % mean** |
| 95 % CI | — | [+0.60 %, +1.95 %] |
| permutation p | — | 0.0013 |
| ITL | 62.41 ms | 61.74 ms (-1.07 %) |
| prefill | — | ≈ unchanged |
| clean-restart cycles | — | **C1 > A0 in 5/5** (sign test p=0.031) |

Pooled A0 baseline over the night (n=18): median 1439.09 tok/s.

### 9.2 BW1101 — EXP-BW1101-E2E, independent double-check

Three paired cycles, six clean-restart lives, twelve formal rounds; same
workload/client as above (bs/concurrency 32, input 65536, output 4096, seed 42,
TP4 · DP2 · EP8 · MTP3, warmup excluded). The single variable per arm is whether
`VLLM_TUNED_CONFIG_FOLDER` points at the package's `moe_configs/` folder.

| pair | A0 (tok/s) | C1 (tok/s) | gain |
|---|---|---|---|
| 1 | 1411.36 | 1453.70 | +3.00 % |
| 2 | 1437.65 | 1461.35 | +1.65 % |
| 3 | 1434.78 | 1454.75 | +1.39 % |

| summary | value |
|---|---|
| A0 median / mean | 1434.78 / 1427.93 tok/s |
| C1 median / mean | 1454.75 / 1456.60 tok/s |
| **median gain** | **+1.39 %** |
| **paired wins** | **3/3** |
| ITL | 62.83 -> 61.58 ms (**-1.98 %**) |
| prefill | 13016.4 -> 13022.7 tok/s (**+0.05 %**, no regression) |
| MTP acceptance | 0.6012 -> 0.6012 (unchanged) |
| stability | 18/18 cases rc=0, 18/18 health 200, no crash / OOM / 5xx |

Reading the numbers honestly:

- The headline is the median **+1.39 %**, not the mean +2.01 %.
- Pair 1's +3.00 % is inflated because its A0 life was the slowest of the three
  (1411.36 tok/s, ITL 63.56 ms).
- Pairs 2 and 3, where A0 sits at its usual level, are +1.65 % and +1.39 %, in
  line with the historical +1.11 %.
- C1's median (1454.75) is within 0.1 % of the historical C1 median (1453.10),
  so the absolute level reproduces as well as the relative gain.
- BW1100 and BW1101 are two **device-name spellings of the same HCU part** in
  this deployment. This is one operating point, not two hardware validations.

## 10. Classification

```text
CONFIRMED_POSITIVE = YES
STRICT_NEW_BEST    = NO
```

Why `CONFIRMED_POSITIVE`:

- correctness: bit-exact, `max_abs = 0.0`;
- local: 23.0-34.4 % per call;
- stage: draft MoE 2.479 -> 1.904 ms/step in the differential trace;
- E2E: +1.11 % median (BW1100, 5/5) and +1.39 % median (BW1101, 3/3 paired);
- regression dimensions: prefill +0.05 %, MTP acceptance unchanged, other
  shapes still use the default;
- reproducible: selector proof per life, clean restarts, no fallback.

Why `STRICT_NEW_BEST = NO`:

- +1.39 % median is below the **project's conservative 1.99 % bar**.
- The 1.99 % bar is a project-level label for "strict new best", not a universal
  performance-value threshold. A ~1 %-class, bit-exact, reproducible gain is
  still a confirmed positive optimization.

What the claim is **not**:

```text
NOT "Marlin kernel +1.39 %"
NOT "Marlin was ported"
NOT a second hardware platform validation
```

The correct claim is:

```text
Marlin-inspired / Marlin-style fixed-shape specialization,
implemented on the existing HCU production path,
produced the C1 gain.
```

## 11. What generalized

1. **The mechanism, not the backend.** The transferable idea was an offline
   shape × M table selecting a pre-tuned launch configuration. The kernel,
   layout, EP semantics, and dataflow stayed with the incumbent.
2. **The default was wrong for the shape, not the backend.** The stock tile was
   sized for something else; the kernel was already correct.
3. **Small, narrow, reversible.** One shape, one dtype, one device-name family,
   one config pointer. Rollback is configuration.
4. **Compute the budget first.** A 23-34 % kernel gain became a ~1 % E2E gain
   because the draft MoE was ~2.4 ms of a ~62 ms step. That is a success because
   the claim was scoped to the budget.
5. **The same signature appears elsewhere.** A dense GEMM family running at
   22.4 µs/call with a vendor default solution is structurally the same
   "default is wrong for the shape" signature. It is a candidate for the same
   method, not a reason to port a backend.
6. **Negative results are part of the method.** The same investigation closed
   the Marlin backend, the vendor decode family, the ep_sort line, the cache
   hint, the expert-local M-grouping line, and the program-count rescue. Those
   closures are why the remaining candidate could be trusted.

## 12. Negative lessons

These are the failure classes a future mechanism-transplant evaluation must not
repeat.

### A. Config sweep failure — parameter search is not mechanism analysis

The Marlin/AITER config sweep did not beat A0. These arms measured the
configuration families on A0's own kernel at the Step5 decode point; they are
config-space results, not a full-Marlin E2E:

```text
vendor decode family BN16/BK512/ns3/nw4   +195.1 % on G1+G2  (+291 % G1)
BK=512 alone                              +3.07 %
BK=512 + ns=3                             OutOfResources (73728 > 65536 B)
ep_sort candidates                        best -2.1 %, all worse
cache modifier on B                       0.03 % (null)
```

The AITER tuned path also lost to A0 after it was made functional and
correctness-validated: prefill -55.55 %, decode -19.87 %, ITL +24.78 %,
TTFT +124.96 % relative to A0. Tuning recovered part of the fallback loss but
did not beat the incumbent.

```text
CONCLUSION: parameter search cannot substitute for architecture/mechanism
analysis. A tuner that searches one family cannot prove global optimality.
```

**Primary class:** `ALREADY_ABSORBED` for the transferable mechanisms;
`RESOURCE_LIMITED` for the LDS-infeasible family; `MEASURED_NULL` for the cache
hint; `REFUTED` for the remaining config direction.

### B. Support kernel C2 — LESS_WORK != LOWER_LATENCY

Two candidates were correct, safe, and slower.

Candidate A — masked-sum dependent-load overlap:

- The diagnosis was confirmed: 8 `scf.if` regions collapsed to 0.
- The implementation did exactly what it was designed to do.
- Result: **0.70 µs/call slower resident, 0.48 µs/call slower streamed.**
- The branch-guarded loads were not on the critical path. Predicating them
  added issue/address work without removing the latency the machine was waiting
  on.

Candidate B — activation-quant padding-row skip:

- Correctness: `PASS`, 85/85 layers; padding-row sentinel survives; downstream
  poison test 0 mismatches.
- 61.4 % of tiles were skipped.
- Result: **0.35 µs/call slower resident, 0.10 µs/call slower streamed.**
- The extra `sorted_ids` load, mask computation, and branch cost more than the
  skipped work.

Combined:

```text
streamed:  -0.049 ms/step
resident:  -0.090 ms/step
gate:      +0.30 ms/step
SUPPORT_KERNEL_C2_STATUS = CLOSED
```

**Primary class:** `MECHANISM_INVALID` for A (critical-path payoff model);
`IMPLEMENTATION_FORM_FAILED` for B (overhead exceeds saving). Both share the
headline: a semantically real work reduction is not automatically a latency
reduction.

### C. D0 M-grouping — the fake win and the corrected verdict

The old result appeared to be a major win:

```text
GEMM2 component   -40 %
total             -14.3 %
projected E2E     ~5 %
```

It was a defect in the experiment's own kernel copy. On the non-`USE_MLS_LOAD`
path (the one GEMM2 uses), the "recompute the B address each K iteration"
variant **omitted the `k * BLOCK_SIZE_K * stride_bk` term**. Every K step
re-loaded the same k=0 B tile, which stays L1/L2 resident. The kernel streamed
**1/6 of GEMM2's weight bytes** (K=1536, BLOCK_K=256).

What caught it: a bit-exactness control against the **installed, untouched**
kernel, per arm, on identical inputs, after zeroing the compared buffer between
arms.

Corrected outcome with the term restored:

```text
every arm bit-exact
pointer style (loop-carried vs recomputed)      0.0 %  (all variants within 0.1 %)
standalone rewritten GEMM2 kernel vs installed  -0.02 % (identical)
residual +7.6 % total vs installed              GEMM1-side structural overhead
```

```text
RESULT.md (old)                         status: SUPERSEDED
"B reuse is measurably valid in GEMM2"  status: REFUTED (it was the defect)
CORRECTION.md                           status: current record
```

The grouping mechanism itself also lost:

```text
GROUP_PENALTY(P0) = +170.3 us (+65.2 %)   G1 +121.0   G2 +49.3
GROUP_PENALTY(P1) = +162.3 us (+61.9 %)   0.75x programs
GROUP_PENALTY(P2) = +150.0 us (+57.1 %)   0.50x programs
noise 0.17 %   every arm bit-exact
best grouped arm still +58 % vs incumbent
registers 61 -> 99 -> 143 (G1), 105 -> 167 -> 247 (G2)
```

**Primary class:** `MEASUREMENT_ARTIFACT` for the -40 %/-14.3 % claim (address
defect); `IMPLEMENTATION_FORM_FAILED` for the grouping (register pressure and
serialized MLP). The semantic reuse was valid; the schedule was not.

### D. Program-count rescue — scheduler tuning cannot rescue a bad decomposition

Lowering the program count shrank the grouped-kernel penalty only slightly:

```text
65.2 % -> 61.9 % -> 57.1 %   as programs went 1.0x -> 0.75x -> 0.50x
P alone on the baseline:     P1G1 +0.34 %, P2G1 +0.62 % (neutral-to-worse)
```

The residual cost was **inside the leader**: one program walked several
m-tiles' K loops serially while holding B, which serializes memory-level
parallelism. A dynamic work queue addresses the assignment, so it cannot touch
this.

```text
PROGRAM_COUNT_RESCUE       = CLOSED
STRAGGLER_HYPOTHESIS       = PARTIALLY_SUPPORTED
dynamic work-queue follow-up = DO NOT ENTER
```

**Primary class:** `MECHANISM_INVALID` / `SCHEDULING_CANNOT_RESCUE`, with
`IMPLEMENTATION_FORM_FAILED` as the secondary form cost.

## 13. Do Not Reopen

Unless a **material condition** changes, do not re-open:

| line | verdict | reason |
|---|---|---|
| whole Marlin backend | REJECTED | non-EP dataflow; missing tables; incumbent already has the mechanisms |
| vendor decode config family | CLOSED | +195 % on G1+G2; LDS-infeasible in the BN=32 form |
| ep_sort specialization | CLOSED | launch floor 46.4 %; every candidate worse |
| cache modifier on MoE GEMM | MEASURED_NULL | 0.03 %, inside noise |
| expert-local M-grouping | CLOSED | best arm +58 % vs incumbent, bit-exact |
| program-count rescue | CLOSED | shrinks the penalty only ~8 pp for 2x P |
| dynamic work queue on this mechanism | CLOSED | residual is intra-leader MLP, not assignment |
| `B_REUSE_MECHANISM_VALID` | REFUTED | G029 address defect |
| `{32,64,128}` as token counts | REFUTED | they are tile parameters, not tokens |

A material change means: a different kernel, a different operating point, a
different hardware/dataflow constraint, a non-register reuse carrier, or a
vendor artifact that was previously unavailable.

## 14. Evidence Paths

Project documents:

```text
vllm-plugin-das/docs/AIOPT-1269_C1_DELIVERY.md
experiments/AIOPT-1269/C1_STACKING_SEARCH.md
experiments/AIOPT-1269/VALIDATED_C1_GIT_DELIVERY.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/00_EXECUTIVE_SUMMARY.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/02_MARLIN_WHY_FAST.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/03_A0_WHY_FAST.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/05_KERNEL_MECHANISM_DELTA.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/07_CANDIDATE_RANKING.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/10_BLOCK_M_SEMANTICS_CORRECTION.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/MARLIN_AITER_MECHANISM_INVENTORY.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/MARLIN_AITER_TO_A0_MECHANISM_MATRIX.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/STEP5_SPECIALIZATION_TOP5.md
experiments/AIOPT-1269/marlin_step5_arch_analysis/FINAL_MARLIN_STYLE_CLOSURE.md
experiments/AIOPT-1269/marlin_specialization_static/01_SLIMQUANT_MARLIN_CALLCHAIN.md
experiments/AIOPT-1269/marlin_specialization_static/04_CONFIG_AND_TUNER_COMPARE.md
experiments/AIOPT-1269/marlin_specialization_static/07_MARLIN_SPECIALIZATION_GO_NOGO.md
experiments/AIOPT-1269/prep-final/MARLIN_CONFIG_FALLBACK_TRACE.md
experiments/AIOPT-1269/prep-final/TRUE_MARLIN_ENABLEMENT.md
experiments/AIOPT-1269/MOJIN_NIGHT_20261016/04_PROGRAM_COUNT_GPU.md
experiments/AIOPT-1269/MOJIN_NIGHT_20261016/07_FINAL_NIGHT_REPORT.md
```

Host evidence directories (not all committed):

```text
experiments/AIOPT-1269/marlin-step5-gpu/results/EXP-005/         tuning sweep
experiments/AIOPT-1269/marlin-step5-gpu/results/EXP-007/         BW1100 E2E
experiments/AIOPT-1269/marlin-step5-gpu/results/EXP-008/         kernel accounting
experiments/AIOPT-1269/marlin-step5-gpu/results/EXP-BW1101/      selector + local smoke
experiments/AIOPT-1269/marlin-step5-gpu/results/EXP-BW1101-E2E/  BW1101 E2E
experiments/AIOPT-1269/marlin-step5-gpu/results/EXP-D0-MGROUP/   old RESULT.md + CORRECTION.md
support_kernel_c2/prep106/05_GPU_VALIDATION_REPORT.md            support C2 closure
```

Shipped config SHA256 (both alias files):
`3000f5aac56558e170d3e10abb6716628a1a7636f82a713ae57f062f04a5f588`.

## 15. Reusable Checklist From This Case

- [ ] Prove the requested backend actually executed (L0).
- [ ] If it missed, classify the fallback before calling it backend performance.
- [ ] Fix the fallback as a separate correctness/selection change.
- [ ] Build the mechanism inventory; mark absorbed mechanisms and remove them.
- [ ] Check whether the source's artifact is reachable at your M.
- [ ] Reconstruct the real shape and the selector key.
- [ ] Find a stable, frequent, generic, correct, narrowly selectable target.
- [ ] Compute `ms/step` and E2E share before implementing.
- [ ] Change launch geometry or selector, not the kernel, when possible.
- [ ] Prove bit-exactness and selector hits per life.
- [ ] Preserve the default entries for other M buckets.
- [ ] Validate with clean-restart paired E2E and regression dimensions.
- [ ] Separate `CONFIRMED_POSITIVE` from `STRICT_NEW_BEST`.
- [ ] Record the negative results and the do-not-reopen conditions.
