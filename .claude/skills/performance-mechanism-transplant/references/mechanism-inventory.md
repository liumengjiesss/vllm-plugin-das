# Mechanism Inventory

The inventory is the core artifact of a mechanism-transplant evaluation. It
converts a vague comparison ("backend A is faster than backend B") into a
per-mechanism gap list with status, evidence, and budget.

## How to Use

For every mechanism in the master table, fill one row of the worksheet:

```text
MECHANISM=
SOURCE_HAS=
CURRENT_PRODUCTION_HAS=
ALREADY_ABSORBED=
REMAINING_GAP=
TRANSFERABLE=
EXPECTED_BUDGET=
EVIDENCE_CLASS=
```

Then delete every row whose status is `ABSORBED`, `NOT_APPLICABLE`,
`UNAVAILABLE_BY_VENDOR_DATA`, or `NO_BUDGET`. What remains is the candidate set.

**Core rule: A mechanism is not a candidate if the incumbent already implements
its useful form.** "Useful form" includes algebraically equivalent
implementations under different names, and a finer-grained version of the same
selector.

## Status Vocabulary

```text
ABSORBED                  The incumbent implements the useful form.
PARTIALLY_ABSORBED        Some axes are covered; a specific axis is not.
NOT_ABSORBED              The mechanism is absent and reachable.
NOT_APPLICABLE            A structural precondition is false (shape, dtype,
                          topology, graph capture, arch).
UNAVAILABLE_BY_VENDOR_DATA The mechanism needs a table/artifact that does not
                          exist for this shape or device.
NO_BUDGET                 The mechanism is excluded by correctness, project
                          policy, or a prior measured sweep.
MEASURED_NULL             Implemented and measured; gain inside noise.
REFUTED                   Measured and worse, or a prior claim was invalidated.
UNKNOWN                   Not enough evidence to classify.
```

Do not collapse `NOT_APPLICABLE` into `NOT_ABSORBED`. The first says the
mechanism cannot apply here; the second says it could and does not.

## Master Table

| Mechanism | Why it can help | How to detect current support | Transfer cost | Typical failure mode |
|---|---|---|---|---|
| **Fixed-shape tile config** | A tile tuned for the actual shape reduces padding, CTA count, or pipeline stalls. | Find the config lookup and the default fallback. Search for `get_config`, `nearest`, `min(... abs(x-M))`, "Using default" warnings, and per-shape JSON files. | Low: a table entry, if the kernel already reads it. | Tuning a shape that does not occur, or letting a decode tile leak into prefill through nearest-M. |
| **Persistent kernel / program count** | One launch of persistent CTAs removes per-tile launch overhead and keeps the pipeline warm. | Search `persistent`, `PERSISTENT_PROGRAMS`, `NUM_SMS`, `tile_id +=`, device-side tile count. | Low if the kernel exists; medium if only a non-persistent kernel exists. | Program count tuned for a tile count that barely exceeds it; little amortization left. |
| **Expert-aware / EP-aware layout** | Work and bytes become proportional to local routes, not all routes. | Search `expert_map`, `local_num_experts`, `ep_size`, sort/align kernels, local-vs-global bound formulas. | Medium: usually changes the layout contract and downstream row bounds. | Porting a non-EP dataflow (global align, unmasked reduce) into an EP path adds work. |
| **Weight preshuffle / packed weights** | A load-time permutation feeds a wide-vector kernel without in-kernel transposes. | Search `process_weights_after_loading`, `shuffle_weight`, `pack`, `repack`, layout reshape/transpose. Check whether the kernel consumes the layout. | High: one-time cost, extra memory, and layout lock-in. | The incumbent's kernel cannot read the packed layout; duplicating weights is infeasible. |
| **Fused activation** | Removes an intermediate tensor and a launch; applies activation at the point of production. | Find kernels between GEMM1 and GEMM2; search `silu`, `swiglu`, `gelu`, `clamp`, `activation`. | Low to medium if the kernel can be patched; high if it changes the math path. | The incumbent already fuses activation, often more than the source's GEMM family does. |
| **Fused quant / dequant** | Avoids materializing a widened tensor or a separate scale pass. | Search per-token/per-channel scale kernels, block-scale loops, `dequant`, `quant`, `fp8`, `int4`, `mxfp4`. | Medium; dtype/quant-scheme-dependent. | The source fusion is gated on a quant form the target does not use (e.g. block scale vs per-channel, fp4 vs fp8). |
| **Fused combine / reduce** | Removes a pass over the output and a launch; can fold shared-expert or routing weights. | Search `moe_sum`, `reduce`, `allreduce`, `combine`, masked reduce, shared-expert add. | Medium; requires matching the routing/layout contract. | The target is latency-bound and serialized per layer, so a "bigger fused reduce" is structurally impossible. |
| **Task map / work distribution** | A better mapping of tiles to programs reduces tail imbalance and improves locality. | Read the pid mapping, `GROUP_SIZE_M`, tile stride, and whether the assignment is static or device-side. | Low to medium if config-only; medium if the kernel must change. | The map is already balanced; the residual is inside the leader (serialized MLP / register pressure). |
| **Dynamic work queue** | Pull-based assignment adapts to skewed task lengths at runtime. | Search for a work counter, atomic claim, or `while` loop over a queue; check whether the current kernel is persistent-static. | Medium to high: usually a kernel rewrite plus synchronization. | The residual is not assignment; a work queue cannot fix intra-program serialization or register pressure. |
| **Launch coalescing** | Fewer launches reduce launch latency and graph overhead. | Count kernels and launches per step; look for tiny kernels and repeated metadata launches. | Low to medium. | Launch count is not the binding cost, or fusion breaks graph capture. |
| **Intermediate-buffer elimination** | Removes zero-fill, permute/unpermute, and copy traffic. | Search `torch.zeros`, `fill_`, `permute`, `unpermute`, workspace allocation, copy kernels. | Medium; changes ownership and masking rules. | Padding rows hold garbage and a downstream consumer reads them without a mask. |
| **Cache policy** | `.cg`/`.cs` hints, L2 swizzle, and matrix loads change reuse and pollution. | Search `cache_modifier`, `matrix_load`, `GROUP_SIZE_M`, `swizzle`, `tl.assume`. | Low (one line) but sign is ambiguous. | The hint is applied to a reused operand and hurts; or it is null because the cache was not the limiter. |
| **Communication overlap** | Hides allreduce/allgather behind compute or removes a serialized collective. | Trace per-layer collectives; check call count, message size, fused GEMM+AR, async paths. | High; topology- and dtype-dependent. | The collective is one-per-layer on the critical path and latency-bound; fusion is impossible. |
| **Specialized config selector** | Routes the real shape to the right config without runtime autotune. | Find the selector key: `(E,N,K,topk,dtype,arch,device)` vs a subset; nearest-M vs exact; fallback policy. | Low. | The key is missing a dimension (e.g. K), so it selects a config tuned for another shape. |
| **Kernel-family / shape × M table** | Chooses among compile-time tile families or kernel variants per operating point. | Search `MODE`, `sol_id`, family flags, compiled-vs-asm branches, per-M tables. | Medium to high: needs the variants to exist and be validated. | The variant is not selected at your M; or the selector is opaque and its tile semantics are inferred, not proven. |
| **Small-M scheduling** | Shrinks the grid/tile to the actual row count instead of the asymptotic M. | Search `M <= E`, `M < BLOCK_SIZE_M`, small-M ladders, grid shrink, tile ladder. | Low if config-only; high if the persistent grid is captured. | Every rule keys on total M, while the target's small quantity is per-expert M_e; the rule does not fire. |
| **Padding / alignment management** | Bounds the padded layout so a static buffer works without host sync. | Find the alignment block, `nvalid`, analytic bound, padding ratio, and whether the bound is consumed device-side. | Low to medium. | The bound is loose (e.g. 1.83x), so it is diagnostic but not a predictor; or the padding rows are charged full work. |

## Detection Recipes

### Fixed-shape tile config

```text
1. Find the lookup function and its key.
2. Find the default branch and the warning it logs.
3. Find where the selected config reaches the kernel launch.
4. Prove the exact entry selected at runtime (log, probe, hash).
5. Record the M selection rule: exact, nearest, smallest >= M.
6. Check the prefill/large-M coverage separately from decode.
```

### Persistent kernel / program count

```text
1. Find the persistent kernel and how its grid is sized.
2. Check whether the grid is a host constant or a device-side valid count.
3. Compute tiles / programs. If it is ~1, the persistent loop barely amortizes.
4. Check whether the program count was tuned jointly with the tile.
5. Do not re-sweep a program count already measured at this operating point.
```

### Expert-aware / EP-aware layout

```text
1. Find the sort/align kernel and the expert space it uses (local vs global).
2. Find the bound formula for the sorted layout.
3. Check whether non-local routes are laid out or dropped.
4. Check the reduce: masked on local experts, or unmasked over all routes?
5. Check whether the layout carries garbage padding and who masks it.
```

### Weight preshuffle / packed weights

```text
1. Find every load-time weight transform.
2. Prove which kernel consumes the transformed layout.
3. Check whether the incumbent's kernel can read it by stride arithmetic.
4. Check memory cost of keeping both layouts.
5. If the layout is exclusive, treat it as a backend-level decision, not a
   mechanism-level one.
```

### Fused activation / quant / combine

```text
1. Enumerate the kernels between the two GEMMs and after the second GEMM.
2. For each, list what it reads, computes, and writes.
3. Check whether the incumbent already fuses more than the source.
4. Compute the intermediate bytes removed; compare with the kernel's cost.
5. Check dtype/quant preconditions (block size, zero-point, per-channel).
```

### Task map / dynamic work queue

```text
1. Record the mapping: tile id -> program, static or dynamic.
2. Record tasks/program under the real routing, not uniform synthetic.
3. Record register pressure and occupancy before/after any grouping.
4. Compute the ceiling: perfect balance x best-case traffic reduction.
5. Compare with the measured structural overhead of the rewritten kernel.
```

### Cache policy

```text
1. Find every load/store site for the relevant operands.
2. Check whether a modifier already exists.
3. Determine reuse per operand per iteration: a hint on a reused operand can hurt.
4. Measure same-session, total-cost verdict, not one component.
5. Classify a null result as MEASURED_NULL, not as "untested".
```

### Communication overlap

```text
1. Count collectives per step and their position on the critical path.
2. Measure per-call latency and message size.
3. Check whether the source fusion requires a dtype or topology you do not have.
4. Check whether the calls are serialized one-per-layer; if so, batching or a
   single fused reduce is structurally impossible.
```

### Config selector

```text
1. Write the exact key: shape, M, dtype, arch, device, version.
2. Check which dimensions are omitted.
3. Check the M rule and the coverage map for decode and prefill.
4. Check the fallback: does a miss return the incumbent's best path, or force a
   weaker generic backend?
5. Verify the selected entry at runtime.
```

## Worksheet

Copy this block per mechanism:

```markdown
### <mechanism>

MECHANISM=
SOURCE_HAS=
CURRENT_PRODUCTION_HAS=
ALREADY_ABSORBED=
REMAINING_GAP=
TRANSFERABLE=
EXPECTED_BUDGET=
STATUS=
EVIDENCE_CLASS=
EVIDENCE=
TRANSFER_COST=
FAILURE_MODE=
```

## Artifact vs Mechanism

| Artifact | Mechanism it may carry | Transfer question |
|---|---|---|
| hand-written asm `.co` / `.so` | a specific tile and pipeline | Is the artifact reachable at your shape? Can a compiler-generated kernel match it? |
| packed weight layout | vectorized wide-N loads | Can the incumbent's kernel consume it? Is the layout exclusive? |
| opaque `MODE` / `sol_id` | compile-time tile-family selection | What does the selector actually select? Is it reached at your M? |
| vendor tuning CSV/JSON | shape × M specialization | Does a row exist for your `(E,N,K,topk,dtype,arch)`? |
| config name / env flag | dispatch policy | Is it consumed by the installed build? Prove the runtime path. |

A mechanism can be transplantable even when the artifact is not. The reverse is
also true: an artifact can be present but not selected, so it contributes
nothing.

## Worked Example — Step5 A0 Inventory (condensed)

The full evidence is in
`case-studies/step5-marlin-style-c1.md` and the AIOPT-1269 evidence pack. This
is the shape of the answer, not the only possible one.

| Mechanism | Source (Marlin/AITER) | A0 production path | Status |
|---|---|---|---|
| shape × M config table | 15-16 M points per shape, `MODE` selects a tile | `ep_table.json`, 21 M buckets, richer key with K/topk | ABSORBED (A0 finer) |
| small-M tile | `BM=16` at total M ≤ 256 | `BM=16` at decode | ABSORBED |
| persistent execution | vendor ASM/workgroup tables | `PERSISTENT_PROGRAMS` 2048/1024 at M=128 | ABSORBED and exhausted |
| EP-aware layout | absent in the W8A8-FP8 runner | local-expert-space sort, non-local routes never laid out | ABSORBED (A0 ahead) |
| fused act + quant | separate kernel in the source runner | `_ep_swiglustep_quant_kernel` also folds the router weight | ABSORBED (A0 ahead) |
| masked reduce + shared expert | unmasked reduce in the source runner | masked reduce with shared expert folded in | ABSORBED (A0 ahead) |
| zero-fill / permute elimination | `torch.empty` caches, global align | zero-fill-free workspace, no permute/unpermute | ABSORBED (A0 ahead) |
| MLS / matrix load | gfx938-only | G1 `USE_MLS_LOAD=true` | ABSORBED |
| L2 swizzle | `GROUP_SIZE_M` | `GROUP_SIZE_M=8` | ABSORBED |
| cache modifiers on MoE GEMM | vendor dense decode uses `.cg` | absent in boltops MoE | NOT_ABSORBED, MEASURED_NULL (0.03 %) |
| wide-N asm tile family | `{32,64,128}x256x128` | A0 16x32x256 / 16x64x256 | NOT_TRANSFERABLE: not selected at M=128; LDS-infeasible in the BN=32 form |
| vendor ASM/CK tier | `asm > moe_c > triton > ck` | Triton | UNAVAILABLE_BY_VENDOR_DATA: no E=44 row |
| per-expert BM rule | `M*topk/E/BM < 0.9` | table keyed on total M | NOT_ABSORBED but REFUTED: selects BM=8, predicted +0.29 ms/step |
| split-K | six implementations, parallelism predicate | off | NOT_INDICATED: 2208 real tiles, weight-bound |
| fused GEMM + allreduce | quantized-only, per-tile handshake | separate bf16 collective | NOT_APPLICABLE: dtype + latency-bound + serialized |
| padding/alignment bound | analytic worst-case bound | device-side `nvalid` | ABSORBED-equivalent (same graph-safe property) |

The important output is not the row count. It is the conclusion: after deleting
absorbed, not-applicable, unavailable, and no-budget rows, only one reachable
cell remained, and it measured null. The correct decision was to stop the
backend-transplant line and search elsewhere for a fixed-shape gap.

## Inventory Quality Checks

Before accepting an inventory:

- [ ] Every `ABSORBED` row cites the incumbent's implementation, not a guess.
- [ ] Every `NOT_ABSORBED` row names the reachable code path.
- [ ] Every `UNAVAILABLE_BY_VENDOR_DATA` row names the missing key/row.
- [ ] Every `NOT_APPLICABLE` row names the false precondition.
- [ ] Every budget is in `ms/step`, not percent alone.
- [ ] The runtime path was verified, not inferred from the config name.
- [ ] At least one mechanism was checked against a measured null/refutation.
- [ ] The inventory distinguishes artefacts from mechanisms.
- [ ] The inventory records what would reopen a closed row.
