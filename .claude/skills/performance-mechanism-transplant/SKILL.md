---
name: performance-mechanism-transplant
description: >
  Use when a high-performance kernel, backend, library, or upstream PR (Marlin,
  AITER, DeepGEMM, CUTLASS, CK, Triton specialized kernels, FlashAttention,
  vendor kernels, upstream performance PRs) is fast elsewhere and the task is to
  determine which mechanisms can be transplanted into the current production
  path. Also use for fixed-shape specialization, production-path gap mining,
  backend-gap analysis, and minimal low-risk performance PoCs.
---

# Performance Mechanism Transplant

## Thesis

**MECHANISM TRANSPLANT, NOT BACKEND TRANSPLANT.**

The operative question is not:

> "Can we move Marlin / AITER / DeepGEMM / CUTLASS / CK into this stack?"

It is:

> "Why is that implementation fast, and which of its mechanisms is the current
> production path still missing?"

A backend is a bundle of mechanisms wrapped in a dataflow, a layout contract, a
dispatch policy, and a set of artefacts. Porting the bundle imports all of the
coupling at once. Transplanting one mechanism keeps the incumbent's proven
dataflow and adds only what is measurably missing.

The canonical result of getting this distinction wrong is AIOPT-1269 Step5:

- The request was "make `slimquant_marlin` fast".
- Marlin never executed at all; the measured regression was a **forced Triton
  fallback**.
- Tuning the Marlin kernel could not have fixed it; the missing thing was one
  config row, and the deeper answer was that the incumbent had already absorbed
  the transferable mechanisms.
- The eventual confirmed positive was **not** a Marlin port. It was a
  Marlin-style, fixed-shape **launch-config specialization** applied to an
  existing correct kernel on the existing HCU production path: +1.11 % median
  E2E (BW1100) and +1.39 % median E2E (BW1101, 3/3 paired wins), bit-exact.

Read `references/case-studies/step5-marlin-style-c1.md` before generalizing from
that case.

---

## Mission

For any external high-performance implementation, answer these questions in
order, and stop at the first one that closes the line:

1. Why is it fast? Decompose into mechanisms, not brand names.
2. Which mechanisms does the current production path already implement?
3. Which mechanisms are genuinely missing, and at what operating point?
4. Is the right move a whole-backend transplant, a mechanism transplant, or no
   action?
5. Where are the real fixed-shape specialization opportunities?
6. What is the smallest PoC that can falsify the hypothesis?
7. How do we get from local gain to aggregate and E2E evidence?
8. How do we avoid repeating a historically closed direction?

The skill produces a **decision**, not an advocacy document. "Do not transplant"
is a successful outcome when the mechanism inventory says so.

---

## When to Use

Use this skill when:

- A vendor kernel or backend is reported fast, and the task is to evaluate it.
- An upstream PR claims a speedup and the task is to see whether it applies.
- A production path is suspected of leaving performance on the table.
- A workload has stable, high-frequency shapes and a generic config.
- A previous optimization attempt failed and the failure class is unclear.
- Someone proposes "just port X" or "just sweep more configs".

Do not use this skill as a replacement for measurement methodology. This skill
decides **what to optimize and what to transplant**. The
`senior-performance-optimization` skill decides **how to measure, verify, and
account**. They are complementary; load both for a real optimization task.

---

## Hard Rules

1. **Runtime path before backend name.** A configuration name is not proof of
   execution. Prove the kernel that actually ran (`L0`).
2. **Mechanism inventory before porting.** A backend port without an inventory
   is a rewrite with extra steps.
3. **An absorbed mechanism is not a candidate.** If the incumbent already
   implements the useful form, re-implementing it under a new name adds risk,
   not speed.
4. **Real operating point before tuning.** Never tune a shape that does not
   occur. Never confuse lookup M with per-expert M.
5. **Budget in `ms/step`, not kernel percent.** A 30 % kernel win on 2 % of the
   step is worth less than a 3 % win on 35 %.
6. **Component gain is not pipeline gain.** Report `G1+G2` or the real stage
   total, never one GEMM alone.
7. **Microbench gain is not E2E gain.** Promotion requires the validation
   ladder, ending in clean-restart paired E2E.
8. **Correctness before timing.** A rewritten kernel or a new selector config
   must pass an independent numerical gate first.
9. **Preserve negative knowledge.** A closed line stays closed until a material
   condition changes: a different kernel, a different operating point, or a
   different hardware/dataflow constraint.
10. **Classify failures, do not binary them.** Use the failure classes in
    `references/failure-classification.md`; "SUCCESS / FAILED" loses the
    information the next engineer needs.

---

## The Algorithm

### STEP 1 — Freeze the real production baseline

Record, with evidence:

```text
RUNTIME_PATH=
HARDWARE=
DTYPE_AND_QUANT_SCHEME=
SHAPE_DOMAIN=
TOPOLOGY=              TP / DP / EP / PP / DCP
WORKLOAD=
METRIC=                decode TPS / ITL / TTFT / prefill TPS / kernel ms
BASELINE_ARTIFACT=     log / trace / profile / benchmark JSON
MEASUREMENT_PROTOCOL=
```

Do not proceed until `EXPECTED_PATH` and `ACTUAL_PATH` are reconciled. If the
"optimized" option silently fell back, the baseline is not the baseline you
think it is. See `references/anti-patterns.md`, `ANTI_PATTERN_09`.

### STEP 2 — Explain WHY the external implementation is fast

Decompose it. Do not accept "it is a better kernel". Use at least these axes:

- tile specialization (`BM/BN/BK`, warps, stages, `GROUP_M`)
- persistent scheduling and program count
- weight packing / preshuffle / load-time layout transform
- layout specialization (sorted, blocked, local-expert-space, cyclic-N)
- task mapping / work distribution / work queue
- fusion (activation, quant, combine, epilogue, producer-consumer)
- launch reduction / launch coalescing
- padding reduction / alignment management
- memory reuse (L2 swizzle, cache hints, buffer elimination)
- config-table specialization (shape × M keyed)
- communication overlap or fused collective
- arch/capability gating and kernel-family selection

For each mechanism, record:

```text
MECHANISM=
WHY_IT_CAN_HELP=       which physical limit it relaxes
EVIDENCE_CLASS=        FACT / INFERENCE / UNKNOWN
ARTEFACT_OR_IDEA=      binary/layout/table is an artefact; tile choice is an idea
REQUIRED_OPERATING_POINT=
```

Two traps:

- **Artifact vs mechanism.** A hand-written `.co`/`.so` is an artifact. "A
  shape-keyed table selects a compile-time tile family" is the mechanism. The
  artifact may not be reachable; the mechanism may still be transplantable.
- **Name vs thing.** One brand can cover several distinct artifacts. In the
  Step5 case, "Marlin" meant upstream weight-only Marlin, vendor LightOp
  W8A8-FP8 Marlin, and the tuning tables, with different mechanisms and
  different reachability.

### STEP 3 — Build the mechanism inventory

Use the worksheet in `references/mechanism-inventory.md`:

```text
MECHANISM=
SOURCE_HAS=
CURRENT_PRODUCTION_HAS=
ALREADY_ABSORBED=
REMAINING_GAP=
TRANSFERABLE=
EXPECTED_BUDGET=
```

Status vocabulary (do not collapse):

```text
ABSORBED
PARTIALLY_ABSORBED
NOT_ABSORBED
NOT_APPLICABLE
UNAVAILABLE_BY_VENDOR_DATA
NO_BUDGET
MEASURED_NULL
REFUTED
UNKNOWN
```

### STEP 4 — Remove already-absorbed mechanisms

Inspect the incumbent's source and runtime, not its marketing. Look for:

- an existing shape × M config table and selector
- persistent execution / program count
- EP-aware or expert-aware layout
- fused activation / quant / combine
- sorted-row bridges, `top_k=1` indexing, MLS/matrix loads
- L2 swizzle / `GROUP_SIZE_M`
- padding bounds and device-side valid counts
- launch coalescing and graph capture

**A mechanism is not a candidate if the incumbent already implements its useful
form.** "Implemented differently but algebraically equivalent" also counts as
absorbed.

### STEP 5 — Reconstruct the real operating points

Measure, do not infer:

```text
M domains:       M_loc / M_ep / M_lookup / per-expert M_e
N, K
E_global, E_local
topk
dtype / quant scheme
calls per step
time share
config key used
```

A single decode step can have several M definitions. Identify which one the
selector consumes. The most expensive mistakes in the Step5 case came from
treating `{32,64,128}` as token counts when they were tile parameters, and from
tuning aggregate M while the interesting distribution was per-expert `M_e`.

### STEP 6 — Find the smallest uncovered specialization gap

Prefer a target that is:

- high frequency (many calls per step)
- fixed or narrow shape
- currently on a generic/default config
- served by an already-correct kernel
- fixable by config, selector, layout, or launch policy
- small to implement
- easy to roll back

Score candidates:

```text
BUDGET_MS_STEP = calls_per_step * (local_ms_before - local_ms_after)
SHARE_E2E      = BUDGET_MS_STEP / E2E_MS
IMPLEMENTATION_COST
CORRECTNESS_RISK
FALLBACK_RISK
```

`references/fixed-shape-specialization.md` gives the full target definition and
selection procedure.

### STEP 7 — Estimate the recoverable budget

Do not stop at "kernel x % faster". Compute:

```text
ms/call before, ms/call after
calls/step
ms/step saved
share of E2E
overhead floor (extra launch, sort, copy, sync)
NET_MS_STEP
```

If the gross benefit is below the overhead floor, close the line before the
PoC. This is how the hot/cold expert split was closed: gross 0.31-0.32 ms/step
against a 1.05-1.87 ms/step overhead floor.

### STEP 8 — Build the minimal PoC

Preference order, cheapest and least invasive first:

1. config entry
2. selector / dispatch key
3. layout or alignment policy
4. launch policy (grid, program count, graph)
5. small epilogue/prologue fusion
6. kernel rewrite

Only escalate when the cheaper form cannot test the mechanism. Use an overlay,
env-redirected table, or experiment copy; do not fork production code to test a
one-line idea.

### STEP 9 — Correctness first

Gate the PoC before reading timing:

```text
BIT_EXACT_VS_INSTALLED=   PASS / FAIL
MAX_ABS=
SHAPE / DTYPE=
INDEX / ADDRESS CHECK=
BUFFER OWNERSHIP=
VALID REGION COMPARED=
ZEROED OR POISONED BETWEEN ARMS=
NaN / Inf CHECK=
```

Rules from the Step5 D0 failure:

- A rewritten kernel must be compared against the **installed, untouched**
  kernel on identical inputs.
- Zero the compared buffer before each arm. Otherwise a kernel that skips stores
  can inherit the previous arm's values and look bit-exact.
- Compare the valid region, not the padded tail.
- Diff a recomputed address term-by-term against the pointer advance it
  replaces. A missing `k * BLOCK_SIZE_K * stride_bk` read the same k=0 tile and
  produced a fake -14.3 % "win".
- If a component win exceeds the mechanism's entire theoretical prize, assume a
  defect until an independent check clears it.

### STEP 10 — Climb the validation ladder

```text
L0  runtime-path proof
L1  real-shape proof
L2  correctness
L3  microbenchmark
L4  aggregate-stage timing
L5  same-session interleaved A/B
L6  clean-restart E2E paired test
L7  regression dimensions
```

Claim rules:

- L3 only → `LOCAL_POSITIVE`. Not a production win.
- L4 → `STAGE_POSITIVE`.
- L5 → `A_B_POSITIVE_PRELIMINARY`.
- L6 + L7 → `CONFIRMED_POSITIVE`.
- L6 without L7 → `E2E_CONFIRMED_REGRESSION_INCOMPLETE`.
- Local positive but E2E null → `LOCAL_POSITIVE_E2E_NULL`; do not argue it into a
  win.

See `references/validation-ladder.md` for the evidence required at each level.

---

## Decision Record Template

Use this exact block for every mechanism-transplant evaluation. It is designed
to be readable without the surrounding chat.

```text
SOURCE_IMPLEMENTATION=

WHY_FAST=

REAL_OPERATING_POINT=

CURRENT_INCUMBENT_PATH=

MECHANISM_INVENTORY=

ALREADY_ABSORBED=

REMAINING_GAPS=

TOP_CANDIDATE=

EXPECTED_LOCAL_GAIN=

EXPECTED_MS_STEP=

MINIMAL_POC=

CORRECTNESS_GATE=

AGGREGATE_GATE=

E2E_GATE=

FAILURE_CLASS=

NEXT_STEP=
```

Optional fields, when they change the decision:

```text
EVIDENCE_CLASS=
ARTEFACT_OR_IDEA=
TRANSFER_COST=
ROLLBACK=
WHAT_WOULD_FALSIFY=
DO_NOT_REOPEN_UNLESS=
```

### Interpreting `FAILURE_CLASS`

Use the taxonomy in `references/failure-classification.md`:

```text
MECHANISM_INVALID
IMPLEMENTATION_FORM_FAILED
ALREADY_ABSORBED
RESOURCE_LIMITED
MEASUREMENT_ARTIFACT
CORRECTNESS_FAILURE
LOCAL_POSITIVE_E2E_NULL
CONFIRMED_POSITIVE
```

Do not use `SUCCESS / FAILED` as the final label. It loses the distinction
between "the idea is wrong", "this form of the idea is wrong", "the incumbent
already has it", "the hardware resource does not fit", "the measurement lied",
and "the gain did not survive E2E".

---

## Stop Rules

Stop the line, with a recorded class, when:

- every mechanism is `ABSORBED`, `NOT_APPLICABLE`, or `UNAVAILABLE_BY_VENDOR_DATA`
- `GROSS_BENEFIT < OVERHEAD_FLOOR` before implementation
- the local gain is below the pre-registered gate
- the local gain does not translate to the pipeline and no explanation exists
- correctness fails or cannot be established
- the candidate requires a production rewrite to test a config-sized idea
- the direction is on the project's do-not-reopen list and no material condition
  changed

A closed line is a deliverable. Record the reason, the evidence, and the
condition that would reopen it.

---

## Source-Family Notes

The same method applies to different external families; what changes is where to
look first.

| Source family | Mechanisms that often transfer | Artefacts that usually do not |
|---|---|---|
| Marlin / LightOp | shape × M tile table, small-M tile, compile-time family selector | hand-written asm binary, backend-specific weight layout |
| AITER | backend tiering, ASM/CK solution tables, fused MoE, persistent workgroups | solution CSVs with no row for your E/shape |
| DeepGEMM | tile/persistent scheduling, fused epilogue, shape-specific configs | dataflow assumptions tied to its dispatch |
| CUTLASS / CK | tile families, swizzled layouts, stream-K / split-K | fixed block geometry and packer lock-in |
| Triton specialized kernel | constexpr specialization, heuristics, autotune winners | tuner coverage and synthetic-routing bias |
| FlashAttention | tile scheduling, online softmax, causal/varlen specialization | attention-specific dataflow |
| Upstream performance PR | one concrete mechanism plus its measured operating point | the PR's benchmark shape, if yours differs |

Always check the operating point the source mechanism needs. A mechanism that
is real, tuned, and fast at one shape can be structurally unavailable at yours.
The Step5 case has three examples: the vendor's small-M grid shrink needs
`A.size(0) < BLOCK_SIZE_M` (false at 128 vs 16); the `M <= E` default needs
`128 <= 44` (false); and the ASM tier has no `E=44` row at all.

---

## References

- `references/mechanism-inventory.md` — the mechanism master table, detection
  recipes, status vocabulary, and worksheet.
- `references/fixed-shape-specialization.md` — how to find and validate a
  fixed-shape specialization target, knob by knob.
- `references/validation-ladder.md` — L0-L7, gates, claim vocabulary, and what
  evidence each level requires.
- `references/anti-patterns.md` — the recurring failure modes, with the correct
  move for each.
- `references/failure-classification.md` — the eight-class taxonomy and how to
  assign primary/secondary classes.
- `references/case-studies/step5-marlin-style-c1.md` — the authoritative worked
  example: fallback root cause, rejected backend transplant, the C1
  specialization, its confirmed-positive evidence, and the negative lessons.

## Relationship to `senior-performance-optimization`

This skill owns:

```text
WHAT TO OPTIMIZE
WHAT TO TRANSPLANT
WHY IT SHOULD HELP
WHETHER IT IS ALREADY PRESENT
```

`senior-performance-optimization` owns:

```text
HOW TO MEASURE
HOW TO VERIFY
HOW TO ACCOUNT FOR THE GAP
HOW TO PROVE THE FINAL WIN
```

When the task is "a high-performance kernel/backend/PR is fast elsewhere; which
mechanisms can be transplanted here?", read this skill first, then use
`senior-performance-optimization` and its references for the measurement,
correctness, gap-accounting, tuning-audit, and E2E protocol.

Do not answer a mechanism-transplant question with a backend port, a broad
config sweep, or a single microbenchmark.
