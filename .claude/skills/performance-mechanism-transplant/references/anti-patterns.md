# Anti-Patterns

Each entry records a recurring failure mode, why it is wrong, the correct move,
and the evidence from the Step5 case where available.

Use the symptom column as a triage table.

| Symptom | Anti-pattern |
|---|---|
| "Backend X is fast, port X" | AP01 |
| Tuning starts before the runtime shape is measured | AP02 |
| A large config grid is expected to close an architectural gap | AP03 |
| One GEMM improved; the pipeline is called faster | AP04 |
| The microbenchmark improved; the service is called faster | AP05 |
| The proposed mechanism already exists in the incumbent | AP06 |
| Production code is rewritten to test a config-sized idea | AP07 |
| A large speedup is reported before correctness | AP08 |
| A fallback path is measured and labeled as the backend | AP09 |
| A synthetic or non-occurring shape is tuned | AP10 |
| Numbers from different sessions are compared | AP11 |
| A legal transformation is treated as a performance win | AP12 |
| A decode config is silently reused for prefill | AP13 |
| A win larger than the theoretical prize is accepted | AP14 |
| Less work is assumed to mean lower latency | AP15 |
| Scheduler tuning is expected to rescue a bad decomposition | AP16 |
| An artifact is confused with a mechanism | AP17 |
| One component of a two-GEMM pipeline decides the verdict | AP18 |

---

## ANTI_PATTERN_01 — Seeing Marlin fast → port Marlin

**Symptom.** A vendor kernel or backend has a reputation for speed, so the task
becomes "make it run here" or "port it".

**Why it is wrong.** It skips the two decisive questions: does it actually
execute at this operating point, and which of its mechanisms is the incumbent
missing? A whole-backend port imports a dataflow, a layout contract, a dispatch
policy, and often a numerical contract.

**Correct move.** Build the mechanism inventory (STEP 2-4). Check runtime path
(STEP 1). If the incumbent already has the useful mechanisms, stop. If one
mechanism is missing, transplant that mechanism onto the incumbent.

**Step5 evidence.** Marlin never executed on the measured `slimquant_marlin`
arm. The whole-backend transplant was rejected because the production path had
already absorbed the transferable mechanisms (persistent execution, EP-aware
layout, fusion, shape/config table, local routing specialization,
activation/quant/combine), and the remaining artifact was not selected at the
operating M. The confirmed positive (C1) was a fixed-shape launch-config
specialization on the existing kernel, not a Marlin port.

**Detection question.** "If this backend is fast, which exact mechanism makes
it fast, and where is that mechanism absent in the incumbent?"

---

## ANTI_PATTERN_02 — Not measuring the runtime shape → start tuning

**Symptom.** A tuner is launched, or a config is changed, before the actual M,
N, K, E, topk, dtype, call count, and selector key are known.

**Why it is wrong.** It optimizes a shape that may not occur, or the wrong M
domain. A single decode step can have `M_loc`, `M_ep`, `M_lookup`, and
per-expert `M_e`. The selector may consume one while the kernel's cost is driven
by another.

**Correct move.** Measure L1 first. Record the shape and the selected config at
runtime. Tune only a shape that occurs.

**Step5 evidence.** `{32,64,128}` in the vendor's asm family were tile
parameters, not token counts. The tuner's `--hidden-size-default` represented
the MoE intermediate dimension, and `--ffn-hidden-size` represented the model
hidden K. Intuitive CLI names were wrong. Aggregate `M=128` was the lookup key,
while the useful distribution was per-expert `M_e` with median 3.

**Detection question.** "What exact shape and M domain does the kernel see, and
what evidence proves it?"

---

## ANTI_PATTERN_03 — Sweeping many configs → expecting the architecture gap to disappear

**Symptom.** The response to a disappointing result is a larger grid: every
`BM/BN/BK/warps/stages/GROUP_M/program count` combination.

**Why it is wrong.** A parameter search cannot fix an architecture or dataflow
gap. It can only find the best point inside the current architecture. A tuner
that searches one kernel family cannot prove global backend optimality. A
one-knob-at-a-time sweep can also miss the interaction cell that matters.

**Correct move.** Explain the mechanism first. Use a small candidate set derived
from the mechanism. Close the line if the architecture is the limit. If a
sweep is justified, record the search space, the interactions, and what it
cannot answer.

**Step5 evidence.** The Marlin config sweep did not beat A0. The vendor decode
family was 3.9x slower on GEMM1 and the `BN=32, BK=512, ns=3` form did not fit
LDS (73728 > 65536 bytes). `EXP-004` was a 24-arm single-knob sweep; the
interaction cell was resolved only by a later targeted run.

**Detection question.** "Which mechanism does this config change test, and what
result would falsify it?"

---

## ANTI_PATTERN_04 — Component gain → claim pipeline gain

**Symptom.** GEMM2 got faster, so the pipeline is reported faster. Or one
kernel in a multi-kernel stage improved and the stage total is not reported.

**Why it is wrong.** Components can anti-correlate. In a two-GEMM pipeline, a
change can help one and hurt the other; the verdict metric is the pair or the
real pipeline total. A per-component win can be a net loss.

**Correct move.** Report `G1`, `G2`, and `G1+G2` for every arm. Decide on the
pair or the real stage total. If the components move in opposite directions,
state the net explicitly.

**Step5 evidence.** The grouped kernel added +121.0 us to GEMM1 and +49.3 us to
GEMM2. Reporting GEMM2 alone would have understated the loss by 3.5x. The
incumbent itself showed +2.6 % G1 / -4.8 % G2 across sessions while `G1+G2` was
stable to 0.04 %.

**Detection question.** "What is the decision metric, and is it one component or
the real total?"

---

## ANTI_PATTERN_05 — Microbenchmark gain → claim E2E gain

**Symptom.** A kernel is 30 % faster in isolation, and the report says the
service is faster.

**Why it is wrong.** The kernel may be a small share of the step; the saved time
may move into sync, copy, communication, launch overhead, a graph break, or
another stage. A local win is not an E2E win.

**Correct move.** Compute the budget in `ms/step` first. Then run L4 (stage) and
L5/L6 (same-session, clean-restart E2E). If the local gain does not translate,
investigate `TIME_MOVED_TO` and classify `LOCAL_POSITIVE_E2E_NULL`.

**Step5 evidence.** C1 improved the draft MoE kernel by 23-34 % per call but
produced a ~1.1-1.4 % E2E gain because the draft MoE was a small share of the
step. That is a valid confirmed positive, not a contradiction: the budget was
computed correctly and the claim matched the level.

**Detection question.** "What is `ms/step` saved, and what share of E2E is
that?"

---

## ANTI_PATTERN_06 — The current path already has the mechanism → re-implement it

**Symptom.** A "new idea" is proposed: persistent execution, a shape-keyed
config table, fused activation, an EP-aware layout, a sorted-row bridge, L2
swizzle, or a small-M tile.

**Why it is wrong.** The incumbent already implements the useful form. The
re-implementation adds code, risk, and a second maintenance path without adding
speed. "Implemented differently but algebraically equivalent" is still
absorbed.

**Correct move.** Run the absorbed-mechanism test in
`mechanism-inventory.md`. Cite the incumbent's implementation for every
`ABSORBED` row. Then search for the actual gap.

**Step5 evidence.** A0 already had: a shape × M table with 21 buckets (finer
than the vendor's 15-16), `BM=16` small-M tile, persistent launch, local-expert
sort, fused act+clamp+quant+router-fold, masked reduce with shared expert,
zero-fill-free workspaces, MLS loads, `GROUP_SIZE_M=8`, and a richer key that
includes K and topk. Much of what "Marlin-style" meant was already there.

**Detection question.** "Where is this mechanism in the incumbent's source, and
what exactly is missing?"

---

## ANTI_PATTERN_07 — Rewriting production to test a small idea

**Symptom.** A one-line config or selector hypothesis is tested by forking the
production kernel or editing site-packages.

**Why it is wrong.** It raises the cost and risk of a cheap experiment, makes
rollback hard, and can contaminate the baseline. It also makes it harder to
attribute the result to the mechanism.

**Correct move.** Use the cheapest form that can test the hypothesis: a config
entry, an env-redirected table, a selector override, a PYTHONPATH overlay, or an
experiment copy. Do not edit site-packages. Keep production untouched until the
mechanism has passed the lower gates.

**Step5 evidence.** The C1 change was a config table plus a selector pointer;
the kernel was untouched. The cache-modifier test used a PYTHONPATH overlay of
the pure-Python kernel source; site-packages was unmodified. The C1 delivery
was reversible by changing one environment variable.

**Detection question.** "What is the cheapest artifact that can falsify this?"

---

## ANTI_PATTERN_08 — Large speedup → skip correctness

**Symptom.** A rewritten kernel measures much faster, and timing is read before
a numerical comparison.

**Why it is wrong.** An address or index rewrite can change which data is read
while leaving the shape and output format unchanged. The result looks fast and
deterministic. A timing-only experiment can ship a fake win.

**Correct move.** L2 before L3. Compare against the installed, untouched kernel
on identical inputs. Zero or poison the compared buffer between arms. Compare
the valid region. Check NaN/Inf. Diff address expressions term-by-term. If the
win exceeds the mechanism's theoretical prize, assume a defect.

**Step5 evidence.** The D0 "B reuse" win measured -40 % on GEMM2 and -14.3 % on
the total. The recomputed B address omitted
`k * BLOCK_SIZE_K * stride_bk`, so every K step re-read the k=0 tile: 1/6 of
GEMM2's weight bytes. After the fix, the pointer style was worth 0.0 %. The
statement "B reuse is measurably valid in GEMM2" is REFUTED.

**Detection question.** "What is the independent correctness control, and was
the buffer clean?"

---

## ANTI_PATTERN_09 — Wrong fallback path → treat it as backend performance

**Symptom.** A quantization/backend option is slower. The conclusion is "the
backend is slow".

**Why it is wrong.** The requested backend may not have executed. A config miss
can force a generic fallback, and the measured path may be a different kernel.
The regression is then a dispatch/correctness problem, not a performance
property of the named backend.

**Correct move.** Prove the runtime path (L0). Check the fallback policy: does a
miss return to the incumbent's best path, or force a weaker generic backend? If
the fallback is wrong, fix the fallback as a separate correctness/selection
change; do not attribute it to the backend.

**Step5 evidence.** `-q slimquant_marlin` regressed decode by ~28 %, but Marlin
execution count was zero. The LightOp table had no `3072_4096` or `4096_1536`
row, and `compressed_tensors_moe_marlin.py:199-221` force-set
`moe_backend="triton"`. The measured path was generic vLLM Triton FP8 MoE,
bypassing the HCU production `DeepGemm`/`ep_fused_moe` path. After the fallback
policy was corrected, `-q slimquant_marlin` no longer materially regressed
decode and was approximately at parity with the default production path. That
is a backend-selection/correctness fix, not a performance optimization.

**Detection question.** "Which kernel actually executed, and did the requested
backend fall back?"

---

## ANTI_PATTERN_10 — Tuning a shape that does not occur

**Symptom.** The tuner covers a convenient grid, but the runtime shape is absent
or rare.

**Why it is wrong.** It consumes GPU time and produces a config that the runtime
never selects, or a nearest-M entry that is wrong for the real bucket.

**Correct move.** L1 first. Map the real shape to the selector key. Tune the
dominant bucket and its immediate neighbors, not an arbitrary grid.

**Step5 evidence.** The real decode bucket was `M_lookup=128`; the interesting
per-expert distribution was `M_e` median 3. Tuning uniform synthetic routing
produced configs that did not match the clumped real routing.

**Detection question.** "Which call sites select this shape, and how often?"

---

## ANTI_PATTERN_11 — Comparing numbers from different sessions

**Symptom.** Candidate numbers from one session are compared against baseline
numbers from another.

**Why it is wrong.** Life-to-life variation can be larger than the effect. In
the Step5 persistent kernel, the life-to-life spread was ~±0.85 ms/step (~4 %),
larger than many candidate effects.

**Correct move.** Same-session interleaved arms, baseline confirmation at the
end, and clean-restart paired E2E for the final claim. If cross-session is
unavoidable, mark it `LOW_CONFIDENCE` and do not promote past L5.

**Step5 evidence.** A0-vs-C1 kernel deltas from separate lives showed an
apparent persistent-kernel saving that was between-life variation, not a C1
effect. The ITL delta matched the draft-MoE delta better than the total kernel
delta.

**Detection question.** "Were baseline and candidate measured in the same
session, and was baseline re-confirmed?"

---

## ANTI_PATTERN_12 — A legal transformation is treated as a performance win

**Symptom.** The reuse is semantically valid, the math is identical, the
transformation is legal, therefore it "should be faster".

**Why it is wrong.** Performance is a property of the schedule on the target
hardware, not of semantic legality. Register pressure, memory-level parallelism,
occupancy, and makespan can dominate the theoretical saving.

**Correct move.** Record two verdicts separately:

```text
SEMANTIC_OPPORTUNITY    = VALID / INVALID
PERFORMANCE_OPPORTUNITY = CONFIRMED / REFUTED / UNKNOWN
```

Never let a valid semantic argument substitute for a GPU verdict. Never write a
refuted performance result as "the mechanism is wrong"; the mechanism may be
fine and the schedule wrong.

**Step5 evidence.** Expert-local M-grouping was semantically sound and bit-exact,
but measured +65.2 % / +61.9 % / +57.1 % against the incumbent. The reuse was
legal; the schedule lost.

**Detection question.** "What is the measured GPU verdict, separately from the
semantic argument?"

---

## ANTI_PATTERN_13 — A decode config is silently reused for prefill

**Symptom.** A decode-tuned entry is selected for a large-M prefill bucket
through nearest-M.

**Why it is wrong.** A decode tile can be catastrophically wrong for prefill.
This is a coverage failure, not a tuning failure.

**Correct move.** Record the coverage map for decode and prefill separately.
Keep default-equivalent entries above the tuned range. Verify the selected
entry for both regimes.

**Step5 evidence.** A decode-tuned AITER set had no large-M entry; runtime
nearest-M selected the `M=160` config for large-M prefill, contributing to a
severe prefill regression. C1 avoided this by keeping `M >= 512` entries at
vLLM's defaults.

**Detection question.** "What does the nearest-M rule select for the largest
real M?"

---

## ANTI_PATTERN_14 — A win larger than the theoretical prize is accepted

**Symptom.** A component improves by more than the entire mechanism could
possibly save.

**Why it is wrong.** It is almost always a defect: wrong data read, skipped
work, a stale buffer, or a measurement artifact.

**Correct move.** Compute the theoretical maximum first
(`MECHANISM_THEORETICAL_MAX_SAVING_MS`). If the measured saving exceeds it,
assume a defect until an independent check clears it.

**Step5 evidence.** The D0 "B reuse" claim was -40 % on GEMM2, while the whole
reuse premise was worth at most ~18 % of that GEMM's bytes. It was an address
defect.

**Detection question.** "What is the maximum this mechanism could save, and does
the measurement exceed it?"

---

## ANTI_PATTERN_15 — Less work is assumed to mean lower latency

**Symptom.** A candidate removes branches, padding rows, or redundant
computation, and is expected to be faster.

**Why it is wrong.** The removed work may not be on the critical path. Predating
loads can add issue/address work without removing the latency the machine is
waiting on. Extra metadata loads, masks, and branches can cost more than the
skipped work.

**Correct move.** Measure the latency, not the instruction count. Check whether
the removed work is on the critical path. A semantically real work reduction can
be a performance loss.

**Step5 evidence.** Support kernel C2:

- Candidate A collapsed 8 `scf.if` regions to zero and was 0.70 us/call slower
  (resident) / 0.48 us/call slower (streamed).
- Candidate B skipped 61.4 % of padding tiles, was correct and safe on all 85
  layers, and was 0.35 us/call slower (resident) / 0.10 us/call slower
  (streamed).
- Combined: -0.090 ms/step resident / -0.049 ms/step streamed, against a
  +0.30 ms/step gate. `SUPPORT_KERNEL_C2_STATUS = CLOSED`.

**Detection question.** "Is the removed work on the measured critical path?"

---

## ANTI_PATTERN_16 — Scheduler tuning is expected to rescue a bad work decomposition

**Symptom.** A grouped/reuse kernel loses, so the next proposal is a dynamic
work queue or a different program count.

**Why it is wrong.** Scheduling can fix assignment and tail imbalance. It
cannot fix serialized memory-level parallelism inside one program, register
pressure, or a fundamentally bad work decomposition.

**Correct move.** Decompose the residual before proposing a scheduler change.
Check program/task mapping, task-length skew, register pressure, MLP, persistent
makespan, and occupancy. Compute the ceiling; if perfect balance still cannot
clear the gate, close the line.

**Step5 evidence.** `GROUP_PENALTY` fell only from 65.2 % to 61.9 % to 57.1 %
when programs were halved twice. The residual was inside the leader (serialized
K loops while holding B). `PROGRAM_COUNT_RESCUE = CLOSED`; the dynamic
work-queue follow-up was `DO NOT ENTER`.

**Detection question.** "What is the residual after perfect balance, and can it
clear the gate?"

---

## ANTI_PATTERN_17 — An artifact is confused with a mechanism

**Symptom.** "Marlin has hand-written asm tiles, therefore transplant the asm
tiles." Or "the source has a packed layout, therefore pack the weights."

**Why it is wrong.** An artifact may be unreachable at your shape, may require
an exclusive layout, or may only be meaningful with a kernel that consumes it.
The mechanism (e.g. shape × M tile selection) can be transplantable while the
artifact is not.

**Correct move.** Classify each item as artifact or mechanism. Ask what
precondition the artifact needs and whether it holds. Prefer the mechanism over
the artifact.

**Step5 evidence.** The `{32,64,128}x256x128` asm family was real, but at total
M=128 the shipped tables selected the compiled family in 5 of 6 sibling shapes;
the artifact was not reached. The packed weight layout was mutually unreadable
by A0's kernel and would have required ~70 GB of duplicated weights per rank.

**Detection question.** "Is the artifact selected at my operating point, and can
the incumbent's kernel consume it?"

---

## ANTI_PATTERN_18 — One component of a two-GEMM pipeline decides the verdict

**Symptom.** GEMM1 or GEMM2 alone is used to accept or reject an arm.

**Why it is wrong.** The components can anti-correlate, and the ratio of their
penalties can change with the arm. A conclusion from one component can be both
quantitatively and qualitatively wrong.

**Correct move.** Report the pair and the real total. Use `G1+G2` or the stage
total as the verdict metric. Do not compare a component delta measured at one
program count / M / shape against a delta measured at another.

**Step5 evidence.** The grouped arm's penalty ratio was 2.45:1 (G1:G2) at P0 and
2.60:1 at P2. The "cheaper component" was not stable.

**Detection question.** "What is `G1+G2`, and is the verdict based on it?"
