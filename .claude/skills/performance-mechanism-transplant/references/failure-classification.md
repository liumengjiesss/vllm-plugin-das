# Failure Classification

A binary `SUCCESS / FAILED` label destroys the information the next engineer
needs. The eight classes below separate **why** a line closed, so the same
analysis is not repeated and a valid mechanism is not wrongly discarded.

## Primary and Secondary Class

Assign one primary class and, when useful, one secondary class:

```text
PRIMARY_CLASS=       the reason the candidate cannot be accepted
SECONDARY_CLASS=     a contributing reason, or a separate lesson
```

Example: the Step5 D0 "B reuse" result is primarily `MEASUREMENT_ARTIFACT`
(address defect) and secondarily `MECHANISM_INVALID` for the grouping schedule
(the reuse did not pay even when correct).

## The Eight Classes

### 1. MECHANISM_INVALID

**Definition.** The claimed causal mechanism or payoff model is false. The
transformation may be semantically legal; the reason it was expected to be
faster is wrong.

**What it means.** Stop trying to optimize this mechanism at this operating
point. The premise, not the implementation, failed.

**What it does not mean.** It does not mean the source backend is bad, or that a
different mechanism in the same backend cannot work.

**Evidence required.** A measured result that contradicts the mechanism's
predicted direction, plus an explanation of which premise was false.

**Subcodes:**

```text
PAYOFF_MODEL_FALSIFIED       the removed work was not on the critical path
LESS_WORK_NOT_LOWER_LATENCY  the work reduction did not reduce latency
SCHEDULING_CANNOT_RESCUE     assignment is not the residual bottleneck
WRONG_COST_MODEL             bytes/blocks/rows were modelled incorrectly
```

**Step5 examples:**

- Support kernel C2 candidate A: the 8 branch-guarded loads were not on the
  critical path. Predicating them collapsed `scf.if` 8 -> 0 and made the kernel
  0.70 us/call slower (resident) / 0.48 us/call slower (streamed). The
  implementation did exactly what it was designed to do; the payoff model was
  false. `A_MECHANISM_GPU_CONFIRMED = NO`.
- Program-count rescue: the grouped kernel's penalty fell only 65.2 % -> 61.9 %
  -> 57.1 % when programs were halved twice. The residual was inside the leader
  (serialized K loops holding B), not in the assignment. A dynamic work queue
  addresses assignment, so it cannot fix the residual.
  `PROGRAM_COUNT_RESCUE = CLOSED`.

**Next action.** Record the falsified premise and the condition that would
change it (different kernel, different operating point, different work
decomposition).

---

### 2. IMPLEMENTATION_FORM_FAILED

**Definition.** The mechanism is valid, but this implementation's overhead,
register pressure, occupancy, launch count, or schedule cost dominates the
benefit.

**What it means.** The idea may be salvageable in a different form; this form is
not. Do not write it up as "the mechanism is wrong".

**What it does not mean.** It does not mean the source mechanism is invalid, and
it does not mean a different implementation will succeed. The burden is on the
next form.

**Evidence required.** Measured net loss with a decomposition that shows the
overhead term. Ideally, a measured or modelled overhead floor and a gross
benefit bound.

**Subcodes:**

```text
REGISTER_PRESSURE            grouping/rewrite raised registers enough to lose occupancy
OCCUPANCY_LOSS               fewer resident work units dominate
STRUCTURAL_OVERHEAD          the rewritten kernel's fixed cost exceeds the prize
LAUNCH_OVERHEAD              extra launches/sorts/copies exceed the saving
OVERHEAD_EXCEEDS_SAVING      the added metadata/mask/branch costs more than the skipped work
```

**Step5 examples:**

- Expert-local M-grouping: semantically sound, bit-exact, but +65.2 % (P0) /
  +61.9 % (P1) / +57.1 % (P2) against the incumbent. Registers rose 61 -> 99 ->
  143 (GEMM1) and 105 -> 167 -> 247 (GEMM2); one program serialized several
  m-tiles' K loops while holding B. The best grouped arm was still +58 %.
- Support kernel C2 candidate B: padding-skip was correct and safe on 85/85
  layers, but the extra `sorted_ids` load, mask computation, and branch cost
  more than the 61.4 % of tiles skipped: 0.35 us/call slower resident / 0.10
  us/call slower streamed.

**Next action.** State the specific overhead term and what would have to change
for the form to pay. Do not schedule a variant search unless the overhead can be
removed.

---

### 3. ALREADY_ABSORBED

**Definition.** The incumbent already implements the useful form of the
mechanism, possibly under a different name or with a finer grid.

**What it means.** There is no incremental value in re-implementing it. The
correct output is the incumbent's implementation, with file/line evidence.

**What it does not mean.** It does not mean every mechanism in the source
backend is absorbed. Re-check the remaining rows.

**Evidence required.** The incumbent's implementation and the equivalence
argument. For a selector/table, the runtime-selected entry.

**Subcodes:**

```text
SAME_IDEA                    the same mechanism exists
EQUIVALENT_FORM              algebraically equivalent under a different flag
FINER_GRID                   the incumbent's coverage is finer
AHEAD_OF_SOURCE              the incumbent fuses more than the source
```

**Step5 examples:** persistent execution, EP-aware local-expert layout, fused
act+clamp+quant+router-fold, masked reduce with shared expert, zero-fill-free
workspaces, shape × M table with 21 buckets, MLS/matrix loads, `GROUP_SIZE_M=8`,
and a selector key richer than the vendor's (includes K and topk). The
whole-backend Marlin transplant was `ALREADY_ABSORBED` at the mechanism level and
`UNAVAILABLE_BY_VENDOR_DATA` at the artifact level.

**Next action.** Remove the row from the candidate set and search for a gap
elsewhere.

---

### 4. RESOURCE_LIMITED

**Definition.** A hard hardware or compile resource is binding: LDS/shared
memory, registers, occupancy, memory capacity, or a fixed CU count.

**What it means.** A configuration that violates the limit is not a tuning
result; it is a closed-form rejection. A configuration that fits but loses
occupancy is also resource-limited.

**What it does not mean.** It does not mean the mechanism is semantically wrong;
it means the hardware cannot host this form.

**Evidence required.** The capacity arithmetic or a compile error, plus the
fitted alternative if one exists.

**Subcodes:**

```text
LDS_OVERFLOW                 shared-memory requirement exceeds the limit
REGISTER_LIMIT               register pressure forces spills/occupancy loss
OCCUPANCY_FLOOR              too few resident CTAs/warps
MEMORY_CAPACITY              weight/layout duplication does not fit
FIXED_CU_COUNT               tuning tied to a CU count that may change
```

**Step5 examples:**

- Vendor decode family `BN=32, BK=512, ns=3` required 73728 bytes of LDS against
  a 65536 limit:
  `(32*512 + 16*512) * 1 * 3 = 73728`, exact.
- The LDS-feasible `BN=16` form measured +291 % on GEMM1 (or +195 % on G1+G2).
  The family was excluded from both directions.
- Duplicating Marlin's `[E, K/64, N*64]` layout alongside A0's `[E, N, K]` would
  cost ~70 GB per rank across the 85 MoE layers.

**Next action.** Record the hard limit and the feasible alternatives. Reopen
only if the hardware, dtype, or kernel changes.

---

### 5. MEASUREMENT_ARTIFACT

**Definition.** The measured result is an artifact of the instrument, the
comparator, the harness, or a correctness defect, not a performance property.

**What it means.** Discard the number. Fix the instrument and re-measure before
classifying the mechanism.

**What it does not mean.** It does not invalidate the mechanism. The mechanism
may still be valid or invalid; it has not been tested yet.

**Evidence required.** The control that caught it, the defect, and the corrected
measurement.

**Subcodes:**

```text
ADDRESS_DEFECT               recomputed address reads different data
BUFFER_REUSE                 compared buffer not zeroed between arms
WRONG_VALID_REGION           padded/unwritten tail compared
CROSS_SESSION_DRIFT          baseline and candidate from different lives
PROFILER_PERTURBATION        profiling changes the measured path
FALSE_FAILURE                comparator reports a difference between identical bytes
```

**Step5 examples:**

- D0 "B reuse": the recomputed B address omitted
  `k * BLOCK_SIZE_K * stride_bk`, so every K step re-read the k=0 tile (1/6 of
  GEMM2's bytes). It measured -40 % on GEMM2 / -14.3 % on the total. The
  bit-exactness control caught it. Corrected: pointer style 0.0 %, standalone
  kernel -0.02 %, residual +7.6 % GEMM1 structural overhead.
  `B_REUSE_MECHANISM_VALID` is `REFUTED`.
- Support kernel C2 harness bug: the `scale` sentinel was allocated internally,
  so the caller's sentinel was never written; a first run appeared to show
  padding rows overwritten. Fixed with an explicit output buffer.
- Support kernel C2 `sorted_w` false failures: `torch.equal` on the whole
  tensor included an uninitialized tail with NaN patterns; `NaN != NaN`
  reported deterministic failures on 9/85 layers. Fixed by bitwise comparison
  over the written region only.
- A guard-branch rule `id < M*topk` misclassified every valid row with `k >= 1`;
  the correct rule is `(sid & 0xFFFFFF) < M`.

**Next action.** Fix the instrument, re-run the control, and only then assign a
performance class.

---

### 6. CORRECTNESS_FAILURE

**Definition.** The output is numerically, structurally, or contractually wrong.
Performance is irrelevant.

**What it means.** Do not read timing. Fix correctness first.

**What it does not mean.** It does not mean the mechanism is slow; it was never
validly measured.

**Evidence required.** The failing comparison, the reference, the valid region,
and the root cause.

**Subcodes:**

```text
NUMERICAL_MISMATCH           max_abs/max_rel beyond tolerance
BIT_EXACT_FAILURE            expected bit-exact, got differences
SHAPE_DTYPE_FAILURE          wrong shape/dtype/layout
INDEX_ADDRESS_FAILURE        reads/writes the wrong element
BUFFER_OWNERSHIP_FAILURE     aliasing/overwrite/uninitialized region
WRONG_FALLBACK               dispatch selects a path that cannot be correct
WRONG_VALID_ROW_RULE         the experiment's own predicate is wrong
```

**Step5 examples:**

- The wrong valid-row rule `id < M*topk`; correct: `(sid & 0xFFFFFF) < M`.
- The `slimquant_marlin` forced-Triton fallback was a backend-selection defect.
  Fixing it is a correctness/selection change, not a performance optimization.
  After the fix, `-q slimquant_marlin` no longer materially regressed decode and
  was approximately at parity with the default production path.

**Next action.** Fix and re-run from L2. Do not carry the timing forward.

---

### 7. LOCAL_POSITIVE_E2E_NULL

**Definition.** The component is faster in isolation, but the pipeline or E2E
does not improve.

**What it means.** The saved time moved, or was never on the critical path, or
the component's share is too small. Investigate before declaring a win.

**What it does not mean.** It does not mean the local measurement was wrong. It
means local gain is not E2E gain.

**Evidence required.** L3 local win, L4 stage total, L5/L6 E2E, and a
`TIME_MOVED_TO` attribution.

**Subcodes:**

```text
SYNC                         new synchronization
COPY                         new copy/layout conversion
COMM                         communication dominates
GRAPH_BREAK                  graph capture lost
LAUNCH                       launch count/overhead rises
SCHEDULING                   work moved to another stage
SHARE_TOO_SMALL              the component is too small a fraction of E2E
UNEXPLAINED                  cost moved, destination unknown
```

**Step5 evidence.** The A0-vs-C1 differential trace is a positive example of
translation: the draft-MoE delta (-0.575 ms/step) matched the ITL delta
(-0.67 ms) far better than the total kernel delta (-1.460 ms). The apparent
persistent-kernel -0.852 ms was between-life variation, not a C1 effect.

**Next action.** Attribute where the time went. If the share is too small, say
so and do not claim a production win.

---

### 8. CONFIRMED_POSITIVE

**Definition.** Correct, reproducible, gain larger than noise, paired
clean-restart E2E, and the regression dimensions checked.

**What it means.** This is the only class that supports a production win claim.

**What it does not mean.** It does not automatically mean `STRICT_NEW_BEST`.
That label is project-specific and may require a larger threshold.

**Evidence required.**

```text
L0 runtime-path proof
L1 real-shape proof
L2 correctness pass
L3 local positive
L4 stage positive / translation
L5 same-session interleaved A/B
L6 clean-restart paired E2E
L7 regression dimensions
GAIN > NOISE
REPRODUCIBLE
```

**Subcodes:**

```text
CONFIRMED_POSITIVE            full L6+L7 pass
E2E_CONFIRMED_REGRESSION_INCOMPLETE  L6 passed, L7 incomplete
STRICT_NEW_BEST               project threshold also met
```

**Step5 example.** C1: bit-exact, +1.11 % median E2E (BW1100, 5/5 clean
restarts), +1.39 % median E2E (BW1101, 3/3 paired wins), ITL -1.98 %, prefill
+0.05 %, MTP acceptance unchanged. `CONFIRMED_POSITIVE = YES`,
`STRICT_NEW_BEST = NO` (below the project's 1.99 % bar).

**Next action.** Deliver with the evidence chain, rollback, and limitations. The
1.99 % bar is a project label, not a universal performance threshold.

## Decision Tree

```text
Did the intended path execute?
  NO  -> CORRECTNESS_FAILURE / WRONG_FALLBACK (fix path; do not classify perf)
  YES
    Is the result correct?
      NO  -> CORRECTNESS_FAILURE
      YES
        Is the measurement trustworthy?
          NO  -> MEASUREMENT_ARTIFACT
          YES
            Is the mechanism already in the incumbent?
              YES -> ALREADY_ABSORBED
              NO
                Did the mechanism's predicted direction fail?
                  YES -> MECHANISM_INVALID
                  NO
                    Is a hard resource binding?
                      YES -> RESOURCE_LIMITED
                      NO
                        Does this implementation's overhead dominate?
                          YES -> IMPLEMENTATION_FORM_FAILED
                          NO
                            Is the local gain positive?
                              NO  -> MECHANISM_INVALID or IMPLEMENTATION_FORM_FAILED
                              YES
                                Does the stage/pipeline improve?
                                  NO  -> LOCAL_POSITIVE_E2E_NULL
                                  YES
                                    Does clean-restart paired E2E improve?
                                      NO  -> LOCAL_POSITIVE_E2E_NULL
                                      YES
                                        Are regression dimensions covered?
                                          NO  -> E2E_CONFIRMED_REGRESSION_INCOMPLETE
                                          YES -> CONFIRMED_POSITIVE
```

## Mapping Table

| Observation | Primary class | Typical secondary |
|---|---|---|
| `max_abs != 0` after a rewrite | CORRECTNESS_FAILURE | MEASUREMENT_ARTIFACT |
| Fast kernel reads 1/6 of the bytes | MEASUREMENT_ARTIFACT | CORRECTNESS_FAILURE |
| Large speedup, no control | MEASUREMENT_ARTIFACT | — |
| Proposed mechanism already in source | ALREADY_ABSORBED | — |
| Correct transformation, slower GPU | MECHANISM_INVALID (payoff) or IMPLEMENTATION_FORM_FAILED (overhead) | — |
| Compile fails on LDS | RESOURCE_LIMITED | — |
| Kernel faster, service flat | LOCAL_POSITIVE_E2E_NULL | — |
| E2E gain, regression dimensions untested | E2E_CONFIRMED_REGRESSION_INCOMPLETE | — |
| Correct, reproducible, paired E2E win | CONFIRMED_POSITIVE | — |
| Below the project's strict threshold | CONFIRMED_POSITIVE | STRICT_NEW_BEST = NO |

## Write-Up Template

```text
CANDIDATE=
PRIMARY_CLASS=
SECONDARY_CLASS=
WHAT_IS_PROVEN=
WHAT_IS_NOT_PROVEN=
EVIDENCE=
WHY_THIS_CLASS=
NEXT_ACTION=
DO_NOT_REOPEN_UNLESS=
```

Do not write only `SUCCESS` or `FAILED`.
