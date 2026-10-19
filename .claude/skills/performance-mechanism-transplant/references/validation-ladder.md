# Validation Ladder

The ladder prevents the two most common false claims:

1. "The kernel is faster, therefore the service is faster."
2. "The numbers look good, therefore the change is correct."

Each level answers a narrower question than the next. A claim may not use a
higher-level word than the highest level it passed.

## The Ladder

| Level | Question answered | Minimum evidence | Claim allowed if passed |
|---|---|---|---|
| **L0** | Did the intended path actually execute? | Runtime log / probe / trace showing the selected backend, kernel, config entry, and absence of fallback. `EXPECTED_PATH == ACTUAL_PATH`. | `PATH_PROVEN` |
| **L1** | Is the shape real? | Measured M domain(s), N, K, E_global/E_local, topk, dtype, calls/step, time share. Selector key and selected entry recorded. | `SHAPE_PROVEN` |
| **L2** | Is the result correct? | Independent numerical comparison against a known-correct reference; bit-exact or documented tolerance; valid region; buffer hygiene; NaN/Inf; shape/dtype; index/address checks. | `CORRECTNESS_PROVEN` |
| **L3** | Is the component faster? | Same-session microbenchmark, real routing/layers, interleaved incumbent control, median over repeats, noise recorded. Component pair (`G1+G2`) or real stage total. | `LOCAL_POSITIVE` |
| **L4** | Does the component gain survive the aggregate stage? | Full-stage/layer timing with the change in place; `LOCAL_GAIN_TRANSLATION` check; overhead accounted. | `STAGE_POSITIVE` |
| **L5** | Does it survive a same-session interleaved A/B? | A0 → candidate → A0 (or more cycles), multiple measured rounds, baseline drift recorded, gain vs noise. | `A_B_POSITIVE_PRELIMINARY` |
| **L6** | Does it survive clean-restart paired E2E? | Clean service restarts, paired arms, multiple cycles, median gain, paired wins, decode/ITL/TTFT/prefill, acceptance metric, no fallback/crash. | `E2E_CONFIRMED_REGRESSION_INCOMPLETE` |
| **L7** | Does it hold across the regression dimensions? | Other shapes, dtypes, layers, backends, nearest-M neighbors, prefill, graph capture, other components, rollback. | `CONFIRMED_POSITIVE` |

**Only L6 and L7 together support `CONFIRMED_POSITIVE`.** L3 alone is
`LOCAL_POSITIVE`; it is not a production win.

## Gate Details

### L0 — Runtime-path proof

Record:

```text
EXPECTED_BACKEND=
EXPECTED_KERNEL=
EXPECTED_CONFIG_ENTRY=
ACTUAL_BACKEND=
ACTUAL_KERNEL=
ACTUAL_CONFIG_ENTRY=
FALLBACK_TRIGGERED=   YES / NO
GRAPH_CAPTURED=       YES / NO
SELECTOR_PROOF=       log line / probe / trace / hash
```

A configuration name is not proof. In the Step5 case, `-q slimquant_marlin`
did not execute Marlin; it forced a Triton fallback because the shape's config
rows were missing. The measured "Marlin" regression was a different kernel.

### L1 — Real-shape proof

Record:

```text
M_LOC=
M_EP / M_LOOKUP=
M_E (per-expert distribution: median, p90, max)
N=
K=
E_GLOBAL=
E_LOCAL=
TOPK=
DTYPE / QUANT=
CALLS_PER_STEP=
TIME_SHARE=
SELECTOR_KEY=
M_RULE=               exact / nearest / smallest >= M
```

Do not tune a shape that does not occur. Do not confuse the selector's M with
the kernel's useful work.

### L2 — Correctness

Minimum:

```text
REFERENCE=            installed kernel / torch reference / project reference
BIT_EXACT=            PASS / FAIL
MAX_ABS=
MAX_REL=
VALID_REGION=         rows/routes/segments actually compared
BUFFER_HYGIENE=       compared buffer zeroed or poisoned before each arm?
UNINITIALIZED_TAIL=   handled?
NaN / Inf=            checked?
SHAPE / DTYPE=        unchanged?
INDEX / ADDRESS=      checked (for rewritten kernels)
BOTH_DIRECTIONS=      arm order swapped once?
```

A `max_abs == 0.0` is only as strong as the comparator. If the buffer was not
zeroed between arms, a kernel that skips stores can inherit the previous arm's
values and look bit-exact. If the compared region includes an uninitialized
tail, a byte-identical result can report a false failure (for example, a float
metadata tail containing NaN patterns).

For a rewritten kernel, compare against the **installed, untouched** kernel on
identical inputs. Diff any recomputed address term-by-term against the pointer
advance it replaces.

### L3 — Microbenchmark

Record:

```text
HARNESS=
REAL_ROUTING=         captured counts, not uniform synthetic
ALL_LAYERS=           YES / NO (n=)
ARMS=                 incumbent + candidates, interleaved
REPEATS=
WARMUP=
TIMER=
STATISTIC=            median preferred
NOISE=                run-to-run range
COMPONENT_PAIR=       G1, G2, G1+G2
```

A microbenchmark winner is not an E2E winner. A component win on one GEMM is not
a pipeline win. If the components anti-correlate, decide on the pair or the real
stage total.

### L4 — Aggregate-stage timing

Record:

```text
STAGE_BEFORE_MS=
STAGE_AFTER_MS=
LOCAL_SAVED_MS=
TOTAL_STEP_BEFORE_MS=
TOTAL_STEP_AFTER_MS=
LOCAL_GAIN_TRANSLATION=  PASS / FAILED / PARTIAL
TIME_MOVED_TO=           sync / copy / comm / graph / launch / scheduling / other
```

If the stage saving does not appear in the total, investigate where the cost
moved. Do not argue the stage win into a pipeline win.

### L5 — Same-session interleaved A/B

Preferred design:

```text
A0 -> candidate -> A0-confirm
or
A0 -> candidate -> candidate -> A0
```

Record:

```text
ROUNDS_PER_ARM=
MEDIAN / MEAN / RANGE=
BASELINE_DRIFT_PERCENT=
GAIN_VS_NOISE=
```

If the gain is comparable to baseline drift or run-to-run noise:

```text
PERFORMANCE_WIN_NOT_PROVEN
```

Do not round noise into a positive result.

### L6 — Clean-restart E2E paired test

Record:

```text
CLEAN_RESTART_PER_ARM=YES
PAIRED_CYCLES=
PAIRED_WINS=          n/N
DECODE_MEDIAN_BEFORE/AFTER
ITL_BEFORE/AFTER
TTFT_BEFORE/AFTER
PREFILL_BEFORE/AFTER
ACCEPTANCE_BEFORE/AFTER
FALLBACK / CRASH / OOM / 5xx
SELECTOR_PROOF_PER_LIFE
```

This is the level that separates a real service win from a benchmark artifact.
Paired clean-restart cycles are stronger than a single long run because they
sample the life-to-life variation that a kernel change may not explain.

### L7 — Regression dimensions

Check at least:

- other shapes and M buckets, especially nearest-M neighbors
- prefill and large-M
- other dtypes / quant schemes
- other layers (e.g. a BF16 layer next to FP8 layers)
- other backends and fallback paths
- graph capture / break
- acceptance metric and output validity
- rollback behavior
- other components' timings (no cost moved)
- reproducibility after a clean restart

If a dimension is not tested, say so. Do not imply coverage.

## Claim Vocabulary

| Highest level passed | Allowed claim | Not allowed |
|---|---|---|
| L0 | `PATH_PROVEN` | "faster" |
| L1 | `SHAPE_PROVEN` | "tuned" |
| L2 | `CORRECTNESS_PROVEN` | "faster" |
| L3 | `LOCAL_POSITIVE` | "production win", "E2E win" |
| L4 | `STAGE_POSITIVE` | "E2E win" |
| L5 | `A_B_POSITIVE_PRELIMINARY` | "confirmed", "new best" |
| L6 | `E2E_CONFIRMED_REGRESSION_INCOMPLETE` | "fully validated" |
| L7 | `CONFIRMED_POSITIVE` | "strict new best" unless the project threshold is also met |

`CONFIRMED_POSITIVE` means: correct, reproducible, gain larger than noise,
paired clean-restart E2E, and the regression dimensions checked.

`STRICT_NEW_BEST` is a **project-specific** label. It may require a larger
threshold than statistical significance. In the Step5 project the conservative
bar was 1.99 %, so a +1.39 % median E2E gain was `CONFIRMED_POSITIVE` but
`STRICT_NEW_BEST=NO`. The 1.99 % number is not a universal performance
threshold; record the project's own bar and do not silently generalize it.

## Demotion Rules

- Local positive, E2E null → `LOCAL_POSITIVE_E2E_NULL`. Investigate where the
  cost moved; do not claim a win.
- Gain comparable to noise → `PERFORMANCE_WIN_NOT_PROVEN`.
- Cross-session comparison only → preliminary at best.
- Correctness unproven → cannot promote past L2.
- Selector not proven → cannot promote past L1.
- Regression dimensions incomplete → cannot use `CONFIRMED_POSITIVE`.
- A fake win larger than the mechanism's theoretical prize → assume a defect
  until an independent check clears it.

## Worked Ladder — Step5 C1

```text
L0  selector log:
    A0 3/3 lives: "Using default MoE config ... BW1101.json"
    C1 3/3 lives: "Using configuration from .../E=44,N=1536,device_name=BW1101.json"
L1  shape: local E=44, N=1536, K=4096, topk=8, bf16, M=1..256 decode
L2  bit-exact: max_abs = 0.0 for every M and routing seed (BW1100 and BW1101)
L3  local: -23.0 % to -34.4 % per call across M=16/32/64/128
L4  aggregate: draft MoE 2.479 -> 1.904 ms/step in the EXP-008 differential trace
    (server-term per-life: 2.359 -> 1.939 and 2.593 -> 1.862 ms/step)
L5  E2E: BW1100 5/5 clean-restart cycles; BW1101 3/3 paired wins
L6  E2E: BW1100 +1.11 % median (95 % CI [+0.60, +1.95], p=0.0013);
    BW1101 +1.39 % median; ITL 62.83 -> 61.58 ms (-1.98 %);
    prefill +0.05 %; MTP acceptance unchanged
L7  other shapes still miss the selector and use the default; fp8 MoE of the
    same E/N uses a different file name; M >= 512 entries keep the default;
    no fallback/crash/OOM/5xx

CLAIM: CONFIRMED_POSITIVE = YES
       STRICT_NEW_BEST    = NO (below the project's 1.99 % bar)
```

## Evidence Package

A complete delivery keeps:

- the decision record
- the mechanism inventory
- L0 proof (logs/probes)
- L1 shape records
- L2 correctness output and comparator hygiene
- L3 raw microbenchmark data and the control arm
- L4 stage/total timing
- L5/L6 benchmark JSON and per-life metadata
- L7 regression outputs
- the exact config/table/code diff
- SHA256 of any shipped config artifact
- the rollback command
- the do-not-reopen condition

## Checklist

- [ ] `EXPECTED_PATH == ACTUAL_PATH` proven.
- [ ] Real shape and selector key measured.
- [ ] Correctness gate passed before timing.
- [ ] Microbenchmark uses real routing and an interleaved control.
- [ ] Component pair or real stage total reported.
- [ ] Local gain translation checked.
- [ ] Same-session A/B with baseline drift.
- [ ] Clean-restart paired E2E.
- [ ] Regression dimensions checked.
- [ ] Claim word matches the highest level passed.
- [ ] Project-specific strict threshold recorded separately.
