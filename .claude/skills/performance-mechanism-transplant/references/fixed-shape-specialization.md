# Fixed-Shape Specialization

Fixed-shape specialization is the cheapest high-value form of mechanism
transplant: keep the incumbent kernel and dataflow, and give the real operating
point the launch geometry or selector entry it is missing.

It is also the easiest place to waste a GPU week. The method below is designed
to prevent that.

## Trigger Conditions

Specialization is worth evaluating only when all four are true:

```text
STABLE_SHAPE=YES        the shape repeats across steps/layers/buckets
HIGH_FREQUENCY=YES      many calls per step, or a large time share
CORRECT_KERNEL=YES      the kernel is already correct; the config is generic
GENERIC_CONFIG=YES      a default/fallback path is actually being used
```

If the shape is not stable, or the kernel is wrong, stop: this is not a
specialization problem.

Typical evidence that a generic config is in use:

- a runtime warning such as `Using default ... config`
- a config-file-not-found message
- a nearest-M hit far from the real M
- a default branch in the selector
- a fallback to a generic backend

## First: Reconstruct the Real Runtime Shape

Before any tuning, write down:

```text
M domains:        M_loc / M_ep / M_lookup / per-expert M_e
N
K
E_global
E_local
topk
dtype / quant scheme
device / arch name strings
calls per step
time share of the step
selector key actually used
selected config at runtime
```

Rules:

1. **Never tune a shape that does not occur.** A shape that appears only in a
   synthetic benchmark or a tuner default is not a target.
2. **Never confuse M domains.** A selector may consume `M_lookup`, while the
   kernel's useful work is driven by per-expert `M_e`. The same decode step can
   have several M definitions.
3. **Never infer the selector key from the filename alone.** Read the lookup
   function and the runtime log.
4. **Record the frequency.** A 30 % win on a kernel called twice per step is
   worth less than a 5 % win on one called two hundred times.

## GOOD_SPECIALIZATION_TARGET

A target is good when it is:

```text
STABLE              the same shape recurs across steps; not a rare ramp bucket
FREQUENT            many calls per step, or a material time share
BUDGET_RELEVANT     the recoverable ms/step can move the E2E metric
NARROWLY_SELECTABLE the selector can hit only this shape; other shapes fall back
EASY_TO_FALLBACK    reverting is a config change, not a code revert
CORRECTNESS_SAFE    the change is launch geometry or dispatch, not math
```

A target that fails `NARROWLY_SELECTABLE` is dangerous: a decode tile selected
by nearest-M can be reused for prefill and cause a much larger regression.

## What Each Knob Can and Cannot Buy

| Knob | What it can buy | What it cannot buy | Check before using |
|---|---|---|---|
| `BLOCK_SIZE_M` | fewer M-tiles / less padding for the actual `M_e` distribution | fewer weight bytes; the weights are `K*N` regardless | MMA minimum (often 16); per-expert `M_e`; padding ratio; block count versus weight sweeps |
| `BLOCK_SIZE_N` | more/fewer CTAs; different A re-read pattern | fewer weight bytes; wide-N does not reduce the weight stream | CTA count vs occupancy; A-tile reuse; LDS budget |
| `BLOCK_SIZE_K` | deeper K pipeline; fewer loop iterations | fewer weight bytes | LDS budget; tail masking; compiler support |
| `GROUP_SIZE_M` | L2 reuse of B across M-tiles | less DRAM traffic if the cache already captures it | tile ordering; whether the grouping survives the persistent map |
| `num_warps` | occupancy vs per-CTA parallelism | more bandwidth than the memory system provides | register pressure; CTA residency; the measured direction |
| `num_stages` | software pipelining / latency hiding | anything when the loop is already latency-bound | LDS budget; the actual dependency chain |
| program count | persistent makespan / tail balance | a fundamentally bad work decomposition | tiles per program; prior sweeps; the cost of a second wave |
| `SPLIT_K` | parallelism when the tile count is too small | fewer bytes in a weight-bound kernel | the vendor's own parallelism predicate; reduction cost; numerics |
| layout / alignment | vectorized loads; less padding | less weight traffic by permutation alone | kernel/layout contract; exclusivity; downstream row bounds |

### LDS / capacity arithmetic

For a tiled GEMM with element width `b` bytes:

```text
LDS_APPROX = (BLOCK_N * BLOCK_SIZE_K + BLOCK_SIZE_M * BLOCK_SIZE_K)
             * b * num_stages
```

If `LDS_APPROX > hardware_limit`, the configuration will not compile. This is a
closed-form rejection, not a tuning result. In the Step5 case:

```text
BN=32, BM=16, BK=512, ns=3, fp8 b=1
= (32*512 + 16*512) * 1 * 3 = 73728 bytes
> 65536 limit  -> OutOfResources
```

The vendor's `BN=16` form fits (`(16*512 + 16*512)*3 = 49152`) but measured
+291 % on GEMM1. The family was excluded from both directions. Record this kind
of proof; it saves a GPU session and prevents a re-open.

## Selection Procedure

### 1. Prove the current path is generic

Find the default branch and the warning. Prove the runtime selected it. Do not
assume "no tuned file" from an empty directory listing; prove the lookup result.

### 2. Establish the selector key and coverage

```text
SELECTOR_KEY=          E,N,K,topk,dtype,device,arch,version
M_RULE=                exact / nearest / smallest >= M
TUNED_RANGE=
DEFAULT_RANGE=
NEIGHBOR_BUCKETS=      what would be selected just outside the target
```

### 3. Measure the baseline across the real range

Use the real routing distribution and all real layers. Report per-call times,
not one aggregate. Mark the measurement protocol.

### 4. Build a small, evidence-based candidate set

Not a broad grid. Start from mechanism reasoning:

- Why is the default wrong for this shape? (padding, CTA count, pipeline depth,
  occupancy, cache reuse)
- Which one or two knobs address that mechanism?
- What is the capacity limit?
- What are the 3-6 candidates that test the hypothesis?

A broad sweep is allowed only when the search space is genuinely the mechanism
and the cost is bounded. It is not a substitute for architecture analysis.

### 5. Run the microbenchmark correctly

```text
same session
same device
same real routing / counts
all real layers
interleaved arms
incumbent rerun as control
median over repeats
report G1, G2, and G1+G2 (or the real stage total)
```

A single-arm comparison against a cross-session baseline is not evidence.

### 6. Correctness first

For a launch-geometry change:

```text
BIT_EXACT_VS_DEFAULT= PASS / FAIL
MAX_ABS=
SHAPE / DTYPE=        unchanged
BUFFER HYGIENE=       zeroed/poisoned between arms
VALID REGION=         compared
OTHER SHAPES=         still select the default
```

A config-only change should normally be bit-exact. If it is not, stop and
explain why before timing.

### 7. Preserve coverage

Keep default-equivalent entries above the tuned range so nearest-M cannot drag
the decode tile into prefill or a ramp bucket. Verify the neighboring buckets
explicitly.

### 8. Climb the ladder

```text
local kernel -> aggregate stage -> same-session A/B -> clean-restart E2E ->
regression dimensions
```

Only the last two levels can support a `CONFIRMED_POSITIVE` claim. A local
23-34 % kernel gain in the Step5 C1 case translated to ~1.1-1.4 % E2E because
the kernel was a small share of the step. That is a good outcome, not a
disappointment; it is also why the budget must be computed before the E2E run.

## Worked Example — Step5 C1

```text
TARGET          MTP draft-layer MoE, layers 88-90, bf16 TritonExperts
SHAPE           local E=44, N=1536, K=4096, topk=8, M=1..256 decode
CURRENT         vLLM generic default; runtime warning "Config file not found"
CONFIG KEY      E=44,N=1536,device_name=BW1100 (and BW1101 alias), no dtype
TILE            BM=16, BN=64, BK=128, GROUP_M=1, warps=2, stages=1
CALLS/STEP      6 (GEMM1/GEMM2 x 3 draft steps)
LOCAL           -23.0 % to -34.4 % per call across M=16/32/64/128
STAGE           2.479 -> 1.904 ms/step (EXP-008 differential trace)
E2E             +1.11 % median (BW1100, 5/5 clean restarts)
                +1.39 % median (BW1101, 3/3 paired wins)
CORRECTNESS     bit-exact, max_abs 0.0
MTP ACCEPTANCE  unchanged
STRICT_NEW_BEST NO (below the project's 1.99 % bar)
```

Why it generalized:

- The default was wrong for the shape, not the backend.
- The kernel was already correct; only launch geometry changed.
- The selector was narrow: one shape, one dtype, one device-name family.
- Other M buckets kept the default, so prefill was untouched.
- Rollback was a config pointer, not a code revert.

Why it did **not** generalize to "port Marlin":

- The production path had already absorbed the transferable mechanisms.
- The Marlin backend's dataflow was non-EP and would have added work.
- The vendor's asm tile family was not selected at this M and was
  LDS-infeasible in the form that kept the incumbent's BN.

## When Not to Specialize

Do not specialize when:

- the shape is not stable, or is a rare bucket
- the current config is already tuned for the shape
- the kernel is missing a mechanism, not a config
- the selector cannot be made narrow (a decode tile would leak into prefill)
- the recoverable budget is below the overhead floor
- the change breaks graph capture, coverage, or correctness
- the target is on the do-not-reopen list and nothing material changed

## Checklist

- [ ] Real shape measured, with the correct M domain.
- [ ] Call frequency and time share measured.
- [ ] Current path proven generic (runtime warning + selector proof).
- [ ] Selector key and coverage map recorded.
- [ ] Candidate set derived from a mechanism, not a broad grid.
- [ ] LDS/register/occupancy limits checked before running.
- [ ] Same-session interleaved microbenchmark with incumbent control.
- [ ] Per-call, aggregate, and `ms/step` budgets computed.
- [ ] Bit-exactness and valid-region comparison passed.
- [ ] Neighboring shapes still hit the default.
- [ ] Fallback/rollback is a config change.
- [ ] E2E validation uses clean restarts and paired arms.
- [ ] Regression dimensions checked (prefill, other dtypes, other layers).
- [ ] Claim vocabulary matches the validation level.
