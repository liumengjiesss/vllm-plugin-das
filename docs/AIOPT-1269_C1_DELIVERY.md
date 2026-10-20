# AIOPT-1269 C1 — MTP draft-layer MoE fixed-shape specialization

Status: rebased onto `HYGON-AI/vllm-plugin-das:v0.28.1-dev`, with one minimal
reviewed-source pin fix required by the deployed Step5 runtime, and revalidated.
This document separates the **historical** 8454f1e validation from the
**v0.28.1-dev** revalidation; only the latter is the acceptance evidence for
this branch.

## 1. Problem

Step5's MTP draft layers (88–90) carry the only unquantized **bf16** MoE in the
model (`moe_intermediate_size` 1536, `hidden_size` 4096, 352 routed experts over
EP8 → 44 local, top-k 8). vLLM resolves Triton MoE tile configs from a per-shape
JSON file; none exists for this shape on this device, so every decode step ran
vLLM's generic default tiling. The server log says so on each start:

```
WARNING [fused_moe.py] Using default MoE config. Performance might be sub-optimal!
Config file not found at .../fused_moe/configs/E=44,N=1536,device_name=BW1101.json
```

The default is wrong for this operating point: ~1–4 rows per expert, so the
kernel is pure weight streaming (18.9 MB per expert pair, bf16), and its
4-warp / 2-stage 16x64 tile leaves bytes in flight on the table.

## 2. Optimization

One tuned tile for that shape, delivered as vLLM's own tuned-config file:

```
BLOCK_SIZE_M 16 · BLOCK_SIZE_N 64 · BLOCK_SIZE_K 128 · GROUP_SIZE_M 1 · num_warps 2 · num_stages 1
```

No kernel change, no math change, no routing/quantization/communication change —
a launch geometry only. The mechanism: halving the warps per 16x64 CTA doubles
resident CTAs, and `num_stages=1` drops the LDS double-buffer that buys nothing
when the K loop is memory-latency bound.

The table keeps vLLM's own default values for prefill-sized batches (M ≥ 512) so
prefill behavior is untouched; those entries must stay because vLLM resolves a
batch to the **nearest** key.

## 3. Target operating point

| key | value | why |
|---|---|---|
| `E` | 44 | local experts after EP8 (352 global) |
| `N` | 1536 | `moe_intermediate_size` (`w2.shape[2]`) |
| `K` | 4096 | `hidden_size` |
| `topk` | 8 | not in the file name; part of the tuning shape |
| dtype | none | bf16/unquantized — the `,dtype=` selector is absent, so the fp8 MoE of the same E/N resolves to a different file name and is unaffected |
| `device_name` | BW1100, BW1101 | both spellings of the same HCU part in this deployment |
| `M` | 1 … 256 tuned | decode-sized; nearest-key lookup |

`BW1100` and `BW1101` are aliases for the same HCU platform; the duplicated
config files tolerate inconsistent device-name strings. This is one operating
point, not two hardware validations.

## 4. Code change

The tuning itself is data. The code only points vLLM's existing lookup at it.

| file | change |
|---|---|
| `vllm_hcu/model_executor/layers/fused_moe/moe_configs/E=44,N=1536,device_name=BW1100.json` | new tuned table, sha256 `3000f5aa…` |
| `vllm_hcu/model_executor/layers/fused_moe/moe_configs/E=44,N=1536,device_name=BW1101.json` | byte-identical alias |
| `vllm_hcu/moe_config_env.py` | new: `configure_tuned_moe_config_folder()` |
| `vllm_hcu/__init__.py` | calls it next to the existing LightOp env bootstrap |
| `setup.py` | ships `moe_configs/*.json` in `package_data` |
| `tests/patch/test_moe_tuned_config.py` | contract tests |

`configure_tuned_moe_config_folder()` sets `VLLM_TUNED_CONFIG_FOLDER` to the
package's `moe_configs/` **only when the user has not set it**. vLLM consults
that folder before its bundled configs and falls through per file name, so
shipping one shape's file cannot affect any other shape, component or device.

C1 code commit:
`1c886480824b3c6448c271d6b6dd2408ea3adece` —
`perf(moe): specialize Step5 MTP draft MoE config`

The deployed Step5 runtime additionally required a 3-line reviewed-source pin
fix (`509affd0…`, see §4.1), so the validated branches are:

```
A0_FIXED = 509affd0df1a5efbe50f1c172c87755b01087dff
C1_FIXED = 09d83cc7fbf01d82269a6028e1c4341e6b3bda23
           (C1 diff 1c88648 cherry-picked onto A0_FIXED; identical 6-file diff)
```

The source commit's test had a latent `str / str` bug
(`MOE_CONFIG_FOLDER` is a `str`, used with `/`); it was fixed test-only by
wrapping both uses in `Path(...)`. No production semantic adaptation was
required on v0.28.1-dev.

### 4.1 Runtime compatibility pin fix

The first A0 launch with the latest plugin failed before weight load:

```
Step5PatchCompatibilityError: Unreviewed Step5 target source:
vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator.
MultiModuleMTPSpeculator._generate_drafts: dcabfd6f…
```

The reviewed pin `88f8e6d3…` differs from the deployed `dcabfd6f…` only by two
trailing comment lines captured by `inspect.getsource()`; the location-stripped
AST, signature, bytecode, marshal bytes and all 169 disassembly operations are
identical. Commit `509affd0…` adds the new hash and retains the historical
pins; no guard was disabled and no C1 code changed. Full proof:
`experiments/AIOPT-1269/C1_REBASE_V0281/07A_RUNTIME_COMPAT_PIN_FIX.md`.

## 5. Compatibility audit (v0.28.1-dev)

TARGET_BASE=567fd248dfc049897b4d01950aa1b040b3912683

- vLLM 0.28.1 still reads `envs.VLLM_TUNED_CONFIG_FOLDER` in
  `get_moe_configs()` and resolves `E=…,N=…,device_name=…[,…].json`.
- Neither the plugin branch nor the vLLM wheel ships an E=44,N=1536 config.
- The plugin's `lightop_env` import-time bootstrap pattern is still present;
  the C1 hook lands in the same place.
- The native csrc gained `close_mem_handle` after the old baseline, so both E2E
  arms were built from the target source, not from the old 8454f1e wheel.
- No C1 code depends on the old `patches/vllm` chain.

## 6. Historical validation (old baseline 8454f1e — not the acceptance evidence)

- Local: `max_abs = 0.0` at M=16/32/64/128, multiple routing seeds.
- BW1100 E2E: 5/5 paired clean-restart wins, decode +1.11 % median,
  ITL −1.07 %, prefill unchanged.
- BW1101 E2E: 3/3 paired wins, decode +1.39 % median, ITL −1.98 %,
  prefill +0.05 %, MTP acceptance unchanged.

These remain historical evidence for the mechanism. They are not reused as
v0.28.1-dev acceptance.

## 7. v0.28.1-dev revalidation

### Environment

- Host nmz106; container `lmj-step5-vllm028`.
- vLLM `0.28.1+dtk2604.torch2110.2609171627.g77acaf.step5.8454f1e.20261012`
  (same for both arms; the deployed 0.28.1 runtime).
- A0 plugin = `567fd24 + pin fix` = `509affd0…`; C1 plugin = `509affd0… +`
  C1 diff `09d83cc…`. The C1 diff is identical to `1c88648`.
- Both plugins built from target source; the same `hcu_ops` `.so`
  (sha256 `b0785af4…`) is used in both isolates.
- Workload: TP4 · DP2 · EP8 · MTP3, bs/concurrency 32, input 65536,
  output 4096, seed 42, `pd_case.py`.
- Interleaved clean-restart A0/C1 cycles; w0 warmup + f1 + f2 formal per life.

### Selector proof

A0 misses and C1 hits at the vLLM API level (clean subprocesses):

```
A0: VLLM_TUNED_CONFIG_FOLDER=None
    WARNING Using default MoE config … Config file not found at …/E=44,N=1536,device_name=BW1101.json
    get_moe_configs(44,1536,None) is None

C1: VLLM_TUNED_CONFIG_FOLDER=<plugin>/moe_configs
    INFO Using configuration from …/E=44,N=1536,device_name=BW1101.json for MoE layer.
    M16 = BM16 BN64 BK128 GM1 w2 s1
```

Live-server selector evidence is recorded per life in
`runs/<RUN_ID>/evidence/selector.txt` during the paired E2E.

### Local correctness

24/24 measurements (HCU4 and HCU5, M=16/32/64/128, seeds 0/1/2):
`max_abs = 0.0`, bit-exact, finite, input preserved.

### Local microbench

CUDA-graph timed, same session/device, weight rotation, 3 rounds × 20 replays,
median:

| M | A0 µs | C1 µs | Δ |
| ---: | ---: | ---: | ---: |
| 16 | 528.4 | 378.4 | −26.94 % |
| 32 | 818.7 | 619.7 | −25.46 % |
| 64 | 1054.5 | 859.3 | −21.17 % |
| 128 | 1606.1 | 1029.7 | −35.01 % |

Overall median −25.72 %; repeated on HCU5 at −25.41 %; 12/12 negative
directions in each session.

### Paired E2E

Five interleaved clean-restart pairs on all 8 HCUs (TP4 · DP2 · EP8 · MTP3),
A0_FIXED vs C1_FIXED, same vLLM runtime and workload:

```
PAIR1: A0=1400.34  C1=1415.85  +1.11%
PAIR2: A0=1388.08  C1=1391.75  +0.26%
PAIR3: A0=1390.28  C1=1404.28  +1.01%
PAIR4: A0=1388.51  C1=1414.42  +1.87%
PAIR5: A0=1394.46  C1=1402.20  +0.55%
```

```
E2E_PAIRS=5
E2E_DECODE_DELTA=+1.01% median (pair median; ratio of medians +1.01%)
E2E_MEAN_DELTA=+0.96%
PAIRED_WINS=5/5
E2E_ITL_DELTA=-0.95%
E2E_PREFILL_DELTA=+0.06%
MTP_ACCEPTANCE_DELTA=0.00% (0.6012 both arms)
SELECTOR=A0 5/5 default miss · C1 5/5 hit
STABILITY=PASS — 10/10 lives READY, 20/20 formal cases rc=0, health 200,
  GPU_RELEASED=YES after every life, no OOM/5xx/compatibility error;
  only shutdown-time EngineDeadError teardown lines in the first three A0/C1
  pairs (present in both arms).
FINAL_RESULT=CONFIRMED_POSITIVE
```

The first three pairs were the pre-registered initial run; pairs 4–5 are the
pre-declared extension after 3/3 positive and stable. The gain is ~1 % class,
consistent with the historical C1 direction; the 1.99 % historical bar is not
used as an acceptance threshold.

## 8. Reproduction

```bash
# contract tests (needs the target vLLM runtime)
pytest tests/patch/test_moe_tuned_config.py -q

# selector proof
VLLM_TUNED_CONFIG_FOLDER=<plugin>/vllm_hcu/model_executor/layers/fused_moe/moe_configs \
python -c "from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_configs; \
print(get_moe_configs(44,1536,None)[16])"

# rollback / disable
VLLM_TUNED_CONFIG_FOLDER=<folder without the file>
```

The E2E harness lives outside git at
`experiments/AIOPT-1269/C1_REBASE_V0281/e2e/`.

## 9. Scope / limitations

- One shape, one dtype, one device family. Everything else keeps vLLM's own
  lookup.
- M=1…256 is tuned; entries above it are copies of `get_default_config` and
  exist only to prevent nearest-key leakage into prefill.
- The tuning is device-name keyed; a third spelling needs one more alias file.
- BW1100 and BW1101 are two spellings of the same HCU part, not two hardware
  platforms.
- The vLLM runtime is the deployed 0.28.1 build, identical for both arms; the
  plugin branch is the controlled variable.

## 10. Rollback

| want | do |
|---|---|
| revert to vLLM's tiling, keep the plugin | point `VLLM_TUNED_CONFIG_FOLDER` at a folder without the file |
| tune a shape yourself | set `VLLM_TUNED_CONFIG_FOLDER` to your own folder (it wins) |
| remove the feature | `git revert` the code commit |

## 11. Evidence

Host paths, kept out of git:
`experiments/AIOPT-1269/C1_REBASE_V0281/`
(`00_BASELINE.md`, `01_COMPATIBILITY_AUDIT.md`, `02_PATCH_PLAN.md`,
`03_STATIC_TESTS.md`, `04_SELECTOR_PROOF.md`, `05_LOCAL_CORRECTNESS.md`,
`06_MICROBENCH.md`, `07_E2E_VALIDATION.md`, `08_GIT_AUDIT.md`,
`09_FINAL_DELIVERY.md`, `runs/`).

Shipped table sha256 = `3000f5aac56558e170d3e10abb6716628a1a7636f82a713ae57f062f04a5f588`
(both alias files).
