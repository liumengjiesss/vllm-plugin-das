# AIOPT-1269 C1 — MTP draft-layer MoE fixed-shape specialization

Status: confirmed positive optimization. **Not a strict NEW_BEST** by the conservative
historical threshold (see §7). Delivered as an opt-out-with-one-variable default; see §10.

## 1. Problem

Step5's MTP draft layers (88–90) carry the only unquantized **bf16** MoE in the model
(`moe_intermediate_size` 1536, `hidden_size` 4096, 352 routed experts over EP8 → 44 local,
top-k 8). vLLM resolves Triton MoE tile configs from a per-shape JSON file; no file exists
for this shape on this device, so every decode step ran vLLM's generic default tiling. The
server log said so on each start:

```
WARNING [fused_moe.py] Using default MoE config. Performance might be sub-optimal!
Config file not found at .../fused_moe/configs/E=44,N=1536,device_name=BW1100.json
```

The default is wrong for this operating point: ~1–4 rows per expert, so the kernel is pure
weight streaming (18.9 MB per expert pair, bf16) and its 4-warp / 2-stage 16x64 tile leaves
bytes in flight on the table.

## 2. Optimization

One tuned tile for that shape, delivered as vLLM's own tuned-config file:

```
BLOCK_SIZE_M 16 · BLOCK_SIZE_N 64 · BLOCK_SIZE_K 128 · GROUP_SIZE_M 1 · num_warps 2 · num_stages 1
```

No kernel change, no math change, no routing/quantization/communication change — a launch
geometry only. The mechanism (INFERENCE, consistent with the measured direction): halving
the warps per 16x64 CTA doubles resident CTAs, and `num_stages=1` drops the LDS
double-buffer that buys nothing when the K loop is memory-latency bound.

The table keeps vLLM's own default values for prefill-sized batches (M ≥ 512) so prefill
behavior is untouched; those entries must stay, because vLLM resolves a batch to the
**nearest** key.

## 3. Target operating point

The selector key is vLLM's file name — shape, dtype and device:

| key | value | why |
|---|---|---|
| `E` | 44 | local experts after EP8 (352 global) |
| `N` | 1536 | `moe_intermediate_size` (`w2.shape[2]`) |
| `K` | 4096 | `hidden_size` |
| `topk` | 8 | not in the file name; part of the tuning shape |
| dtype | none | bf16/unquantized — the `,dtype=` selector is absent, so **fp8 MoE of the same E/N resolves to a different file name and is unaffected** |
| `device_name` | BW1100, BW1101 | both spellings of the same HCU part in this deployment |
| `M` | 1 … 256 tuned | decode-sized; nearest-key lookup |

`TARGET_MODEL/COMPONENT` = Step5 MTP draft-layer MoE (layers 88–90).
`TARGET_LAYER_TYPE` = unquantized bf16 `TritonExperts` (`fused_moe_kernel`), 6 calls/step
(gemm1/gemm2 x 3 draft steps).

**BW1100 and BW1101 are aliases for the same HCU platform in the current deployment;
the duplicated config files are provided solely to tolerate inconsistent device-name
strings.** This is one operating point, not two hardware validations.

## 4. Code change

The tuning itself is data. The code only points vLLM's existing lookup at it:

| file | change |
|---|---|
| `vllm_hcu/model_executor/layers/fused_moe/moe_configs/E=44,N=1536,device_name=BW1100.json` | new: the tuned table |
| `vllm_hcu/model_executor/layers/fused_moe/moe_configs/E=44,N=1536,device_name=BW1101.json` | new: byte-identical alias |
| `vllm_hcu/moe_config_env.py` | new: `configure_tuned_moe_config_folder()` |
| `vllm_hcu/__init__.py` | calls it next to the existing LightOp env bootstrap |
| `setup.py` | ships `moe_configs/*.json` in `package_data` |
| `tests/patch/test_moe_tuned_config.py` | new contract tests |

`configure_tuned_moe_config_folder()` sets `VLLM_TUNED_CONFIG_FOLDER` to the package's
`moe_configs/` **only when the user has not set it**. vLLM consults that folder before its
bundled configs and falls through per file name, so shipping one shape's file cannot affect
any other shape, component or device.

## 5. Correctness

- **Bit-exact** against the default tile: `max_abs = 0.0` for every M and routing seed, on
  both device-name keys (BW1100: EXP-005, 3 sweeps; BW1101: EXP-BW1101, 24/24 measurements).
- The tile changes only `BLOCK_SIZE_*`/`GROUP_SIZE_M`/`num_warps`/`num_stages`. `SPLIT_K` is
  forced to 1 at the launch site, so the table not carrying it is inert.
- E2E: MTP acceptance unchanged, no fallback, no crash, in all 10 formal rounds of EXP-007.
- Other shapes still miss the selector and keep `get_default_config` (verified for
  `E=352,N=1536`, `E=44,N=1536,dtype=fp8_w8a8`, `E=128,N=1024`, `E=48,N=512`).

## 6. Local performance

Per `fused_experts` call, CUDA-graph timed, 1 HCU:

| M | default (µs) | C1 (µs) | BW1100 delta | BW1101 delta |
|---|---|---|---|---|
| 16 | 531.6 / 532.6 | 388.6 / 380.2 | −26.9 % | −28.6 % |
| 32 | 830.2 / 832.4 | 607.3 / 593.4 | −26.9 % | −28.7 % |
| 64 | 1063.2 / 1057.3 | 819.1 / 798.9 | −23.0 % | −24.4 % |
| 128 | 1529.2 / 1519.3 | 1003.7 / 1022.9 | −34.4 % | −32.7 % |

Same direction and magnitude on both keys. In server terms the draft MoE drops
2.359 → 1.939 and 2.593 → 1.862 ms/step, i.e. **−0.42 … −0.73 ms/decode step** (EXP-008).

## 7. E2E performance

### 7.1 Original run — EXP-007, BW1100 key, interleaved clean restarts

| metric | A0 | C1 |
|---|---|---|
| decode (median, n=10) | 1437.12 tok/s | **1453.10 tok/s** |
| gain | — | **+1.11 % median, +1.28 % mean** |
| 95 % CI | — | [+0.60 %, +1.95 %] |
| permutation p | — | 0.0013 |
| ITL | 62.41 ms | 61.74 ms (−1.07 %) |
| prefill | — | ≈ unchanged |
| clean-restart cycles | — | **C1 > A0 in 5/5** (sign test p=0.031) |

Pooled A0 baseline over the night (n=18): median 1439.09 tok/s.

**C1 is a confirmed positive optimization, not a strict NEW_BEST by the conservative
historical threshold**: +1.11 % median is below the 1.99 % noise band this project uses to
declare one. The evidence for "positive" is the CI excluding 0, p=0.0013 and 5/5 clean
restarts — not a claim of a new best.

### 7.2 Independent double-check — EXP-BW1101-E2E, BW1101 key

Re-ran the whole A/B on a host that reports the other device-name spelling, to confirm the
shipped alias is not a silent no-op. 3 paired cycles, 6 clean-restart lives, 12 formal
rounds; same workload and client as §7.1 (bs/concurrency 32, input 65536, output 4096,
seed 42, TP4 · DP2 · EP8 · MTP3, warmup excluded). The single variable per arm is whether
`VLLM_TUNED_CONFIG_FOLDER` points at the package's `moe_configs/` folder, i.e. the bytes
committed here.

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
| ITL | 62.83 → 61.58 ms (**−1.98 %**) |
| prefill | 13016.4 → 13022.7 tok/s (**+0.05 %**, no regression) |
| MTP acceptance | 0.6012 → 0.6012 (unchanged) |
| stability | 18/18 cases rc=0, 18/18 health 200, no crash / OOM / 5xx |

**Reading the numbers honestly.** The headline is the median **+1.39 %**, not the mean
+2.01 %: pair 1 is inflated because its A0 life was the slowest of the three (1411.36 tok/s,
ITL 63.56 ms — the worst of both arms). Pairs 2 and 3, where A0 sits at its usual level, are
+1.65 % and +1.39 %, in line with the historical +1.11 % of §7.1. C1's median here (1454.75)
lands within 0.1 % of the historical C1 median (1453.10), so the absolute level reproduces
as well as the relative gain. Conclusion: **a stable ~1 %-class gain reproduced independently
on the BW1101 key, 3/3 paired wins** — still not a strict NEW_BEST (below the 1.99 % band),
and not a second hardware platform: BW1100 and BW1101 are two device-name spellings of the
same HCU part.

Selector evidence, per life (this is what rules out a silent no-op):

```
A0 3/3 lives  Using default MoE config … Config file not found at …BW1101.json
C1 3/3 lives  Using configuration from …/E=44,N=1536,device_name=BW1101.json for MoE layer.
```

Local numerical check on the same key: 24/24 measurements bit-exact (`max_abs = 0.0`) — see
§5. Runtime per life: `vllm 0.28.1+…step5.8454f1e.20261012`, the same build as §7.1; all
lives started clean and released the GPUs (`GPU_RELEASED=YES`).

Report: `experiments/AIOPT-1269/marlin-step5-gpu/results/EXP-BW1101-E2E/RESULT.md`.
Raw evidence: the same directory (driver log, per-life `run_meta.env` / `server.log` /
`evidence/selector.txt` / `bench/*.json`).

## 8. Reproduction

Microbench (single HCU, ~1 min):

```bash
CFG=<plugin>/vllm_hcu/model_executor/layers/fused_moe/moe_configs   # package folder
python scripts/microbench_draft.py --ms 16,32,64,128 --seeds 3 \
       --configs <cands.json with the tuned tile> --out sweep.json
```

Selector smoke, with vLLM's own lookup (both keys):

```bash
VLLM_TUNED_CONFIG_FOLDER=$CFG python scripts/selector_smoke.py    # experiment copy
```

Server A/B (frozen A0 argv/env; the only difference is the one variable):

```bash
export VLLM_TUNED_CONFIG_FOLDER=$CFG          # the C1 arm only
vllm serve /model/step-5-preview-0913-fp8-channel --tensor-parallel-size 4 \
  --data-parallel-size 2 --enable-expert-parallel ...   # unchanged A0 argv
```

Unit tests (no HCU needed for the first group):

```bash
pytest tests/patch/test_moe_tuned_config.py -q
```

## 9. Scope / limitations

- **One shape, one dtype, one device family.** Everything else keeps vLLM's own lookup.
- The M=1…256 range is tuned; entries above it are copies of `get_default_config` and exist
  only so the nearest-key lookup cannot drag the decode tile into prefill.
- `M` between the tuned keys resolves to the nearest one (vLLM's rule, not a new behavior).
- Both device-name spellings carry their own E2E evidence: BW1100 in §7.1 (5/5 clean-restart
  cycles), BW1101 in §7.2 (3/3 paired cycles, median +1.39 %). The second is a repeat of the
  same operating point on the same part under the other name — not a second platform.
- Both E2E rounds are single-host, few-cycle experiments. §7.1 has the stronger design
  (5 cycles, 10 vs 10 formal rounds, permutation test); §7.2 is the delivery double-check.
- The tuning is device-name keyed; a device that reports a third spelling gets vLLM's
  default and needs one more alias file.

## 10. Rollback

Nothing is forked or patched; rollback is configuration.

| want | do |
|---|---|
| revert to vLLM's tiling, keep the plugin | `export VLLM_TUNED_CONFIG_FOLDER=<any folder without the file>` |
| tune a shape yourself | set `VLLM_TUNED_CONFIG_FOLDER` to your own folder (it wins) |
| remove the feature | `git revert` the commit |

Evidence (host paths, kept out of git):
`experiments/AIOPT-1269/marlin-step5-gpu/results/EXP-005/` (tuning sweep),
`EXP-007/` (BW1100 A/B), `EXP-008/` (kernel accounting),
`EXP-BW1101/` (BW1101 selector smoke + local kernel smoke),
`EXP-BW1101-E2E/` (BW1101 E2E double-check; report in its `RESULT.md`).
Shipped table sha256 = `3000f5aac56558e170d3e10abb6716628a1a7636f82a713ae57f062f04a5f588`
(both alias files).
