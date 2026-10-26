# Step5 Prefill / Decode 优化复现 Cookbook

> Repository: `WenQ7/vllm-plugin-das`
>
> Branch: `codex/step5-pd-optimizations-20261023`
>
> 本文校准 commit: `857d114a56846b7a8f6831592bc735adc9923510`（2026-10-26）
>
> 目标软件栈: vLLM 0.28.1 / vllm-plugin-das `0.28.1rc1.dev491` / DTK2610 / gfx938

## 1. 文档范围

本文档用于复现本 branch 已记录的 Step5 **Prefill / Decode 分阶段性能配置**，包括外部资产准备、四节点启动、profile 选择、dry-run、benchmark 口径、结果解释和常见问题排查。

**本文档不是 online P/D disaggregated serving 部署指南。** 本 branch 记录的 Prefill 和 Decode 是分别启动、分别 benchmark、分别记录性能的 phase-isolated 测量；同一组四节点 × 四卡一次只运行一个 phase。

```text
The recorded Prefill and Decode profiles in this branch are
phase-isolated performance measurements.

They are not a simultaneous online P/D disaggregated-serving deployment.
```

如果要部署真正的 P/D disaggregated serving（client → proxy → `kv_producer` → KV transfer → `kv_consumer`），请参考仓库现有的 `docs/pd分离使用.md`、`docs/pd分离mooncake使用.md`、`docs/pd分离du_swift_connector_dp使用.md` 及对应 connector/example。不要把那套在线拓扑与本文档的 phase-isolated 性能复现流程混用。

本文档覆盖的 profile 以当前 branch HEAD 的 `benchmarks/step5/launch_multinode.sh` 为准。后续 HEAD 若发生变化，以同一 commit 的 launcher 和已提交 verification/evidence 为 source-of-truth，并同步更新本文档。

## 2. 版本、repository 与 branch

| 项目 | 值 |
| --- | --- |
| Repository | `WenQ7/vllm-plugin-das` |
| Branch | `codex/step5-pd-optimizations-20261023` |
| 本文校准 HEAD | `857d114a56846b7a8f6831592bc735adc9923510` |
| Plugin base version | `0.28.1rc1.dev491`（`setup.py`） |
| 已记录测量镜像 tag | `0.28.1rc1-ubuntu22.04-dtk2610-py3.12-scalex40-20261021-0235` |
| 已记录 vLLM package | `0.28.1+dtk2610.torch2110.2610191943.g29071f` |
| 已记录 torch package | `2.11.0+das.opt1.dtk2610.2609201632.gf74cd7` |
| 已记录 AITER package | `0.1.7+dtk2610.torch2110.2610201311.gbc033a` |
| 目标架构 | gfx938（BW1300_LC） |

Source-of-truth 优先级：

```text
1. 当前远端 branch 最新 HEAD 的 source
2. 当前 HEAD 已提交的 verification / evidence
3. 四份 1021 人工 reference（historical/operator reference）
4. 本文档
```

发生冲突时使用 `CURRENT_HEAD_WINS`。人工 reference 保留实际操作经验，但不是当前 repository 的 API contract。

## 3. phase-isolated Prefill/Decode 与 online disaggregated serving 的区别

| 维度 | 本文档（phase-isolated） | Online P/D disaggregated serving |
| --- | --- | --- |
| Prefill / Decode 关系 | 分别启动、分别 benchmark、分别记录 | 同时在线运行 |
| 资源占用 | 同一组四节点 × 四卡一次只跑一个 phase | Prefill 与 Decode 各自占用资源并同时工作 |
| 数据路径 | benchmark client → 单个 Prefill 或 Decode 服务 | client → proxy → `kv_producer` → KV transfer → `kv_consumer` |
| KV transfer | 不属于本性能 profile | 必要组件 |
| 入口 | `benchmarks/step5/launch_multinode.sh` | `docs/pd分离*.md` 及 connector/example |
| 目的 | 复现优化后的 Prefill/Decode 性能口径 | 部署在线服务 |

本 branch 中 `VLLM_USE_PD_SPLIT=1` 是 Step5 优化路径的启动环境之一，**不表示**本文档会启动 proxy 或执行在线 KV transfer。不要因为看到 `P/D` 字样就按 online disaggregated serving 部署。

## 4. 硬件与 topology

已记录环境为四节点，每节点四张 BW1300_LC / gfx938 卡，共 16 卡。

| Phase | 节点 × 卡 | TP | DP | EP | 每节点 local DP | 默认 API 端口 | max-num-batched-tokens | max-model-len | gpu-memory-utilization | 已记录 workload |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prefill | 4 × 4 | 1 | 16 | 16 | 4 | 8018 | 8192 | 70000 | 0.875 | input 65536 / output 1 / concurrency 64 |
| Decode | 4 × 4 | 4 | 4 | 16 | 1 | 18118 | 32768 | 1048576 | 0.85 | input 65536 / output 4096 / concurrency 64 |

launcher 固定的公共参数：

```text
--max-num-seqs 64
--kv-cache-dtype fp8_e4m3
--no-enable-prefix-caching
--max-cudagraph-capture-size 64
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE",
                      "custom_ops":["+swiglustep_and_mul"],
                      "compile_sizes":[16,64]}
```

graph region 默认值：`VLLM_STEP4_DSA_GRAPH_REGIONS=8704`；`HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736` 时改为 8736；`dp16_native` Decode 改为 0。`HCU_STEP5_DP_FABRIC` / `HCU_STEP5_CONTEXT_GRAPH_REGIONS` 自身默认均为 0。

性能 profile 固定 synthetic MTP3：

```text
[0.838992, 0.598322, 0.362686]
```

这组 synthetic acceptance 只用于性能复现。质量验证必须使用单独的 standard-MTP3 run，不能把 synthetic-MTP 结果当作 accuracy evidence。

已记录 Decode 的 hot-context graph 覆盖最多 69888 context tokens；更长的 batch 走 native synchronized eager fallback。`max-model-len=1048576` 是配置上限，已在记录的验证中保留，但没有执行实际 1M-token end-to-end workload。

## 5. external runtime assets

源码仓库和 wheel 不携带以下资产：

- Step5 模型权重（`Step5-Channel-FP8-w8a8`）。
- benchmark tokenizer。
- Prefill expert placement PT。
- Decode expert placement PT。
- E23 fixed static expert map。
- 外部 MoE tuned table（`STEP5_EP_TABLE`）。
- 外部 Dense / ASM tuning profile（`HCU_STEP5_TUNING_CONFIG`）。
- benchmark client、dataset、原始 predictions、trace archive。
- 匹配的已编译 `hcu_ops` native extension / DTK / torch 运行环境。

在四个节点上设置当前 launcher 支持的变量：

| Variable | Required | Default at HEAD | 用途 |
| --- | --- | --- | --- |
| `STEP5_HEAD_IP` | 是 | 无 | 所有节点可达的 head 通信 IP |
| `VLLM_HOST_IP` | 是 | 无 | 当前节点自身通信 IP |
| `STEP5_TOKENIZER` | 是 | 无 | benchmark tokenizer 目录 |
| `STEP5_STATE_DIR` | 是 | 无 | 数据盘上的 fresh run directory |
| `STEP5_EP_TABLE` | 是 | 无（HEAD 不再提供 in-tree 默认表） | 外部 tuned MoE JSON |
| `STEP5_SOURCE_DIR` | 否 | 当前 checkout 根目录 | plugin source / `PYTHONPATH` |
| `STEP5_MODEL_DIR` | 否 | `/data/model/Step5-Channel-FP8-w8a8` | 模型权重 |
| `STEP5_PREFILL_PLACEMENT` | 1021 Prefill profile 必需 | 无 | 外部 Prefill placement PT |
| `HCU_STEP5_TUNING_CONFIG` | 1021 Prefill profile 必需 | 无 | 外部 Dense/ASM tuning JSON |
| `STEP5_DECODE_PLACEMENT` | `itl6479` 必需 | 无 | 外部 Decode placement PT |
| `STEP5_PREFILL_STATIC_MAP` | `e23_asm_1021` 必需 | 无 | E23 fixed static map |
| `STEP5_API_HOST` | 否 | Prefill `127.0.0.1`；Decode `0.0.0.0` | API bind host |
| `STEP5_PORT` | 否 | Prefill `8018`；Decode `18118` | API port |
| `STEP5_RPC_PORT` | 否 | `18192` | DP RPC port |
| `STEP5_TMP_DIR` | 否 | `/var/tmp/xuwq_step5` | ZMQ / 临时目录 |
| `STEP5_CACHE_DIR` | 否 | `$STEP5_STATE_DIR/cache` | 编译与框架 cache |
| `NCCL_SOCKET_IFNAME` | 否 | `eno2` | NCCL 通信网卡 |
| `GLOO_SOCKET_IFNAME` | 否 | 继承 `NCCL_SOCKET_IFNAME` | GLOO 通信网卡 |

Baseline 示例：

```bash
export STEP5_HEAD_IP=<head-node-communication-ip>
export VLLM_HOST_IP=<this-node-communication-ip>
export STEP5_SOURCE_DIR=<path-to-this-checkout>
export STEP5_MODEL_DIR=<path-to-Step5-Channel-FP8-w8a8>
export STEP5_TOKENIZER=<path-to-benchmark-tokenizer>
export STEP5_STATE_DIR=<fresh-data-backed-run-directory>
export STEP5_EP_TABLE=<path-to-qualified-moe-table.json>
```

### 5.1 `STEP5_EP_TABLE` 合同

launcher 在启动前把 `STEP5_EP_TABLE` 导出为 `EP_FUSED_MOE_TABLE`，并运行 `benchmarks/step5/check_ep_table.py`。该表是外部运行输入，HEAD 已删除仓库内的 `vllm_hcu/model_executor/layers/fused_moe/ep_table.json`。

表 key 格式：

```text
E=<physical_experts / ep_size>,N=3072,K=4096,topk=8
```

- 默认 `physical_experts=352`，EP16 时 key 为 `E=22,N=3072,K=4096,topk=8`。
- `e23_asm_1021` 使用 `physical_experts=368`，EP16 时 key 为 `E=23,N=3072,K=4096,topk=8`。
- 每个 key 的值是 `M -> [gemm1_config, gemm2_config]`；运行时选择 nearest M。
- 两个 config 的 `BLOCK_SIZE_M` 必须相同；`BLOCK_SIZE_M/N/K` 必须是正整数。
- `e23_asm_1021` 额外要求表中存在显式的 `M=65536` row（launcher 传入 `--require-m 65536`）。

不要从旧 checkout 随意复制一份 JSON 当成当前配置。`check_ep_table.py` 失败时真实启动会在加载模型前退出；dry-run 不会执行该检查。

### 5.2 `HCU_STEP5_TUNING_CONFIG` 合同

这是外部 Dense / ASM tuning JSON，由 `vllm_hcu/ops/step5/tuning.py` 读取。最低要求：

```json
{
  "schema_version": 1,
  "architecture": "gfx938",
  "dense": {
    "hipblaslt_sha256": "<64-hex-char sha256>",
    "algorithms": {"<M,N,K>": <algorithm-id>}
  },
  "asm_g1": {
    "block_m": 128,
    "solution": <solution-id>,
    "persist_by_local_experts": {"22": <persist-id>, "23": <persist-id>}
  }
}
```

要点：

- 按 profile 需要相应的 section：`dense_1021` / `g2_1021` 需要 `dense`；`asm_g1_1021` / `e23_asm_1021` 需要 `dense` 和 `asm_g1`。缺少 section 时对应优化会静默回退，launcher 的“文件非空”检查不会报错。
- `schema_version` 必须为 `1`，`architecture` 必须为 `gfx938`。
- Dense recipe 的 `hipblaslt_sha256` 必须与当前加载的 hipBLASLt library 一致；library 变化后需要重新 tuning。
- Dense recipe 只为 startup 的 8192-row unsplit profile 预热；M16/M64 等尺寸走原有 fallback。
- ASM G1 需要 `block_m=128`、正的 `solution`，以及按 local expert 数索引的 `persist_by_local_experts`。E22 的 local experts 为 22，E23 为 23；记录的 E22 / E23 测量均使用 solution 13001，persist 分别为 256 / 320。
- 构建带 Dense 算子的匹配 wheel 时设置 `VLLM_HCU_BUILD_STEP5_DENSE=1`；默认构建不直接链接 hipBLASLt。

不要把 machine-specific tuning JSON 提交进 Git。

### 5.3 placement 与 E23 static map

- `STEP5_PREFILL_PLACEMENT`：外部 Prefill expert placement PT；所有 1021 Prefill profile 都要求。
- `STEP5_DECODE_PLACEMENT`：外部 Decode expert placement PT；`itl6479` 必需。
- `STEP5_PREFILL_STATIC_MAP`：E23 fixed static map；`e23_asm_1021` 必需，且 launcher 在 dry-run 前就要求文件可读。

E23 static map 必须是 version-2 offline EPLB JSON，包含：

```text
version = 2
format  = "vllm_offline_eplb_physical_to_logical_by_model"
model_maps["Step5ForCausalLM"]
  record_kind = "proposal"
  model_name = <configured model path>
  model_class = "Step5ForCausalLM"
  num_moe_layers = 88
  num_logical_experts = 352
  num_physical_experts = 368
  num_redundant_experts = 16
  physical_to_logical_map = 88 rows × 368 columns
```

每行必须覆盖全部 352 个 logical expert ID。runtime binding 默认要求 `record_kind="proposal"`。

已记录的 E22 placement / E23 map pair：

```text
placement SHA256 = 876ac0f5963c51d7e9e0a4629e589de70ef2d2e9ba86b9b20437d8d0beb1bb32
map SHA256       = 6444cd5e4c7af18d51c6a660462c053248ac7274dc484403f622cf705aa61051
```

E23 必须使用与记录一致的 placement / static-map pair；不要把 legacy Prefill placement 再重复应用一次。

### 5.4 禁止提交的资产

不要提交：

```text
模型权重
tuning JSON
placement PT
static map
native binary / hcu_ops
dataset / 原始 predictions / trace archive
private path artifact
```

## 6. NODE_RANK 与四节点准备

四个节点的 rank 固定为：

```text
node 1 (p5c1) -> NODE_RANK=0
node 2 (p5c2) -> NODE_RANK=1
node 3 (p5c3) -> NODE_RANK=2
node 4 (p5c4) -> NODE_RANK=3
```

每个节点单独设置，例如：

```bash
export NODE_RANK=0
```

`STEP5_HEAD_IP` 在四个节点上相同；`VLLM_HOST_IP` 必须是当前节点自身的通信 IP。

建议每个 phase 使用新的 `STEP5_STATE_DIR`，避免 cache / state / log 混在同一目录。`STEP5_STATE_DIR` 必须位于数据盘支撑的目录，不要把模型、cache 或大日志写进节点 home。

网络相关：

```bash
export NCCL_SOCKET_IFNAME=<this-cluster-communication-interface>
export GLOO_SOCKET_IFNAME=<this-cluster-communication-interface>
```

launcher 的默认网卡名是 `eno2`；不要直接复用其他集群的网卡名或 topology file。

## 7. canonical launcher

当前 HEAD 的 canonical reproduction entrypoint 仍然是：

```text
benchmarks/step5/launch_multinode.sh
```

正式启动命令：

```bash
bash benchmarks/step5/launch_multinode.sh prefill "$NODE_RANK"

bash benchmarks/step5/launch_multinode.sh decode "$NODE_RANK"
```

rank 0 启动 API server；其他 rank 以 `--headless --data-parallel-start-rank rank*local_dp` 方式加入：

| Phase | local DP | rank 1 start-rank | rank 2 start-rank | rank 3 start-rank |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 4 | 4 | 8 | 12 |
| Decode（dp4tp4） | 1 | 1 | 2 | 3 |

launcher 在每次启动时会先清理可能从 shell 继承的旧值，包括：

```text
VLLM_STEP5_EXPERT_PLACEMENT
VLLM_HCU_DP_PACKED_AG
HCU_STEP5_ROUTER_SHARED_MAX_M
NCCL_MAX_NCHANNELS / NCCL_MIN_NCHANNELS
VLLM_PLUGINS
ROCBLAS_TENSILE_LIBPATH
HIPBLASLT_TUNING_OVERRIDE_FILE
```

因此仍需显式设置 profile 所需的 placement、packed-AG、router 和 graph 开关；不要依赖上一轮 shell 的残留环境。

任何正式运行之前先执行 dry-run：

```bash
STEP5_DRY_RUN=1 bash benchmarks/step5/launch_multinode.sh prefill "$NODE_RANK"
STEP5_DRY_RUN=1 bash benchmarks/step5/launch_multinode.sh decode "$NODE_RANK"
```

dry-run 只打印最终环境和完整启动命令，不启动模型服务。其执行顺序和检查边界见第 10 节。

## 8. Prefill reproduction

### 8.1 当前 profile inventory

| Profile | 状态 | 外部资产 | launcher 关键有效开关 |
| --- | --- | --- | --- |
| `baseline`（默认） | 历史 launcher 默认，不是 1021 measured profile | `STEP5_EP_TABLE` | `HCU_STEP5_DENSE_TUNING=0`；G2/ASM 关闭 |
| `dense_1021` | 已记录 Dense baseline | EP table + Prefill placement + tuning config | Dense tuning=1；G2=0；ASM G1=0 |
| `g2_1021` | opt-in candidate | EP table + Prefill placement + tuning config | Dense tuning=1；G2=1；ASM G1=0 |
| `asm_g1_1021` | opt-in E22 cached ASM G1 + fused G2 | EP table + Prefill placement + tuning config | Dense tuning=1；ASM G1=1；G2=1 |
| `e23_asm_1021` | opt-in fixed E23 cached ASM G1 + fused G2 | 以上 + readable static map + E=23/M65536 EP row | Dense tuning=1；ASM G1=1；G2=1；GPU_MAX_HW_QUEUES=8 |
| `g1_1021` | HISTORICAL ONLY，当前 HEAD 不接受 | — | 旧 Triton G1 specialization 已移除 |

注意：

- canonical launcher 自身的默认 `STEP5_PREFILL_PROFILE` 是 `baseline`。要复现 1021 Dense，必须显式设置 `STEP5_PREFILL_PROFILE=dense_1021`。
- `asm_g1_1021` 是当前 HEAD 的 E22 cached ASM G1 路径（`HCU_STEP5_PREFILL_ASM_G1`），**不是**已删除的 Triton G1（`HCU_STEP5_PREFILL_G1`）。不要混用这两个实现。
- 所有 1021 profile 都要求 `STEP5_PREFILL_PLACEMENT` 和 `HCU_STEP5_TUNING_CONFIG`。
- 所有 1021 profile 都要求 `STEP5_PREFILL_MOE_BACKEND=auto` 和 `STEP5_PREFILL_EP_FUSED_MOE=1`；`aiter` backend 或 fused-MoE 关闭只用于 baseline screening。
- 所有 profile 都要求 `STEP5_EP_TABLE`。
- `dense_1021` / `g2_1021` 必须保持 `STEP5_PREFILL_REDUNDANT_EXPERTS=0` 且不设置 `STEP5_PREFILL_STATIC_MAP`。launcher 的 E23 EPLB 分支由这两个变量触发，误设会把 physical experts 改成 368 并附加 `--require-m 65536`，不再是 Dense/G2 口径。

### 8.2 `dense_1021`（已记录 Dense baseline）

四节点均设置：

```bash
export STEP5_PREFILL_PROFILE=dense_1021
export STEP5_PREFILL_DP_SIZE=16
export STEP5_PREFILL_BATCHED_TOKENS=8192
export STEP5_PREFILL_MAX_MODEL_LEN=70000
export STEP5_PREFILL_GPU_MEMORY_UTILIZATION=0.875
export STEP5_PREFILL_MOE_BACKEND=auto
export STEP5_PREFILL_EP_FUSED_MOE=1
export STEP5_PREFILL_PLACEMENT=<path-to-prefill-placement.pt>
export HCU_STEP5_TUNING_CONFIG=<path-to-qualified-dense-asm-tuning-profile>
```

先 dry-run：

```bash
STEP5_DRY_RUN=1 bash benchmarks/step5/launch_multinode.sh prefill "$NODE_RANK"
```

确认无误后，每个节点分别运行：

```bash
bash benchmarks/step5/launch_multinode.sh prefill "$NODE_RANK"
```

### 8.3 `g2_1021`（opt-in G2 candidate）

只切 profile，其余 1021 条件保持不变：

```bash
export STEP5_PREFILL_PROFILE=g2_1021
```

再执行同样的 dry-run 和正式启动。`g2_1021` 是显式 opt-in candidate，不要描述为默认行为。

### 8.4 `asm_g1_1021`（E22 cached ASM G1 + fused G2）

```bash
export STEP5_PREFILL_PROFILE=asm_g1_1021
export STEP5_PREFILL_PLACEMENT=<path-to-measured-e22-prefill-placement.pt>
export HCU_STEP5_TUNING_CONFIG=<path-to-qualified-dense-asm-tuning-profile>
export STEP5_PREFILL_REDUNDANT_EXPERTS=0
unset STEP5_PREFILL_STATIC_MAP
```

launcher 会设置：

```text
HCU_STEP5_PREFILL_ASM_G1=1
HCU_STEP5_PREFILL_G2=1
GPU_MAX_HW_QUEUES=4
```

外部 tuning profile 必须包含 `asm_g1` section，且 `persist_by_local_experts["22"]` 对应记录的 E22 persist 值（记录为 256）。

### 8.5 `e23_asm_1021`（fixed E23 cached ASM G1 + fused G2）

```bash
export STEP5_PREFILL_PROFILE=e23_asm_1021
export STEP5_PREFILL_PLACEMENT=<path-to-measured-e22-prefill-placement.pt>
export HCU_STEP5_TUNING_CONFIG=<path-to-qualified-dense-asm-tuning-profile>
export STEP5_PREFILL_REDUNDANT_EXPERTS=16
export STEP5_PREFILL_STATIC_MAP=<path-to-readable-e23-static-map.json>
```

launcher 会设置：

```text
HCU_STEP5_PREFILL_ASM_G1=1
HCU_STEP5_PREFILL_G2=1
GPU_MAX_HW_QUEUES=8
--enable-eplb --eplb-config {"num_redundant_experts":16,"use_async":false,
                             "disable_rearrange":true,
                             "expert_map_path":"<static-map>",
                             "static_dispatch_policy":"token_hash"}
physical_experts=368
check_ep_table.py --require-m 65536
```

E23 的 EP table 必须包含 `E=23,N=3072,K=4096,topk=8` 且存在 `M=65536` row。外部 tuning profile 必须包含 `asm_g1.persist_by_local_experts["23"]`（记录为 320）。

E23 是显式 opt-in candidate，不是 accuracy-equivalence claim。已记录的 E23 性能与 full GSM8K 结果属于其 pinned measurement runtime；不要把它提升为默认配置。

如果当前目标只是复现 1021 Dense / G2 helper，则不需要切到 ASM/E23。

## 9. Decode reproduction

### 9.1 已记录的 `itl6479` profile（DP4/TP4）

四节点均设置：

```bash
export STEP5_DECODE_PARALLEL=dp4tp4
export STEP5_DECODE_PROFILE=itl6479
export STEP5_DECODE_GPU_MEMORY_UTILIZATION=0.85
export STEP5_DECODE_PLACEMENT=<path-to-decode-placement.pt>
export STEP5_ROUTER_SHARED_MAX_M=64
export STEP5_SHARED_STREAM_MIN_TOKENS=64
export STEP5_DECODE_EXPERT_GRID=1
export STEP5_DECODE_FUSED_ROUTER_STREAM=0
unset STEP5_DECODE_NCCL_CHANNELS
unset STEP5_PREFILL_EP_FUSED_MOE
export HCU_STEP5_DP_FABRIC=1
export HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736
export STEP5_API_HOST=0.0.0.0
export STEP5_PORT=18118
```

说明：

- `STEP5_DECODE_PROFILE=itl6479` 必须显式设置；launcher 没有默认 Decode measured profile。
- `STEP5_ROUTER_SHARED_MAX_M=64` 是记录口径；launcher 对 `itl6479` 的内部默认值是 32，所以必须显式覆盖。
- `STEP5_SHARED_STREAM_MIN_TOKENS=64` 与 launcher 默认值一致。
- `STEP5_DECODE_EXPERT_GRID=1` 是记录口径；launcher 默认值为 0。
- `STEP5_DECODE_FUSED_ROUTER_STREAM=0` 与记录口径一致；该 candidate 未提高吞吐，保持关闭。
- `STEP5_DECODE_NCCL_CHANNELS` 应 unset，让 NCCL 使用自动 channel；只有在明确实验时才固定为 1/2/4/8/16/32/64。
- `STEP5_PREFILL_EP_FUSED_MOE` 会被 launcher 同时用于 Decode 的 `VLLM_HCU_EP_FUSED_MOE`。正式 Decode 前应 unset（默认即为 1），避免从 Prefill screening 继承 `0`。

先 dry-run：

```bash
STEP5_DRY_RUN=1 bash benchmarks/step5/launch_multinode.sh decode "$NODE_RANK"
```

重点检查输出至少包含：

```text
VLLM_HCU_DP_PACKED_AG=1
HCU_STEP5_ROUTER_SHARED_MAX_M=64
HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=64
VLLM_STEP4_DSA_GRAPH_REGIONS=8736
HCU_STEP5_DENSE_TUNING=0 HCU_STEP5_PREFILL_G2=0
HCU_STEP5_PREFILL_ASM_G1=0 GPU_MAX_HW_QUEUES=4
VLLM_HCU_DP_CUSTOM_AG=0
HCU_STEP5_DP_FABRIC=1 HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736
HCU_STEP5_EXPERT_GRID=1
HCU_STEP5_FUSED_ROUTER_STREAM=0
```

然后每个节点分别运行：

```bash
bash benchmarks/step5/launch_multinode.sh decode "$NODE_RANK"
```

`HCU_STEP5_DP_FABRIC=1` 只允许用于 DP4/TP4 + `itl6479`；`HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736` 也有同样的 guard。不要把这两个配置直接迁移到其他 Decode topology。

### 9.2 备选 `dp16_native` profile（非本文档记录的 1021 Decode 口径）

当前 HEAD 还支持：

```bash
export STEP5_DECODE_PARALLEL=dp16tp1
export STEP5_DECODE_PROFILE=dp16_native
export STEP5_DECODE_GPU_MEMORY_UTILIZATION=0.91
export STEP5_DECODE_MOE_CHUNK=65536
export STEP5_DECODE_LONG_PREFILL_TOKEN_THRESHOLD=2048
```

launcher 会设置 `VLLM_STEP4_DSA_GRAPH_REGIONS=0`、`HCU_STEP5_BOUNDED_EP_WORKSPACE=1`，并使用 DP16/TP1。该 profile 不使用 `VLLM_HCU_DP_PACKED_AG` 或 `STEP5_DECODE_PLACEMENT`，也不允许 Fabric / 8736 hot-context graph。它不是本文档记录的 1021 Decode 成绩来源，使用前需要单独 live validation。

### 9.3 Guard 汇总

| 配置 | 要求 | launcher 行为 |
| --- | --- | --- |
| `itl6479` | DP4/TP4 | 设置 packed AG，要求 `STEP5_DECODE_PLACEMENT` |
| `HCU_STEP5_DP_FABRIC=1` | DP4/TP4 + `itl6479` | 设置 `VLLM_HCU_DP_CUSTOM_AG=0` |
| `HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736` | DP4/TP4 + `itl6479` | 设置 `VLLM_STEP4_DSA_GRAPH_REGIONS=8736` |
| `dp16_native` | DP16/TP1 | 设置 DSA graph regions 0、bounded workspace |
| `STEP5_DECODE_NCCL_CHANNELS` | 1/2/4/8/16/32/64 | 同时设置 NCCL min/max |

Prefill 启动会 `unset HCU_STEP5_DP_FABRIC` / `HCU_STEP5_CONTEXT_GRAPH_REGIONS`；Decode 启动会清除 Prefill Dense/G2/ASM 开关。不要跨 phase 继承候选开关。

## 10. dry-run

### 10.1 launcher 执行顺序

当前 launcher 的顺序是：

```text
1. 解析 phase 和 NODE_RANK
2. 检查 STEP5_HEAD_IP / VLLM_HOST_IP / STEP5_TOKENIZER / STEP5_STATE_DIR / STEP5_EP_TABLE
3. 设置公共默认值，清理继承的旧环境
4. 执行 phase-specific profile / topology / guard 校验
5. 组装完整 serve command
6. 如果 STEP5_DRY_RUN=1：打印有效环境和完整 command，然后 exit 0
7. 真实启动：检查 placement 文件存在性
8. 真实启动：运行 check_ep_table.py
9. 真实启动：创建 cache/state 目录并 exec 服务
```

因此 dry-run 能验证：

- profile 是否存在、是否与 topology 匹配。
- guard 是否满足（Fabric、hot-context graph、DP size、batch-token budget、context limit、GPU memory utilization 等）。
- 最终的 `STEP5_*` / `HCU_*` / `VLLM_*` 有效值和完整 CLI。

dry-run **不会**验证：

- `STEP5_EP_TABLE` 的 JSON 内容、key 或 `M=65536` row。
- placement PT 是否存在、是否属于当前 topology。
- `HCU_STEP5_TUNING_CONFIG` 的 schema、hash 或 recipe。
- E23 static map 的 JSON schema / 行内容（但 launcher 在 dry-run 前会要求该路径可读）。
- GPU、模型、native extension 是否可用。

`e23_asm_1021` 的 dry-run 需要 readable `STEP5_PREFILL_STATIC_MAP`；只有文件可读性会在 dry-run 前检查，内容由后续 model loader 校验。不要用伪造的 static map 声称 E23 artifact 已验证。

### 10.2 Prefill dry-run 检查清单

- `STEP5_PREFILL_PROFILE` 与本轮目标一致。
- TP1 / DP16 / local DP4（1021 profiles）。
- `max-num-batched-tokens=8192`。
- `max-model-len=70000`。
- `gpu-memory-utilization=0.875`。
- `VLLM_HCU_EP_FUSED_MOE=1`。
- `--long-prefill-token-threshold 0`。
- `STEP5_PREFILL_PLACEMENT` 指向正确外部 PT。
- `STEP5_EP_TABLE` 和 `HCU_STEP5_TUNING_CONFIG` 属于当前 gfx938 / 当前软件环境。
- 1021 profiles 的公共有效值：

```text
HCU_DP_MOE_PIPE_MIN_TOKENS=65536
HCU_DBO_MIN_TOKENS=4096
HCU_EP_DIRECT_SUM=65536
HCU_DP_MOE_PIPE_UNIFORM=1
EP_FUSED_MOE_CHUNK=65536
HCU_STEP5_BOUNDED_EP_WORKSPACE=1
HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
HCU_STEP5_DENSE_TUNING=1
NCCL_MAX_NCHANNELS=32
```

- profile-specific：

```text
dense_1021    HCU_STEP5_PREFILL_G2=0   HCU_STEP5_PREFILL_ASM_G1=0
g2_1021       HCU_STEP5_PREFILL_G2=1   HCU_STEP5_PREFILL_ASM_G1=0
asm_g1_1021   HCU_STEP5_PREFILL_G2=1   HCU_STEP5_PREFILL_ASM_G1=1   GPU_MAX_HW_QUEUES=4
e23_asm_1021  HCU_STEP5_PREFILL_G2=1   HCU_STEP5_PREFILL_ASM_G1=1   GPU_MAX_HW_QUEUES=8
```

### 10.3 Decode dry-run 检查清单

- DP4 / TP4 / EP16。
- `STEP5_DECODE_PROFILE=itl6479`。
- `max-num-batched-tokens=32768`。
- `max-model-len=1048576`。
- Fabric 与 hot-context graph 的两个开关都命中上述 guard。
- expert placement 为本轮 measured Decode placement。
- NCCL channel 没有被遗留环境变量偷偷固定，除非实验明确要求。
- Prefill 候选开关已清零：`HCU_STEP5_DENSE_TUNING=0`、`HCU_STEP5_PREFILL_G2=0`、`HCU_STEP5_PREFILL_ASM_G1=0`、`HCU_DP_MOE_PIPE_UNIFORM=0`。

## 11. historical 1021 helper mapping

人工 reference 目录：

```text
/public/home/liumengjie/step5-pd-服务启动脚本/
```

包含：

```text
launch_pd_common.sh
start_prefill_1021.sh
start_decode_1021.sh
使用说明.txt
```

调用关系：

```text
start_prefill_1021.sh
    -> launch_pd_common.sh prefill NODE_RANK

start_decode_1021.sh
    -> launch_pd_common.sh decode NODE_RANK
```

这些 helper 是 **historical/operator reference**，不是当前 branch 的 canonical interface。`d0061be` 曾加入 measured start scripts，`07e318f` 已 revert；不要重新把 `start_prefill_1021.sh`、`start_decode_1021.sh`、`launch_pd_common.sh` 提交进 repository。

### 11.1 helper → 当前 canonical launcher 的映射

| helper 行为 | 当前 HEAD 对应 |
| --- | --- |
| `start_prefill_1021.sh "$NODE_RANK"` 默认 `dense_1021` | `STEP5_PREFILL_PROFILE=dense_1021` + canonical launcher |
| `start_prefill_1021.sh "$NODE_RANK" g2_1021` | `STEP5_PREFILL_PROFILE=g2_1021` |
| `start_prefill_1021.sh "$NODE_RANK" g1_1021` | 当前 HEAD 不接受；旧 Triton G1 已移除。E22 ASM G1 使用 `asm_g1_1021`，两者不是同一实现 |
| `start_decode_1021.sh "$NODE_RANK"` | `STEP5_DECODE_PROFILE=itl6479` + placement + Fabric + hot-context graph |
| `launch_pd_common.sh` 要求 `STEP5_SOURCE_DIR` | 当前 launcher 可选，默认当前 checkout 根目录 |
| `launch_pd_common.sh` 默认使用 in-tree `ep_table.json` | 当前 launcher 要求外部 `STEP5_EP_TABLE` |
| helper 没有 `HCU_STEP5_TUNING_CONFIG` | 当前 1021 profiles 要求该外部 tuning profile |
| helper 内出现 `172.20.2.*`、`/opt/xuwq_*`、`/var/lib/xuwq_*` | 历史现场值，仅作来源说明；正式操作使用 `STEP5_*` |
| 同一组 GPU 一次只运行一个 phase | 与当前 launcher 一致 |

### 11.2 历史 helper 的关键默认值

`start_prefill_1021.sh` 的历史语义：

```text
STEP5_PREFILL_PROFILE=dense_1021
STEP5_PREFILL_DP_SIZE=16
STEP5_PREFILL_BATCHED_TOKENS=8192
STEP5_PREFILL_MAX_MODEL_LEN=70000
STEP5_PREFILL_GPU_MEMORY_UTILIZATION=0.875
STEP5_PREFILL_MOE_BACKEND=auto
STEP5_PREFILL_EP_FUSED_MOE=1
```

`start_decode_1021.sh` 的历史语义：

```text
STEP5_DECODE_PARALLEL=dp4tp4
STEP5_DECODE_PROFILE=itl6479
STEP5_DECODE_GPU_MEMORY_UTILIZATION=0.85
STEP5_ROUTER_SHARED_MAX_M=64
STEP5_SHARED_STREAM_MIN_TOKENS=64
STEP5_DECODE_EXPERT_GRID=1
STEP5_DECODE_FUSED_ROUTER_STREAM=0
HCU_STEP5_DP_FABRIC=1
HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736
```

这些值用于说明历史操作，不替代当前 launcher 的 guard 和外部资产合同。

## 12. benchmark / evidence contract

仓库保留 server-side reproduction entrypoint 和结果 evidence，但原始 benchmark client、数据集及部分外部资产不随源码提交。

当前 HEAD 的主要 Step5 evidence：

| 文件 | 内容 |
| --- | --- |
| `benchmarks/step5/results-pd-image1021-20261023.json` | Dense / G2 / 历史 Decode Fabric 结果 |
| `benchmarks/step5/results-pd-image1021-20261024.json` | fresh consolidated-source Decode、E22/E23 Prefill、accuracy |
| `benchmarks/step5/results-full-gsm8k-image1021-20261024.json` | 1319-example full GSM8K 对比 |
| `benchmarks/step5/results-deepep-e23-20261026.json` | 单独的 DeepEP installed-wheel Prefill 路径 |
| `benchmarks/step5/results-decode-*.json` | 历史 Decode profile / candidate 记录 |
| `benchmarks/step5/results-prefill-*.json` | 历史 Prefill profile / candidate 记录 |

正式性能数据必须沿用冻结的 benchmark client / dataset / workload 口径，不要仅凭服务启动成功重写历史 throughput。

每次正式记录至少包括：

```text
source commit / source snapshot hash
container image tag + image ID
torch / vllm / aiter / plugin package versions
model / tokenizer
placement / static map / tuning artifact hash
phase profile
parallel topology (TP / DP / EP / local DP)
input / output length
concurrency
max-num-batched-tokens / max-model-len
warmup 轮数
formal-round throughput
ITL / TTFT 等对应指标
request completion / error / preemption
JIT / cache update 审计
```

至少三轮 clean formal rounds、完整 warmup、每轮后检查 JIT/cache update，才能形成可比较的 performance record。

机器可读 evidence 的目录：

```text
docs/step5_pd_optimizations_20261023.md
docs/step5_pd_verification_20261024.md
docs/step5_prefill_asm_g1.md
docs/step5_prefill_g2.md
docs/step5_full_gsm8k_20261024.md
```

## 13. 结果如何解释

以下数字均为 **historical measured result**，属于其 pinned measurement runtime。运行本 Cookbook 不保证得到相同数值。

| Phase / variant | Topology | Throughput | ITL | Evidence |
| --- | --- | ---: | ---: | --- |
| Prefill Dense baseline | DP16/TP1/EP16 | 41,580.14 tokens/s | — | `results-pd-image1021-20261023.json` |
| Prefill G2 candidate（G1 off） | DP16/TP1/EP16 | 42,224.93 tokens/s | — | `results-pd-image1021-20261023.json` |
| Prefill E22 cached ASM G1 + fused G2 | DP16/TP1/EP16 | 42,565.786 tokens/s | — | `results-pd-image1021-20261024.json` |
| Prefill E23 fixed cached ASM G1 + fused G2 | DP16/TP1/EP16 | 42,666.451 tokens/s | — | `results-pd-image1021-20261024.json` |
| Decode fresh consolidated-source | DP4/TP4/EP16 | 3,177.991 tokens/s | 56.441650 ms | `results-pd-image1021-20261024.json` |
| Decode historical Fabric run | DP4/TP4/EP16 | 3,149.197 tokens/s | 57.038124 ms | `results-pd-image1021-20261023.json` |

Full GSM8K（1319 examples，standard MTP3，single stochastic pass）：

| Profile | Legacy evaluator | Request errors |
| --- | ---: | ---: |
| Decode | 1273/1319 (96.513%) | 0 |
| Prefill E22 | 1275/1319 (96.664%) | 0 |
| Prefill E23 | 1272/1319 (96.437%) | 0 |

解释注意事项：

- **ITL 是原 benchmark 的 streaming chunk interval**。MTP chunk 可能包含多个 output token，因此不能描述成 per-token TPOT。
- Prefill 的 throughput 是 65536 input / 1 output、C64、8192 batched-token 口径。
- Decode 的 throughput 是 65536 input / 4096 output、C64、32768 batched-token 口径。
- 这些结果用于 reproduction comparison，不是“当前 checkout 一定能得到”的硬编码验收值。
- E22/E23 的 local operator 加速、complete-model throughput 和 GSM8K accuracy 是不同层级的 evidence，不能互相替代。
- Full GSM8K 是单次 stochastic pass；同一 source/profile 的 100-example 样本也曾出现 97/100 与 98/100 的差异。不要把它当成 numerical equivalence 证明。
- E23 的 full GSM8K 结果来自其 pinned inference code；E23 仍是 opt-in candidate，不是默认配置。
- 45000 Prefill tokens/s 目标未达到；Decode ITL < 60ms 目标在记录中已由 fresh run 达到，但仍是 pinned runtime 的结果。

## 14. troubleshooting

### `STEP5_EP_TABLE` 未设置

当前 launcher 故意要求显式外部 MoE table：

```text
Set an external tuned MoE table; machine tuning data is not shipped in the source
```

不要从旧 checkout 随意复制一份 JSON 当成当前配置。先用 `check_ep_table.py` 检查 key、pair 和 `M=65536` row。

### `HCU_STEP5_TUNING_CONFIG` 未设置

1021 Dense/G2/ASM/E23 Prefill reproduction 都需要对应的 qualified Dense/ASM tuning profile。缺失时 launcher 在 profile guard 阶段退出。检查：

- `schema_version=1`；
- `architecture=gfx938`；
- `dense.hipblaslt_sha256` 与当前 library 一致；
- ASM profile 含 `asm_g1.block_m=128`、`solution`、对应 local expert 的 persist 值。

### `STEP5_PREFILL_PROFILE=g1_1021` 报 Unsupported

当前 HEAD 已移除 Triton G1 和 `g1_1021`。E22 cached ASM G1 使用 `asm_g1_1021`，但不要在文档或结果中把它与旧 Triton G1 当作同一实现。

### placement 文件不存在

`itl6479` Decode 和所有 1021 Prefill profile 都依赖外部 placement。dry-run 不检查 `-f`；真实启动在 dry-run 之后检查，缺失时退出。确认路径存在、文件属于目标 topology，并核对 evidence 中的 hash。

### E23 static map 不可读

`e23_asm_1021` 在 dry-run 前就要求 `STEP5_PREFILL_STATIC_MAP` 可读。真实启动还会校验 version-2 offline EPLB schema、88 × 368 行、352 logical experts 和 `record_kind=proposal`。不要用空 JSON 或旧 map 冒充当前 artifact。

### source 与已编译 `hcu_ops` 不匹配

如果通过 `STEP5_SOURCE_DIR` 切换源码，必须确保 Python source 与匹配的已编译 native extension / DTK / torch 环境一致。不要把“Python 文件能 import”当成 native ABI 已匹配。

### Dense 路径在 startup 报 hipBLASLt hash 不一致

`HCU_STEP5_TUNING_CONFIG.dense.hipblaslt_sha256` 必须匹配当前加载的 hipBLASLt library。换镜像或 library 后需要重新 tuning；不要复用旧 hash。

### ZMQ Unix socket 路径过长

launcher 默认使用较短 `TMPDIR`（`/var/tmp/xuwq_step5`），因为 ZMQ 会在路径后追加 UUID（107-byte path limit）。不要把 `STEP5_TMP_DIR` 指向层级很深的目录。

### NCCL / GLOO 网络错误

确认 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 对应当前机器实际通信网卡。launcher 默认 `eno2`；不要直接复用其他集群的网卡名或 topology file。

### Prefill 与 Decode 相互污染

进入另一个 phase 前先完整停止前一阶段。使用新的 `STEP5_STATE_DIR`，并检查遗留进程、端口和环境变量。launcher 会清理部分继承变量，但不会替代人工确认。

### synthetic MTP 被误用于质量测试

launcher 固定 synthetic acceptance 是性能测试口径。GSM8K / 质量检查必须使用 separate standard-MTP run。不要把 synthetic-MTP 的 throughput 或 acceptance 当作 accuracy evidence。

## 15. known limitations

- 本文档的验证是 **static double check**：已核对 current HEAD source、launcher dry-run、guard 和 evidence；没有在 4-node / 16-GPU 环境执行 live reproduction。
- `LIVE_16GPU_REPRODUCTION=NOT_RUN`，`LIVE_RUNTIME_DOUBLE_CHECK=PENDING`。
- `max-model-len=1048576` 是配置上限；记录的 hot graph 覆盖 69888 context tokens，实际 1M-token end-to-end workload 未运行。
- E22 / E23 ASM profile、E23 static map、DeepEP 路径都需要各自的 external artifacts；没有这些 artifact 不能完成 live validation。
- E23 仍是 opt-in candidate；E23 complete-model quality 和 performance 不是默认配置，也不构成 accuracy equivalence。
- Full GSM8K 是单次 stochastic pass，不建立 full-dataset equivalence 或 cross-run reproducibility。
- 历史 throughput / ITL / accuracy 只属于其 pinned measurement runtime；本文档不声称当前 HEAD 能复现相同数值。
- 本文档不覆盖 online P/D disaggregated serving、proxy、KV transfer 或 `kv_producer` / `kv_consumer` 部署。
- 当前 HEAD 另有单独的 DeepEP installed-wheel Prefill 路径（`docs/step5_deepep_installed_reproduction.md`、`benchmarks/step5/serve_deepep_prefill.py`）。它使用不同的入口和资产 manifest，不属于本文档的 1021 `launch_multinode.sh` 复现路径；其 45,621.7 / 45,575.4 tokens/s 记录不能被当作 1021 Dense/G2/ASM profile 的成绩。
- `dp16_native` Decode profile 是当前 HEAD 的备选 topology，不是本文档记录的 DP4/TP4 `itl6479` 成绩来源。
- 不修改 runtime、kernel、MoE path、communication、scheduler、graph semantics 或 benchmark workload 来迁就本文档；source 与文档冲突时以 source 为准。

## 16. provenance

```text
repository: WenQ7/vllm-plugin-das
branch:     codex/step5-pd-optimizations-20261023
calibrated: 857d114a56846b7a8f6831592bc735adc9923510 (2026-10-26)
scope:      Step5 phase-isolated Prefill / Decode optimization reproduction
```

主要 source / evidence：

```text
benchmarks/step5/launch_multinode.sh
benchmarks/step5/check_ep_table.py
benchmarks/step5/results-pd-image1021-20261023.json
benchmarks/step5/results-pd-image1021-20261024.json
benchmarks/step5/results-full-gsm8k-image1021-20261024.json
docs/step5_pd_optimizations_20261023.md
docs/step5_pd_verification_20261024.md
docs/step5_prefill_asm_g1.md
docs/step5_prefill_g2.md
docs/step5_full_gsm8k_20261024.md
vllm_hcu/ops/step5/tuning.py
vllm_hcu/ops/step5/prefill_asm_g1.py
vllm_hcu/model_executor/layers/fused_moe/static_eplb.py
```

人工 reference：

```text
/public/home/liumengjie/step5-pd-服务启动脚本/launch_pd_common.sh
/public/home/liumengjie/step5-pd-服务启动脚本/start_prefill_1021.sh
/public/home/liumengjie/step5-pd-服务启动脚本/start_decode_1021.sh
/public/home/liumengjie/step5-pd-服务启动脚本/使用说明.txt
```

这些 reference 只作为 historical/operator reference；当前 repository contract 以同一 commit 的 launcher、source 和 verification/evidence 为准。旧 standalone helper 不重新提交，machine tuning JSON、模型权重、placement PT、static map、native binary、dataset 和 private path artifact 都不进入 Git。

本文档的 static validation 包括：

```text
current remote HEAD checked
candidate cookbook read
four reference files read
launcher / source / evidence cross-checked
launcher dry-run for Prefill dense_1021 / g2_1021 / asm_g1_1021
e23_asm_1021 launcher-contract dry-run (readable static-map path only; content not validated)
launcher dry-run for Decode itl6479 rank 0..3
negative guard checks (profile/topology/guard failures)
git diff --check
```

状态：

```text
COOKBOOK_STATIC_DOUBLE_CHECK=PASS
LIVE_RUNTIME_DOUBLE_CHECK=PENDING
```
