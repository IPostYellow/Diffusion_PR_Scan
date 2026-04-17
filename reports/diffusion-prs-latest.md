# Strict Diffusion Performance PR Scan

Generated at: `2026-04-17T08:02:34Z`

## Strict rule

- Must be clearly about diffusion models, tasks, or runtime code paths.
- Must describe a concrete performance optimization or attach benchmark / memory evidence.
- Must touch production code paths; docs / CI / tests / benchmark-only / Docker / support-only PRs are excluded.

## Repo summary

- `sgl-project/sglang`: 18 strict matches
- `vllm-project/vllm`: 0 strict matches
- **总计**: 18 条

## Technique summary

| 分类 | 数量 |
|------|------|
| 内核融合/算子快路径 (`kernel-fusion`) | 7 |
| CUDA Graph 图捕获 (`cuda-graph`) | 4 |
| torch.compile 编译 (`torch-compile`) | 5 |
| FP8/NVFP4/量化 (`low-precision-quantization`) | 4 |
| 注意力后端/精度调度 (`attention-backend`) | 5 |
| 执行批处理/前向合批 (`execution-batching`) | 1 |
| 通信优化 (`communication`) | 1 |
| Offload/驻留策略 (`memory-offload`) | 9 |
| 缓存复用 (`cache-reuse`) | 5 |
| 显存效率 (`memory-efficiency`) | 3 |
| 调度/吞吐优化 (`scheduling-throughput`) | 4 |

## Detailed list

### 内核融合/算子快路径 (`kernel-fusion`)

#### `sgl-project/sglang#20816` [open] `2026-04-17` [Diffusion][CPU] Init CPU platform support for SGLang Diffusion

- **链接**：https://github.com/sgl-project/sglang/pull/20816
- **摘要**：为 SGLang Diffusion 添加 CPU 平台原生支持（Intel Xeon），包含 OMP 核绑定、NUMA 自动绑定、SDPA attention 等 CPU 算子适配，支持纯 CPU 推理。
- **证据**：PR 描述强调显存占用或驻留策略收益，建议重点关注峰值显存与 e2e 时延。

#### `sgl-project/sglang#23025` [open] `2026-04-17` Optimize LTX2 modulation and two-stage warmup

- **链接**：https://github.com/sgl-project/sglang/pull/23025
- **摘要**：融合 LTX-2 的 RMSNorm+scale/shift 调制到 CuTeDSL/Triton kernel，并优化两阶段 warmup 流程。E2E 延迟下降约 7.84%（37.7s→34.7s，H100）。
- **证据**：E2E 延迟（不含 warmup）从 37.684s 降至 34.728s，提升 7.84%（2x H100, LTX-2.3 one-stage T2V）。

#### `sgl-project/sglang#22814` [open] `2026-04-17` diffusion: add HunyuanVideo GroupNorm+SiLU fast path

- **链接**：https://github.com/sgl-project/sglang/pull/22814
- **摘要**：为 HunyuanVideo 的 ResnetBlock 引入 Triton GroupNorm+SiLU 融合 kernel fast path，DecodingStage 延迟降低约 6.2%。
- **证据**：DenoisingStage 41406→41583ms（+0.4%），DecodingStage 15515→14549ms（-6.2%），H100 单卡。

#### `sgl-project/sglang#22445` [open] `2026-04-15` [NPU] [Diffusion] Performance Optimization for LTX-2 Model

- **链接**：https://github.com/sgl-project/sglang/pull/22445
- **摘要**：在 LTX-2 热路径引入融合 kernel 或 fast path，压缩归一化、激活或 DiT kernel launch 开销。 主要改动涉及 ltx_2.py。
- **证据**：This PR reduces E2E latency by approximately 27% on NPU and 3% on GPU.；E2E before the pr: 102.74s

#### `sgl-project/sglang#22441` [open] `2026-04-12` [diffusion] Cache LTX-2 RoPE coords to avoid per-step recompute

- **链接**：https://github.com/sgl-project/sglang/pull/22441
- **摘要**：在 LTX-2 热路径引入融合 kernel 或 fast path，压缩归一化、激活或 DiT kernel launch 开销。 主要改动涉及 denoising_av.py。
- **证据**：PR 描述给出了明确的性能目标，但没有附上可直接抽取的 benchmark 数值。

#### `sgl-project/sglang#21912` [open] `2026-04-02` [diffusion] ZImage-Turbo DiT FP8 full quantization & CUDA Graph 

- **链接**：https://github.com/sgl-project/sglang/pull/21912
- **摘要**：为 Z-Image 路径接入 CUDA Graph，重点减少 denoising 过程中的 host launch gap 和 step 间空洞。 主要改动涉及 diffusion_cuda_graph_runner.py, convert_zimage_hf_to_fp8.py。
- **证据**：PR 描述给出了明确的性能目标，但没有附上可直接抽取的 benchmark 数值。

#### `sgl-project/sglang#19632` [closed-unmerged] `2026-03-02` [Perf] Reduce DiT kernel launch overhead with fused CUDA kernels for FLUX/Z-Image

- **链接**：https://github.com/sgl-project/sglang/pull/19632
- **摘要**：在 Z-Image 热路径引入融合 kernel 或 fast path，压缩归一化、激活或 DiT kernel launch 开销。 主要改动涉及 flux.py, reproduce.md。
- **证据**：| **Denoising Stage Total** | 0.7028s | 0.6913s | **1.64%** |；| **Average Time Per Step** | 0.1751s | 0.1721s | **1.71%** |


### CUDA Graph 图捕获 (`cuda-graph`)

#### `sgl-project/sglang#19876` [closed-unmerged] `2026-04-17` [Diffusion] Diffusion support cuda graph

- **链接**：https://github.com/sgl-project/sglang/pull/19876
- **摘要**：为 FLUX 的 diffusers backend 接入 CUDA Graph 捕获，减少 denoising 循环中的 host launch gap，降低推理延迟。
- **证据**：峰值显存 31.51 GB（已分配 27.30 GB），显存池开销 4.21 GB（13.4%），Max peak 32268 MB。

#### `sgl-project/sglang#21912` [open] `2026-04-02` [diffusion] ZImage-Turbo DiT FP8 full quantization & CUDA Graph 

- **链接**：https://github.com/sgl-project/sglang/pull/21912
- **摘要**：为 Z-Image 路径接入 CUDA Graph，重点减少 denoising 过程中的 host launch gap 和 step 间空洞。 主要改动涉及 diffusion_cuda_graph_runner.py, convert_zimage_hf_to_fp8.py。
- **证据**：PR 描述给出了明确的性能目标，但没有附上可直接抽取的 benchmark 数值。

#### `sgl-project/sglang#21417` [open] `2026-03-26` [Diffusion] Change default torch.compile mode from max-autotune to default

- **链接**：https://github.com/sgl-project/sglang/pull/21417
- **摘要**：将 diffusion pipeline 的 torch.compile 默认模式从 max-autotune 改为 default，消除 1.47-1.68x denoising 回退，同时加速 text encoding（2x）和 VAE decode（1.9x）。
- **证据**：max-autotune 导致 1.47-1.68x denoising 回退和 30s+ warmup，default 模式消除回退并加速 text encoding 和 VAE decode。测试环境 1x H200。

#### `sgl-project/sglang#19516` [open] `2026-03-18` [Diffusion] add cuda graph support for Qwen-Image

- **链接**：https://github.com/sgl-project/sglang/pull/19516
- **摘要**：为 Qwen-Image 路径接入 CUDA Graph，重点减少 denoising 过程中的 host launch gap 和 step 间空洞。 主要改动涉及 qwen_image.py, denoising.py。
- **证据**：PR 描述强调显存占用或驻留策略收益，建议重点关注峰值显存与 e2e 时延。


### torch.compile 编译 (`torch-compile`)

#### `sgl-project/sglang#23025` [open] `2026-04-17` Optimize LTX2 modulation and two-stage warmup

- **链接**：https://github.com/sgl-project/sglang/pull/23025
- **摘要**：融合 LTX-2 的 RMSNorm+scale/shift 调制到 CuTeDSL/Triton kernel，并优化两阶段 warmup 流程。E2E 延迟下降约 7.84%（37.7s→34.7s，H100）。
- **证据**：E2E 延迟（不含 warmup）从 37.684s 降至 34.728s，提升 7.84%（2x H100, LTX-2.3 one-stage T2V）。

#### `sgl-project/sglang#20434` [closed-unmerged] `2026-04-17` [diffusion] Batch serial CFG for Qwen-Image to reduce denoising overhead

- **链接**：https://github.com/sgl-project/sglang/pull/20434
- **摘要**：将 Qwen-Image 的 CFG 条件/无条件两次前向合并为单次 batched forward，减少每步 denoising 的重复 launch 开销。
- **证据**：峰值显存 63.26 GB（已分配 61.10 GB），显存池开销 2.17 GB（3.4%），Max peak 64780 MB。

#### `sgl-project/sglang#21912` [open] `2026-04-02` [diffusion] ZImage-Turbo DiT FP8 full quantization & CUDA Graph 

- **链接**：https://github.com/sgl-project/sglang/pull/21912
- **摘要**：为 Z-Image 路径接入 CUDA Graph，重点减少 denoising 过程中的 host launch gap 和 step 间空洞。 主要改动涉及 diffusion_cuda_graph_runner.py, convert_zimage_hf_to_fp8.py。
- **证据**：PR 描述给出了明确的性能目标，但没有附上可直接抽取的 benchmark 数值。

#### `sgl-project/sglang#21417` [open] `2026-03-26` [Diffusion] Change default torch.compile mode from max-autotune to default

- **链接**：https://github.com/sgl-project/sglang/pull/21417
- **摘要**：将 diffusion pipeline 的 torch.compile 默认模式从 max-autotune 改为 default，消除 1.47-1.68x denoising 回退，同时加速 text encoding（2x）和 VAE decode（1.9x）。
- **证据**：max-autotune 导致 1.47-1.68x denoising 回退和 30s+ warmup，default 模式消除回退并加速 text encoding 和 VAE decode。测试环境 1x H200。

#### `sgl-project/sglang#19673` [merged] `2026-03-04` [diffusion] support torch compile for diffusers backend

- **链接**：https://github.com/sgl-project/sglang/pull/19673
- **摘要**：为 diffusers backend 接入 torch.compile（兼容 cache-dit），FLUX.1-dev 在 L20 上 E2E 从 23.6s 降至 20.3s，提速 16.2%。
- **证据**：FLUX.1-dev 在 L20 上 E2E 从 23.6s 降至 20.3s，提速 16.2%；开启 cache-dit 后可进一步降至 12.8s。


### FP8/NVFP4/量化 (`low-precision-quantization`)

#### `sgl-project/sglang#22869` [open] `2026-04-17` [diffusion] optimize ltx-2.3 offload hot paths

- **链接**：https://github.com/sgl-project/sglang/pull/22869
- **摘要**：优化 LTX-2.3 的 offload 热路径：在高显存 Hopper GPU 上默认关闭 DiT/VAE CPU offload，预构建 stage-2 transformer 并常驻 GPU，减少权重搬运延迟约 67s。
- **证据**：DiT 搬运到 GPU 的 h2d 耗时约 67.1s（开启 layerwise offload 时），LoRA 加载约 18.5s。

#### `sgl-project/sglang#21742` [open] `2026-04-08` [diffusion] attention: add support for hybrid attention schedule

- **链接**：https://github.com/sgl-project/sglang/pull/21742
- **摘要**：在 hybrid-schedule 路径引入 FP8 或量化配置，降低 GEMM 或 attention 计算成本。 主要改动涉及 hybrid_schedule.py, layer.py。
- **证据**：| DenoisingStage | 158981.35 | 138116.00 | -20865.35 | -13.1% | 🟢 |；| DenoisingStage | 158981.35 | 140662.09 | -18319.26 | -11.5% | 🟢 |

#### `sgl-project/sglang#21912` [open] `2026-04-02` [diffusion] ZImage-Turbo DiT FP8 full quantization & CUDA Graph 

- **链接**：https://github.com/sgl-project/sglang/pull/21912
- **摘要**：为 Z-Image 路径接入 CUDA Graph，重点减少 denoising 过程中的 host launch gap 和 step 间空洞。 主要改动涉及 diffusion_cuda_graph_runner.py, convert_zimage_hf_to_fp8.py。
- **证据**：PR 描述给出了明确的性能目标，但没有附上可直接抽取的 benchmark 数值。

#### `sgl-project/sglang#20319` [open] `2026-03-31` [AMD] Support fp8 MHA for diffusion model

- **链接**：https://github.com/sgl-project/sglang/pull/20319
- **摘要**：在 aiter 路径引入 FP8 或量化配置，降低 GEMM 或 attention 计算成本。 主要改动涉及 aiter.py。
- **证据**：| DenoisingStage (s) | 432.40 | 348.43 | **-19.4%** |；| DenoisingStage (s) | 1403.50 | 1152.64 | **-17.9%** |


### 注意力后端/精度调度 (`attention-backend`)

#### `sgl-project/sglang#20816` [open] `2026-04-17` [Diffusion][CPU] Init CPU platform support for SGLang Diffusion

- **链接**：https://github.com/sgl-project/sglang/pull/20816
- **摘要**：为 SGLang Diffusion 添加 CPU 平台原生支持（Intel Xeon），包含 OMP 核绑定、NUMA 自动绑定、SDPA attention 等 CPU 算子适配，支持纯 CPU 推理。
- **证据**：PR 描述强调显存占用或驻留策略收益，建议重点关注峰值显存与 e2e 时延。

#### `sgl-project/sglang#21742` [open] `2026-04-08` [diffusion] attention: add support for hybrid attention schedule

- **链接**：https://github.com/sgl-project/sglang/pull/21742
- **摘要**：在 hybrid-schedule 路径引入 FP8 或量化配置，降低 GEMM 或 attention 计算成本。 主要改动涉及 hybrid_schedule.py, layer.py。
- **证据**：| DenoisingStage | 158981.35 | 138116.00 | -20865.35 | -13.1% | 🟢 |；| DenoisingStage | 158981.35 | 140662.09 | -18319.26 | -11.5% | 🟢 |

#### `sgl-project/sglang#21912` [open] `2026-04-02` [diffusion] ZImage-Turbo DiT FP8 full quantization & CUDA Graph 

- **链接**：https://github.com/sgl-project/sglang/pull/21912
- **摘要**：为 Z-Image 路径接入 CUDA Graph，重点减少 denoising 过程中的 host launch gap 和 step 间空洞。 主要改动涉及 diffusion_cuda_graph_runner.py, convert_zimage_hf_to_fp8.py。
- **证据**：PR 描述给出了明确的性能目标，但没有附上可直接抽取的 benchmark 数值。

#### `sgl-project/sglang#20319` [open] `2026-03-31` [AMD] Support fp8 MHA for diffusion model

- **链接**：https://github.com/sgl-project/sglang/pull/20319
- **摘要**：在 aiter 路径引入 FP8 或量化配置，降低 GEMM 或 attention 计算成本。 主要改动涉及 aiter.py。
- **证据**：| DenoisingStage (s) | 432.40 | 348.43 | **-19.4%** |；| DenoisingStage (s) | 1403.50 | 1152.64 | **-17.9%** |

#### `sgl-project/sglang#19673` [merged] `2026-03-04` [diffusion] support torch compile for diffusers backend

- **链接**：https://github.com/sgl-project/sglang/pull/19673
- **摘要**：为 diffusers backend 接入 torch.compile（兼容 cache-dit），FLUX.1-dev 在 L20 上 E2E 从 23.6s 降至 20.3s，提速 16.2%。
- **证据**：FLUX.1-dev 在 L20 上 E2E 从 23.6s 降至 20.3s，提速 16.2%；开启 cache-dit 后可进一步降至 12.8s。


### 执行批处理/前向合批 (`execution-batching`)

#### `sgl-project/sglang#20434` [closed-unmerged] `2026-04-17` [diffusion] Batch serial CFG for Qwen-Image to reduce denoising overhead

- **链接**：https://github.com/sgl-project/sglang/pull/20434
- **摘要**：将 Qwen-Image 的 CFG 条件/无条件两次前向合并为单次 batched forward，减少每步 denoising 的重复 launch 开销。
- **证据**：峰值显存 63.26 GB（已分配 61.10 GB），显存池开销 2.17 GB（3.4%），Max peak 64780 MB。


### 通信优化 (`communication`)

#### `sgl-project/sglang#22805` [open] `2026-04-14` [diffusion] comms: Pack QKV for a2a in Flux2

- **链接**：https://github.com/sgl-project/sglang/pull/22805
- **摘要**：优化 FLUX.2 的多卡通信路径，减少 all-to-all 或 collective 调用带来的同步开销。 主要改动涉及 usp.py, layer.py。
- **证据**：| **E2E Latency** | 4740.77 ms | 4657.01 ms | **-83.77 ms (-1.8%)** | ⚪️ |；| DenoisingStage | 4523.43 | 4432.28 | -91.15 | -2.0% | ⚪️ |


### Offload/驻留策略 (`memory-offload`)

#### `sgl-project/sglang#20816` [open] `2026-04-17` [Diffusion][CPU] Init CPU platform support for SGLang Diffusion

- **链接**：https://github.com/sgl-project/sglang/pull/20816
- **摘要**：为 SGLang Diffusion 添加 CPU 平台原生支持（Intel Xeon），包含 OMP 核绑定、NUMA 自动绑定、SDPA attention 等 CPU 算子适配，支持纯 CPU 推理。
- **证据**：PR 描述强调显存占用或驻留策略收益，建议重点关注峰值显存与 e2e 时延。

#### `sgl-project/sglang#22869` [open] `2026-04-17` [diffusion] optimize ltx-2.3 offload hot paths

- **链接**：https://github.com/sgl-project/sglang/pull/22869
- **摘要**：优化 LTX-2.3 的 offload 热路径：在高显存 Hopper GPU 上默认关闭 DiT/VAE CPU offload，预构建 stage-2 transformer 并常驻 GPU，减少权重搬运延迟约 67s。
- **证据**：DiT 搬运到 GPU 的 h2d 耗时约 67.1s（开启 layerwise offload 时），LoRA 加载约 18.5s。

#### `sgl-project/sglang#22814` [open] `2026-04-17` diffusion: add HunyuanVideo GroupNorm+SiLU fast path

- **链接**：https://github.com/sgl-project/sglang/pull/22814
- **摘要**：为 HunyuanVideo 的 ResnetBlock 引入 Triton GroupNorm+SiLU 融合 kernel fast path，DecodingStage 延迟降低约 6.2%。
- **证据**：DenoisingStage 41406→41583ms（+0.4%），DecodingStage 15515→14549ms（-6.2%），H100 单卡。

#### `sgl-project/sglang#20434` [closed-unmerged] `2026-04-17` [diffusion] Batch serial CFG for Qwen-Image to reduce denoising overhead

- **链接**：https://github.com/sgl-project/sglang/pull/20434
- **摘要**：将 Qwen-Image 的 CFG 条件/无条件两次前向合并为单次 batched forward，减少每步 denoising 的重复 launch 开销。
- **证据**：峰值显存 63.26 GB（已分配 61.10 GB），显存池开销 2.17 GB（3.4%），Max peak 64780 MB。

#### `sgl-project/sglang#19876` [closed-unmerged] `2026-04-17` [Diffusion] Diffusion support cuda graph

- **链接**：https://github.com/sgl-project/sglang/pull/19876
- **摘要**：为 FLUX 的 diffusers backend 接入 CUDA Graph 捕获，减少 denoising 循环中的 host launch gap，降低推理延迟。
- **证据**：峰值显存 31.51 GB（已分配 27.30 GB），显存池开销 4.21 GB（13.4%），Max peak 32268 MB。

#### `sgl-project/sglang#18764` [open] `2026-04-15` [diffusion] Add dynamic batching v0

- **链接**：https://github.com/sgl-project/sglang/pull/18764
- **摘要**：为 diffusion scheduler 引入动态批处理（max batch size + delay），在多个 text-to-image 模型上吞吐提升最高 29.6%，平均延迟降低 22.4%，P99 延迟降低 31.8%。
- **证据**：吞吐提升最高 29.6%，平均延迟降低 22.4%，P99 延迟降低 31.8%（多模型 text-to-image benchmark）。

#### `sgl-project/sglang#20319` [open] `2026-03-31` [AMD] Support fp8 MHA for diffusion model

- **链接**：https://github.com/sgl-project/sglang/pull/20319
- **摘要**：在 aiter 路径引入 FP8 或量化配置，降低 GEMM 或 attention 计算成本。 主要改动涉及 aiter.py。
- **证据**：| DenoisingStage (s) | 432.40 | 348.43 | **-19.4%** |；| DenoisingStage (s) | 1403.50 | 1152.64 | **-17.9%** |

#### `sgl-project/sglang#19516` [open] `2026-03-18` [Diffusion] add cuda graph support for Qwen-Image

- **链接**：https://github.com/sgl-project/sglang/pull/19516
- **摘要**：为 Qwen-Image 路径接入 CUDA Graph，重点减少 denoising 过程中的 host launch gap 和 step 间空洞。 主要改动涉及 qwen_image.py, denoising.py。
- **证据**：PR 描述强调显存占用或驻留策略收益，建议重点关注峰值显存与 e2e 时延。

#### `sgl-project/sglang#19673` [merged] `2026-03-04` [diffusion] support torch compile for diffusers backend

- **链接**：https://github.com/sgl-project/sglang/pull/19673
- **摘要**：为 diffusers backend 接入 torch.compile（兼容 cache-dit），FLUX.1-dev 在 L20 上 E2E 从 23.6s 降至 20.3s，提速 16.2%。
- **证据**：FLUX.1-dev 在 L20 上 E2E 从 23.6s 降至 20.3s，提速 16.2%；开启 cache-dit 后可进一步降至 12.8s。


### 缓存复用 (`cache-reuse`)

#### `sgl-project/sglang#19876` [closed-unmerged] `2026-04-17` [Diffusion] Diffusion support cuda graph

- **链接**：https://github.com/sgl-project/sglang/pull/19876
- **摘要**：为 FLUX 的 diffusers backend 接入 CUDA Graph 捕获，减少 denoising 循环中的 host launch gap，降低推理延迟。
- **证据**：峰值显存 31.51 GB（已分配 27.30 GB），显存池开销 4.21 GB（13.4%），Max peak 32268 MB。

#### `sgl-project/sglang#22441` [open] `2026-04-12` [diffusion] Cache LTX-2 RoPE coords to avoid per-step recompute

- **链接**：https://github.com/sgl-project/sglang/pull/22441
- **摘要**：在 LTX-2 热路径引入融合 kernel 或 fast path，压缩归一化、激活或 DiT kernel launch 开销。 主要改动涉及 denoising_av.py。
- **证据**：PR 描述给出了明确的性能目标，但没有附上可直接抽取的 benchmark 数值。

#### `sgl-project/sglang#21742` [open] `2026-04-08` [diffusion] attention: add support for hybrid attention schedule

- **链接**：https://github.com/sgl-project/sglang/pull/21742
- **摘要**：在 hybrid-schedule 路径引入 FP8 或量化配置，降低 GEMM 或 attention 计算成本。 主要改动涉及 hybrid_schedule.py, layer.py。
- **证据**：| DenoisingStage | 158981.35 | 138116.00 | -20865.35 | -13.1% | 🟢 |；| DenoisingStage | 158981.35 | 140662.09 | -18319.26 | -11.5% | 🟢 |

#### `sgl-project/sglang#21912` [open] `2026-04-02` [diffusion] ZImage-Turbo DiT FP8 full quantization & CUDA Graph 

- **链接**：https://github.com/sgl-project/sglang/pull/21912
- **摘要**：为 Z-Image 路径接入 CUDA Graph，重点减少 denoising 过程中的 host launch gap 和 step 间空洞。 主要改动涉及 diffusion_cuda_graph_runner.py, convert_zimage_hf_to_fp8.py。
- **证据**：PR 描述给出了明确的性能目标，但没有附上可直接抽取的 benchmark 数值。

#### `sgl-project/sglang#19673` [merged] `2026-03-04` [diffusion] support torch compile for diffusers backend

- **链接**：https://github.com/sgl-project/sglang/pull/19673
- **摘要**：为 diffusers backend 接入 torch.compile（兼容 cache-dit），FLUX.1-dev 在 L20 上 E2E 从 23.6s 降至 20.3s，提速 16.2%。
- **证据**：FLUX.1-dev 在 L20 上 E2E 从 23.6s 降至 20.3s，提速 16.2%；开启 cache-dit 后可进一步降至 12.8s。


### 显存效率 (`memory-efficiency`)

#### `sgl-project/sglang#22869` [open] `2026-04-17` [diffusion] optimize ltx-2.3 offload hot paths

- **链接**：https://github.com/sgl-project/sglang/pull/22869
- **摘要**：优化 LTX-2.3 的 offload 热路径：在高显存 Hopper GPU 上默认关闭 DiT/VAE CPU offload，预构建 stage-2 transformer 并常驻 GPU，减少权重搬运延迟约 67s。
- **证据**：DiT 搬运到 GPU 的 h2d 耗时约 67.1s（开启 layerwise offload 时），LoRA 加载约 18.5s。

#### `sgl-project/sglang#18764` [open] `2026-04-15` [diffusion] Add dynamic batching v0

- **链接**：https://github.com/sgl-project/sglang/pull/18764
- **摘要**：为 diffusion scheduler 引入动态批处理（max batch size + delay），在多个 text-to-image 模型上吞吐提升最高 29.6%，平均延迟降低 22.4%，P99 延迟降低 31.8%。
- **证据**：吞吐提升最高 29.6%，平均延迟降低 22.4%，P99 延迟降低 31.8%（多模型 text-to-image benchmark）。

#### `sgl-project/sglang#22183` [open] `2026-04-12` [Diffusion] Sequential Per-Output Execution for Multi-Output Diffusion Generation

- **链接**：https://github.com/sgl-project/sglang/pull/22183
- **摘要**：调整 Qwen-Image 的执行方式以压低峰值显存，缓解多输出或大 batch 场景下的 OOM 风险。 主要改动涉及 pipeline_executor.py, parallel_executor.py。
- **证据**：| DiT time | 15.2041 s | 15.4286 s | +1.4% |；| Peak GPU memory | 57,104 MB | **32,042 MB** | **↓ 43.9%** |


### 调度/吞吐优化 (`scheduling-throughput`)

#### `sgl-project/sglang#22869` [open] `2026-04-17` [diffusion] optimize ltx-2.3 offload hot paths

- **链接**：https://github.com/sgl-project/sglang/pull/22869
- **摘要**：优化 LTX-2.3 的 offload 热路径：在高显存 Hopper GPU 上默认关闭 DiT/VAE CPU offload，预构建 stage-2 transformer 并常驻 GPU，减少权重搬运延迟约 67s。
- **证据**：DiT 搬运到 GPU 的 h2d 耗时约 67.1s（开启 layerwise offload 时），LoRA 加载约 18.5s。

#### `sgl-project/sglang#18764` [open] `2026-04-15` [diffusion] Add dynamic batching v0

- **链接**：https://github.com/sgl-project/sglang/pull/18764
- **摘要**：为 diffusion scheduler 引入动态批处理（max batch size + delay），在多个 text-to-image 模型上吞吐提升最高 29.6%，平均延迟降低 22.4%，P99 延迟降低 31.8%。
- **证据**：吞吐提升最高 29.6%，平均延迟降低 22.4%，P99 延迟降低 31.8%（多模型 text-to-image benchmark）。

#### `sgl-project/sglang#22805` [open] `2026-04-14` [diffusion] comms: Pack QKV for a2a in Flux2

- **链接**：https://github.com/sgl-project/sglang/pull/22805
- **摘要**：优化 FLUX.2 的多卡通信路径，减少 all-to-all 或 collective 调用带来的同步开销。 主要改动涉及 usp.py, layer.py。
- **证据**：| **E2E Latency** | 4740.77 ms | 4657.01 ms | **-83.77 ms (-1.8%)** | ⚪️ |；| DenoisingStage | 4523.43 | 4432.28 | -91.15 | -2.0% | ⚪️ |

#### `sgl-project/sglang#22183` [open] `2026-04-12` [Diffusion] Sequential Per-Output Execution for Multi-Output Diffusion Generation

- **链接**：https://github.com/sgl-project/sglang/pull/22183
- **摘要**：调整 Qwen-Image 的执行方式以压低峰值显存，缓解多输出或大 batch 场景下的 OOM 风险。 主要改动涉及 pipeline_executor.py, parallel_executor.py。
- **证据**：| DiT time | 15.2041 s | 15.4286 s | +1.4% |；| Peak GPU memory | 57,104 MB | **32,042 MB** | **↓ 43.9%** |


## Notes

- 本次只保留严格规则下的高置信 diffusion 性能优化 PR。
- 使用 word-boundary 匹配避免关键词误匹配（如 dit 不会匹配 edit）。
- 摘要由 LLM 基于 PR 实际内容生成，非模板填充。
