# Strict Diffusion Performance PR Scan

Generated at: `2026-04-17T03:04:39Z`

## Strict rule
- Must be clearly about diffusion models, tasks, or runtime code paths.
- Must describe a concrete performance optimization or attach benchmark / memory evidence.
- Must touch production code paths; docs / CI / tests / benchmark-only / Docker / support-only PRs are excluded.

## Repo summary
- `sgl-project/sglang`: 19 strict matches
- `vllm-project/vllm`: 0 strict matches

## Technique summary
- `kernel-fusion` / 内核融合/算子快路径: 4
- `cuda-graph` / CUDA Graph 图捕获: 3
- `torch-compile` / torch.compile 编译: 2
- `low-precision-quantization` / FP8/NVFP4/量化: 3
- `attention-backend` / 注意力后端/精度调度: 2
- `execution-batching` / 执行批处理/前向合批: 1
- `communication` / 通信优化: 1
- `memory-offload` / Offload/驻留策略: 1
- `cache-reuse` / 缓存复用: 3
- `memory-efficiency` / 显存效率: 2
- `scheduling-throughput` / 调度/吞吐优化: 1

## Detailed list
### 内核融合/算子快路径 (`kernel-fusion`)
- `sglang#22814` [open] `2026-04-17` diffusion: add HunyuanVideo GroupNorm+SiLU fast path
  摘要：为 `HunyuanVideoResnetBlockCausal3D` 增加 Triton GroupNorm+SiLU 快路径，减少 3D ResNet 块中的归一化与激活开销。优化主要落在 decoding 阶段。
  证据：H100 上总时延 57.23s->56.45s，DecodingStage 15.51s->14.55s。
- `sglang#22786` [open] `2026-04-16` [AMD][diffusion] Add FlyDSL fused normalization kernels for ROCm diffusion models optimization
  摘要：在 ROCm diffusion 路径里引入 FlyDSL 融合归一化、scale/shift 和 GeLU 快路径，并保留 shape、dtype、import 三类回退逻辑。核心目标是压缩 denoising 热路径上的 norm/modulation kernel 数量。
  证据：Wan2.2 T2V 在 MI355X 上无 compile 总时延 143.41s->138.70s；compile 场景 131.60s->129.28s，denoising 子阶段改善更明显。
- `sglang#22445` [open] `2026-04-15` [NPU] [Diffusion] Performance Optimization for LTX-2 Model
  摘要：给 LTX-2 换成 custom RMSNorm 与 NPU 融合 RMSNorm，并在全 1 mask 情况下绕过低效 attention 路径。收益主要来自归一化与 attention 前处理热路径的收缩。
  证据：E2E 在 NPU 上 102.74s->75.69s，在 GPU 上 136.62s->131.70s。
- `sglang#19632` [closed-unmerged] `2026-03-02` [Perf] Reduce DiT kernel launch overhead with fused CUDA kernels for FLUX/Z-Image
  摘要：在 FLUX/FLUX.2/Z-Image 的 DiT 热路径里引入融合 residual+norm+modulation、QK norm 与 gated residual kernel，目标是减少 kernel launch 与内存带宽开销。
  证据：4-step 1024x1024 benchmark 里 DenoisingStage 0.7028s->0.6913s；作者预计标准 20-50 步场景收益会更明显。

### CUDA Graph 图捕获 (`cuda-graph`)
- `sglang#19876` [closed-unmerged] `2026-04-17` [Diffusion] Diffusion support cuda graph
  摘要：把 CUDA Graph 预捕获扩展到通用 diffusion/FLUX 路径，通过预捕获 CFG positive graph 减少 kernel 间空洞。优化很直接，主要作用在 denoising 的 host launch 开销。
  证据：FLUX.1-dev 50 步里单步 0.1495s->0.1449s，Pixel data 总时长 17.61s->15.02s。
- `sglang#21912` [open] `2026-04-02` [diffusion] ZImage-Turbo DiT FP8 full quantization & CUDA Graph 
  摘要：给 Z-Image-Turbo 做完整 FP8 量化（attention+FFN）并接入 whole/step-level CUDA Graph，顺带修复 FFN FP8 scale 丢失等加载问题。组合目标是同时压缩 GEMM 成本和 host 发射开销。
  证据：H20 上 1024x1024 的 E2E 3607.24ms->1837.81ms；512x512 为 923.06ms->504.06ms。
- `sglang#19516` [open] `2026-03-18` [Diffusion] add cuda graph support for Qwen-Image
  摘要：针对 Qwen-Image 里约 30% 的 CPU launch overhead，按 txt/img 子路径拆分并捕获 CUDA Graph，尽量只对 prompt 维做 padding。PR 的重点是消除 host launch gap，而不是改模型算法。
  证据：给出了 profile 与输出一致性验证；属于 profile-based 的性能 PR。

### torch.compile 编译 (`torch-compile`)
- `sglang#21417` [open] `2026-03-26` [Diffusion] Change default torch.compile mode from max-autotune to default
  摘要：把 diffusion 默认 `torch.compile` mode 从 `max-autotune-no-cudagraphs` 改成 `default`，修复“开 compile 反而更慢”的运行时策略问题。它不是新增算子，而是 compile policy 调优。
  证据：作者给出 Qwen/Flux.2 测试，说明旧默认模式会导致 1.47-1.68x denoising regression，而新默认模式能改善 text encoding 与 VAE decode。
- `sglang#19673` [merged] `2026-03-04` [diffusion] support torch compile for diffusers backend
  摘要：让 diffusers backend 也能安全启用 `torch.compile`，并补上 `warmup_steps` 来配合 cache-dit。它把 compile 收益从自研 backend 扩展到了 diffusers 路径。
  证据：FLUX.1-dev 在 L20 上 23.6s->20.3s；与 cache 叠加后到 12.8s。

### FP8/NVFP4/量化 (`low-precision-quantization`)
- `sglang#21912` [open] `2026-04-02` [diffusion] ZImage-Turbo DiT FP8 full quantization & CUDA Graph 
  摘要：给 Z-Image-Turbo 做完整 FP8 量化（attention+FFN）并接入 whole/step-level CUDA Graph，顺带修复 FFN FP8 scale 丢失等加载问题。组合目标是同时压缩 GEMM 成本和 host 发射开销。
  证据：H20 上 1024x1024 的 E2E 3607.24ms->1837.81ms；512x512 为 923.06ms->504.06ms。
- `sglang#20319` [open] `2026-03-31` [AMD] Support fp8 MHA for diffusion model
  摘要：将 AMD diffusion attention 的 FP8 per-tensor flash attention 替换成 MLA prefill ASM kernel，并在不满足 tile 约束时回退到 BF16。收益集中在 Wan2.2 的 attention 热点。
  证据：MI355X 上 81 帧总时长 442.17s->357.75s，161 帧总时长 1426.08s->1175.45s。
- `sglang#20361` [merged] `2026-03-17` [Diffusion] Bump up cache-dit & support quant for diffusers backend
  摘要：升级 cache-dit 集成，让 diffusers backend 能直接加载更完整的 cache、parallel 与 FP8 配置。它的本质是把 cache-dit 的优化能力完整打通到 SGLang diffusion。
  证据：FLUX.1-dev 在 L20 上 20.46s->13.81s，在 H200 上 3.73s->2.77s。

### 注意力后端/精度调度 (`attention-backend`)
- `sglang#21742` [open] `2026-04-08` [diffusion] attention: add support for hybrid attention schedule
  摘要：引入 hybrid attention schedule，在扩散前后若干步使用高精后端，中间步切到更快的低精后端，以降低 artifacts 同时保留大部分性能收益。
  证据：Wan2.2 T2V 上，相比纯 AITER 基线，hybrid 调度的 DenoisingStage 下降约 11.5%。
- `sglang#20319` [open] `2026-03-31` [AMD] Support fp8 MHA for diffusion model
  摘要：将 AMD diffusion attention 的 FP8 per-tensor flash attention 替换成 MLA prefill ASM kernel，并在不满足 tile 约束时回退到 BF16。收益集中在 Wan2.2 的 attention 热点。
  证据：MI355X 上 81 帧总时长 442.17s->357.75s，161 帧总时长 1426.08s->1175.45s。

### 执行批处理/前向合批 (`execution-batching`)
- `sglang#20434` [closed-unmerged] `2026-04-17` [diffusion] Batch serial CFG for Qwen-Image to reduce denoising overhead
  摘要：把 Qwen-Image 串行 CFG 的 cond/uncond 两次 transformer forward 合并成一次 batched forward，再按原公式合成 guidance。算法不变，主要减少每步框架与 launch 开销。
  证据：50 步 denoising 的单步时间 0.3052s->0.2553s，总 denoising 15.2640s->12.7667s。

### 通信优化 (`communication`)
- `sglang#22805` [open] `2026-04-14` [diffusion] comms: Pack QKV for a2a in Flux2
  摘要：在 Flux2 的 USP attention 中把 Q/K/V 三次 all-to-all 合并成一次打包通信，减少多 GPU 通信调用和拆分开销。收益集中在 denoising 通信热段。
  证据：8xB200 与 8xMI355X 的 1024/2048 分辨率测试中，DenoisingStage 约下降 0.7%-2.0%，总体延迟下降约 0.5%-1.8%。

### Offload/驻留策略 (`memory-offload`)
- `sglang#22869` [open] `2026-04-17` [diffusion] optimize ltx-2.3 offload hot paths
  摘要：围绕 LTX-2.3 两阶段管线，减少 DiT/VAE 的 CPU offload 和请求期 `module.to(cpu/cuda)` 切换，新增 snapshot/resident 模式与预合并 stage-2 transformer。重点是把权重搬运和 LoRA 切换的同步开销从请求路径里移走。
  证据：同一条链路下 e2e 从 legacy 33.93s 降到 snapshot 30.16s，再到 resident 21.42s；resident 模式显存峰值提升到约 99.31GB。

### 缓存复用 (`cache-reuse`)
- `sglang#22441` [open] `2026-04-12` [diffusion] Cache LTX-2 RoPE coords to avoid per-step recompute
  摘要：对 LTX-2 音视频 denoising 阶段中每步重复构造的 video/audio RoPE 坐标做 LRU 缓存，把与分辨率、帧数相关但跨步不变的准备工作移出循环。属于典型的 per-step 常量复用优化。
  证据：A800 上单步 2.3484s->2.1664s，整段 denoising 93.94s->86.66s。
- `sglang#21573` [open] `2026-04-01` Add persistent diffusion workspace reuse and extend random benchmarks for image-conditioned tasks
  摘要：在多个 diffusion denoising loop 里复用 persistent workspace，避免每步重复分配临时 buffer。它属于低风险的 allocator/temporary tensor 开销优化。
  证据：H200 单卡测试里，Wan DMD、Flux.2 与 Helios 都有小幅平均延迟或 P99 改善，并在部分模型上略降峰值显存。
- `sglang#20361` [merged] `2026-03-17` [Diffusion] Bump up cache-dit & support quant for diffusers backend
  摘要：升级 cache-dit 集成，让 diffusers backend 能直接加载更完整的 cache、parallel 与 FP8 配置。它的本质是把 cache-dit 的优化能力完整打通到 SGLang diffusion。
  证据：FLUX.1-dev 在 L20 上 20.46s->13.81s，在 H200 上 3.73s->2.77s。

### 显存效率 (`memory-efficiency`)
- `sglang#22183` [open] `2026-04-12` [Diffusion] Sequential Per-Output Execution for Multi-Output Diffusion Generation
  摘要：把 `num_outputs_per_prompt` 从 denoising 内部大 batch 改成执行器层的 sequential per-output，解决多输出时 OOM 和模型不兼容问题。这个 PR 主要优化的是显存占用而非单次速度。
  证据：4 输出 FLUX 测试里 DiT 时间基本持平，但峰值显存 57.10GB->32.04GB。
- `sglang#21573` [open] `2026-04-01` Add persistent diffusion workspace reuse and extend random benchmarks for image-conditioned tasks
  摘要：在多个 diffusion denoising loop 里复用 persistent workspace，避免每步重复分配临时 buffer。它属于低风险的 allocator/temporary tensor 开销优化。
  证据：H200 单卡测试里，Wan DMD、Flux.2 与 Helios 都有小幅平均延迟或 P99 改善，并在部分模型上略降峰值显存。

### 调度/吞吐优化 (`scheduling-throughput`)
- `sglang#18764` [open] `2026-04-15` [diffusion] Add dynamic batching v0
  摘要：给 diffusion scheduler 增加动态 batching（max batch size + delay），在不改模型算法的前提下提高 serving 吞吐并压低尾延迟。适用范围目前是 prompt-only 的 t2i/t2v。
  证据：H100 上 Qwen-Image 吞吐 +29.6%，mean latency -22.4%，P99 -31.8%。

## Notes
- 本次清除了旧的宽松扫描结果，当前仓库只保留严格规则下的高置信 diffusion 性能 PR。
- 由于 `vllm-project/vllm` 在这轮严格规则下没有命中项，当前数据全部来自 `sgl-project/sglang`。
