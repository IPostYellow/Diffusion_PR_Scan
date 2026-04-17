# Diffusion PR Sync Report

- run_id: `20260417T022125Z`
- window_start_utc: `1970-01-01T00:00:00Z`
- window_end_utc: `2026-04-17T02:21:25Z`
- scanned_pr_candidates: `40`
- matched_diffusion_prs: `13`
- delta_added: `13`
- delta_updated: `0`

## Delta

| Kind | Repo | PR | Title | Techniques |
| --- | --- | --- | --- | --- |
| added | sgl-project/sglang | [#23014](https://github.com/sgl-project/sglang/pull/23014) | [CI] install rust toolchain in docker images | kernel-fusion-compiler, quality-ci |
| added | vllm-project/vllm | [#26624](https://github.com/vllm-project/vllm/pull/26624) | [Feature][V1][P/D]: Support P/D num_cached_tokens in response usage | memory-management, quality-ci, runtime-io, scheduler-sampler, serving-pipeline |
| added | vllm-project/vllm | [#40084](https://github.com/vllm-project/vllm/pull/40084) | [WIP][Draft]Ray async scheduling with pp | kernel-fusion-compiler, memory-management, parallel-distributed, precision-quantization, quality-ci, scheduler-sampler, serving-pipeline |
| added | vllm-project/vllm | [#32415](https://github.com/vllm-project/vllm/pull/32415) | [Fix] Fix buffer sizing and memory handling for routed-expert return in TP | kernel-fusion-compiler, memory-management, parallel-distributed, precision-quantization, quality-ci, scheduler-sampler, serving-pipeline |
| added | vllm-project/vllm | [#32476](https://github.com/vllm-project/vllm/pull/32476) | [Doc] Clarify comment regarding partial block loading | memory-management, parallel-distributed, quality-ci, scheduler-sampler |
| added | sgl-project/sglang | [#21758](https://github.com/sgl-project/sglang/pull/21758) | multi-item scoring test | kernel-fusion-compiler, memory-management, model-architecture, precision-quantization, quality-ci, runtime-io, scheduler-sampler, serving-pipeline |
| added | sgl-project/sglang | [#21761](https://github.com/sgl-project/sglang/pull/21761) | dynamic batch tokenizer testcases | kernel-fusion-compiler, memory-management, model-architecture, parallel-distributed, quality-ci, runtime-io, scheduler-sampler, serving-pipeline |
| added | sgl-project/sglang | [#22015](https://github.com/sgl-project/sglang/pull/22015) | EPD test | model-architecture, parallel-distributed, quality-ci, runtime-io, scheduler-sampler |
| added | sgl-project/sglang | [#22339](https://github.com/sgl-project/sglang/pull/22339) | Used for RL acceleration, historical request memory, and scheduling. | memory-management, quality-ci, runtime-io, scheduler-sampler, serving-pipeline |
| added | sgl-project/sglang | [#20863](https://github.com/sgl-project/sglang/pull/20863) | [Diffusion] Add mixed-resolution benchmark support (for #20762) | memory-management, quality-ci, serving-pipeline |
| added | sgl-project/sglang | [#23015](https://github.com/sgl-project/sglang/pull/23015) | Used for RL acceleration, historical request memory, and scheduling. | memory-management, quality-ci, runtime-io, scheduler-sampler, serving-pipeline |
| added | sgl-project/sglang | [#22998](https://github.com/sgl-project/sglang/pull/22998) | Skip torch.cuda.empty_cache() in weight update flush path | memory-management, parallel-distributed, quality-ci, scheduler-sampler, serving-pipeline |
| added | sgl-project/sglang | [#22997](https://github.com/sgl-project/sglang/pull/22997) | [Whisper] Automatic language detection via structured generation | kernel-fusion-compiler, memory-management, quality-ci, runtime-io, scheduler-sampler, serving-pipeline |


## Chinese Summaries

- `sgl-project/sglang#20863`: 该 PR 为 diffusion benchmark 增加 mixed-resolution 随机请求配置。代码主要修改 bench_serving.py、bench_offline_throughput.py 和 datasets.py，引入 --random-request-config 与 --random-request-seed，支持按权重采样不同分辨率和 inference steps 并传入基准请求。
- `sgl-project/sglang#21758`: 该 PR 为 Ascend NPU 的 multi-item scoring 增加专项测试。改动主要落在 python/sglang/test/ascend/ 与 NPU CI workflow，覆盖 raw logits 与 softmax 两种返回模式、语义排序验证以及公共测试辅助逻辑。
- `sgl-project/sglang#21761`: 该 PR 为 Ascend NPU 的 dynamic batch tokenizer 增加完整测试矩阵。代码集中在 python/sglang/test/ascend/ 与 NPU CI workflow，新增并发、batch size、timeout、sampling、TP/DP 等 13 个用例，用于验证不同 batch_wait_timeout 和并行模式下的行为。
- `sgl-project/sglang#22015`: 该 PR 为 Ascend NPU 上的 VLM encoder-prefill disaggregation 补充测试。主要新增 test_npu_adaptive_dispatch_to_encoder.py 与 test_npu_disaggregated_vlm.py，覆盖 --encoder-only、--language-only、transfer backend 和 adaptive dispatch 等多进程参数组合。
- `sgl-project/sglang#22339`: 该 PR 想针对 RL 多轮重复请求的 decode 阶段引入历史请求记忆调度。代码主要修改 schedule_batch.py、scheduler.py 与 server_args.py，通过记录历史推理长度来调整新一轮 decode 的请求优先级，并同步补充 server argument 文档。
- `sgl-project/sglang#22997`: 该 PR 为 Whisper 转写增加自动语言检测，避免未显式传 language 时默认英语。代码主要修改 serving_transcription.py、transcription_adapters、processors/whisper.py 和 warmup.py，用单次 regex-constrained structured generation 融合语言检测与转写，并新增 warmup 预编译 xgrammar FSM。
- `sgl-project/sglang#22998`: 该 PR 调整权重更新 flush 路径，默认跳过 torch.cuda.empty_cache() 以减少并发推理时的 CUDA 同步开销。代码主要修改 scheduler.py、scheduler_update_weights_mixin.py 和 io_struct.py，统一多个 UpdateWeights 请求结构里的 torch_empty_cache 开关，同时保留 KV cache pool flush。
- `sgl-project/sglang#23014`: 该 PR 为多个 Docker 镜像补上 Rust toolchain，修复包含 setuptools-rust 扩展时 pip install 因找不到 rustc/cargo 而失败的问题。代码主要修改 docker/Dockerfile、docker/diffusion.Dockerfile、docker/npu.Dockerfile 等镜像定义，在 framework、diffusion、NPU、XPU 和 Xeon 镜像里安装 rustup 并显式设置 PATH。
- `sgl-project/sglang#23015`: 该 PR 想针对 RL 多轮重复请求的 decode 阶段引入历史请求记忆调度。代码主要修改 schedule_batch.py、scheduler.py 与 server_args.py，通过记录历史推理长度来调整新一轮 decode 的请求优先级，并同步补充 server argument 文档。
- `vllm-project/vllm#26624`: 该 PR 修复 vLLM 在 P/D 部署下 response usage.prompt_tokens_details.cached_tokens 统计错误的问题。代码主要修改 vllm/v1/core/sched/scheduler.py 和 vllm/v1/request.py，把 Prefill 阶段真正命中的 prefix cached tokens 传给 Decode，而不是错误地返回 pulled KV tokens。
- `vllm-project/vllm#32415`: 该 PR 修复 Tensor Parallel 加 routed-expert return 场景下的缓冲区 sizing 和内存处理问题。代码主要修改 routed_experts_capturer.py、scheduler.py、gpu_model_runner.py 及对应测试，引入 /dev/shm 检测与 mmap 回退、按 num_gpu_blocks 定尺寸、增长式扩容、异步 D2H 拷贝和更窄 buffer dtype。
- `vllm-project/vllm#32476`: 该 PR 只改注释，不改行为。代码在 offloading_connector.py 中把“可以加载少于一个 block”改成“不能加载 partial offloaded blocks”，明确当前 offloading 逻辑只支持 block 粒度加载。
- `vllm-project/vllm#40084`: 该 Draft PR 试图在 Ray 分布式执行器上为 pipeline parallel (pp>1) 打通 --async-scheduling。代码主要修改 core.py、ray_executor.py、ray_utils.py 和相关测试，引入 step_id 标记的 DAG 协议、P2P sampled-token 同步、异步 output materialization 与预取 output RPC，但描述中也明确记录了长输入和 rate-limited 场景下的已知退化问题。

## Technique Counts

| Technique | Count |
| --- | --- |
| kernel-fusion-compiler | 6 |
| memory-management | 11 |
| model-architecture | 3 |
| parallel-distributed | 6 |
| precision-quantization | 3 |
| quality-ci | 13 |
| runtime-io | 7 |
| scheduler-sampler | 11 |
| serving-pipeline | 10 |

