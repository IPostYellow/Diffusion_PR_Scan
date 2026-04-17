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

