# Diffusion_PR_Scan

Strict scan results for diffusion performance optimization PRs in:

- `sgl-project/sglang`
- `vllm-project/vllm`

## Strict filter

Only keep PRs that satisfy all of the following:

- The title/body/code path is clearly about diffusion models or diffusion runtime paths.
- The PR is explicitly about performance, throughput, latency, memory, compilation, quantization, caching, or communication optimization.
- The PR changes production code paths and includes a concrete optimization mechanism or benchmark evidence.

Default exclusions:

- docs / CI / tests / benchmark-only / Docker changes
- generic support or portability PRs without a direct performance goal
- PRs that only match broad words like `scheduler` but do not describe a diffusion optimization

## Current rebuild snapshot

- `sgl-project/sglang`: 19 strict matches
- `vllm-project/vllm`: 0 strict matches

## Data layout

- `data/diffusion_prs/prs.json`: canonical per-PR records, including `summary_zh`, techniques, and evidence
- `data/diffusion_prs/techniques-index.json`: grouped index by optimization technique
- `data/diffusion_prs/techniques/*.json`: one file per optimization technique for direct category browsing
- `data/diffusion_prs/runs/20260417T030439Z.json`: rebuild snapshot for this strict run
- `reports/diffusion-prs-latest.md`: latest human-readable summary report
- `state/diffusion_checkpoint.json`: checkpoint metadata for the strict filter version
