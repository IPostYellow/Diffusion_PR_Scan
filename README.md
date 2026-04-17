# Diffusion_PR_Scan

Incremental scan results for diffusion-related PRs in:

- `sgl-project/sglang`
- `vllm-project/vllm`

## Data layout

- `data/diffusion_prs/prs.json`: canonical per-PR records, including `summary_zh`
- `data/diffusion_prs/techniques-index.json`: grouped index by optimization technique
- `data/diffusion_prs/techniques/*.json`: per-technique materialized views
- `data/diffusion_prs/runs/*.json`: per-run deltas
- `reports/diffusion-prs-latest.md`: latest summary report with Chinese summaries for the delta
- `state/diffusion_checkpoint.json`: incremental cursor checkpoint
