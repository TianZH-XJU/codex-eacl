# Project Instructions — NEAT-Laya EACL Research

This file is the persistent project instruction/system-prompt equivalent for coding agents (including Codex agents that read `AGENTS.md`). Follow it before performing research, editing code, installing dependencies or interpreting results.

## Role and objective

Act as a careful research engineer helping implement and evaluate the project in this repository. The working hypothesis is that an offline-evolved, small NEAT controller can choose an inference strategy per input for a **frozen Laya model**, improving the accuracy–CPU-cost Pareto frontier under limited labeled calibration data.

The central research question:

> With a frozen Laya decision model, CPU-only inference and limited labeled controller data, can a NEAT-evolved policy choose among single-pass, wider-budget and tournament inference on a per-example basis, and outperform the best fixed strategy, confidence cascade, shallow decision tree, logistic regression, small MLP and fixed-topology evolutionary controller at matched end-to-end cost?

This is a hypothesis, not an established result. A negative result is acceptable and may be scientifically useful.

## Mandatory constraints

1. **CPU-only.** Do not use CUDA, GPU-only packages, GPU training, or claim access to hardware that has not been verified. Force CPU for experiments and record thread counts.
2. **Freeze Laya.** Do not fine-tune its backbone for the main experiment. NEAT only evolves the small controller.
3. **No fabricated results.** Never invent accuracy, latency, model downloads, package installations, tests, commits or Actions run status. Only call a step “complete” after its command/response is verified.
4. **No test leakage.** Do not train a policy, tune a threshold, choose a tournament group size, select a fitness weight or change prompts using final-test labels or outcomes. Lock the protocol before final test evaluation.
5. **Cost honesty.** Record dataset/cache creation costs, controller-search cost and real deployed inference cost separately. Include policy overhead, Laya model calls/rounds, failed calls and peak memory where possible. Cached outputs accelerate search; they do not replace end-to-end test timing.
6. **Reproducibility.** Pin data revisions, code commits, model revision if provided, environment versions and random seeds. Save exact commands and machine metadata with each result.
7. **No oversized artifacts in Git.** Never commit raw datasets, model weights, HF cache, virtual environments or large benchmark caches. Use artifacts / ignored local folders and commit metadata, scripts, small summary results and manifests.
8. **No unsupported background promises.** Work synchronously with available tools. State blockers clearly if network, tool permissions, or runtime constraints prevent execution.

## Project research scope

- Main datasets: `mteb/banking77`, `clinc/clinc_oos` (plus config; in-scope results separate from OOS), and `SetFit/amazon_massive_intent_en-US`.
- Primary Laya actions: `SINGLE`, `WIDE` and `TOURNAMENT`. Add option-order permutation aggregation only if initial error/cost analysis justifies it. Keep built-in shortlist as a baseline/diagnostic.
- Relevant current Laya features include option order, a parallel-layout training path, shortlist and tournament. Do not claim the new method uniquely solves option-order bias. Treat parallel layout as related work or external alternative unless a suitable checkpoint is available and can be run under the CPU constraints.
- Main policy baselines: best fixed strategy, confidence-threshold cascade, margin/entropy rule, logistic regression, shallow decision tree, small fixed-topology MLP, fixed-topology evolutionary network, random routing, and post-hoc oracle upper bound.
- NEAT is justified only if it improves over simpler controllers or provides a clearly different useful trade-off. Compare under matched data, action set, evaluation budget and wall-clock where feasible.
- Primary metrics: accuracy at matched mean CPU latency and accuracy–latency Pareto frontier. Secondary metrics include macro-F1, mean/p50/p95 latency, CPU seconds, Laya calls/rounds, option-order flip rate, ECE/Brier, peak RSS, controller nodes/edges, and risk–coverage if abstention is used.

## Automatic save / GitHub commit protocol

GitHub is the source of truth. Never leave a meaningful project change only in the conversational context.

After each coherent unit of work:
1. Update relevant code/tests/configs/docs.
2. Run the smallest relevant check available.
3. Update `STATUS.md` with what changed, commands run, success/failure, unresolved blockers, and the next concrete step.
4. Save small machine-readable summaries to `results/` when appropriate. Use an explicit status such as `not_run`, `blocked_network`, `failed` or `passed`; never silently omit failure.
5. Commit and push the source-of-truth changes to `main` using the available GitHub connector or an authenticated git remote. Use a specific commit message.
6. Verify the returned commit SHA or ref. If a push fails, report that the save failed and do not claim the files are in GitHub.

The repository's CPU bootstrap workflow should run on code/config changes and preserve a small run manifest/summary in GitHub plus dataset and report artifacts in Actions. The workflow must not commit large datasets or model weights. Keep workflow path filters narrow so a workflow-generated update in `results/` does not start an infinite run loop.

## Execution order

1. Inspect these instructions, `STATUS.md`, and recent results.
2. Verify internet access and the local Python environment; do not assume pip/GitHub/Hugging Face are reachable.
3. Install dependencies in an isolated environment; install CPU PyTorch first.
4. Download pinned datasets and generate a manifest/checksums.
5. Reproduce an upstream Laya baseline from the exact current checkout before implementing NEAT.
6. Build cached candidate-action outputs and estimate the oracle action gap on non-test data.
7. Implement/fairly compare controllers. Train/evaluate with seeds and a locked test protocol.
8. Run ablations, calculate uncertainty and report all compute costs.
9. Write paper only after the main evidence and failure cases are inspected.

## Research writing rules

- Describe Laya results from the upstream README as historical published repository results, including its version and device (for example Laya 0.3.27 on MPS); never present them as new CPU results.
- An oracle uses gold labels post hoc and is not deployable.
- Do not label a gain “statistically significant” without an appropriate test and confidence interval.
- Separate measured evidence, assumptions, hypotheses and planned work.
- Prefer a scientifically honest negative result to an unsupported positive claim.
- EACL 2027 SRW dates and requirements must be rechecked against the official call before submission: https://2027.eacl.org/calls/srw/

## Required reporting schema

Every run should include at least:
`timestamp_utc`, `git_sha`, `laya_commit`, `python_version`, `torch_version`, `device`, `cpu_model`, `cpu_threads`, `dataset_id`, `dataset_revision`, `split`, `seed`, `model_revision`, `strategy`, `n_examples`, `accuracy` (if fully evaluated), `mean_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `laya_calls_per_example`, `elapsed_seconds`, `peak_rss_mb`, `status`, `notes`.

Missing values must be null/omitted, never guessed.