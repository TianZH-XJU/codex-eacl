# NEAT-Laya Budget-Aware NLP Decisions

Research repository for an EACL 2027 Student Research Workshop project: evaluate whether a small NEAT-evolved controller can choose among inference strategies for a frozen Laya System-1 decision model under CPU-only and low-label constraints.

> **Research status:** project instructions, plan, data-download scripts and CPU bootstrap workflow are committed. The local runtime cannot reach GitHub/PyPI due to DNS, so installation/data download/baseline are pending verification on GitHub Actions. No new accuracy or speed gains are claimed until measured.

## Project instructions

Read [`AGENTS.md`](AGENTS.md) before editing code. It is the project's persistent agent instruction file (system-prompt equivalent for coding agents that honor repository instructions).

## Research plan

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for problem definition, model/data choices, baseline matrix, architecture, fitness function, ablations, compute-accounting rules, expected narratives and timeline.

## Status

See [`STATUS.md`](STATUS.md). Each meaningful implementation step should update the status and be committed to GitHub.

## CPU-only setup

Recommended: Python 3.12 in a virtual environment. Install CPU PyTorch first, then project dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
```

The current Laya source can be installed from its upstream repository:

```bash
git clone https://github.com/NandhaKishorM/laya.git external/laya
git -C external/laya rev-parse HEAD
python -m pip install -e external/laya
```

Check dependencies and device:

```bash
python scripts/check_environment.py --output results/environment.json
```

Download the exact datasets used by the upstream tournament benchmark:

```bash
python scripts/download_datasets.py --output-dir data/raw
```

Run the upstream baseline smoke test (requires `external/laya`, network access for the model checkpoint, and the data dependencies):

```bash
python scripts/run_baseline_smoke.py --laya-root external/laya --limit 5 --output-dir results/smoke
```

## Automatic saving and reproducibility

- Source code, prompts, configurations and documentation are versioned in GitHub.
- A coding agent must commit each coherent work unit instead of leaving changes only in chat/session memory.
- Pushes to relevant code/config/workflow paths trigger the CPU bootstrap workflow. It installs dependencies, downloads pinned public datasets, runs a small baseline smoke test and stores the report/dataset bundle as GitHub Actions artifacts.
- The workflow records a small machine-readable run status in `results/`; raw datasets, model weights, local environments and large model caches are not committed to Git.
- Do not report a run as successful unless its command exit code and outputs have been checked. Any unrun/failed step must be stated explicitly.

Workflow status and artifacts: [GitHub Actions](https://github.com/TianZH-XJU/codex-eacl/actions).

## Important baseline warning

The current Laya repository is evolving. Its current package version and API may differ from the historical benchmark results documented for Laya 0.3.27. All new results must record the exact Laya commit, checkpoint/model revision, Python/PyTorch versions, CPU device, data revisions, seeds, latency and model-call counts. Historical MPS/GPU results are context only, not expected CPU results.

## License and data

Code authored for this project will be released under the MIT License unless otherwise noted. Dataset terms remain governed by each dataset's own license. Do not commit downloaded model checkpoints or raw dataset files to the repository.