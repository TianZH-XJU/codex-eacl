# NEAT-Laya Budget-Aware NLP Decisions

Research repository for an EACL 2027 Student Research Workshop project: evaluate whether a small NEAT-evolved controller can choose among inference strategies for a frozen Laya System-1 decision model under CPU-only and low-label constraints.

> **Research status:** according to the user's local Windows logs, Python dependencies, the Laya model, 8 pinned dataset files, the 3-dataset five-example smoke test and 3 structural tests are ready/passed. The research computer is AMD Ryzen 5 with 16 GB installed RAM but only about 4 GB may be used by this project. No research-level accuracy/latency result has been established yet. All experiments are local-only; the cloud bootstrap workflow was removed on request.

## Project instructions

Read [`AGENTS.md`](AGENTS.md) before editing code. It is the project's persistent agent instruction file (system-prompt equivalent for coding agents that honor repository instructions).

## Research plan

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the research design and [`docs/LOW_MEMORY_EXECUTION_PLAN.md`](docs/LOW_MEMORY_EXECUTION_PLAN.md) for Windows/Git sync steps, a 4 GB RAM budget, the first 20-row pilot, and go/no-go gates.

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

## Local-only execution and autosave

- Run installs, downloads and inference on the user's own Windows computer. Do not run the experiments on ChatGPT's runtime or GitHub-hosted Actions runners.
- The previous cloud bootstrap workflow was removed. GitHub is for source control and backup, not remote model execution.
- The current folder must be a real Git clone or connected local Git repository for autosave/push to work. The user's current extracted folder had no `.git` directory; follow the safe setup instructions in `docs/LOW_MEMORY_EXECUTION_PLAN.md` before the next experiment. After Git is connected, use `scripts/sync_checkpoint.ps1` for a reviewed one-command checkpoint save/push.
- Commit coherent code/docs changes and small JSON summaries to GitHub. Keep `.venv`, `external/laya`, model weights, raw datasets, HF caches and large output caches local and ignored.
- Do not report success unless the local command return code and result files are checked. Any unrun/failed step must be stated explicitly.

## Important baseline warning

The current Laya repository is evolving. Its current package version and API may differ from the historical benchmark results documented for Laya 0.3.27. All new results must record the exact Laya commit, checkpoint/model revision, Python/PyTorch versions, CPU device, data revisions, seeds, latency and model-call counts. Historical MPS/GPU results are context only, not expected CPU results.

## License and data

Code authored for this project will be released under the MIT License unless otherwise noted. Dataset terms remain governed by each dataset's own license. Do not commit downloaded model checkpoints or raw dataset files to the repository.