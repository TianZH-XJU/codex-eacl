# Execution Status

**Last updated:** 2026-10-26  
**Phase:** repository scaffold committed; CPU bootstrap and upstream baseline workflow configured  
**Research result status:** no new accuracy result has been measured yet.

## GitHub autosave / source of truth

- Repository: https://github.com/TianZH-XJU/codex-eacl
- Persistent project instructions: [AGENTS.md](AGENTS.md)
- Full plan: [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
- CPU bootstrap workflow: [.github/workflows/cpu-bootstrap.yml](.github/workflows/cpu-bootstrap.yml)
- Main setup scripts: `scripts/check_environment.py`, `scripts/download_datasets.py`, `scripts/run_baseline_smoke.py`
- The initial scaffold has been committed to `main` in multiple commits (the GitHub connector returned successful commit SHAs for README, AGENTS.md, STATUS.md, .gitignore, requirements-cpu.txt and experiment config; scripts/docs/workflow are present on main and are linked above).
- Project instructions require each coherent change to be committed, the status updated, and the commit verified. GitHub Actions is configured to autosave small run manifests/results; raw data and model weights are uploaded as artifacts and not committed.

## Verified directly in the current interactive runtime

- Python 3.13.5.
- PyTorch 2.10.0+cpu.
- CUDA unavailable.
- PyTorch CPU threads: 3.
- Previously checked: NumPy, pandas and scikit-learn are installed; Laya, Transformers, neat-python and datasets were absent before attempted installation.
- Shell `git clone https://github.com/NandhaKishorM/laya.git` failed with DNS resolution error (`Could not resolve host: github.com`).
- `python -m pip install neat-python datasets pyarrow` failed because the package index hostname could not be resolved. This is a current runtime network restriction, not evidence that the packages are unavailable upstream.
- The current Laya upstream `pyproject.toml` was read through GitHub and declares version 0.4.16, Python >=3.10 and base dependencies on torch, transformers, safetensors, huggingface_hub and numpy.
- The official current upstream benchmark script and README were read through GitHub; the project scripts pin the same three dataset revisions.

## Status of install / download / benchmark

| Step | Status | Evidence / next action |
|---|---|---|
| Write project/system instructions to GitHub | **Committed** | See `AGENTS.md` and `docs/PROJECT_PLAN.md` |
| Configure code/result autosave | **Configured; CI execution must be verified** | GitHub Actions workflow commits small reports and uploads artifacts |
| Install Laya and experiment dependencies in this runtime | **Blocked by DNS/network** | Let GitHub-hosted Actions try the installation |
| Download BANKING77, CLINC150 and MASSIVE files locally | **Blocked by DNS/network** | CI script downloads pinned revisions and writes SHA-256 manifest |
| Download Laya model checkpoint locally | **Not run / blocked by network** | CI smoke test will attempt checkpoint download |
| Run CPU baseline smoke test | **Not run yet** | Inspect Actions logs/artifact; fix API/dependency failures if any |
| NEAT controller experiment | **Not run** | Do only after baseline and oracle-gap checks pass |

## Current pinned datasets

- BANKING77: `mteb/banking77`, revision `18072d2685ea682290f7b8924d94c62acc19c0b2`.
- CLINC150: `clinc/clinc_oos`, revision `155b9c710419136e17307b80d0a13e68cd46b4ec`; use the `plus` split and report in-scope and OOS separately.
- MASSIVE en-US: `SetFit/amazon_massive_intent_en-US`, revision `f7672a018e8ceb37fc0184dcfbb7e665155ffea6`.

## Next actions

1. Verify whether the GitHub Actions workflow is enabled and inspect the run triggered by the initial pushes.
2. If the run has not started, manually dispatch it from the Actions tab (workflow_dispatch is enabled).
3. Fix any dependency/API/model/data issue based on the actual log, not assumptions.
4. Record the Laya commit and model revision, then run a slightly larger validation benchmark before implementing NEAT.
5. Build cached per-example action outputs on train/validation only, calculate the oracle action gap, and stop if there is insufficient headroom.
6. Implement simple controllers first; introduce NEAT only with matched-budget comparisons.

## Autosave rule

Each coherent code/documentation unit must be committed to GitHub. Every run must write a small machine-readable report with a truthful status (`not_run`, `blocked_network`, `failed`, `partial_failure`, or `passed`). Never mark a step passed unless the command exit code and output are verified.