# Execution Status

**Last updated:** 2026-10-26  
**Phase:** project scaffold committed; CPU bootstrap workflow configured  
**Current research status:** no new benchmark accuracy or speed results have been measured yet.

## Verified environment facts

- Repository `TianZH-XJU/codex-eacl` is public and uses default branch `main`.
- Current interactive runtime: Python 3.13.5; PyTorch 2.10.0+cpu; CUDA unavailable; 3 PyTorch CPU threads.
- Current Laya source metadata inspected through GitHub: upstream `pyproject.toml` declares Laya 0.4.16, Python >=3.10, and base dependencies on torch, transformers, safetensors, huggingface_hub and numpy.
- Upstream Laya tournament benchmark pins BANKING77, CLINC OOS and MASSIVE en-US dataset revisions. Those exact revisions are recorded in `configs/experiment.yaml` and `scripts/download_datasets.py`.
- Shell networking in the current interactive runtime cannot resolve `github.com` or the package index. The attempted shell clone and pip install failed due to DNS resolution. This is a runtime connectivity blocker, not proof that upstream packages/data are unavailable.

## Committed project scaffolding

- `AGENTS.md`: persistent project instructions, research constraints, no-fabrication rule, test-leakage guardrails and automatic-save/commit protocol.
- `docs/PROJECT_PLAN.md`: detailed research question, hypotheses, packages/models, datasets, literature, baselines, architecture, fitness, ablations, metrics and possible storylines.
- `configs/experiment.yaml`: CPU-first configuration and pinned dataset revisions.
- `requirements-cpu.txt`: starter dependencies (CPU PyTorch is installed separately).
- `scripts/check_environment.py`: machine-readable environment report.
- `scripts/download_datasets.py`: pinned dataset downloads and SHA-256 manifest.
- `scripts/run_baseline_smoke.py`: tiny CPU run using the upstream Laya benchmark.
- `tests/test_config_and_scripts.py`: structural checks.
- `.github/workflows/cpu-bootstrap.yml`: hosted runner workflow to install dependencies, download data, run tests and baseline smoke, and upload artifacts.
- `.gitignore`: excludes model weights, local datasets, caches, environments and large raw outputs.

## Run status

| Step | Status | Evidence / blocker |
|---|---|---|
| Repository and project instructions | committed | GitHub file fetches confirmed |
| Local package installation | blocked | package index DNS resolution failed |
| Local Laya clone | blocked | github.com DNS resolution failed |
| Dataset files downloaded in current runtime | not completed | network blocker; CI workflow is configured to attempt it |
| Model checkpoint downloaded | not completed | network blocker; not attempted to completion |
| Laya CPU baseline smoke test | not run locally | dependencies, source checkout and model weights unavailable locally |
| Hosted GitHub Actions bootstrap | pending verification | workflow committed; inspect Actions run and artifacts before calling it successful |
| NEAT experiments | not run | must wait for successful baseline and cache/oracle-gap analysis |

## Automatic saving

GitHub is the source of truth. Coherent code/docs changes are committed directly to the repository. Pushes to relevant code, config and documentation paths trigger the CPU bootstrap workflow; the workflow stores environment reports, dataset manifest, smoke-test output and downloaded dataset bundle as Actions artifacts. Large data and model weights are not committed to Git. This is not a background editor autosave: every code change still needs a verified commit/push.

## Next actions

1. Inspect the first GitHub Actions run for install, dataset and model download results.
2. Fix any dependency or current-Laya API mismatch revealed by logs.
3. Save a tested dependency lockfile and pin the exact Laya commit/checkpoint revision.
4. Run the 5-example CPU smoke test, then a small development baseline.
5. Build cached action outputs and calculate the oracle gap before spending compute on NEAT.
