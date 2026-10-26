# Execution Status

**Last updated:** 2026-10-26  
**Run location:** user's own Windows PC, not ChatGPT or GitHub-hosted runners  
**Current phase:** dependencies/data/model prepared; tiny smoke tests passed; low-memory action caches and targeted latency retests completed; controller baselines pending  
**Research result status:** no statistically meaningful new accuracy result has been established yet.

## Local machine and verified steps (from the user's pasted PowerShell log)

- Machine: AMD Ryzen 5, 16 GB installed RAM, but project should use at most about 4 GB because the computer must remain available for work.
- Project directory in the log: `C:\Users\32085\Desktop\codex-eacl-main\codex-eacl-main`.
- Python 3.12.12 virtual environment: `.venv`.
- PyTorch 2.14.1+cpu; CUDA unavailable; 3 PyTorch CPU threads.
- Laya 0.4.16, local source commit `b9fe351ce81e2d974bca2bf219c1e4030c0a05ea`.
- neat-python 2.0.0, datasets 5.1.0 and pyarrow 26.0.0 installed.
- Model cache: `convaiinnovations/laya`, revision `7b928d828b7b0e022f929d9bd2e44165aa270148`; the reported cached file size was about 846 MB.
- Eight pinned dataset split files were downloaded and `results/dataset_manifest.json` reports status passed.
- The five-row-per-dataset CPU smoke test reported status passed, and the subprocess return code for BANKING77, CLINC150 and MASSIVE en-US was 0.
- `pytest tests/ -q` reports 3 passed after the local smoke-runner test was corrected.
- Detailed per-action accuracy/latency and peak RSS are not present in the pasted terminal excerpt; read the local JSON reports before quoting them.
- The local folder has no `.git` directory yet, so code changes there are not automatically committed to GitHub.

The above local setup and smoke-test facts are based on the user's terminal log pasted into the conversation. Do not treat a five-example smoke test as a research result.

## Persistent instructions and plans

- [`AGENTS.md`](AGENTS.md): research guardrails and save/commit protocol.
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md): full research plan.
- [`docs/LOW_MEMORY_EXECUTION_PLAN.md`](docs/LOW_MEMORY_EXECUTION_PLAN.md): local-only plan for roughly 4 GB available RAM.
- [`configs/experiment.yaml`](configs/experiment.yaml): CPU settings and pinned dataset revisions.
- [`scripts/check_environment.py`](scripts/check_environment.py): environment report.
- [`scripts/download_datasets.py`](scripts/download_datasets.py): pinned dataset downloader and SHA-256 manifest.
- [`scripts/run_baseline_smoke.py`](scripts/run_baseline_smoke.py): small CPU smoke test.
- [`.gitignore`](.gitignore): ignores model data, raw dataset files, virtual environment and the separate upstream Laya checkout.

## Dataset and model pins

- BANKING77: `mteb/banking77`, revision `18072d2685ea682290f7b8924d94c62acc19c0b2`.
- CLINC150: `clinc/clinc_oos`, revision `155b9c710419136e17307b80d0a13e68cd46b4ec`, plus split; in-scope and OOS should be evaluated separately.
- MASSIVE en-US: `SetFit/amazon_massive_intent_en-US`, revision `f7672a018e8ceb37fc0184dcfbb7e665155ffea6`.
- Laya model revision: `7b928d828b7b0e022f929d9bd2e44165aa270148`.

## Status matrix

| Step | Status | Evidence / next action |
|---|---|---|
| Full plan and project instructions in GitHub | committed | See the docs above |
| Local Python and dependencies | passed | Python 3.12.12; PyTorch CPU; Laya and NEAT-related dependencies installed |
| Model checkpoint download | passed | Model revision above is in local HF cache |
| Public dataset download | passed | Eight pinned files and SHA-256 manifest |
| Tiny CPU smoke test | passed | Three datasets return code 0 |
| Unit/structure tests | passed | 3 passed from `pytest tests/ -q` |
| Detailed baseline accuracy and latency | passed (pilot) | BANKING77 train 50, CLINC150 validation 30, MASSIVE validation 30 action caches; see pilot table |
| Fixed train/dev manifest | passed | `scripts/make_split_manifest.py`; `results/splits/banking77_train_dev_manifest.json` (77 classes, 385 train, 231 dev, disjoint) |
| Per-example action cache | passed | `scripts/build_action_cache.py`; SINGLE-first features, cascade costs, latency percentiles, pairwise overlaps |
| Oracle action gap | measured (pilot) | BANKING77 +0.12, CLINC150 +0.033, MASSIVE +0.067 over best fixed strategy |
| NEAT and controller baselines | not run | Implement baselines first; NEAT only if cache and headroom remain credible |
| Local Git/GitHub sync | passed | Commit `7a3458f0211ec1df2a311dec6517fe8d9e0fde59` pushed to `main` |
| GitHub-hosted experiments | disabled/not used | User requested all installation, data and model runs be performed on the user's computer. The CPU bootstrap GitHub Actions workflow has been removed to avoid executing experiments in GitHub's environment. |

## Controller baselines on BANKING77 (2026-10-26)

Train cache: 77 class-balanced 1-shot examples from official `train`. Dev cache: 231 held-out examples from the same fixed manifest. All methods use only initial SINGLE features and cumulative cascade costs.

| Method | Dev accuracy | Mean cumulative ms | Mean cumulative calls |
|---|---:|---:|---:|
| Best fixed (tournament) | 0.5541 | 5766.4 | 3.000 |
| Confidence threshold | 0.5584 | 5131.2 | 2.810 |
| Margin threshold | 0.5584 | 5131.2 | 2.810 |
| Entropy threshold | 0.5584 | 5131.2 | 2.810 |
| Logistic regression (binary upgrade) | 0.4242 | 2799.0 | 1.080 |
| Decision tree (depth 3, binary) | 0.4329 | 2889.4 | 1.100 |
| Small MLP (8 units, binary) | 0.4026 | 2182.7 | 1.040 |
| Random routing | 0.5152 | 4102.3 | 2.030 |
| Oracle (post-hoc) | 0.7056 | 2870.9 | 1.416 |

Interpretation:
- Threshold cascades improve slightly over the best fixed strategy at lower cumulative cost.
- Learned controllers using only initial confidence, margin, entropy, probabilities, text length and option count underperform the fixed strategy on this 77-example training set.
- The oracle gap remains large (0.7056), so there is still potential headroom, but the current feature set/training size is insufficient for the learned controllers.
- Before NEAT, inspect feature quality, class balance, and whether additional cheap features or calibration are needed.

## Low-memory pilot results (2026-10-26)

All pilots used the frozen Laya 0.4.16 checkpoint, CPU threads=2, and `train`/`validation` splits only. Raw per-example caches are local JSONL; summaries are small JSON files under `results/pilot/`.

| Dataset | Split | N | Best fixed | Best fixed acc. | Oracle acc. | Oracle gain | Peak RSS (MB) | Notes |
|---|---|---:|---|---:|---:|---:|---:|---|
| BANKING77 | train | 50 | WIDE | 0.6400 | 0.7600 | +0.1200 | 2144.9 | single 0.46, wide 0.64, tournament 0.54; mean latency single 2037ms, wide 2399ms, tournament 3871ms |
| CLINC150 | validation | 30 | TOURNAMENT | 0.8333 | 0.8667 | +0.0333 | 2166.1 | single 0.60, wide 0.7333, tournament 0.8333; modest oracle headroom |
| MASSIVE en-US | validation | 30 | WIDE | 0.6667 | 0.7333 | +0.0667 | 2116.4 | two cached examples had extreme latency (single 1.87e6 ms, tournament 8.8e5 ms); targeted retests of indices 8 and 9 were normal (single 1.68-1.77s, tournament 2.57-2.87s), so the extremes are treated as transient, not deleted |

Interpretation:
- There is measurable oracle headroom on BANKING77 (+0.12) and smaller headroom on CLINC150 (+0.033) and MASSIVE (+0.067).
- Peak RSS stayed below 2.2 GB, so the 3.5 GB ceiling was not approached.
- WIDE is a strong fixed baseline; TOURNAMENT is expensive (about 2 Laya calls per example) and not uniformly better.
- Before NEAT, implement controller baselines and lock a protocol. Test leakage must be avoided: use only train/validation labels for controller training and thresholds.

## Next actions

1. Read `results/smoke/baseline_smoke_report.json`, `results/dataset_manifest.json` and `results/environment.json` locally.
2. Establish Git sync without discarding the local virtual environment, model cache or datasets.
3. Run one BANKING77 train pilot with 20 rows and SINGLE+TOURNAMENT only, with CPU threads limited to 2.
4. Inspect elapsed time and peak RAM; expand to 50 rows only if stable.
5. Build an output cache, check action error overlap and calculate the oracle gap on development data.
6. Implement simple controller baselines and NEAT only if the action set has sufficient headroom.

## Resource limits and research integrity

- Run one inference process at a time; set `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, and `TOKENIZERS_PARALLELISM=false`.
- Keep peak memory under roughly 3.5 GB and stop if Windows becomes sluggish.
- Do not commit model weights, raw datasets, the virtual environment or large caches.
- Do not tune anything using final-test results.
- No new accuracy gains are claimed until measured, saved and checked.
