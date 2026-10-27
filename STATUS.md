# Execution Status

**Last updated:** 2026-10-27
**Run location:** user's own Windows PC, not ChatGPT or GitHub-hosted runners
**Current phase:** three-action offline evaluation + temperature calibration done on fixed BANKING77 caches; 5-shot and NEAT paused  
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
- [`.gitignore`](.gitignore): ignores model data, raw dataset files, virtual environment, the separate upstream Laya checkout, and the local Claude Code permission config.
- [`.claude/settings.local.json`](.claude/settings.local.json): local (git-ignored) Claude Code permission config for this folder only — `bypassPermissions` mode plus an explicit allow list, per user request. This file is machine-local and is never committed or pushed.

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
| NEAT and controller baselines | baselines done; NEAT paused | Binary + three-action + calibration diagnostics on fixed caches; NEAT only after reliable baseline |
| Three-action offline evaluation | done 2026-10-27 | `scripts/run_three_action.py`; 1-shot/3-shot Pareto in `results/pilot/banking77_three_action_*.json` |
| Temperature calibration | done 2026-10-27 | `scripts/calibrate_temperature.py`; T fit on train only; `results/pilot/banking77_calibration_3shot.json` |
| CV stability (3-fold OOF on 3-shot train) | done 2026-10-27 | `scripts/run_cv_stability.py`; choice beats numeric under both weight settings; `results/pilot/banking77_cv_stability.json` |
| Local Git/GitHub sync | passed | Commit `7a3458f0211ec1df2a311dec6517fe8d9e0fde59` pushed to `main` |
| GitHub-hosted experiments | disabled/not used | User requested all installation, data and model runs be performed on the user's computer. The CPU bootstrap GitHub Actions workflow has been removed to avoid executing experiments in GitHub's environment. |

## Three-action offline evaluation + calibration (2026-10-27)

All offline on fixed caches (train 1-shot n=77 / 3-shot n=231, dev n=231). Decisions use TRAIN-estimated mean costs only, never per-sample dev latency. 13/13 tests pass.

| Method (dev n=231) | 1-shot acc / ms | 3-shot acc / ms |
|---|---:|---:|
| Best fixed (tournament) | 0.5541 / 5766 | 0.5541 / 5766 |
| Best threshold cascade | 0.5584 / 5131 | 0.5541 / 4661 |
| Binary decision tree | 0.4329 / 2889 | 0.5325 / 4413 |
| Tri-action lam=0 | 0.4848 / 3993 | 0.5195 / 3798 |
| Tri-action lam=2e-05 | 0.4329 / 3058 | 0.4805 / 3214 |
| Tri-action + choice ablation lam=0 (3-shot only, C=0.2) | — | 0.5541 / 4026 |
| Tri-action + choice ablation lam=2e-05 | — | 0.5325 / 3406 |
| Oracle (post-hoc) | 0.7056 / 2871 | 0.7056 / 2871 |

Full Pareto tables: `results/pilot/banking77_three_action_1shot.json`, `results/pilot/banking77_three_action_3shot.json`.

Interpretation:
- Tri-action without category info beats binary LR but stays below best fixed; the binary action design was indeed part of the limitation, but features remain the binding constraint.
- The single controlled choice ablation (6 numeric + one-hot initial_choice, C=0.2, 3-shot only) matches best-fixed accuracy (0.5541) at ~30% lower mean cost (4026 vs 5766 ms). Hypothesis only: 77-category one-hot on 231 train samples carries overfit risk; needs seed/subset replication before any claim.
- Calibration (T=4.60 fit on 3-shot train): dev NLL 4.95→2.87, Brier 0.91→0.79, ECE 0.36→0.16. Routing with calibrated confidence: dev 0.5541 at 4921 ms vs raw 0.5541 at 4661 ms — reliability improves, routing does not. Keep calibration as analysis, not a core contribution.
- No 5-shot cache, no NEAT. Next: replicate choice-ablation stability across seeds/subsets before deciding on 5-shot.

## CV stability on 3-shot train cache (2026-10-27, offline)

3-fold OOF within `banking77_train3shot_cache.jsonl` (231 samples; fold k holds out the k-th sample of each class → 77 OOF / 154 fit; ≈2-shot scale). Unweighted LR is the primary probability estimator, balanced LR the control. Costs from fold-train only; fixed lambda grid; 16/16 tests pass.

| OOF (n=231) | unweighted acc / ms | balanced acc / ms |
|---|---:|---:|
| numeric lam=0 | 0.5844 / 5502 | 0.5455 / 3718 |
| choice lam=0 | 0.6061 / 5493 | 0.5801 / 3954 |
| numeric lam=2e-05 | 0.5758 / 4690 | 0.4978 / 3110 |
| choice lam=2e-05 | 0.6061 / 4738 | 0.5195 / 3199 |
| fixed tournament (train ref) | 0.6017 | 0.6017 |

Paired choice-vs-numeric OOF diff with stratified bootstrap 95% CI:
- unweighted lam=0: +0.0216 [0.0000, 0.0433]; lam=2e-05: +0.0303 [0.0087, 0.0519]
- balanced lam=0: +0.0346 [0.0087, 0.0606]; lam=2e-05: +0.0216 [0.0000, 0.0433]

Interpretation:
- The choice advantage replicates under BOTH probability settings (not an artifact of class_weight), at lam=0 and lam=2e-05 with lower or equal cost. Per the audit decision table this points to an independent confirmation experiment, not to 5-shot or NEAT.
- At higher lambdas both models collapse to near-single and the gap vanishes, as expected.
- Next (gated): hold out a 77-sample confirmation set from manifest train samples unused so far, run new Laya inference once with the locked config, evaluate once. No 5-shot, no test touch, no NEAT until then.

## Controller baselines on BANKING77 (2026-10-26)

Train cache: 77 class-balanced 1-shot examples from official `train`. Dev cache: 231 held-out examples from the same fixed manifest. All methods use only initial SINGLE features and cumulative cascade costs.

| Method | Dev accuracy | Mean cumulative ms | Mean cumulative calls |
|---|---:|---:|---:|
| Best fixed (tournament) | 0.5541 / 0.5541 | 5766.4 | 3.000 |
| Confidence threshold | 0.5584 / 0.5541 | 5131.2 / 4660.6 | 2.810 / 2.593 |
| Margin threshold | 0.5584 / 0.5541 | 5131.2 / 4660.2 | 2.810 / 2.593 |
| Entropy-confidence threshold | 0.5584 / 0.5541 | 5131.2 / 4660.6 | 2.810 / 2.593 |
| Logistic regression (binary, scaled) | 0.4329 / 0.4848 | 2827.1 / 3508.0 | 1.082 / 1.394 |
| Decision tree (depth 3, binary) | 0.4329 / 0.5325 | 2889.4 / 4412.6 | 1.100 / 2.065 |
| Small MLP (8 units, binary, scaled) | 0.4069 / 0.4069 | 2174.6 / 2043.8 | 1.035 / 1.000 |
| Random routing | 0.5152 | 4102.3 | 2.030 |
| Oracle (post-hoc) | 0.7056 | 2870.9 | 1.416 |

Columns for learned/threshold rows are 1-shot / 3-shot dev accuracy and cost. Labels are binary upgrade targets tied to the concrete TRAIN-chosen upgrade (tournament in both runs). Features are the locked 6-feature set with scaler fitted on train only for LR/MLP.

Interpretation:
- Threshold cascades match or slightly beat best fixed at lower cost, but the 1-shot gain is 1/231 samples and not a stable accuracy claim.
- After the label fix, decision tree improves from 0.4329 (1-shot) to 0.5325 (3-shot), closer to threshold behavior; LR also improves with more data.
- Dev failure modes (n=231): 27 tournament-only fixes, 12 wide-only fixes, 30 fixed by both, 31 single-correct-but-upgrade-wrong, 68 all-wrong, 63 all-correct.
- Upgrade-gain cases have lower median confidence (0.73) than upgrade-loss cases (0.99), so confidence carries signal but overlaps heavily.
- No 5-shot cache and no NEAT yet; decided after this correction.

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

## Next actions (updated 2026-10-27; test split stays locked)

1. Decide whether to launch the gated independent confirmation run: 77 held-out train-manifest samples, one locked-config inference, single evaluation. Needs new Laya inference (~30-60 min, single process, 2 threads).
2. Only if confirmation replicates the choice advantage: consider 5-shot cache, then NEAT vs fixed-topology baselines under matched budgets.
3. If confirmation fails: keep threshold cascade as the main baseline and do error analysis; do not scale up.
4. Keep all test splits locked; no test use for tuning, thresholds, model selection, or fitness.

## Resource limits and research integrity

- Run one inference process at a time; set `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, and `TOKENIZERS_PARALLELISM=false`.
- Keep peak memory under roughly 3.5 GB and stop if Windows becomes sluggish.
- Do not commit model weights, raw datasets, the virtual environment or large caches.
- Do not tune anything using final-test results.
- No new accuracy gains are claimed until measured, saved and checked.
