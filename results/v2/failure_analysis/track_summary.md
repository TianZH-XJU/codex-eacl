# Post-gate diagnostic track — stage verdicts (A0–A3 executed, A4+ blocked)

All runs: local Windows CPU, single model process, 2 threads. No test labels used.
New-track manifests: `results/v2/data_manifest/`. Frozen protocol: `results/v2/audit/protocol_freeze.json`.

## A0 audit fixes — PASS
- Group-aware OOF, train-only majority, cluster bootstrap, paired-join hash checks, semantic Noul schemas, function/report/problem group manifests, protocol freeze. 12 new tests pass (60/60 total at commit).
- Recomputation with fixed stats confirms old Gate 1 verdicts (LR OOF below majority everywhere).

## A1 native baselines — recorded (no gate to pass)
- ProcessBench Laya single: error_acc 0.21/0.29 (train/dev), correct_acc **0.0/0.0** (never predicts all-correct), harmonic F1 = 0. Random-step expected ≈ 0.14–0.16.
- Code TF-IDF LogReg macro-F1 0.606 / LinearSVM 0.600 (search_dev) vs Laya single 0.52 pilot → model adds nothing over bag-of-words.
- FinDVer TF-IDF macro-F1 0.42–0.51, near chance like both models.
- Vela paired baselines: math 0.058 vs Laya 0.125; code 0.483/0.500 vs 0.517/0.533; findver 0.433/0.533 vs 0.567/0.617. All paired CIs cover zero or favor Laya.

## A2 representation ablation — DONE, no representation rescues the tasks
| Setting (n=20, new clean rows) | legacy Noul | semantic Noul | Choice |
|---|---|---|---|
| Laya code (gold 14S/6V) | 0.50 | 0.35 (collapse→vulnerable) | 0.80 (collapse→secure, +2 over majority) |
| Laya findver (gold 12T/8F) | 0.60 | 0.70 | 0.60 (all-True = majority) |
| Vela code | — | 0.70 (all-secure = majority) | 0.60 |
| Vela findver | — | 0.50, AUROC **0.135** (inverted) | 0.60 (all-True = majority) |
- Noul AUROCs: Laya code legacy 0.75 vs semantic 0.62; Laya findver 0.56/0.59; Vela code 0.62.
- Threshold sweep (exploratory, same 20 rows): best balanced accuracies 0.68/0.71/0.62/0.50 — post-hoc on n=20, not deployable.
- Conclusion: representation flips the collapse direction; nothing discriminates credibly.

## A3 parameter sensitivity (OFAT, Laya, semantic Noul, same 20 rows) — FAIL
| Config | code Δ (CI) | findver Δ (CI) | flips |
|---|---|---|---|
| head_max_len=96 | 0.00 [0,0], 0 flips | 0.00 [0,0], 0 flips | none — pure cost change |
| max_len=384 | −0.05 [−0.15,0], 1 flip (break) | −0.15 [−0.30,0], 3 flips (all breaks) | harmful-only |
- No config improves quality repeatably on either task. Vela has no per-call budget knobs (audited), so no Vela OFAT exists to run.
- **Gate verdict: A3 FAIL → A4 interaction/A5 ES/A6 conditional-NEAT/A7-A10 blocked per protocol.**

## What remains allowed
- Reporting, paper updates, and any *new* track registered separately with fresh data.
- No locked-test evaluation (no gate passed). No optimizer packages installed (correctly so).
- ONNX backends: not_run (Torch stable, RSS < 2GB).
