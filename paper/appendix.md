# Appendix: full per-seed tables (v4.1)

_Generated from committed result JSONs; numbers identical to main-text claims. All CIs are paired group-bootstraps (5000 resamples) unless noted._

## A. GA train-selected: dev report

| task | seed | ga_report | vs switch-only est [95% CI] | sig |
|---|---|---|---|---|
| code_laya | 17 | 0.4500 | -0.2000 [-0.3833, -0.0167] | SIG |
| code_laya | 23 | 0.4500 | -0.2000 [-0.3833, -0.0167] | SIG |
| code_laya | 41 | 0.4500 | -0.2000 [-0.3833, 0.0] | ns |
| code_vela | 17 | 0.6667 | 0.0000 [0.0, 0.0] | ns |
| code_vela | 23 | 0.6667 | 0.0000 [0.0, 0.0] | ns |
| code_vela | 41 | 0.6500 | -0.0167 [-0.05, 0.0] | ns |
| findver_laya | 17 | 0.6271 | -0.0169 [-0.0896, 0.0492] | ns |
| findver_laya | 23 | 0.6949 | 0.0508 [-0.0645, 0.1875] | ns |
| findver_laya | 41 | 0.7119 | 0.0678 [-0.0597, 0.2364] | ns |
| findver_vela | 17 | 0.4576 | 0.0000 [-0.12, 0.0909] | ns |
| findver_vela | 23 | 0.4576 | 0.0000 [-0.1207, 0.0909] | ns |
| findver_vela | 41 | 0.4576 | 0.0000 [-0.1137, 0.0921] | ns |

## B. GA train-selected: confirm report

| task | seed | ga_report | vs switch-only est [95% CI] | sig |
|---|---|---|---|---|
| code_laya | 17 | 0.5500 | -0.0167 [-0.2167, 0.1667] | ns |
| code_laya | 23 | 0.5500 | -0.0167 [-0.2, 0.1667] | ns |
| code_laya | 41 | 0.5500 | -0.0167 [-0.2, 0.1667] | ns |
| code_vela | 17 | 0.5667 | 0.0000 [0.0, 0.0] | ns |
| code_vela | 23 | 0.5667 | 0.0000 [0.0, 0.0] | ns |
| code_vela | 41 | 0.5333 | -0.0333 [-0.1167, 0.05] | ns |
| findver_laya | 17 | 0.6136 | -0.0227 [-0.0667, 0.0] | ns |
| findver_laya | 23 | 0.5455 | -0.0909 [-0.2, 0.0213] | ns |
| findver_laya | 41 | 0.5000 | -0.1364 [-0.25, -0.02] | SIG |
| findver_vela | 17 | 0.7045 | 0.0227 [-0.0488, 0.1026] | ns |
| findver_vela | 23 | 0.7045 | 0.0227 [-0.05, 0.102] | ns |
| findver_vela | 41 | 0.7045 | 0.0227 [-0.0488, 0.1] | ns |

## C. ES frozen-best: dev + confirm vs switch-only

| task | seed | dev est [CI] | sig | confirm est [CI] | sig |
|---|---|---|---|---|---|
| code_laya | 17 | -0.1833 [-0.3167, -0.05] | SIG | 0.0500 [-0.0833, 0.1833] | ns |
| code_laya | 23 | -0.2000 [-0.3667, -0.0167] | SIG | -0.0167 [-0.2, 0.1833] | ns |
| code_laya | 41 | -0.1667 [-0.3333, 0.0167] | ns | -0.0167 [-0.2, 0.1667] | ns |
| code_vela | 17 | 0.0000 [0.0, 0.0] | ns | 0.0000 [0.0, 0.0] | ns |
| code_vela | 23 | 0.0000 [0.0, 0.0] | ns | 0.0000 [0.0, 0.0] | ns |
| code_vela | 41 | 0.0000 [0.0, 0.0] | ns | 0.0000 [0.0, 0.0] | ns |

## D. NSGA-II dev-selected deploy vs fixed baseline

| task | seed | deploy genome | dev est [CI] | sig | confirm est [CI] | sig |
|---|---|---|---|---|---|---|
| code_laya | 17 | [0.3288, 0.8152, 0.0000] | -0.2000 [-0.3833, -0.0167] | SIG | -0.0167 [-0.2167, 0.1667] | ns |
| code_laya | 23 | [0.7211, 0.9486, 1.0000] | -0.1667 [-0.3, -0.0333] | SIG | 0.0333 [-0.1167, 0.1833] | ns |
| code_laya | 41 | [0.3452, 0.5531, 1.0000] | -0.1667 [-0.3333, 0.0167] | ns | -0.0167 [-0.2, 0.1667] | ns |
| code_vela | 17 | [0.2896, 0.7661, 1.0000] | 0.0000 [0.0, 0.0] | ns | -0.0167 [-0.0837, 0.05] | ns |
| code_vela | 23 | [0.1303, 0.9486, 1.0000] | 0.0000 [0.0, 0.0] | ns | 0.0000 [0.0, 0.0] | ns |
| code_vela | 41 | [0.4770, 0.9020, 0.0000] | -0.0333 [-0.15, 0.0833] | ns | 0.0000 [-0.15, 0.15] | ns |
| math_laya | 17 | [0.1509, 0.0000, 0.8152] | 0.0000 [0.0, 0.0] | ns | 0.0000 [0.0, 0.0] | ns |
| math_laya | 23 | [0.0836, 0.0000, 0.2210] | 0.0000 [0.0, 0.0] | ns | 0.0000 [0.0, 0.0] | ns |
| math_laya | 41 | [0.1903, 0.0000, 0.9926] | 0.0000 [0.0, 0.0] | ns | 0.0000 [0.0, 0.0] | ns |
| math_vela | 17 | [0.2709, 0.0000, 0.9605] | -0.0667 [-0.1364, -0.0159] | SIG | -0.0500 [-0.1129, 0.0] | ns |
| math_vela | 23 | [0.2674, 0.0000, 0.2210] | -0.0667 [-0.1333, -0.0161] | SIG | -0.0500 [-0.1111, 0.0] | ns |
| math_vela | 41 | [0.3810, 0.0000, 0.9926] | -0.0833 [-0.1607, -0.0172] | SIG | -0.0833 [-0.1613, -0.0167] | SIG |

## E. Held-out transfer (frozen train genomes -> target dev/confirm)

| source -> target | dev mean_est | dev SIG cells | confirm mean_est | confirm SIG cells |
|---|---|---|---|---|
| code_laya_on_code_laya | -0.2000 | 17,23,41 | -0.0167 | none |
| code_laya_on_code_vela | -0.0333 | none | 0.0000 | none |
| code_laya_on_findver_laya | 0.0678 | none | -0.1364 | 17,23,41 |
| code_laya_on_findver_vela | 0.0000 | none | 0.0227 | none |
| code_vela_on_code_laya | -0.0167 | none | -0.0111 | none |
| code_vela_on_code_vela | -0.0056 | none | -0.0111 | none |
| code_vela_on_findver_laya | 0.0169 | none | -0.0227 | none |
| code_vela_on_findver_vela | -0.0056 | none | 0.0151 | none |
| findver_laya_on_code_laya | -0.2000 | 17,23,41 | -0.0167 | none |
| findver_laya_on_code_vela | -0.0056 | none | 0.0056 | none |
| findver_laya_on_findver_laya | 0.0339 | none | -0.0833 | 41 |
| findver_laya_on_findver_vela | 0.0000 | none | 0.0076 | none |
| findver_vela_on_code_laya | -0.2000 | 17,23,41 | -0.0167 | none |
| findver_vela_on_code_vela | -0.0333 | none | 0.0000 | none |
| findver_vela_on_findver_laya | 0.0678 | none | -0.1364 | 17,23,41 |
| findver_vela_on_findver_vela | 0.0000 | none | 0.0227 | none |

## F. Fault-map cells (discover severity, Wilson LB)

| cell | n | S_discover | Wilson LB | dev confirmed | confirm confirmed |
|---|---|---|---|---|---|
| math/vela/mid | 14 | 0.9286 | 0.6853 | True | True |
| math/laya/mid | 13 | 0.9231 | 0.6669 | True | True |
| math/vela/low | 45 | 0.8889 | 0.7650 | True | True |
| math/laya/high | 41 | 0.7561 | 0.6066 | True | True |
| code/vela/mid | 30 | 0.5000 | 0.3315 | False | False |
| findver/laya/low | 14 | 0.4286 | 0.2138 | False | False |
| findver/vela/mid | 30 | 0.4000 | 0.2459 | True | True |
| findver/laya/mid | 18 | 0.3889 | 0.2030 | True | True |
| findver/laya/high | 28 | 0.3571 | 0.2071 | True | True |
| code/laya/low | 25 | 0.3200 | 0.1720 | True | False |
| findver/vela/high | 19 | 0.3158 | 0.1536 | True | False |
| code/laya/high | 14 | 0.2857 | 0.1172 | True | False |
| code/vela/low | 25 | 0.2800 | 0.1428 | True | True |
| findver/vela/low | 11 | 0.2727 | 0.0975 | True | True |
| code/laya/mid | 21 | 0.0476 | 0.0085 | True | True |

## G. Action complementarity (EXP-01)

| cell | base | base_acc | oracle | headroom [CI] | rep/harm/neut |
|---|---|---|---|---|---|---|
| code_laya | noul | 0.5500 | 0.7833 | 0.2333 [0.1333, 0.35] | 14/15/31 |
| code_vela | switch | 0.5167 | 0.5833 | 0.0667 [0.0167, 0.1333] | 4/10/46 |
| findver_laya | noul | 0.5333 | 0.6167 | 0.0833 [0.0182, 0.1538] | 5/6/49 |
| findver_vela | switch | 0.5500 | 0.6500 | 0.1000 [0.0351, 0.1692] | 6/7/47 |
| math_laya | single rep | 0.2000 | 0.2000 | 0.0 | - |
| math_vela | single rep | 0.1000 | 0.1000 | 0.0 | - |

## H. Utility predictability (EXP-02)

| cell | champion | OOF AUROC (logreg/tree) | dev net | confirm net | n_pos tr/dev/conf |
|---|---|---|---|---|---|
| code_laya | logreg | 0.6739/0.7391 | 0.2000 | 0.0167 | 14/23/17 |
| code_vela | logreg | 0.7160/0.6460 | 0.0333 | -0.0167 | 10/7/10 |
| findver_laya | tree | 0.4545/0.4655 | 0.0169 | 0.0227 | 5/6/7 |
| findver_vela | logreg | 0.9191/0.7938 | 0.0339 | -0.0227 | 7/6/2 |

## I. Reproducibility index

See `results/v4_1/manifests/protocol_metrics_index.json` (machine-readable file inventory) and `results/v4_1/manifests/derived_provenance.json` (generator commands per derived file). Full test log: `results/v4_1/manifests/pytest_full_*.log`.
## J. Intent-track supplement (fixed-retrieval effects, not evolutionary gains)

Moved here from §9 to keep the main results focused on the evolutionary-routing
question. On a separate intent track (CLINC150/BANKING77, v4 group manifests,
full label sets, same-row Laya/Vela pairs; no GP/NSGA-II search run; no test
split touched):

**A1 native baselines (full label sets, search_train n=100/row-matched).**
CLINC150: Laya 0.61 vs Vela 0.57 (paired diff −0.04, 95% CI [−0.16,+0.08]).
BANKING77: Laya 0.44 vs Vela 0.54 (diff +0.10, CI [−0.02,+0.21]).
Neither gap is significant; Vela is 2.5–4× faster per call. A subset-candidate pilot
that inflated Laya CLINC150 to 0.86 was caught and fixed by the full-set rerun.

**E2 menu headroom (dev n=60/task, TF-IDF top-10 menus, identical rows full-vs-menu).**
Menu recall@10 is 0.9367 (CLINC150) / 0.9767 (BANKING77) vs random 0.11.
Restricted-menu Choice beats full-menu Choice in 3/4 paired CIs:
BANKING77 Laya +0.333 [0.2,0.4667], Vela +0.133 [0.0,0.2667] (ns);
CLINC150 Laya +0.25 [0.1333,0.3667], Vela +0.217 [0.1,0.3333].
Menus also cut mean latency (e.g. Laya CLINC150 2256→585 ms).
This is a *fixed-retrieval* effect, not an evolutionary gain.

## K. Paired macro-F1 CIs, GA-best vs fixed switch and vs random (all pools)

| task | pool | comp | SIG seeds (est [CI]) |
|---|---|---|---|
| code_laya | train | ga_minus_switchonly | all ns |
| code_laya | train | ga_minus_random | all ns |
| code_laya | dev | ga_minus_switchonly | all ns |
| code_laya | dev | ga_minus_random | all ns |
| code_laya | confirm | ga_minus_switchonly | all ns |
| code_laya | confirm | ga_minus_random | all ns |
| code_vela | train | ga_minus_switchonly | all ns |
| code_vela | train | ga_minus_random | all ns |
| code_vela | dev | ga_minus_switchonly | all ns |
| code_vela | dev | ga_minus_random | all ns |
| code_vela | confirm | ga_minus_switchonly | all ns |
| code_vela | confirm | ga_minus_random | all ns |
| findver_laya | train | ga_minus_switchonly | all ns |
| findver_laya | train | ga_minus_random | all ns |
| findver_laya | dev | ga_minus_switchonly | all ns |
| findver_laya | dev | ga_minus_random | all ns |
| findver_laya | confirm | ga_minus_switchonly | all ns |
| findver_laya | confirm | ga_minus_random | all ns |
| findver_vela | train | ga_minus_switchonly | 17: 0.1202 [0.0043, 0.2271]; 23: 0.1202 [0.0062, 0.2282]; 41: 0.1202 [0.0067, 0.2229] |
| findver_vela | train | ga_minus_random | all ns |
| findver_vela | dev | ga_minus_switchonly | all ns |
| findver_vela | dev | ga_minus_random | all ns |
| findver_vela | confirm | ga_minus_switchonly | all ns |
| findver_vela | confirm | ga_minus_random | all ns |

Primary comparison (pre-registered): GA vs fixed switch on confirm, macro-F1 paired CI: all 12 cells ns. All other cells exploratory, no family-wise control.
