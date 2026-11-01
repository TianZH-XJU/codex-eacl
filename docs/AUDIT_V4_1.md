# AUDIT_V4_1 — Stage 0 verification (2026-11-01, offline, zero new inference)

Scope: verify E2 menu-pilot evidence chain and correct remote-audit path errors.
Method: `scripts/audit_e2_results.py` (18 checks, ALL PASS at commit `e4b61d6`).

## 1. Path-existence corrections (evidence, not assertion)

A remote check reported five scripts "not found". Re-checked against local `main`
(`e4b61d6`, in sync with `origin/main`):

| Claimed path | Verdict | Evidence |
|---|---|---|
| `scripts/prepare_processbench.py` | EXISTS | listed in `scripts/`, has `tests/test_prepare_processbench.py` |
| `scripts/prepare_codexglue_defect.py` | EXISTS | listed in `scripts/`, has `tests/test_prepare_code_finance.py` |
| `scripts/prepare_findver.py` | EXISTS | listed in `scripts/` |
| `scripts/evaluate_strategy_complementarity.py` | EXISTS | listed in `scripts/`, has `tests/test_evaluate_strategy_complementarity.py` |
| `scripts/run_budgeted_controller_cv.py` | MISSING | confirmed absent; BLSC multi-step controller has no runner yet |
| `scripts/run_menu_evo_pilot.py` | WRONG NAME | actual file is `scripts/run_evo_menu_pilot.py` (transposed); exists with `tests/test_evo_menu_pilot.py` |

Conclusion: 4 of 6 "not found" verdicts were wrong-path/stale-view lookups. Only
`run_budgeted_controller_cv.py` is genuinely unimplemented. Corrections applied here;
no source files were renamed (history stays stable).

## 2. E2 evidence chain (18/18 checks pass)

- `project_git_sha` in all four pilot files resolves to real commits; each SHA is the
  end-of-run HEAD. `git log --follow scripts/run_evo_menu_pilot.py` shows a single
  commit (`43f2d31`): the runner code was identical across all four runs, so end-of-run
  SHAs are valid provenance (mid-run commits touched only docs/results).
- Consistency: all four runs agree on task=banking77, n_rows=30, n_train=2000, seed=17,
  device=cpu, threads=2. Budgets as designed (24/24/12/12).
- Recomputed from raw pilot files: Vela range 0.6333–0.80, Laya 0.667–0.80; flip sets
  7/12 Vela and 6/12 Laya; effective distinct space 12 (norm no-op); best{} pointers
  reference true argmax genomes; determinism files match the joint analysis exactly.

## 3. STATUS.md known conflicts (fix plan, v3 numbers untouched)

1. Header says A1 covers math/code/findver/intent paired baselines, but the detailed A1
   sections emphasize CLINC150/BANKING77 intent numbers; math/code/findver native tables
   live in older sections with different row sets. Fix: STATUS gets a "current track"
   block (v4 intent+E2) separated from "historical track" blocks (old NEAT/BANKING77);
   no old numbers rewritten.
2. Test-count wording mixes eras (`3 passed` from the old smoke era vs current full suite).
   Fix: single authoritative count = latest full `pytest tests/ -q` (86 at `e4b61d6`);
   historical counts stay labeled with their commit/date.
3. Old NEAT controller history sits beside EvoTIDE state. Fix: NEAT sections keep a
   `HISTORICAL — frozen, do not extend` banner; current mainline is v4 menu/track work.

## 4. E2 conceptual status (confirmed, not downgraded)

- Exhaustive-24, random-12, and config enumeration are search-space/landscape tools,
  NOT evolutionary algorithms (no population, selection, variation operators).
- E2 verdict stands: fixed-retrieval menu effect real (3/4 paired CIs); no genome
  reliably beats best-fixed menu; GP unjustified at n=30 noise floor.
- Cost projections in `e2_joint_analysis.json` (n=500, GP hours) are projections,
  explicitly labeled, never executed. Any future reader must not cite them as runs.

## 5. Stage 0 exit gate

- [x] E2 evidence independently reproduced from raw files (18/18).
- [x] Path corrections recorded with file-level evidence.
- [x] SHA provenance explained (end-of-run HEAD + frozen runner).
- [x] STATUS conflict list with fix plan that freezes v3 numbers.
- [ ] STATUS.md restructure itself → next commit (docs-only, offline).
- [ ] `run_budgeted_controller_cv.py` still missing → Stage 6, not Stage 0.

## 6. v4.1 addendum (2026-11-01)

- Full suite is now 130 passed (\pytest tests/ -q\); 86 was the count at e4b61d6.
- Cited CIs re-verified offline into results/v4_1/manifests/recomputed_cis.json
  (A1 paired diffs [-0.04 CI -0.16/0.08; +0.10 CI -0.02/0.21]; surrogate
  agreements 0.6167/0.85/0.75/0.7833, all < 0.9 bar). No v4 result file was modified.
- Train/dev/confirm split discipline enforced by scripts/eval_trainselected_on_dev.py
  (pool-disjointness assertion) and scripts/test_evo_significance.py (paired-join check).
- Known wording fixes applied to paper/STATUS: transfer max off-diagonal is 0.0
  (not -0.006); ablation-vs-switch and traditional ga-vs-threshold are 11/12 ns
  (not all-ns); there is no results/v4/menu path (always results/v2/menu).

## 7. Follow-up fixes applied (this round)

- Budget pre-check + memoization in all runners; dead budget params wired ((mu,lambda) in NSGA-II).
- Paired-join and pool-disjoint guards in eval/selection scripts with regression tests.
- Fitness task-primary metric plumbed (accuracy default frozen legacy; v4.1 uses macro-F1).
- Weight hash pinned and verified at Vela load; laya package version checked.
- Full suite: 132 passed.
