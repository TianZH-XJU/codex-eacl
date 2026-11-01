# v4.1 Phase-1 completion record (fix & calibrate)

Base: `dd52fe9`. HEAD at writing: see `git log` (this file is informational;
SHAs below are the commits that landed each fix).

## C-01..C-10 (pre-existing fixes, verified not reverted)

| ID | Fix | Proof |
|----|-----|-------|
| C-01 | Genome round-trip | `tests/test_evolution.py::test_repair_and_vector_roundtrip`, `test_abstain_roundtrip_nontrivial_genes` |
| C-02 | TF-IDF leak guard | `scripts/build_native_matrix.py::code_rows_for_groups` + `tests/test_build_native_matrix.py` (4 tests, incl. cross-split duplicate rejection) |
| C-03 | Train-members matrix | same as C-02 + `results/v2/data_manifest/*` + `tests/test_v2_group_splits.py` |
| C-04 | Output validation | `src/models/base.py::validate_answers/validate_prob_map` + `tests/test_model_interface.py` (span/set rejected, NaN/Inf rejected) |
| C-05 | Fitness fail-closed | `src/evolution/fitness.py::clean_p` + `evaluate` missing-branch + `tests/test_evolution.py::test_evaluate_garbage_rows_fail_closed_and_empty_raises` |
| C-06 | Budget boundaries | `BudgetTracker.charge_eval` pre-charge + `tests/test_evolution.py` budget tests (incl. `test_budget_always_enforced_without_caller_tracker`) |
| C-07 | Partial-generation merge | `es.py`/`nsga2.py` scored-children merge + `test_mid_generation_budget_keeps_scored_children_{es,nsga2}` |
| C-08 | Cache/memo protection | memo-key round-trip + `test_ga_memoizes_duplicates_without_unique_charge` |
| C-09 | Model provenance | `build_policy_cache.py` provenance block (weights SHA/verified, revision) + `tests/test_v4_1_protocol.py` |
| C-10 | Interface tests | `tests/test_model_interface.py` (9 tests) |

Four-layer acceptance: (1) unit tests — 196 passed at HEAD; (2) end-to-end —
`tests/test_run_policy_evolution.py` runs GA+ES on synthetic caches with dev
reporting; faultmap/transfer/significance have integration tests on real caches;
(3) traceability — every search file carries `run_manifest` + `project_git_sha`,
every derived file is logged in `results/v4_1/manifests/derived_provenance.json`;
(4) full suite log — `results/v4_1/manifests/pytest_full_*.log` (see below).

## C-11..C-14 (audit-found, fixed this phase)

| ID | Fix | Proof |
|----|-----|-------|
| C-11 | Significance tags per-algo | `test_evo_significance.py` f-string tags; `es_significance.json` now `es_minus_random`; `tests/test_run_policy_evolution.py::test_significance_tags_carry_algorithm` |
| C-12 | ES gens from actual logs | `ga_gens_effective()` + post-loop `max(len(log))` with `gens_rule`; es_train files gens_effective 2->3, search outputs bit-identical |
| C-13 | Dynamic dev keys | `entry[tag]` keyed by `args.algo`; `test_entry_keys_follow_algorithm` + leak-guard negative test |
| C-14 | NSGA-II dev selection | `scripts/eval_nsga2_deploy.py` (frozen train pareto -> dev rule select -> confirm report, paired CIs); `nsga2_deploy_devconfirm.json`; supersedes in-sample `nsga2_deploy_significance.json` |

## E-01..E-07 (experiment/statement fixes)

- E-01: GA tables carry split/metric/n/algo/baseline (`train_search_*eval.json` keys + paper §9 + STATUS rows).
- E-02: `scripts/eval_es_deploy.py` + `es_deploy_devconfirm.json` (frozen ES best -> dev+confirm, paired CIs vs switch-only; dev 2/6 SIG negative, confirm 0/6).
- E-03: in-sample NSGA-II verdict retained with in-sample label; held-out verdict from C-14 file.
- E-04: transfer scheme A (in-sample label in file + paper + STATUS) + optional EXP-05 held-out evidence (separate file, frozen genomes).
- E-05: "E5 diagnostic fault map (MAP-Elites-inspired, not an evolutionary optimizer)" unified in STATUS/paper/script docstring.
- E-06: paper wording = "raising the cap produced no gain under current GA config; the cap was underutilized (22/17/23 of 48), not proven sufficient".
- E-07: +6/60 chain = `complementarity_screen.json` + `scripts/complementarity_screen.py` + cited in STATUS/paper.

## Paper statement table (plan 1.4)

All 8 rows applied in `paper/main.md` v0.3 + §6/§7/§8 rewrite (H1/H2/H3 + EXP-01..05 numbers). No "no evolutionary search" narrative remains; NEAT/GP blocking stated with reopen criteria.

## Phase-1 gate verdict

- [x] No known code P0/P1 open (4 audit rounds closed; suite green).
- [x] No train/held-out mixup (C-14; pool labels in every file; RNG-order caveat documented).
- [x] No statement contradicts evidence (docs audit + fixes).
- [ ] Locked test: NOT run (no locked cache built; pipeline not frozen). Correctly pending per protocol — must not run until freeze.

Phase-1: PASSED (locked test explicitly deferred, not skipped).
