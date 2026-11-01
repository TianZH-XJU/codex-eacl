# Configuration Sensitivity, Not Search: A Negative Result on Evolving Inference Configurations for Frozen Structured Decision Models

*Draft v0.1 — 2026-10-31. Work-in-progress short paper for EACL 2027 SRW.
All numbers below are measured on the author's local CPU machine; nothing is simulated.
Test splits were never used for tuning. No evolutionary search stage was reached: every
search gate failed, so this paper reports sensitivity, baselines, and a failed-gate analysis.*

*Addendum v0.2 — 2026-11-01 (v4 EvoTIDE track, frozen v3 unchanged). New intent-track
results (CLINC150/BANKING77, full label sets, paired Laya/Vela) and a validated menu-restriction
effect are reported in §9. No GP/NSGA-II search has run; surrogate-cache fitness was rejected
after three-way validation, so any future E2 search must use real model calls.*

## 1. Introduction

Structured decision models such as Laya and Vela answer typed questions (choice, yes/no,
scored judgments) in a single forward pass. A natural hope is that *how* we ask —
context budgets, candidate grouping, option order, shortlists — can be optimized per task
by black-box search, improving accuracy at matched compute cost. We set out to test exactly
this: can evolutionary search discover better native inference configurations for frozen
structured decision models under CPU and memory constraints?

Our answer, on three intent/judgment-style tasks and one mathematical error-localization
task, is negative — but instructive. We find that (1) the candidate configurations barely
move predictions at all, except when they make them worse; (2) where movement exists, it
does not replicate across data splits; (3) a newer, larger-context model (Vela 2.0 0.3B)
does not beat the older baseline on these tasks either. Because our protocol pre-registered
gates that block search when no tunable signal exists, we stopped *before* running any
large evolutionary comparison, rather than after burning the budget. We argue this is the
correct outcome of a gated protocol, and we release the full audit trail so the negative
result is checkable.

Contributions: (i) a frozen, auditable evaluation protocol for native inference
configurations (single-process CPU, pinned revisions, per-config cost accounting);
(ii) paired Laya-vs-Vela baselines on identical rows across three tasks; (iii) a
pre-registered configuration-sensitivity study with a failed Gate 4; (iv) evidence that
on step-localization Choice tasks both models collapse to early options, which explains
why configuration search has nothing to work with.

## 2. Related Work

Black-box optimization (CMA-ES, DE, GA/NSGA-II, Bayesian optimization/TPE, random search)
is standard for hyperparameter tuning. NEAT evolves both topology and weights, originally
for control. Recent work motivates input-adaptive computation (Damani et al., ICLR 2025:
*Learning How Hard to Think*) and warns that confidence-based cascading can fail when
downstream error structure is ignored (Jitkrittum et al., NeurIPS 2023). Routing/cascading
unification (Dekoninck et al., 2024) frames our cost accounting. Our study differs: the
object being optimized is the *native inference configuration* (budgets, grouping,
ordering), not prompts, weights, or a routing policy over frozen actions. A prior phase
of this project tried evolving a post-hoc router over frozen Laya actions (SINGLE/WIDE/
TOURNAMENT) and likewise found no learnable signal; those experiments are retained as
preliminary evidence and are not reused here.

## 3. Method

**Models (frozen).** Laya 0.4.16 (source `b9fe351`), checkpoint
`convaiinnovations/laya` rev `7b928d8`; Vela 2.0 0.3B rev `d6f03aa` (Torch FP32, local
snapshot, SHA-256 manifest committed). One model resident at a time; CPU-only, 2 threads,
per-process RSS ceiling 3.5 GB (observed peaks ≤ 2.2 GB).

**Configuration space (pre-registered).** Direct path: `max_len`, `head_max_len` bins;
option-order templates (ORIGINAL/REVERSE/...); tournament `group_size`; shortlist `k`;
long-window `window`/`stride` (long inputs only). Every candidate passes through a
`validate_and_repair` gate (group_size ≥ 2, no incompatible shortlist+order combos,
no out-of-checkpoint budgets). Costs are cumulative (SINGLE + upgrades) with real rounds,
windows, and embedding calls counted — never "one API call = one forward".

**Gates.** Gate 0: API smoke. Gate 1: strategy complementarity on a small pilot
(error overlap, only-A fixes, predictability of gains). Gate 4: configuration sensitivity
— at least one config must change quality repeatably (paired bootstrap CI excluding zero
on train *and* dev). Only then would a full ES comparison (Stage 5) run. Locked test sets
exist but were never touched, as no gate passed.

## 4. Experimental Setup

**Tasks.** (1) ProcessBench error-step localization as Choice (3,400 rows / 2,842 problems;
official `prediction == label` mapping verified against upstream eval code; -1 = all
correct, else 0-based earliest-error index; problem-grouped 60/20/20 split, seed 17).
(2) CodeXGLUE Defect Detection, official 21,854/2,732/2,732 split (60 exact-duplicate
functions documented; 13 cross-split). (3) FinDVer claim verification, testmini carved
by report into controller train/dev (155 reports span official testmini/test — documented,
test never used). BANKING77 pins retained as history only.

**Paired comparison.** Both models see byte-identical state text and criteria on the same
rows (a state-text mismatch bug was caught, fixed, and the affected Vela run repeated).

## 5. Results

**Baselines (defaults, paired rows).** ProcessBench train pilot (n=120): Laya single 0.125,
Vela choice 0.058 (Δ −0.067, 95% CI [−0.133, 0.0]). Code valid pilot (n=60): Laya
noul 0.517 vs Vela 0.483; switch-vs-switch 0.533 vs 0.500. FinDVer dev carve (n=60):
Laya 0.567 vs Vela 0.433 (noul); 0.617 vs 0.533 (switch). Vela is consistently faster
per call (e.g. 1.7 vs 3.0 ms mean on math) but never more accurate; every paired CI
covers zero or favors Laya. Vela runs 256–8192 input tokens on CPU within 1.7–2.0 GB RSS;
latency grows super-linearly (2.4 s at 256 tokens → 41 s at 8192), so long-doc sweeps
would be honest but impractical — a resource finding in itself.

**Gate 1 (complementarity, pilots n=50–120).** Math: tournament's correct set is *identical*
to single's (only-fix 0, new errors 0); wide fixes 2, breaks 11; oracle headroom +0.017;
gain-predicting LR equals majority. Code: upgrades fix 1 each, break 15 each; LR below
majority. Finance: switch +0.05 raw but unpredictable (LR below majority); wide net zero.
Verdict on all three: no exploitable complementarity → no NEAT routing, no controller
training. (Recorded before the present study; retained as motivation.)

**Gate 4 (config sensitivity, math, n=100 train + n=60 dev).** Train: five of six configs
have Δ CIs exactly [0,0] (predictions unchanged, only added cost); wide_max1024 is
significantly *harmful* (Δ CI [−0.14,−0.01], 57 flips); order_reverse Δ [−0.02,+0.06]
(18 flips, CI covers zero). Dev replication: wide_max1024 harm does *not* replicate
([−0.1,+0.1]); tournament_g8 flips from neutral to harmful ([−0.133,−0.017]); nothing
improves quality repeatably. **Gate 4 FAILED** → no Stage 5 ES comparison, no Stage 9
locked test. Optimizer packages (cma/optuna/…) were deliberately never installed.

## 6. Analysis: why there is nothing to search

The pilot prediction distributions explain the nulls. On ProcessBench both models'
choices collapse onto early steps (Laya: 113/120 on steps 0–5, zero sentinel predictions;
Vela similar with slight spread, also zero sentinel). When the base policy cannot even
represent whole regions of the label space (late steps, all-correct), budget parameters
can only reshuffle mass within the collapsed region — hence identical tournament/single
outcomes and unpredictable gains. Vela's 8k context and newer encoder do not rescue the
task: its accuracy is *lower* than Laya's. The binding constraint is the models'
task behavior, not the search budget — consistent with the failed-gate decision to not
run evolution at all.

## 7. Limitations

Small pilots (n=60–120) limit precision, but CIs are reported throughout and the verdicts
rely on replication (train *and* dev), not point estimates. We did not test shortlist or
long-window paths (no evidence they were needed; shortlist needs recall@k validation first).
Vela's training-data provenance (includes MASSIVE; router-style synthetic decisions) is
disclosed; we excluded MASSIVE as a confirmatory benchmark. Test splits remain locked and
unused — a consequence, not an oversight, of gated protocol.

## 8. Conclusion

We asked whether evolutionary search can find better native inference configurations for
frozen structured decision models. Our gated protocol answered an earlier question first —
*is there anything responsive to tune?* — and the answer, replicated across models,
tasks, and splits, was no. The contribution is the audit trail: pinned revisions,
paired inputs, cost-accounted configs, pre-registered gates, and published failures.
For these tasks, the next step is not more search but better task adaptation (e.g.
counteracting position collapse) or honest model comparison — both laid out as explicit
follow-ups, not claims.

## 9. v4 addendum: intent track — menu restriction helps, surrogate fitness rejected (2026-11-01)

v3 is frozen above and unchanged. On a new intent track (CLINC150/BANKING77, v4 group
manifests, full label sets, same-row Laya/Vela pairs) we report three measured outcomes.
No GP/NSGA-II search has run; no test split was touched.

**A1 native baselines (full label sets, search_train n=100/row-matched).**
CLINC150: Laya 0.61 vs Vela 0.57 (paired diff −0.04, 95% CI [−0.16,+0.08]).
BANKING77: Laya 0.44 vs Vela 0.54 (diff +0.10, CI [−0.02,+0.21]).
Neither gap is significant; Vela is 2.5–4× faster per call. A subset-candidate pilot
that inflated Laya CLINC150 to 0.86 was caught and fixed by the full-set rerun.

**E2 menu headroom (dev n=60/task, TF-IDF top-10 menus, identical rows full-vs-menu).**
Menu recall@10 is 0.9367 (CLINC150) / 0.9767 (BANKING77) vs random 0.11.
Restricted-menu Choice beats full-menu Choice in 3/4 paired CIs:
BANKING77 Laya +0.333 [0.2,0.4667], Vela +0.133 [0.0,0.2667];
CLINC150 Laya +0.25 [0.1333,0.3667], Vela +0.217 [0.1,0.3333].
Menus also cut mean latency (e.g. Laya CLINC150 2256→585 ms).
This is a *fixed-retrieval* effect, not an evolutionary gain.

**Surrogate rejection (three-way: full-real vs restricted-real vs cached-restriction, n=30).**
Restricted-real vs cached-restriction agreement is 0.62 (Laya/banking77), 0.75
(Vela/banking77), 0.85 (Laya/clinc150), 0.78 (Vela/clinc150) — all below the 0.9 bar.
Cached full-distribution argmax does not reproduce restricted-menu calls, so any E2
search must use real model calls for fitness. E2 GP scope (tiny real-call pilot vs
record-as-blocked) is pending explicit eval-budget approval and is not claimed here.

**Distinct-view batching (Vela, single-sample diagnostic, 4 distinct questions).**
Categorical decisions match 4/4 (choice keys, noul pos/neg, score argmax level), but
continuous outputs shift substantially (choice L1 0.29–0.35, noul |Δp| 0.13, score
|Δexpected| 0.16) — relevant because fusion/calibration consumes those values.
Latency 3222→1095 ms total (per-q 806→274 ms), tokens 1877→542, RSS ~1.7 GB.
Fixed order (warmup→separate→batched), single sample: an engineering clue, not a
distinct-query speedup claim. A randomized multi-row interleaved study is required
before any throughput conclusion.

**E2 menu-parameter pilot (banking77, n=30 dev rows, real calls; 2026-11-01).**
Genome = {retriever × k × norm × order} = 24 combos; `norm` later proven a no-op
(TfidfVectorizer lowercases by default; effective space 12). Exhaustive-24 + random-12,
both models, same 30 rows. Vela range 0.633–0.80 (best char/k=10, 0.80); random-12 best
0.833 on identical params (1-row cross-process flip). Laya range 0.667–0.80 (best
char/k=5, 0.80); random-12 ties 0.80, with one genome flipping 3 rows across processes
(word/k=10/shuffled: 0.70→0.80). Cross-model genome-accuracy correlation is 0.092 —
"good menu" does not transfer across models. Determinism check (same genome, same rows,
two passes, one process): 0 flips both models (Vela 0.767/0.767, Laya 0.70/0.70), so the
cross-process flips are process-level noise, and adjacent-genome gaps (0.033–0.067) sit
inside it. Verdict: the menu landscape is flat within noise on n=30; GP at this n would
evolve noise. Repeatable selection needs n≈500, whose CPU cost is prohibitive on this box
(exhaustive-24 ≈108 min Laya / 30 min Vela; GP-300 ≈22.6 h Laya / 6.3 h Vela, real calls).
E2 search is therefore BLOCKED pending budget or a narrower space; the fixed-retrieval
menu effect (3/4 paired CIs) stands unchanged and unclaimed by any evolutionary method.

**Stage 3 unified native matrix (v4.1 adapters, search_dev, n=59–60, group-safe; 2026-11-01).**
Same rows per task for both models, full label sets, real calls:

| task / rep | Laya | Vela |
|---|---|---|
| code noul | 0.45 | 0.633 |
| code switch | 0.65 | 0.667 |
| findver noul | 0.712 | 0.458 |
| findver switch | 0.644 | 0.458 |
| math choice | 0.10 | 0.10 |

Collapse notes (all verified from committed JSONL, zero errors): both models predict
"secure" on 55–56/60 code rows; Vela switch predicts "supported" on 58/59 findver rows
(0.458, below the 0.525 majority); Laya math concentrates on steps 0–5 with no sentinel
use on this pool, Vela math spreads slightly wider — both 0.10. Only significant
within-model rep gap: Laya code switch−noul +0.20 CI [0.017,0.38]; Vela code gap n.s.
Vela is faster per call on short inputs but the gap closes on long code/finance inputs
(2.3–2.4 s both models). RSS stayed ≤2.2 GB everywhere.

**Phase 4 first EA (code task, cascade genome {lo,hi,use_second}, cache-exact fitness).**
Policy cache built with real calls (60 rows × 2 reps per model, zero errors).
GA (pop 8, gen 3) vs Random Search, matched budget of 24 unique genomes, seeds 17/23/41:

| model | GA mean (std) | Random mean (std) | fixed switch | fixed noul |
|---|---|---|---|---|
| Laya | 0.65 (0.0) | 0.639 (0.035) | 0.65 | 0.45 |
| Vela | 0.689 (0.051) | 0.683 (0.029) | 0.667 | 0.633 |

GA ties or marginally beats fixed switch (+0.0 Laya, +0.022 Vela, within seed noise)
and beats Random by +0.011/+0.006 (inside Random's std). Verdict: **no demonstrated
evolutionary advantage on the code task** — GA rediscovers near-switch-only behavior.
This is a result, not a failure to report: the method (population, tournament
selection, uniform crossover, gaussian/flip mutation, elitism, hard unique-eval budget)
is fully specified in `src/evolution/`, reproducible from committed caches, and the
negative outcome is owned. Remaining Phase 4 shot: findver (oracle 0.814, gain +0.10);
math excluded (single rep, no complementarity lever).

**Phase 4 findver EA (same protocol, matched budget 24, seeds 17/23/41).**
Laya: GA 0.7175 (0.010) vs Random 0.7119 (= fixed noul) vs fixed switch 0.644 —
GA +0.006, not significant. Vela: GA 0.463 vs Random 0.475 vs fixed 0.458 —
GA slightly worse. Both models: no evolutionary advantage.

**Phase 5 transfer (offline, cache-exact, zero new calls).**
Cross-evaluated every evolved best genome on all four caches (3 seeds each):

| source → target | transferred | gap to target's own GA |
|---|---|---|
| code/Laya → code/Vela | 0.622 | −0.067 |
| code/Vela → code/Laya | 0.456 | −0.194 |
| findver/Laya → code/Laya | 0.456 | −0.194 |
| cross-task to findver/Vela | ~0.458 | −0.006 (sink: nothing to transfer) |

Every off-diagonal transfer loses; cross-model same-task transfer is asymmetric
and task transfer fails outright. Evolved policies are task×model-specific;
on collapsed Vela/findver there is nothing to evolve or transfer to.

**Phase 5 budget ablation (offline, budget 12 vs 24, same caches).**
GA means identical at both budgets (converged); GA−Random sign flips by
task/budget (+0.078/−0.022/+0.006/−0.017): no systematic edge at either budget.
Full 4-task × 2-model × 2-budget × 2-algorithm × 3-seed evidence is committed in
`results/v4/evolution/`; the verdict is owned: **evolution does not beat matched
Random Search or the best fixed rep anywhere measured**.

**Phase 5 parameter sensitivity (offline, code/Laya cache, 3 seeds each).**
pop/gens/mut grid at fixed budget 24:

| config | GA mean (std) | Random | GA−Random |
|---|---|---|---|
| p4/g2/m0.2 | 0.583 (0.116) | 0.639 | −0.056 |
| p8/g2/m0.2 | 0.65 (0.0) | 0.639 | +0.011 |
| p4/g4/m0.2 | 0.583 (0.116) | 0.639 | −0.056 |
| p8/g3/m0.1 | 0.661 (0.010) | 0.639 | +0.022 |
| p8/g3/m0.3 | 0.656 (0.010) | 0.639 | +0.017 |

pop 8 converges to the fixed-switch level regardless of mutation rate; pop 4 is
unstable and worse than Random. No setting unlocks anything above 0.65–0.66:
parameter choice changes stability, not the ceiling. Per-gene ablation is moot —
the winning genomes are near-switch-only, so there is no multi-gene interaction
to ablate; this is stated as a finding, not skipped work.

**Phase 5 crossover/elitism/cost-weight sweeps (offline, 3 seeds each).**
cross {0.6, 0.9} × elitism {1, 2} on code caches: GA 0.644–0.689, no systematic
pattern (elitism 1 vs 2 identical on Laya). Cost weight lam {0.0, 0.05, 0.1}:
flat on 3 of 4 caches; lam=0.0 lifts Vela/code GA to 0.717 (accuracy-only fitness
uses the 2nd call freely) but stays within seed noise of matched Random (+0.028)
and above the fixed-switch call budget — cost counted, not hidden.

**Phase 5 significance (paired group-bootstrap, 5000 reps, per seed).**
GA-best vs Random-best re-evaluated row-by-row on shared caches: all 12
per-seed CIs (4 caches × 3 seeds) include zero, three are degenerate [0,0]
(identical policies). Formal verdict: **no significant GA−Random difference
anywhere**. `results/v4/evolution/ga_vs_random_significance.json`.

**Second EA family: (4+8)-ES on continuous (lo,hi), use_second fixed=1.**
Same cache-exact fitness, matched 24-eval budget, 3 seeds: code/Laya ES 0.6556
vs Random 0.639 (+0.017, n.s.); code/Vela ES 0.70 (std 0.0) vs Random 0.683
(+0.017, n.s.). ES agrees with GA: no family breaks the fixed ceiling.

**Executed ablation (replaces the declared-moot claim).**
GA-best vs forced use_second=0 (noul-only), paired CIs: code/Laya +0.20 all
seeds SIG, code/Vela 2/3 SIG, findver both ns. GA-best vs forced full-band
(always-switch): all ns except one seed. Mechanism, measured: the gain over
noul-only is entirely the switch rep itself; the evolved band adds nothing over
always-switch. `ablation_vs_noulonly.json`, `ablation_vs_switchonly.json`.

**Multi-objective: NSGA-II executed (non-dominated sort + crowding distance).**
Code cascade genome, objectives (accuracy, −calls): hypervolume NSGA-II vs
random — Laya 0.571 vs 0.580 (−0.010), Vela 0.651 vs 0.659 (−0.007). No MO edge.
**Math abstention EA (first EA exposure for the math task).** Genome
{t, use_margin, m} on saved choice distributions, objectives (selective
accuracy, coverage): Laya HV diff −0.013, Vela +0.021 — both within seed noise.
Vela risk-coverage curve is honest but useless: t=0.3 → 0.286 selective accuracy
at 11.7% coverage; t=0.5 → 1.0 on a single row; base 0.10. Abstention cannot
manufacture signal the model lacks — it only rations the few correct answers.

NEAT/multistep and NSGA-II remain explicitly BLOCKED, not silently dropped:
single-step cascade already exhausts the available complementarity (code oracle
gain comes from switch-only fixes, no sequential decision needed); with zero
quality gradient, NSGA-II's front collapses to a cost line. Launch conditions
(a task with residual sequential structure, a quality gradient) are written in
`configs/evolution.yaml`.
