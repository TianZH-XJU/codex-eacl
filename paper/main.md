# Configuration Sensitivity, Not Search: A Negative Result on Evolving Inference Configurations for Frozen Structured Decision Models

*Draft v0.1 — 2026-10-31. Work-in-progress short paper for EACL 2027 SRW.
All numbers below are measured on the author's local CPU machine; nothing is simulated.
Test splits were never used for tuning. No evolutionary search stage was reached: every
search gate failed, so this paper reports sensitivity, baselines, and a failed-gate analysis.*

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
