# E5 Failure Landscape (diagnostic, from measured evidence — no new inference)

Date: 2026-10-31. Track: post_gate_evolutionary_diagnostic.
Status: E5-analysis DELIVERED from measured bank + pilots. E5-active evolutionary
search (new NSGA-II/MAP-Elites runs): BLOCKED — see §5.

## 1. Measured failure modes (each with committed evidence)

| # | Failure mode | Evidence | Location |
|---|---|---|---|
| F1 | Positional collapse (early steps) | Laya predicts steps 0–5 on 113/120 rows; zero sentinel; Vela similar | Gate 1 analysis |
| F2 | Sentinel blindness | Laya correct_acc = 0.0 on train AND dev (never predicts all-correct) | native_metrics |
| F3 | Schema-induced collapse flip | Code: semantic Noul → all-vulnerable (0.35); Choice → mostly secure (0.80). Vela: opposite direction (all-secure 0.70) | A2 ablations |
| F4 | Inverted confidence | Vela FinDVer P(true) AUROC 0.135 (worse than chance) | A2 |
| F5 | Budget knobs harm-only or no-op | h96: 0 flips; max384: breaks only (findver Δ −0.15); max1024: train-harmful, dev-null | Gate 4 + OFAT |
| F6 | Tournament == single | Identical correct sets; extra cost, zero gain | Gate 1 |
| F7 | No learnable routing signal | LR OOF ≤ majority everywhere (fixed stats, cluster CIs) | audit recompute |
| F8 | Vela weaker-or-equal everywhere paired | All 5 paired diffs ≤ 0 (CIs cover zero or favor Laya) | model_comparison |

## 2. Flatness / harmfulness of the evaluated bank (24 real-measured configs)

- Best bank accuracy overall: Laya findver Choice 0.80 (n=20, +2 over majority) — single small-sample cell, not replicated at n=59 (0.617).
- No config beats its task's majority baseline by a replicable margin.
- Only statistically significant moves are harmful (wide_max1024 train; tournament_g8 dev).
- Selection regret of ANY optimizer restricted to legal measured cells: bounded by bank best minus default — and the bank is already exhaustively evaluated, so running NSGA-II/GA/DE here would only re-query known cells (theater, not science).

## 3. Why E5-active search is blocked (not skipped)

1. The bank of all currently-legal, validated configurations is fully measured; an EA cannot discover anything the bank doesn't already contain.
2. Expanding the space (new templates, orderings, evidence programs) without any quality signal would violate the protocol's own ban on blind generations.
3. Failure *mechanisms* are already identified with CIs (F1–F8); an EA re-discovery run would add cost, not knowledge.
4. E5-active may proceed ONLY as a separately pre-registered follow-up with: new frozen search pool, new config space audit, per-eval budget, and a falsifiable hypothesis (e.g., "MAP-Elites finds a report subgroup where evidence packing beats full context by >X with CI excluding zero").

## 4. What WOULD falsify this verdict

- A new legal config with replicated paired gain (train + held-out groups, CI excludes zero).
- A representation that removes collapse on EITHER model (sentinel predicted at base rate; AUROC > 0.7 with CI excluding 0.5).
- An optimizer beating exhaustive bank best on a held-out group split (impossible on the current bank by construction — requires a genuinely larger validated space).
