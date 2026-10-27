#!/usr/bin/env python3
"""Cache-only 3-fold OOF NEAT evaluation against locked controller baselines.

No Laya calls and no dev/confirmation/test split access. Each class's 3 cached
train samples is held out once (77 test / 154 fit). Cost estimates, scalers,
encoders, LR comparators and NEAT genomes are fitted inside each fold. The
primary NEAT input is six numeric SINGLE features plus one-hot initial_choice;
numeric-only is an ablation. Fitness uses correctness minus lambda times the
fold-train mean cumulative cost of the selected action.
"""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

for k, v in {
    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ[k] = v

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_controller_baselines import FEATURE_NAMES, features, single_row
from run_three_action import STRATEGIES, choose_action, group, load_cache
from run_cv_stability import (
    BOOTSTRAP_REPS, BOOTSTRAP_SEED, fit_choice_models,
    fit_numeric_models, fold_costs, predict_proba_choice, predict_proba_numeric,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

DEFAULT_LAMBDAS = (0.0, 2e-5, 5e-5)
DEFAULT_SEEDS = (17, 23, 41)

def label(r):
    return str(r.get("gold_normalized") or r.get("gold") or "").replace("_", " ").strip()

def raw_x(by, ids):
    import numpy as np
    return np.asarray([features(single_row(by[i])) for i in ids], dtype=float)

def build_stratified_folds(by):
    """Hold out one sample/class per fold for the fixed 3-shot or 5-shot cache."""
    by_label = defaultdict(list)
    for idx, items in by.items():
        by_label[label(single_row(items))].append(idx)
    for gold in by_label:
        by_label[gold].sort()
    counts = {len(v) for v in by_label.values()}
    if len(counts) != 1 or next(iter(counts)) not in (3, 5):
        raise ValueError("expected exactly 3 or 5 cached samples per class")
    k_folds = next(iter(counts))
    all_ids = set(by)
    folds = []
    for k in range(k_folds):
        test = sorted(v[k] for v in by_label.values())
        test_set = set(test)
        train = sorted(all_ids - test_set)
        folds.append({"train": train, "test": test})
    return folds, k_folds

def fit_threshold(by, train_ids, target):
    vals = sorted({float(single_row(by[i]).get("initial_answer_confidence") or 0.0) for i in train_ids})
    best_t, best_acc = None, -1.0
    for t in vals:
        hit = 0
        for i in train_ids:
            conf = float(single_row(by[i]).get("initial_answer_confidence") or 0.0)
            action = "single" if conf >= t else target
            hit += int(bool(next(r for r in by[i] if r["strategy"] == action)["correct"]))
        acc = hit / len(train_ids)
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t if best_t is not None else 1.0

def metrics(by, choices):
    import numpy as np
    from sklearn.metrics import f1_score
    ids = sorted(choices)
    golds, preds, hits, costs, calls = [], [], [], [], []
    mix = {s: 0 for s in STRATEGIES}
    for i in ids:
        action = choices[i]
        r = next(x for x in by[i] if x["strategy"] == action)
        golds.append(label(single_row(by[i])))
        preds.append(str(r.get("choice") or "").replace("_", " ").strip())
        hits.append(int(bool(r["correct"])))
        costs.append(float(r["cumulative_latency_ms"]))
        calls.append(float(r["cumulative_calls"]))
        mix[action] += 1
    return {
        "n": len(ids),
        "accuracy": round(float(np.mean(hits)), 4),
        "macro_f1": round(float(f1_score(golds, preds, labels=sorted(set(golds)),
                                          average="macro", zero_division=0)), 4),
        "mean_cumulative_ms": round(float(np.mean(costs)), 1),
        "p50_cumulative_ms": round(float(np.percentile(costs, 50)), 1),
        "p95_cumulative_ms": round(float(np.percentile(costs, 95)), 1),
        "mean_cumulative_calls": round(float(np.mean(calls)), 3),
        "action_mix": mix,
    }

def fit_aux_correctness_models(X_train, train_ids, by, family, seed):
    """Three train-fold correctness models with the same action set as NEAT."""
    models = {}
    diagnostics = {}
    for ai, action in enumerate(STRATEGIES):
        y = [int(bool(next(r for r in by[i] if r["strategy"] == action)["correct"]))
             for i in train_ids]
        if family == "decision_tree":
            model = DecisionTreeClassifier(max_depth=3, random_state=seed + ai, class_weight=None)
        elif family == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(8,), max_iter=300,
                                  random_state=seed + ai, early_stopping=False)
        else:
            raise ValueError(f"Unknown auxiliary model family: {family}")
        model.fit(X_train, y)
        models[action] = model
        diagnostics[action] = {
            "n_iter": int(getattr(model, "n_iter_", 0)),
            "reached_max_iter": bool(family == "mlp" and getattr(model, "n_iter_", 0) >= 300),
        }
    return models, diagnostics

def auxiliary_probabilities(models, X):
    """Return P(action correct) for each action; handle single-class folds safely."""
    import numpy as np
    result = {s: np.zeros(len(X), dtype=float) for s in STRATEGIES}
    for action, model in models.items():
        classes = list(model.classes_)
        prob = model.predict_proba(X)
        if 1 in classes:
            result[action] = np.asarray(prob[:, classes.index(1)], dtype=float)
        elif len(classes) == 1 and classes[0] == 1:
            result[action] = np.ones(len(X), dtype=float)
    return result

def pareto_frontier(report):
    """Deployable accuracy-vs-mean-cost frontier; excludes post-hoc oracle."""
    points = []
    for name, item in report.items():
        if name.startswith("oracle::posthoc") or "accuracy" not in item or "mean_cumulative_ms" not in item:
            continue
        points.append({"policy": name, "accuracy": item["accuracy"],
                       "mean_cumulative_ms": item["mean_cumulative_ms"],
                       "mean_cumulative_calls": item.get("mean_cumulative_calls")})
    frontier = []
    for candidate in points:
        dominated = any(
            other["accuracy"] >= candidate["accuracy"]
            and other["mean_cumulative_ms"] <= candidate["mean_cumulative_ms"]
            and (other["accuracy"] > candidate["accuracy"]
                 or other["mean_cumulative_ms"] < candidate["mean_cumulative_ms"])
            for other in points if other["policy"] != candidate["policy"]
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda p: (p["mean_cumulative_ms"], -p["accuracy"]))

def config_text(n_inputs, pop_size, fixed_topology=False):
    conn_add = 0.0 if fixed_topology else 0.2
    conn_delete = 0.0 if fixed_topology else 0.05
    node_add = 0.0 if fixed_topology else 0.1
    node_delete = 0.0 if fixed_topology else 0.03
    activation_mutate = 0.0 if fixed_topology else 0.02
    initial_connection = "full_direct" if fixed_topology else "partial_direct 0.15"
    return f"""[NEAT]
fitness_criterion = max
fitness_threshold = 2.0
pop_size = {pop_size}
reset_on_extinction = False
no_fitness_termination = False

[DefaultGenome]
activation_default = tanh
activation_mutate_rate = {activation_mutate}
activation_options = tanh sigmoid relu
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_init_type = gaussian
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = {conn_add}
conn_delete_prob = {conn_delete}
enabled_default = True
enabled_mutate_rate = 0.01
feed_forward = True
initial_connection = {initial_connection}
node_add_prob = {node_add}
node_delete_prob = {node_delete}
num_hidden = 0
num_inputs = {n_inputs}
num_outputs = 3
response_init_mean = 1.0
response_init_stdev = 0.0
response_init_type = gaussian
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_init_type = gaussian
weight_max_value = 30.0
weight_min_value = -30.0
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 10
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
min_species_size = 2
"""

def evolve(X_train, train_ids, by, costs, lam, seed, pop_size, generations, fixed_topology=False):
    import neat
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    text = config_text(X_train.shape[1], pop_size, fixed_topology=fixed_topology)
    with tempfile.TemporaryDirectory(prefix="neat_laya_cfg_") as tmp:
        cfgpath = Path(tmp) / "config.ini"
        cfgpath.write_text(text, encoding="utf-8")
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfgpath))
        row_at = {i: j for j, i in enumerate(train_ids)}
        penalty = {s: lam * float(costs[s]["mean_cumulative_ms"]) for s in STRATEGIES}
        def eval_genomes(genomes, _):
            for _, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, cfg)
                total = 0.0
                for i in train_ids:
                    out = net.activate(X_train[row_at[i]])
                    action = STRATEGIES[int(np.argmax(out))]
                    r = next(r for r in by[i] if r["strategy"] == action)
                    total += float(bool(r["correct"])) - penalty[action]
                genome.fitness = total / len(train_ids)
        pop = neat.Population(cfg)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        best = pop.run(eval_genomes, generations)
        net = neat.nn.FeedForwardNetwork.create(best, cfg)
        detail = {
            "fitness": float(best.fitness),
            "nodes": len(best.nodes),
            "connections_total": len(best.connections),
            "connections_enabled": sum(bool(c.enabled) for c in best.connections.values()),
            "topology_type": "fixed_full_direct" if fixed_topology else "neat_evolved",
            "generations_evaluated": len(stats.get_fitness_mean()),
        }
        return net, detail, best, text

def paired_ci(by, base, candidate, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    import numpy as np
    diffs, golds = [], []
    for i in sorted(base):
        rb = next(r for r in by[i] if r["strategy"] == base[i])
        rc = next(r for r in by[i] if r["strategy"] == candidate[i])
        diffs.append(int(bool(rc["correct"])) - int(bool(rb["correct"])))
        golds.append(label(single_row(by[i])))
    strata = defaultdict(list)
    for g, d in zip(golds, diffs):
        strata[g].append(d)
    rng = np.random.RandomState(seed)
    draws = np.zeros(reps, dtype=float)
    n = len(diffs)
    for g in sorted(strata):
        arr = np.asarray(strata[g], dtype=float)
        sampled = rng.randint(0, len(arr), size=(reps, len(arr)))
        draws += arr[sampled].mean(axis=1) * (len(arr) / n)
    return {"candidate_minus_baseline_accuracy": round(float(np.mean(diffs)), 4),
            "ci95_lo": round(float(np.percentile(draws, 2.5)), 4),
            "ci95_hi": round(float(np.percentile(draws, 97.5)), 4),
            "n": len(diffs), "bootstrap_reps": reps, "seed": seed}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="results/pilot/banking77_train3shot_cache.jsonl")
    ap.add_argument("--out", default="results/pilot/banking77_neat_cv.json")
    ap.add_argument("--progress", default=None)
    ap.add_argument("--pop-size", type=int, default=30)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--seeds", default="17,23,41")
    ap.add_argument("--lambdas", default="0,2e-5,5e-5")
    args = ap.parse_args()
    if args.pop_size < 10 or args.generations < 2:
        raise SystemExit("Require pop-size >= 10 and generations >= 2.")
    cache_path, out_path = [(ROOT / p).resolve() for p in (args.train_cache, args.out)]
    progress_path = Path(args.progress).resolve() if args.progress else out_path.with_suffix(".progress.jsonl")
    if not cache_path.is_file():
        raise SystemExit(f"train cache not found: {cache_path}")
    if out_path.exists() or progress_path.exists():
        raise SystemExit(f"Refusing to overwrite output/progress file: {out_path} / {progress_path}")
    try:
        import neat  # noqa: F401
        import numpy as np
        from threadpoolctl import threadpool_limits
    except Exception as exc:
        raise SystemExit(f"Required offline dependency unavailable: {exc}") from exc
    by = group(load_cache(cache_path))
    if len({label(single_row(items)) for items in by.values()}) != 77:
        raise SystemExit("Expected all 77 BANKING77 classes in the training cache")
    folds, examples_per_class = build_stratified_folds(by)
    if len(by) != 77 * examples_per_class:
        raise SystemExit("Training cache size does not match its class-stratified fold structure")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    if not seeds or not lambdas:
        raise SystemExit("Seeds and lambdas must be non-empty lists.")
    created = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.touch(exist_ok=False)
    baseline_oof, neat_oof, fold_log = {}, {}, []

    with threadpool_limits(limits=2), progress_path.open("a", encoding="utf-8") as log:
        for fi, fold in enumerate(folds):
            tr, te = list(fold["train"]), list(fold["test"])
            costs = fold_costs(by, tr)
            nsc, nmodels = fit_numeric_models(by, tr, class_weight=None)
            csc, enc, cmodels = fit_choice_models(by, tr, class_weight=None)
            nfn = lambda row, s=nsc, m=nmodels: predict_proba_numeric(s, m, row)
            cfn = lambda row, s=csc, e=enc, m=cmodels: predict_proba_choice(s, e, m, row)
            train_acc = {s: sum(int(bool(next(r for r in by[i] if r["strategy"] == s)["correct"])) for i in tr) / len(tr)
                         for s in STRATEGIES}
            fixed = max(STRATEGIES, key=lambda s: train_acc[s])
            target = max(("wide", "tournament"), key=lambda s: train_acc[s])
            threshold = fit_threshold(by, tr, target)
            baseline_oof.setdefault("fixed_best_train", {}).update({i: fixed for i in te})
            baseline_oof.setdefault("threshold_trainfit", {}).update({
                i: ("single" if float(single_row(by[i]).get("initial_answer_confidence") or 0.0) >= threshold else target)
                for i in te
            })
            for lam in lambdas:
                for mode, fn in (("numeric", nfn), ("choice", cfn)):
                    baseline_oof.setdefault(f"{mode}_lr_lam_{lam:g}", {}).update({
                        i: choose_action(fn(single_row(by[i])), costs, lam) for i in te
                    })
            Xtrn = nsc.transform(raw_x(by, tr))
            Xten = nsc.transform(raw_x(by, te))
            cats_tr = np.asarray([[single_row(by[i]).get("initial_choice") or ""] for i in tr])
            cats_te = np.asarray([[single_row(by[i]).get("initial_choice") or ""] for i in te])
            Xtrc = np.hstack([csc.transform(raw_x(by, tr)), enc.transform(cats_tr).toarray()])
            Xtec = np.hstack([csc.transform(raw_x(by, te)), enc.transform(cats_te).toarray()])
            matrices = (("numeric", Xtrn, Xten), ("choice", Xtrc, Xtec))
            for mode_i, (mode, Xtr, Xte) in enumerate(matrices):
                for family in ("decision_tree", "mlp"):
                    aux_models, aux_diag = fit_aux_correctness_models(
                        Xtr, tr, by, family, seed=17 + fi * 100 + mode_i * 10)
                    probs_by_action = auxiliary_probabilities(aux_models, Xte)
                    for lam in lambdas:
                        baseline_oof.setdefault(f"{family}_{mode}_lam_{lam:g}", {}).update({
                            i: choose_action({s: float(probs_by_action[s][j]) for s in STRATEGIES},
                                             costs, lam)
                            for j, i in enumerate(te)
                        })
                    fold_log.append({
                        "fold": fi, "mode": mode, "family": family,
                        "diagnostics": aux_diag,
                    })
            fold_log.append({
                "fold": fi, "n_train": len(tr), "n_test": len(te),
                "train_costs": costs, "train_accuracy_by_action": train_acc,
                "fixed_best_train": fixed, "threshold_upgrade_target": target,
                "threshold": threshold, "choice_encoder_categories": len(enc.categories_[0]),
            })
            print(f"Fold {fi + 1}/{len(folds)}: train={len(tr)} test={len(te)}", flush=True)
            for mode_i, (mode, Xtr, Xte) in enumerate(matrices):
                for li, lam in enumerate(lambdas):
                    for seed0 in seeds:
                        actual_seed = seed0 + fi * 100000 + mode_i * 10000 + li * 1000
                        net, genome, _best_genome, _config_text = evolve(Xtr, tr, by, costs, lam, actual_seed,
                                             args.pop_size, args.generations)
                        choices = {i: STRATEGIES[int(np.argmax(net.activate(Xte[j])))] for j, i in enumerate(te)}
                        key = f"neat_{mode}_lam_{lam:g}_seed_{seed0}"
                        neat_oof.setdefault(key, {}).update(choices)
                        rec = {
                            "fold": fi, "mode": mode, "lambda_per_ms": lam, "seed": seed0,
                            "actual_seed": actual_seed, "pop_size": args.pop_size,
                            "generations": args.generations, "best_genome": genome,
                            "fold_oof_metrics": metrics(by, choices),
                        }
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        print(f"  {key} fold={fi + 1}: acc={rec['fold_oof_metrics']['accuracy']:.4f}, "
                              f"mean_ms={rec['fold_oof_metrics']['mean_cumulative_ms']:.0f}, "
                              f"fitness={genome['fitness']:.4f}", flush=True)
                        # Matched fixed-topology evolutionary ablation for the primary
                        # choice-feature / lambda=2e-5 setting, same fold and search budget.
                        if mode == "choice" and abs(lam - 2e-5) < 1e-12:
                            fixed_seed = actual_seed + 500
                            fnet, fgenome, _fbest, _fcfg = evolve(
                                Xtr, tr, by, costs, lam, fixed_seed,
                                args.pop_size, args.generations, fixed_topology=True)
                            fchoices = {i: STRATEGIES[int(np.argmax(fnet.activate(Xte[j])))]
                                        for j, i in enumerate(te)}
                            fkey = f"fixed_ea_choice_lam_{lam:g}_seed_{seed0}"
                            neat_oof.setdefault(fkey, {}).update(fchoices)
                            frec = {
                                "fold": fi, "mode": "choice", "topology_type": "fixed_full_direct",
                                "lambda_per_ms": lam, "seed": seed0, "actual_seed": fixed_seed,
                                "pop_size": args.pop_size, "generations": args.generations,
                                "best_genome": fgenome, "fold_oof_metrics": metrics(by, fchoices),
                            }
                            log.write(json.dumps(frec) + "\n")
                            log.flush()
                            print(f"  {fkey} fold={fi + 1}: acc={frec['fold_oof_metrics']['accuracy']:.4f}, "
                                  f"mean_ms={frec['fold_oof_metrics']['mean_cumulative_ms']:.0f}, "
                                  f"fitness={fgenome['fitness']:.4f}", flush=True)

    all_policies = {f"baseline::{k}": v for k, v in baseline_oof.items()}
    all_policies.update({f"neat::{k}": v for k, v in neat_oof.items()})
    report = {k: metrics(by, v) for k, v in all_policies.items()}
    oracle = {}
    for i in by:
        oracle[i] = max(STRATEGIES, key=lambda s: (
            int(bool(next(r for r in by[i] if r["strategy"] == s)["correct"])),
            -float(next(r for r in by[i] if r["strategy"] == s)["cumulative_latency_ms"])))
    report["oracle::posthoc"] = metrics(by, oracle)
    comparisons = {}
    for neat_key, chosen in neat_oof.items():
        mode = neat_key.split("_")[1]
        rhs = neat_key.split("_lam_", 1)[1]
        lam_text, seed_text = rhs.rsplit("_seed_", 1)
        lam = float(lam_text)
        other_mode = "choice" if mode == "numeric" else "numeric"
        keys = [
            "fixed_best_train", "threshold_trainfit",
            f"{mode}_lr_lam_{lam:g}", f"{other_mode}_lr_lam_{lam:g}",
            f"decision_tree_{mode}_lam_{lam:g}", f"decision_tree_{other_mode}_lam_{lam:g}",
            f"mlp_{mode}_lam_{lam:g}", f"mlp_{other_mode}_lam_{lam:g}",
        ]
        if neat_key.startswith("fixed_ea_"):
            keys.append(f"neat_choice_lam_{lam:g}_seed_{seed_text}")
        elif mode == "choice" and abs(lam - 2e-5) < 1e-12:
            keys.append(f"fixed_ea_choice_lam_{lam:g}_seed_{seed_text}")
        comparisons[neat_key] = {k: paired_ci(by, baseline_oof[k], chosen) for k in keys}

    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sha = None
    out = {
        "created_utc": created, "project_git_sha": sha,
        "dataset": "banking77", "split": f"train_{examples_per_class}shot_{examples_per_class}fold_oof",
        "examples_per_class": examples_per_class,
        "train_cache": str(cache_path.relative_to(ROOT)) if cache_path.is_relative_to(ROOT) else str(cache_path),
        "n_examples": len(by), "n_classes": len({label(single_row(x)) for x in by.values()}),
        "config": {
            "population": args.pop_size, "generations": args.generations,
            "seeds": seeds, "lambdas_per_ms": lambdas,
            "features": {"numeric": FEATURE_NAMES, "choice": "numeric + one-hot initial_choice; encoder fold-train only"},
            "fitness": "mean(correct_selected - lambda * fold-train mean cumulative cost for selected action)",
            "cost_estimation": "fold-train only; held-out per-example latency never used to choose actions",
            "laya_calls": 0, "test_split_touched": False, "confirmation_split_touched": False,
        },
        "folds": fold_log, "metrics": report, "paired_bootstrap_neat_minus_baselines": comparisons,
        "pareto_frontier_deployable": pareto_frontier(report),
        "progress_log": str(progress_path.relative_to(ROOT)) if progress_path.is_relative_to(ROOT) else str(progress_path),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "status": "passed" if len(neat_oof) == 2 * len(lambdas) * len(seeds) + len(seeds) else "partial_failure",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "n_neat_policies": len(neat_oof),
                      "elapsed_seconds": out["elapsed_seconds"],
                      "metrics": {k: v for k, v in report.items() if k.startswith(("baseline::", "neat::"))}},
                     indent=2), flush=True)
    return 0 if out["status"] == "passed" else 1

if __name__ == "__main__":
    raise SystemExit(main())
