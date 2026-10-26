# Full Research Plan: NEAT-Laya Budget-Aware NLP Decisions

## 0. Executive summary

**Working title:** Do Evolved Controllers Help? Budget-Aware Strategy Selection for Frozen System-1 NLP Decisions.

**Question:** Given a frozen Laya model, CPU-only inference and a small labeled calibration set, can a compact NEAT-evolved controller choose a per-example inference strategy and improve the accuracy–latency Pareto frontier over fixed strategies and simpler controllers?

**Method:** Freeze Laya. Generate candidate outputs for SINGLE, WIDE and TOURNAMENT on train/calibration data. Cache those outputs and their measured costs. Train controllers on the cache so evolutionary search does not repeatedly invoke the language model. Compare NEAT with a best fixed action, confidence/margin rules, logistic regression, shallow tree, small fixed-topology MLP and fixed-topology evolutionary network. Final test evaluation must execute the selected action for real and report end-to-end cost, not pretend cached model calls are free.

This is a research hypothesis. No improvement is assumed. A negative result can be publishable if the comparisons are fair and the failure analysis is strong.

## 1. Problem, motivation, method and significance

### Problem
Many-option classification with a frozen decision model has a quality–cost trade-off. Single-pass inference is cheap, while wider budgets or tournament-style multi-stage prediction can help on some tasks but cost more. One fixed action may be wasteful on easy examples and insufficient on difficult examples.

### Proposed method: NEAT-Laya Budget Controller (NLBC)
A small feed-forward policy maps cheap statistics from the initial Laya output and input metadata to an action:
- SINGLE: return the initial result.
- WIDE: repeat prediction with a wider head/context budget.
- TOURNAMENT: use Laya's tournament strategy.
- Optional only after evidence supports it: permutation aggregation and abstention.

The backbone is frozen. NEAT evolves the controller's topology and weights. Its fitness balances correctness, normalized latency/call cost and optional controller complexity:
F(theta) = mean(correctness - lambda * normalized_cost) - beta * complexity(theta).
Train separate policies across a pre-registered range of cost weights or budgets and report the Pareto frontier. Do not tune weights on test results.

### Why it matters
The work tests whether a small, CPU-feasible controller can spend inference compute selectively rather than applying an expensive strategy to every input. It also tests the stronger claim that NEAT's topology evolution adds value beyond simpler rules and models. This is a more defensible contribution than claiming that NEAT itself is new.

### Falsifiable hypotheses
H1: per-example policy selection improves the accuracy–latency frontier over the best fixed action.
H2: NEAT outperforms simpler policies under matched labeled data and search/runtime budgets.
H3: adaptive selection is especially helpful on large label sets and examples whose single-pass predictions are uncertain.
H4: if NEAT gains disappear at matched latency, the apparent gain came from extra inference rather than better allocation.

## 2. Environment, dependencies and models

### Environment requirements
- Python 3.12 recommended; use an isolated virtual environment.
- CPU-only PyTorch; set CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS.
- Record CPU model, core/thread count, Python/PyTorch versions, OS, Laya commit, model revision and random seeds.
- The currently available interactive runtime reported Python 3.13.5, torch 2.10.0+cpu, CUDA unavailable and 3 PyTorch threads. Its shell could not resolve GitHub or the package index, so package installation and dataset/model download are blocked there. A GitHub Actions hosted runner is configured to attempt these tasks.

### Packages
Install CPU torch first, then requirements-cpu.txt: neat-python, datasets, huggingface_hub, pyarrow, numpy, pandas, scikit-learn, scipy, matplotlib, psutil, threadpoolctl, tqdm, pytest. Laya base requirements currently include torch, transformers, safetensors, huggingface_hub and numpy. Lock the resolved working environment after a successful install.

### Model
Primary model: the English checkpoint from convaiinnovations/laya, loaded through the upstream Laya API. Freeze all backbone weights. Record resolved model revision. Do not download or commit model weights into Git. A multilingual model can be a later extension, not a requirement for the first complete experiment.

### Setup sequence
1. Inspect this repository's AGENTS.md and STATUS.md.
2. Create Python 3.12 venv and install CPU torch.
3. Install Laya from a pinned upstream commit or pinned package version.
4. Install experiment dependencies and save a lock file.
5. Run scripts/check_environment.py.
6. Download pinned datasets and verify hashes.
7. Run the 5-example upstream benchmark smoke test on CPU.
8. Only after this passes, proceed to cache construction and NEAT.

## 3. Datasets and protocol

Use the exact revisions referenced by the upstream Laya tournament benchmark:
- BANKING77: mteb/banking77, revision 18072d2685ea682290f7b8924d94c62acc19c0b2, 77 intents.
- CLINC150: clinc/clinc_oos, revision 155b9c710419136e17307b80d0a13e68cd46b4ec; use the plus split and report 150 in-scope classes separately from OOS.
- MASSIVE en-US: SetFit/amazon_massive_intent_en-US, revision f7672a018e8ceb37fc0184dcfbb7e665155ffea6, 60 intents.

Use the upstream benchmark's prompt and label treatment for the initial replication: one choice question, label underscores converted to spaces, no extra label descriptions. Do not simultaneously change prompts and strategy implementation.

Use official train/validation/test splits where possible. For low-label controller training, prefer per-class 1-, 3- and 5-shot calibration sets rather than a fixed total sample count, since CLINC150 has 150 labels. Fix seeds and repeat sampled calibration subsets. Never tune on the final test split. Dataset manifest must record repository, revision, file, size and SHA-256.

The upstream benchmark has used 600 development examples sampled from train for BANKING77 and from validation for CLINC/MASSIVE. Reuse this only if split boundaries are respected and documented.

## 4. Codebases, theory and related literature

### Codebases
- Laya: https://github.com/NandhaKishorM/laya — primary model and APIs.
- Laya many-option benchmark: https://github.com/NandhaKishorM/laya/tree/main/research/benchmarks/tournament — replication target and pinned dataset revisions.
- NEAT-Python: https://github.com/CodeReclaimers/neat-python — evolutionary controller.
- BANKING77 source: https://github.com/PolyAI-LDN/task-specific-datasets
- CLINC OOS: https://github.com/clinc/oos-eval
- MASSIVE: https://github.com/alexa/massive
- autoBOT paper/code context: https://pmc.ncbi.nlm.nih.gov/articles/PMC8550026/

### Core reading
- Stanley & Miikkulainen (2002), Evolving Neural Networks through Augmenting Topologies, foundational NEAT: https://doi.org/10.1162/106365602320169811
- autoBOT, evolutionary feature/text representation: https://doi.org/10.1007/s10994-021-05968-x
- EvoPrompt, evolutionary prompt optimization: https://arxiv.org/abs/2309.08532
- Promptbreeder, self-referential prompt evolution: https://arxiv.org/abs/2309.16797
- Seq103 (recent evolved sequence architecture work; verify exact bibliographic metadata before citation): https://arxiv.org/abs/2606.07664
- Laya repository's benchmark and current option-order/parallel-layout implementation: https://github.com/NandhaKishorM/laya
- Fast Models, Slow Evidence (2026 preprint, treat as preliminary and re-test current Laya): https://arxiv.org/abs/2610.02267

Related literature already studies neuroevolution, evolutionary prompts/agents, calibration, cascades and adaptive inference. Novelty must be framed narrowly: low-label, CPU-only, cached-output policy learning for per-example inference action selection on a frozen structured NLP decision model, with cost-matched baselines and complete end-to-end accounting. Verify all citation metadata and latest papers before submission.

## 5. Baselines

### Frozen Laya action baselines
1. SINGLE / default one-pass.
2. WIDE / larger head_max_len and max_len.
3. TOURNAMENT / default and development-selected group size.
4. Built-in SHORTLIST as a diagnostic, not presumed strong.
5. Optional permutation aggregation only if initial tests show it is useful.
6. Parallel-layout model/checkpoint only if an actual usable checkpoint and fair CPU setup are available; do not imply it is already available.

### Action-selection baselines
1. Best fixed action selected on validation.
2. Confidence-threshold cascade.
3. Margin/entropy threshold rules.
4. Logistic regression.
5. Shallow decision tree.
6. Small fixed-topology MLP.
7. Fixed-topology evolutionary controller (weights evolve; topology fixed).
8. NEAT (topology and weights evolve).
9. Random action routing.
10. Oracle per-example action choice using gold labels, strictly as an analysis-only upper bound, never a deployable result.

The most important comparison is NEAT vs best fixed action, simple uncertainty rules, shallow tree and fixed-topology EA at matched data and compute budgets.

## 6. Fusion architecture

Pipeline:
1. Text and fixed candidate labels -> frozen Laya single pass.
2. Extract cheap features: candidate count K, input/label lengths, top confidence, top-1/top-2 margin, entropy, score distribution, winner position and relevant truncation/budget metadata.
3. Small controller selects action.
4. Execute SINGLE, WIDE or TOURNAMENT and map output to a canonical label.
5. Record accuracy, latency, model calls/rounds, RSS and errors.

Avoid feeding the raw text or all label IDs to the first controller. Begin with low-dimensional, interpretable features to reduce overfitting and support error analysis.

### Cache-based evolutionary search
For calibration and training examples, run each candidate action once and cache predictions, available scores, action correctness labels (for training data only), and measured action costs. NEAT evaluates candidate policies from the cache, not by repeatedly invoking Laya. The final test must execute the selected policy with real Laya calls. Report cache generation time separately, amortized and non-amortized costs, search wall-clock and deployment cost.

### Fitness
F(theta) = mean(correctness - lambda * normalized_cost) - beta * complexity(theta).
Evaluate multiple pre-registered lambda/budget values. Do not choose fitness weights based on test results.

### Oracle gap gate
Before implementing NEAT at scale, compute the best fixed-action score and an oracle per-example upper bound on validation/calibration data. If oracle improvement is negligible, do not spend time on a large evolutionary search. Oracle results must never be used as a deployable baseline.

## 7. Ablations
- NEAT topology evolution vs fixed-topology EA.
- Calibration size: 1/3/5 examples per class, with repeated subset seeds.
- Feature sets: confidence only; confidence+margin; add entropy; full metadata.
- Action ablation: remove WIDE or TOURNAMENT; add permutation only if justified.
- Fitness: accuracy-only vs cost-sensitive, with multiple pre-registered lambda/budget values.
- Search budget: population size and generations; plot performance vs evaluations and wall-clock.
- Controller complexity penalty and number of nodes/edges.
- Tournament group size chosen only on development data.
- Original vs shuffled candidate order as a diagnostic; do not claim unique order-bias solution.
- Optional OOS/abstention and multilingual generalization as extensions.

## 8. Metrics, fairness and statistics

Primary:
- Accuracy at matched mean CPU latency / fixed latency budget.
- Accuracy–latency Pareto frontier.

Secondary:
- Macro-F1; mean, p50, p95 latency; CPU seconds; cache generation and search time; model calls/rounds; peak RSS; ECE/Brier where probabilities are available; option-order flip rate; controller nodes/edges; risk–coverage if abstention is included.

Use multiple calibration subset seeds and at least 3 controller seeds (for example 17, 23, 41). Report mean and standard deviation plus paired bootstrap confidence intervals; use paired tests such as McNemar only where assumptions fit. Record every run's command and environment. Keep test set untouched until prompts, action set, thresholds, group sizes and fitness weights are frozen.

Cost reporting must distinguish:
- candidate-output cache construction;
- evolutionary search and controller training;
- actual deployed inference, including the initial pass and any second-stage calls;
- total experiment wall-clock and CPU time.
Do not hide preprocessing/model-call costs.

## 9. Expected results and storylines

A. NEAT wins at matched cost: strongest claim — topology-evolved policy improves the frontier over simpler controls and generalizes across tasks/seeds.
B. Adaptive policies win, but tree/logistic regression matches NEAT: useful negative algorithmic finding — adaptation helps, NEAT complexity is unnecessary.
C. Best fixed strategy wins and oracle gap is small: current action library lacks complementary strengths; pivot to careful strategy reliability/cost characterization.
D. Accuracy rises only with extra cost: benefit is additional inference, not efficient allocation; report the frontier honestly.
E. Benefit is localized to high-K/low-margin examples: focus story on when extra computation is valuable and which features predict it.
F. No reproducible gain: document this as a negative result if data, costs, uncertainty and baselines are rigorous; do not cherry-pick seeds.

## 10. Project structure and automation

- AGENTS.md: persistent coding/research instructions.
- docs/PROJECT_PLAN.md: this plan.
- configs/experiment.yaml: dataset revisions and initial protocol.
- scripts/check_environment.py: environment report.
- scripts/download_datasets.py: pinned dataset downloader and SHA-256 manifest.
- scripts/run_baseline_smoke.py: tiny CPU baseline test using upstream Laya benchmark.
- tests/: fast structural tests.
- .github/workflows/cpu-bootstrap.yml: CI install/download/smoke-test workflow.
- STATUS.md: human-readable project state.
- results/: small reports only; big raw files and model caches are artifacts or local ignored files.

GitHub is the source of truth. Commit each coherent work unit, update STATUS, and verify the commit. CI artifacts retain logs/manifests/dataset bundle; do not commit raw data or weights.

## 11. Submission planning

Target EACL 2027 SRW. Re-check official dates, eligibility, formatting and current call before submitting: https://2027.eacl.org/calls/srw/. Keep the scope small enough to finish a complete CPU-only comparison. If mentor feedback is sought, prepare a draft with real baseline evidence rather than hypothetical gains.

## 12. Immediate execution checklist

- [ ] Commit scaffold and this plan.
- [ ] Run CI on a hosted runner and inspect install/download logs.
- [ ] Fix dependency/API errors; pin exact working environment.
- [ ] Download data at exact revisions and save SHA-256 manifest.
- [ ] Run 5-example CPU smoke test for all three datasets.
- [ ] Reproduce upstream baseline on a small development subset.
- [ ] Build candidate-action cache and estimate oracle gap.
- [ ] Only then implement NEAT and simple controller baselines.
- [ ] Run locked test evaluation, ablations, cost audit and statistical analysis.
