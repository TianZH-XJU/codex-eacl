# 多领域研究计划：NEAT 驱动的 Laya 自适应推理与预算分配

**工作标题（暂定）**：Beyond One-Pass Decisions: Budget-Aware Neuroevolution for Mathematical, Code, and Financial Judgments  
**当前状态**：研究计划，不代表新实验已经运行；BANKING77 上的 NEAT OOF 已完成，但预设门槛未通过。  
**执行约束**：所有模型推理、数据处理和实验只在用户本地 Windows PC 执行；GitHub 只用于版本控制与备份，不使用 GitHub Actions/云 runner 做实验。  
**硬件约束**：CPU-only、建议最多 2 个并行线程、实验进程 RSS 尽量低于 3.5–4 GB。  
**主模型**：Laya 英文 checkpoint，沿用当前固定 revision `7b928d828b7b0e022f929d9bd2e44165aa270148`；Laya 代码版本先固定为本地已验证 commit `b9fe351ce81e2d974bca2bf219c1e4030c0a05ea` / Laya 0.4.16。扩展前必须先确认该 pinned 版本实际支持哪些 API，不直接把上游最新 README 的行为当成本地版本事实。

---

## 执行摘要

本项目不再把“在 BANKING77 上继续增加 NEAT 代数”作为默认下一步。现有证据表明，在 3-shot BANKING77、6 个数值特征加初始预测类别特征的设定下，NEAT 没有可靠胜过匹配的逻辑回归或阈值级联；跨折动作分布大幅摆动，策略正确性预测的 AUC 也不稳定。下一步要检验的是：**换到具有真实推理/判断难度的不同领域后，Laya 的不同推理动作是否具有互补错误，以及控制器能否在有限预算内识别何时值得升级。**

建议保留三类场景，但逐个进入，而不是同时启动所有数据集：

1. **数学推理过程错误定位**：ProcessBench，`choice` 任务，首选主实验。
2. **代码缺陷/漏洞判断**：CodeXGLUE Defect Detection（Devign 来源），作为二元判断的领域扩展；先验证 Laya `noul` 与二选一 `choice` 的行为，再决定动作适配器。
3. **财报证据判断**：FinDVer 为首选主任务，其 claim entailment 标签直接匹配“判断财报声明是否有证据支持”；FinQA、TAT-QA、ConvFinQA 是后续数值推理扩展。

每个领域先通过“任务适配与策略互补性门槛”，再做 NEAT。若所有可用策略在同一批样本上犯同样的错，控制器无从创造新能力；若策略互有胜负但特征无法预测增量收益，则应优先研究信号与数据，而不是盲目增加进化代数。

## 1. 针对什么问题、提出什么方法、为什么、怎么做、研究意义

### 1.1 问题定义

在数学、代码和金融判断任务中，冻结的结构化决策模型可能因候选项数量、输入长度、表格/代码结构、错误步骤定位难度或证据复杂性而出现不同的准确率与 CPU 成本。对所有样本始终运行最昂贵策略会浪费资源；对所有样本只运行单次推理又可能漏掉能被更强推理程序纠正的错误。

因此研究问题为：

> 在固定的 Laya checkpoint、有限的少样本标注和本地 CPU 预算下，能否训练一个轻量控制器，逐样本选择何时停止、何时升级到另一种有效推理策略，并在严格匹配的总推理预算下改善准确率—成本 Pareto frontier？这种策略在数学、代码和金融三个不同场景中能否泛化？NEAT 的拓扑进化是否比逻辑回归、阈值、树或固定拓扑网络提供额外价值？

### 1.2 方法：Budgeted Laya Strategy Controller（BLSC，名称暂定）

保持 Laya 主模型冻结；把 NEAT 当作 **Laya 外部的策略控制器**，不是直接进化 Laya 的 4 亿级参数。控制器根据初次 Laya 输出的廉价特征、任务元数据、当前策略历史与剩余预算，决定下一步执行哪个任务适配动作。

核心目标分成三层：

- **模型层**：Laya 生成结构化判断（choice / score / noul）。
- **策略层**：在当前任务有效的动作里选择 STOP、宽预算重跑、Tournament、证据复核或固定多视角聚合。
- **预算层**：任何升级都要满足剩余预算与效益门槛；预算不满足时必须停止并返回当前结果。

只将当前 Laya 代码中已验证的行为称为“原生策略”。任务适配器里新定义的复核/多视角动作必须标注为本项目新增实验，不可称作 Laya 内置的 WIDE 或 TOURNAMENT。

### 1.3 为什么可能有效

- 样本难度并不相同；简单样本可能一遍即可，困难样本可能值得花额外计算。
- 相同模型的不同推理策略可能利用不同的候选分组、token/head 预算或输入表示，因此可能在部分样本上互相纠错。
- 成本敏感控制器可以学习预计增量收益，而非只看置信度。已有级联研究指出，单纯基于 confidence 的升级规则在某些下游专家/分布偏移场景会失败；输入自适应计算分配研究则强调，应预测不同输入和计算预算对应的预期收益。见 `When Does Confidence-Based Cascade Deferral Suffice?` 与 `Learning How Hard to Think`。论文元数据与结果已放入 Undermind workspace。

### 1.4 本项目不能预设的结论

- 不能假设更难的数据集必然让 NEAT 获益；更难的任务也可能让 Laya 所有策略同样失败。
- 不能把增加的推理调用带来的准确率提升，直接说成控制策略更聪明。必须在相同总成本下比较。
- 不能把“NEAT 学会了选择策略”说成“NEAT 修改或训练了 Laya 本身”。本项目主线仍是冻结 Laya 的策略选择；如果以后要微调 Laya 权重，必须登记为不同研究方向。
- 负结果可接受：若简单阈值/线性模型与 NEAT 一样好，结论可以是拓扑进化没有提供额外收益。

### 1.5 目标研究贡献

1. 为冻结结构化决策模型建立统一、透明的逐样本计算预算评测协议。
2. 对比固定策略、阈值级联、学习控制器、NEAT 和固定拓扑进化方法。
3. 通过三种领域和不同任务类型检验策略互补性与控制器泛化，而不是只依赖一个意图分类数据集。
4. 报告端到端 CPU 成本：首次调用、升级调用、控制器时间、缓存构建成本、调用次数、均值/中位数/P95、内存与错误率。

## 2. 环境准备：软件包、模型和本地复现要求

### 2.1 冻结环境

沿用现有本地环境，不因新任务随意升级依赖：

- Windows + Python 3.12.12 venv `.venv`
- PyTorch CPU 版本；不启用 CUDA
- Laya 0.4.16 / 本地源码 commit `b9fe351ce81e2d974bca2bf219c1e4030c0a05ea`
- Laya checkpoint `convaiinnovations/laya` revision `7b928d828b7b0e022f929d9bd2e44165aa270148`
- neat-python 2.0.0
- 已安装的 datasets、pyarrow、numpy、pandas、scikit-learn、scipy、psutil、threadpoolctl、tqdm、pytest

可视化与错误分析可用 matplotlib。数学答案规范化可按需使用 SymPy；代码任务只有在确实需要 AST/函数切分时才考虑 tree-sitter。第一版不额外下载第二个大型 embedding 模型，避免把模型成本和内存混入主方法。若 shortlist 需要 embedding 模型，作为独立基线报告额外模型加载与推理成本。

### 2.2 线程与内存

每个本地长任务设置：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

每次只启动一个实验进程。长时间 Laya 推理需要开启系统休眠保护；原始耗时全部保留，离群延迟只加标记，不覆盖、不删除。记录 Python、torch、Laya commit、checkpoint revision、数据 revision、线程数、CPU、峰值 RSS、暖机方式和命令。

### 2.3 必须先做的 Laya API 预检

在新任务上做 20–50 条的 CPU smoke test 前，先通过本地 pinned 版本验证：

1. `choice` / `score` / `noul` 三种类型能否按本项目格式返回结构化结果。
2. `WIDE` 的实际实现是否只是使用更大的 `head_max_len` / `max_len` 重新执行同一判断；不要把它写成自回归“继续思考”。
3. `predict_tournament` 是否可用、每个样本实际用了多少轮/forward，以及候选数小于分组大小时如何处理。
4. 当前 pinned 版本对 `noul` 的 `true` / `false` criteria 和 labels override 如何渲染；Laya README 报告过英文 checkpoint 的 `noul` 标签敏感性，因此需在真实任务上验证，不能直接相信 P(true) 校准。
5. 对 `score` 检查分数方向、尺度、顺序偏差及与真实等级的关系。

**预检失败时停止该类型的 NEAT 实验**，先修复任务适配器或选择可靠的 typed representation。不可把 API 不匹配造成的坏结果归因于 NEAT。

## 3. 数据准备：三种场景与候选数据集

### 3.1 主实验选择（分阶段）

| 场景 | 首选任务 | 主数据来源 | Laya 输出类型 | 为什么先选它 | 状态 |
|---|---|---|---|---|---|
| 数学 | 定位推理过程中最早的错误步骤 | [ProcessBench](https://github.com/QwenLM/ProcessBench), [Hugging Face](https://huggingface.co/datasets/Qwen/ProcessBench) | Choice | 原生步骤和错误位置标签；约 3,400 条，Apache-2.0 数据卡；比短文本意图更接近真正推理判断 | 第一扩展主场景 |
| 代码 | 函数是否存在缺陷/漏洞 | [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) 的 Defect Detection（来源于 Devign） | 首先试 Noul；同时做二选一 Choice 对照 | 有明确二元标签和官方 split；代码长度/结构为不同领域挑战 | 第二阶段 |
| 金融 | 判断财报中的声明是否被长文本/表格证据支持 | [FinDVer](https://github.com/yilunzhao/FinDVer)；后续扩展 FinQA/TAT-QA/ConvFinQA | 原生 Noul 与二选一 Choice 对照 | 直接带 claim entailment 标签，覆盖信息抽取、数值推理与知识型验证 | 第三阶段；先做长文档/截断与 API smoke |

#### 数学：ProcessBench

- 官方仓库与 Hugging Face 数据卡已验证；数据记录含 `problem`、`steps`、`final_answer_correct`、`label` 等字段，数据卡当前显示总量约 3,400 条，多个源任务 split。
- 将每条记录转换为“问题 + 完整推理步骤”的状态，候选集合是每一步的编号/文本，并加入“所有步骤都正确”的候选项。金标为数据集标注的最早错误位置；必须照官方评测代码核对 `label` 到候选项的映射（特别是全正确的 sentinel 与编号是否 0/1 起始），不能猜测或自行重标。
- 按原始 `problem` / 原始题目 ID 分组切分；同题不同生成器/不同轨迹不可跨 train/dev/test。对 generator 来源做分层报告，避免把同一题的多个轨迹泄漏到不同集合。
- 主指标：最早错误步骤 exact accuracy、macro-F1；辅助指标：全正确识别率、错误定位偏差（差几步）、按步骤数/题目来源/输入长度分层。
- PRM800K（OpenAI 官方仓库，约 800,000 个步骤级正确性标签）和 Math-Shepherd 可做后续 verifier/特征研究，不作为首个控制器数据集直接混入 ProcessBench 训练。跨数据集联合时必须严格切分题目来源与去重。

#### 代码：CodeXGLUE Defect Detection / Devign

- CodeXGLUE 官方任务将其定义为二元分类：0 为安全、1 为存在不安全/缺陷代码；其 Defect Detection 数据来源于 Devign。以官方下载脚本、split 和 license 文档为准。
- 主指标不要只看 accuracy：同时报告 macro-F1、balanced accuracy、AUROC、AUPRC、正例率和混淆矩阵，因为漏洞/缺陷数据通常不均衡。
- 代码长度可能超过 Laya 可用输入预算。记录截断率和 token 长度；不要默默截断后声称覆盖完整代码。切分应尽量按项目/仓库和函数去重，避免相同/高度相似函数跨 split 泄漏。
- `noul` 在语义上适合“这个函数是否有漏洞？”但它没有天然的 WIDE/Tournament 候选分组语义。首期要比较两种形式：原生 `noul`；把同一判断表达成两项 `choice`（“secure / vulnerable”）以复用多选策略。二者视为不同任务适配，不可把结果合并成一个分数。对 Noul 的额外动作需要独立定义，例如 STOP、固定替代表述的再次判断、两个预注册视角的概率聚合；每个动作必须先通过单测与小样本有效性检查。
- 暂不从 SWE-bench 起步：它是生成补丁、应用到软件仓库并运行测试的完整软件工程任务，不等同于一个 Noul 判断问题，而且需要较重的 Docker/仓库环境。

#### 金融：FinDVer 财报声明验证（主任务）

- 官方仓库：https://github.com/yilunzhao/FinDVer；论文：https://aclanthology.org/2024.emnlp-main.818/。数据包含 statement、explanation、entailment_label（bool）、relevant_context 与 report；子集覆盖信息抽取（FDV-IE）、数值推理（FDV-MATH）与知识密集推理（FDV-KNOW）。这比人工从普通 QA 构造标签更直接地对应财报证据支撑判断。
- 官方论文与后续公开仓库的样本数量描述并不完全一致；正式准备时必须 pin 仓库 commit，读取实际文件清单和 license，并记录每个 split、子集、报告数量与行数。README 说明 testmini/test 的解释、entailment 和 evidence 标注已公开。
- 输入为声明及对应财报上下文。可用 relevant_context 做一个 oracle-evidence 诊断条件，但不能冒充端到端检索；完整报告/长上下文或固定检索条件需单独报告。记录截断率、证据覆盖率及文本/表格比例。
- 采用原生 Noul 和二选一 Choice 两种 adapter 预检。由于 Laya 英文 checkpoint 对 Noul 的 true/false 词面可能敏感，须在 train/dev 验证 label rendering 与校准；test 不用于选择 adapter。
- 仓库提供 testmini 和 test。推荐 testmini 内按 report 分组切 controller train/dev，test 锁定一次评估；先检查报告是否跨 split。若同一 report 跨 split，重新按报告去重切分，并明确是 project-defined split。
- 主指标：balanced accuracy、macro-F1、AUROC/AUPRC；按 IE/MATH/KNOW 子集、长短文、表格、截断和证据覆盖率分层报告。FinQA、TAT-QA、ConvFinQA 留作后续数值/混合证据扩展。

### 3.2 用户给出的候选数据集清单：优先级与风险

标记说明：**A** = 可优先核实并进入准备；**B** = 有实际用途但作为扩展；**C** = 需要构造任务、环境太重或数据名尚未核实，不能直接假设可用。

#### 数学推理

| 数据/任务 | 级别 | 用法与注意事项 |
|---|---|---|
| ProcessBench | A | 首选：错误步骤定位 Choice；严格照官方标签映射与 split |
| PRM800K | B | 大规模步骤级标签，可训练或评估 verifier；和 ProcessBench 做严格题目去重 |
| Math-Shepherd | B | 过程监督/步骤正确性；核实原始数据版本、许可和合成标签质量 |
| GSM8K / MATH | B | 原始数学题集；不能单独充当错误步骤标注集，适合构造生成/验证任务 |
| miniF2F / ProofNet | C | 形式化定理证明/证明库，需要 Lean/证明验证环境；“严谨度评分”不是原生标签，先不纳入 CPU 主实验 |
| TheoremQA | B/C | 知识/定理问答，需核实当前版本和题型，可作为 theorem selection 的候选 |
| LIMO-Strategy | C | 当前尚未确认与该精确名称对应的稳定公开标准数据集；需用户提供准确仓库/论文或另行核实 |
| Arithm-Entailment | C | 作为任务描述暂存；须核实准确名称、schema、来源及许可 |

#### 代码判断

| 数据/任务 | 级别 | 用法与注意事项 |
|---|---|---|
| CodeXGLUE Defect Detection / Devign | A | 首选二元缺陷判断；有官方项目；注意类不均衡与项目泄漏 |
| Big-Vul | B | 大规模漏洞分类；需去重、按项目切分、核实许可和漏洞年代偏差 |
| BigCloneBench | B | 克隆/语义相似判断；配对数据会有相似性泄漏，需按函数簇划分 |
| POJ-104 | B | 程序检索/克隆分类基准；注意数据授权与多语言适用性 |
| CodeSearchNet | B | 代码检索语料；`CodeSearchNet-Rerank` 更像任务构造而非统一的官方独立数据集名，候选与负例需预注册 |
| StackOverflow-Retrieval | B/C | 需明确具体公开基准版本、查询和相关性标注；不要只用泛称 |
| HumanEvalFix | B/C | 需要核对确切论文/仓库和生成/修复/判定协议，不能默认是单一标准数据集名 |
| Defects4J | B/C | 真实 Java 程序缺陷/修复基准，执行测试较重；作为后期外部有效性场景 |
| SWE-bench / Lite / Verified | C | issue→patch→Docker tests 的完整工程任务，资源开销和实验变量太大；不作为第一代码场景 |
| ToolBench / Gorilla API | B/C | API/tool calling 方向可用于工具路由，但需要统一工具集合与接口，且运行/依赖更复杂 |
| jevgrep-Bench | C | 当前尚未核实精确公开来源与 schema，列为待核实项 |
| CodeReviewer | B/C | 需明确是哪个公开数据/版本及所对应标签；可做风险/评论判断，不直接作为统一 score benchmark |
| RiskEval-C | C | 当前尚未核实该精确名称的公开标准数据集；需补来源链接 |

#### 金融判断

| 数据/任务 | 级别 | 用法与注意事项 |
|---|---|---|
| FinDVer | A | 首选直接的财报 claim verification；有 entailment 标注和 evidence metadata；需 pin 最新版本并审计长文档输入 |
| FinQA | B | 原生数值问答/程序推理；如构造 answer verification，标为 derived task，不当作原生二分类 benchmark |
| TAT-QA | B | 表格+文本 QA，可扩展到证据支持/答案正确性；需检查数据许可与题目去重 |
| ConvFinQA | B | 多轮财报 QA；对话上下文不能跨 split 泄漏 |
| Financial PhraseBank | B | 可作容易且稳定的多类情感分类 sanity check，但不足以单独代表高难金融推理 |
| FiQA sentiment | B | 金融情感分析候选；需锁定具体数据版本、标签定义与许可 |
| TweetFinSent | C | 先核实精确名称/官方来源；不要与其它 FinSent/Twitter sentiment 数据集混淆 |
| Credit Card Fraud / IEEE-CIS Fraud | B/C | 部分版本经 Kaggle 分发，访问条款/账户要求需核实；严重类别不平衡和时间泄漏风险大 |
| PaySim / AMLSim | B | 合成欺诈/反洗钱数据；适合机制测试，但不代表真实金融数据分布 |
| LendingClub / Home Credit Default / German Credit | B/C | 信用风险数据涉及公平性、历史制度偏差和许可；German Credit 规模较小且历史较久；不做首发任务 |
| Consumer Financial Protection Bureau (CFPB) complaints | B | 可作官方投诉文本的替代来源，但它不是用户清单中 `Financial-HelpDesk` 等名字的同义数据集；优先级/严重性标签需验证 |
| FinCEN SAR | C | SAR 信息涉及敏感/受限内容；不能假设可公开使用完整真实报送数据 |
| EDGAR / 10-K | B/C | SEC 文件公开，但“合规命中”标签不天然存在；需要自行定义、审计规则和标注流程 |
| Financial-HelpDesk / FinRouting / Complaint-Severity | C | 当前未确认精确名称对应单一标准公开基准；需准确链接或按明确规范自行构造并命名为 derived dataset |
| KYC-Check / RiskEval-C / Expense-Tag / Bank-Transaction-Categorization | C | 名称可能是内部任务称呼、课程数据或项目名；确认仓库、schema、license 后再纳入，不能当作已验证公共基准 |

### 3.3 通用数据与拆分规约

- 每个数据集保存 `dataset_manifest.json`：来源 URL、revision/commit、文件 hash、许可、split 名、行数、标签分布、原始 id 字段。
- 测试集永不参与控制器训练、预算阈值选择、fitness 调参或动作集合设计。
- 三领域都以原始问题/函数/报告/对话为 group key 切分，避免派生样本或负例跨 split。
- 所有派生标签和干扰候选都保存生成规则、seed、源 id、审计字段。
- 先做 20–50 条格式 smoke，然后 100–200 条行动缓存 pilot。仅在所有策略在预测质量上有可观察互补性后扩全量缓存。
- 对代码和金融数据分别审计 class imbalance、近重复、时间/项目泄漏、截断率和敏感属性风险。

## 4. 代码库复用、Undermind 原理与论文汇总

### 4.1 代码库清单与直接复用点

| 代码库 | 链接 | 复用什么 | 不要做什么 |
|---|---|---|---|
| 当前项目 | https://github.com/TianZH-XJU/codex-eacl | split manifest、行动缓存、三策略评测、LR/树/MLP、OOF、paired bootstrap、NEAT CV、错误分析与休眠保护 | 不要重写已经通过测试的缓存/统计基础设施；不要在云端跑实验 |
| Laya 源码 | https://github.com/NandhaKishorM/laya | typed `choice`/`score`/`noul`、CPU 推理、token/head budget、`predict_tournament`、`predict_shortlist`、`option_order`、置信度和 hook | 新实验固定本地 commit；不要直接拉最新 main 后静默改变语义 |
| Laya tournament benchmark | https://github.com/NandhaKishorM/laya/tree/main/research/benchmarks/tournament | pinned 任务格式、候选策略和 benchmark protocol | 公开 benchmark 的设备/版本/耗时不能当作本机 CPU 结果 |
| NEAT-Python | https://github.com/CodeReclaimers/neat-python | population/species、mutation/crossover、genome config、seed、节点/边统计 | 不要因为 NEAT 原理允许复杂网络，就直接扩大搜索预算；先证明信号有效 |
| ProcessBench | https://github.com/QwenLM/ProcessBench | 论文提供的格式、评测代码和 prompt 模板 | 不手工猜 label 编号；不要让同题不同轨迹跨 split |
| PRM800K | https://github.com/openai/prm800k | 步骤级标签 schema 和人类标注说明，作为后续 verifier 数据参考 | 不直接将其题目/标签混入 ProcessBench test；先检查 license / overlap |
| CodeXGLUE | https://github.com/microsoft/CodeXGLUE | Defect Detection 的任务定义、数据处理、官方 split 与现有基线 | 不照搬 GPU CodeBERT 结果来冒充本地 Laya 对照 |
| FinDVer | https://github.com/yilunzhao/FinDVer | 直接复用 claim/entailment/evidence 标注与长财报场景 | pin commit，区分 oracle evidence 与端到端上下文 |
| FinQA | https://github.com/czyssrs/FinQA | 财报 QA schema、gold answer / program、题目 ID 与数据处理 | 不把衍生的 answer verification 说成原生 FinQA leaderboard |
| TAT-QA | https://github.com/NExTplusplus/TAT-QA | 表格和文本混合证据数据，可作为金融外部验证扩展 | 不在首期同时添加多个 QA 任务 |
| ConvFinQA | https://github.com/czyssrs/ConvFinQA | 多轮财报 QA 任务；后续检验对话上下文 | 需要按原始问题链分组切分 |
| Undermind workspace | https://app.undermind.ai/projects/b21f5fa9-2c07-4d02-aa95-6c94ab3d591b | 组织论文、deep search、PDF、文献综述与后续研究笔记 | 文献搜索结果是候选证据；在最终论文引用前还要核实出版状态、DOI、许可和全文细节 |

### 4.2 文献原理：每条线对本研究的作用

1. **NEAT 原理**：Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies*（2002）。关键原理是创新标记、物种保护结构创新、从简单拓扑开始再逐渐复杂化。它说明 NEAT 如何运行，但 NEAT 本身不是本项目的新颖性。
2. **自适应计算分配**：Damani et al., *Learning How Hard to Think: Input-Adaptive Allocation of LM Computation*（ICLR 2025）。直接支持“按输入预测额外计算是否有价值”的研究动机；必须比较预算匹配下的质量与成本。
3. **级联与拒绝/升级**：Jitkrittum et al., *When Does Confidence-Based Cascade Deferral Suffice?*（NeurIPS 2023）。强调仅用初始 confidence 可能忽略下游策略错误结构；是阈值级联基线与理论动机。
4. **模型路由/级联统一**：Dekoninck et al., *A Unified Approach to Routing and Cascading for LLMs*（2024）。帮助定义何时从固定路由转为级联；本项目必须把总成本与质量估计误差一起报告。
5. **小网络/进化 NLP**：autoBOT、有限资源文本分类的进化架构搜索等工作。提醒审稿人“用进化算法做 NLP”不是新贡献；贡献应聚焦冻结 Laya、按样本策略分配、严格成本对照与跨领域验证。
6. **演化 prompt / agent harness**：EvoPrompt、Promptbreeder、EvoTest、TTHE 等。与本项目的搜索目标不同，但说明“演化带来提升”这一泛化主张已经拥挤。需要突出的是预算约束的 typed-decision 控制，不宣称泛化的自我改进。
7. **代码/金融 benchmark**：CodeXGLUE、FinDVer、FinQA、TAT-QA、ProcessBench 提供不同标签和数据来源，但这些 benchmark 本身不保证 Laya 的策略适用；需做 adapter validation。

### 4.3 当前 Undermind 研究区

工作区：[EACL SRW idea research: NEAT + LAYA/JEV on CPU](https://app.undermind.ai/projects/b21f5fa9-2c07-4d02-aa95-6c94ab3d591b)

既有深度搜索：
- [Adaptive inference and selective routing for frozen language-model decisions](https://app.undermind.ai/projects/b21f5fa9-2c07-4d02-aa95-6c94ab3d591b?path=%2FAdaptive%20inference%20and%20selective%20routing%20for%20frozen%20language-model%20decisions)
- [Laya and Jev current implementation and evaluation methods](https://app.undermind.ai/projects/b21f5fa9-2c07-4d02-aa95-6c94ab3d591b?path=%2FLaya%20and%20Jev%20current%20implementation%20and%20evaluation%20methods)
- [NEAT and neuroevolution for NLP under CPU constraints](https://app.undermind.ai/projects/b21f5fa9-2c07-4d02-aa95-6c94ab3d591b?path=%2FNEAT%20and%20neuroevolution%20for%20NLP%20under%20CPU%20constraints)
- [数学推理数据集深度搜索](https://app.undermind.ai/projects/b21f5fa9-2c07-4d02-aa95-6c94ab3d591b?path=%2Fdomains%2FMath%20reasoning%20verification%20datasets)
- [代码判断数据集深度搜索](https://app.undermind.ai/projects/b21f5fa9-2c07-4d02-aa95-6c94ab3d591b?path=%2Fdomains%2FCode%20judgment%20benchmarks%20for%20budget-aware%20decisions)
- [金融判断数据集深度搜索](https://app.undermind.ai/projects/b21f5fa9-2c07-4d02-aa95-6c94ab3d591b?path=%2Fdomains%2FFinancial%20reasoning%20and%20decision%20datasets)

上述三个新领域 deep search 已完成首轮检索。摘要已指出：数学优先 ProcessBench；代码优先 CodeXGLUE Defect Detection，并严格按项目去重；金融优先 FinDVer，而非先自行合成 FinQA 二分类。进入正式实验前还需 pin 数据版本、核实 license、检查论文全文和 split/重复样本。

## 5. 必须对比的基线算法

### 5.1 每个领域都要有的共同基线

1. **Laya SINGLE-only**：一次调用后返回结果，是最低成本参考点。
2. **Laya WIDE-only**：每个样本始终采用较大的 head/context budget；仅对 `choice` 且本地 API 确认支持的任务使用。
3. **Laya TOURNAMENT-only**：每个样本始终用原生 tournament；候选数小/不适用时记录并按预注册处理。
4. **其他 Laya 原生策略（可用时）**：固定 option-order permutation aggregation、built-in shortlist、confidence calibration。Shortlist 若需另一个 embedding model，要把 embedder 的加载/CPU 时间计入。
5. **固定平均成本策略**：固定策略无法自然命中任意预算，可报告 Pareto 端点，不可把不同预算点上的结果混作同一比较。
6. **随机路由**：动作比例由训练/开发预先固定；以相同平均预算比较，仅作 sanity check。
7. **Oracle 上界**：利用留出标签事后选择每样本最佳策略，只用来衡量理论可改进空间；永远不能作为可部署基线或调阈值目标。

### 5.2 控制器基线

- 置信度 threshold cascade（最高概率/answer_confidence；阈值仅在训练/开发选择）。
- margin / entropy / 置信度组合规则。
- Logistic Regression（数值特征；以及 + 初始预测类别的 one-hot 变体）。
- 浅层 Decision Tree（深度 2–4）。
- 小型 MLP（例如 8 个隐藏单元；限制迭代、记录收敛）。
- 固定拓扑 EA（相同输入/输出和 fitness，拓扑不变），用于检验 NEAT 的拓扑演化是否有额外价值。
- NEAT（相同训练数据、seed、fitness evaluation count 和预算网格）。
- **增量收益基线**：直接预测升级相对当前结果的 Δcorrect 与 Δcost，按预计增量收益/增量成本选择。它比仅预测“某一策略是否正确”更接近目标决策。
- 对代码缺陷与金融判断，额外提供 TF-IDF + Logistic Regression / Linear SVM 等廉价任务基线；它们不是 Laya 路由基线，但能帮助判断任务本身是否可由简单文本分类解决。

### 5.3 预算匹配原则

每个策略和控制器都在相同数据 split 上评估实际累计成本。对每个预算值，阈值、LR、树、MLP、固定 EA 和 NEAT 只能在训练/开发集选择相应工作点，再在 OOF/锁定测试上评估。匹配预算基于 **总累计 CPU latency / forward rows 或 calls**，包括首次 SINGLE、升级调用、控制器开销、重跑/复核和必要的额外模型开销；不能只按最后一个 Laya call 的时间算。

## 6. 总体融合架构、SINGLE/WIDE/TOURNAMENT 与迭代预算

### 6.1 原生 Choice 策略的准确含义

- **SINGLE**：先调用一次 Laya 单次 choice，生成初步答案与置信/概率等特征；在控制器动作里，它也表示“现在停止，直接返回初步答案”。不是要在每次选择后再额外运行一次 SINGLE。
- **WIDE**：重新运行相同 choice 问题，但使用更大的 `head_max_len` / `max_len` 或该 pinned 版本定义的宽预算。它不是让 Laya 接着上一次隐藏状态“继续想”，而是另一种配置的推理调用。准确率可能提高，也可能因选项仍被截断/位置偏差而不提高；成本需要实测。
- **TOURNAMENT**：使用 Laya 的 tournament 实现，把候选分成若干组，先在每组选择候选，再把组优胜者进行后续比较。它通常包含多阶段调用，适合候选很多的 Choice 任务；不是对任何 Noul/Score 任务都天然有效。
- **PERMUTATION / SHORTLIST**：若本地 pinned API 可用，可作为额外原生基线。Permutation 改变选项顺序并聚合分数，shortlist 先筛选候选；任何额外 embedder、调用和缓存开销都要入账。

目前已有的 `build_action_cache.py` 采用 SINGLE-first 成本定义：`SINGLE cost = 初次 SINGLE`，`WIDE cumulative cost = 初次 SINGLE + WIDE 调用`，`TOURNAMENT cumulative cost = 初次 SINGLE + tournament 所需调用`。这个定义必须延续到后续公平比较。Tournament 自身的轮数和 forward rows 另外记录，不能把它误记成单次调用。

### 6.2 统一流程

```text
原始样本 x
  │
  ▼
Laya SINGLE（必须先运行，用来得到初步答案和控制器可见特征）
  │  输出：答案、概率/置信度、top-1/top-2 margin、熵、输入长度、候选数量等
  ▼
预算控制器 π(s, B_remaining)
  ├── STOP / SINGLE：保留初步结果，不再调用 Laya
  ├── WIDE：以更宽 head/context budget 重新执行 Choice（仅适用的任务）
  ├── TOURNAMENT：运行候选分组比较并选出最终答案（仅适用的任务）
  └── task-specific VERIFY / ENSEMBLE：Noul 或 Score 的任务适配动作，必须单独验证和命名
  │
  ▼
如果本阶段结果满足停止条件或预算不足：返回结果
否则（仅迭代版）：更新状态、剩余预算与已执行动作，再决定 STOP 或执行一个尚未用过的有效动作
  │
  ▼
记录最终答案、正确性、总累计 latency、各次调用、forward rows/calls、预算使用率、峰值内存
```

### 6.3 迭代式 Laya + NEAT：作为第二阶段扩展，不冒充当前实现

当前已跑的 NEAT 是“一次 SINGLE 后选一个 action”的 one-shot 控制器。用户设想的“NEAT 选择之后，再让 Laya 多判断一次”属于下一阶段的 **iterative budgeted controller**：

1. 第一次 SINGLE：获取初步答案。
2. NEAT 选择 STOP 或一个升级动作；例如 WIDE/Tournament，或一个经过设计的 VERIFY 动作。
3. Laya 执行动作并输出新结果；控制器更新当前答案、置信、与初次答案是否一致、耗时和剩余预算。
4. 第二次控制决策只在预算还够时做，动作集合排除已经执行过的重复动作，且最多允许预注册的 N 次升级（建议先最多 1 次额外策略调用；后续再测 2 次）。达到预算/轮数上限必须停止。
5. 输出最终答案。

`VERIFY` 如果让 Laya 看到原始问题、证据和前一次答案并判断是否成立，是一个**本项目新增的输入模板/动作**；Laya 不是生成式长推理模型，不能仅仅把它称作“继续思考”。需要单独和 `SINGLE→WIDE`、`SINGLE→TOURNAMENT`、固定两遍调用以及无迭代控制器比较。

### 6.4 Controller 状态与输入特征

首期只使用不泄漏 gold 的可观测特征：

- Laya SINGLE 输出：answer_confidence、confidence、top-1 概率、top-1/top-2 margin、概率熵、是否存在异常/截断。
- 任务元数据：候选项数量、每项平均/最大文本 token 数、问题/代码/财报上下文长度、步骤数、表格是否存在、输入截断比率。
- 已执行的动作历史：当前动作、答案是否改变、不同动作是否冲突、已调用次数。
- 预算状态：已消耗估计/实际 latency、剩余 budget、当前动作的训练集 cost quantile。

不得把 gold label、其他策略在该样本上的正确性、test latency 或测试标签编码到特征中。缓存里的未来动作结果只用于训练标签/fitness，不作为部署期输入。

### 6.5 NEAT 的预算函数：不能让每个样本都升级

只优化 raw accuracy 会倾向总选最贵动作；只在 fitness 加一个原始毫秒惩罚，也不能保证硬预算一定满足。采用“双保险”：

**训练目标**（每个预算档单独训练/评估）：

- 软约束版本：`fitness = mean(correct) - λ × mean(cumulative_cost_normalized)`，成本归一化规则在实验前冻结；如额外加复杂度惩罚，作为独立预注册实验，不能和旧版混称。
- 硬约束版本：目标最大化准确率，且只允许平均累计 cost 不超过预算 B；在训练/开发中调节策略阈值/λ，确定可行工作点。

**运行期预算门**：

- 总预算 B 至少覆盖首次 SINGLE 与预计升级成本。
- 如果剩余预算不够当前动作的训练期 cost 上分位估计（例如 P90/P95），禁用该动作；只在可行动作中选择。
- 如果预测的增量收益不足以补偿增量成本，选择 STOP。
- 设定每样本最大调用数/最大升级次数，防止无限迭代。
- 按每个领域独立定义预算，不把数学题的毫秒数与代码/金融任务的毫秒数直接混合。

**预算网格**建议先用相对 SINGLE-only 基线的平均累计成本倍数：`1.00×`、`1.25×`、`1.50×`、`2.00×`。每档的具体 threshold/λ 只在训练/开发确定。还要报告实际 latency、p50/p95、调用数和预算违反率；“目标预算”不是假设它一定被遵守。

## 7. 消融实验设计

消融实验按阶段做，不要所有维度一次性全排列。

1. **策略动作消融**：SINGLE-only；SINGLE/WIDE；SINGLE/TOURNAMENT；三策略；如果可用，再加 permutation/shortlist；任务特定的 VERIFY/ENSEMBLE 单独评估。
2. **控制器消融**：confidence threshold；数值 LR；LR + initial_choice/任务标签（只允许部署可见内容）；tree；小 MLP；固定拓扑 EA；NEAT。
3. **预算目标消融**：纯准确率、soft cost-penalty、hard average budget、相同最大调用数。复杂度惩罚如要测试，独立登记，不可在已看过结果后临时加入并与旧 fitness 混合。
4. **样本量消融**：每类 1/3/5-shot（若每类结构适用），至少 3 个 subset seeds；ProcessBench 等非均衡轨迹数据不强行按类别“每类 k 条”，改为按训练总量与标签分层的固定样本量曲线。
5. **特征消融**：confidence only；confidence + margin；再加 entropy；长度/候选数量；完整廉价元数据；加入初始预测类别的 one-hot（只在类别词表固定且不泄漏时）。
6. **迭代消融**：one-shot；固定 `SINGLE→WIDE`；固定 `SINGLE→TOURNAMENT`；固定 `SINGLE→VERIFY`；NEAT 控制的一次升级；NEAT 控制最多两次升级。所有方法匹配相同预算，不用更多调用来换准确率而不披露。
7. **任务类型消融**：Choice 对原生 Noul；Score 只在输出有效性与标定通过后加入；不把语义不同的 action set 直接合并做 pooled score。
8. **稳健性消融**：题目来源/generator、长度/步骤数、代码仓库、财报题类型、输入截断、候选顺序变换、prompt 的轻微固定变体；每次只改变一个因素。

## 8. 对比与统计协议

### 8.1 三阶段评测

- **Gate 0：task/API sanity**：每个领域抽样 20–50 条，验证格式、金标映射、Laya 输出与 action API；检查 Noul/Score collapse、label-order bias 和截断。不计算研究结论。
- **Gate 1：策略互补性 pilot**：每个样本缓存各有效策略输出。报告固定策略之间 paired errors、仅策略 A 修复、仅策略 B 修复、所有策略都错、所有策略都对。若没有可测互补性，暂停 NEAT。
- **Gate 2：控制器 OOF**：仅在非 test 样本上做分层/分组 OOF；所有 scaler/encoder、成本估计、阈值、λ、策略和 genome 都只在当前 fold-train 拟合。对固定模型、阈值、LR/tree/MLP、fixed EA、NEAT 做配对比较。
- **Gate 3：锁定确认/test**：只在 OOF 门槛通过、方法与预算完全冻结后启动。新确认集需要与训练/开发/旧确认集不重叠；test 只做最终一次评估。当前项目的 test 仍保持锁定。

### 8.2 匹配预算方式

每个领域将策略成本记录成训练期可估计的 `expected cost / P90 or P95 cost`。在开发集选每种基线的工作点，使其平均累计 latency 尽量落在相同 B 的预注册容差（例如 ±5%）；如无法精确匹配，报告真实成本并画完整 Pareto curve，不插值假造结果。另报告 call count、forward rows/rounds、total CPU time 与 cache creation/search time。

### 8.3 指标

- 所有任务：主指标 = 在预算 B 下的准确率/任务主分数 + quality-cost Pareto frontier；另报告累计 mean/p50/p95 latency、平均调用数、策略动作混合、超预算比例、控制器 CPU 时间、训练搜索时间、峰值 RSS。
- 数学步骤定位：exact first-error accuracy、macro-F1、全正确识别、定位偏差。
- 代码缺陷：macro-F1、balanced accuracy、AUROC、AUPRC、precision/recall、正例率。
- 财报判断：balanced accuracy、macro-F1、AUROC/AUPRC，并按计算题类型分层。
- 稳定性：至少 3 个 controller seeds；bootstrap 应对同一原始题目/函数/报告按 group 配对采样，避免把派生候选当作独立样本。
- 统计：paired bootstrap confidence interval；必要时 McNemar 作为辅助。报告差异及区间，不能仅根据单次点估计说“显著提升”。

## 9. 可能结果、研究意义与论文故事线

### A. NEAT 在多个场景、匹配预算下稳定优于简单控制器

故事线：简单置信度并不总能识别额外计算的边际收益；NEAT 学到非线性、任务条件化的预算决策。需要至少在两个领域/多个 seed 有相同方向的证据，并在独立测试确认。不能只挑最好任务。

### B. 自适应策略有价值，但 LR/树/MLP 达到同等效果

故事线：按样本分配预算值得做，但 NEAT 的拓扑进化不是必要的；低成本、可解释控制器更适合本地 CPU 使用。这是实用且可信的负算法结论。

### C. 多策略存在互补错误，但特征无法预测增量收益

故事线：瓶颈在可观测状态信号/小样本学习，不是搜索代数。可研究增量收益标签、task metadata、少样本规模和分布偏移；不应直接宣称 NEAT 无效于所有任务。

### D. 不同领域结果明显不同

故事线：预算控制依赖任务结构；数学步骤定位、代码缺陷、财报证据判断对 Laya 的信号和策略互补性要求不同。应报告条件化结论，不把三个领域简单平均成一个总分。

### E. 升级策略提高准确率但没有匹配成本优势

故事线：收益来自更多计算而非更好的预算分配。结论应是固定升级策略带来质量收益，但控制器没有改善效率。

### F. Laya 原生策略在某些任务上均不够好

故事线：当前决策引擎或输入/输出结构是主要瓶颈，控制器不能创造模型未提供的能力。此时应转向可解释的任务适配、Laya fine-tuning/头部训练或更合适模型对比，作为单独研究；不应继续对失败的 action set 做无止境路由搜索。

## 10. 实施路线与停止门槛

1. **整理现状并复用现有代码**：不重跑 BANKING77 NEAT、不启动新确认；保留 74eb143 与 2f4fb36 作为旧实验事实。
2. **Math gate（优先）**：ProcessBench 下载/许可/hash；官方 label mapping 单测；固定题目级 split；Laya API smoke；比较 SINGLE/WIDE/TOURNAMENT；仅缓存 train/dev 的输出。
3. **Math gate pass 才训练控制器**：如果策略间存在互补性且某些 upgrade gain 可由特征预测，跑阈值/LR/tree/MLP/NEAT 的预算匹配 OOF。若失败则记录并继续 code adapter，不加代数。
4. **Code gate**：CodeXGLUE Defect/Devign，先完成 Noul vs binary Choice 的 API/标签敏感性对照；建立项目级分割和类别不平衡指标。只在有效动作家族定义清楚后做路由实验。
5. **Finance gate**：优先 FinDVer claim verification；pin 官方仓库版本，审计 report 分组、证据标注、输入长度/截断及 Noul/Choice adapter。FinQA-derived answer verification 仅作后续扩展，须固定负例规则并人工抽查。
6. **跨域总结**：分别报告每领域和 task type，不能把动作不兼容的任务强行汇总。若方法只在 Choice 任务可行，诚实缩小主张；若 Noul/Score 的策略适配通过验证，再报告跨类型泛化。
7. **最终确认**：只有 OOF 门槛通过后才冻结代码、预算、split 和哈希；test 集最后使用一次。

### 10.1 每阶段交付物

- `configs/tasks.yaml`：任务 schema、原始数据 revision、label 映射、action adapter、预算档。
- `scripts/prepare_processbench.py`、`scripts/prepare_codexglue_defect.py`、`scripts/prepare_findver.py`（FinQA 衍生任务后续可另加 `scripts/prepare_finqa_verification.py`）：可重复数据适配。
- `scripts/laya_task_smoke.py`：三种 typed primitive 与策略 adapter 的真实 API smoke。
- `scripts/build_action_cache.py`：保留，增加统一 task adapter 与实际调用/forward row 记录。
- `scripts/evaluate_strategy_complementarity.py`：新增；报告 pairwise errors、Δcorrect、Δcost 和可预测性。
- `scripts/run_budgeted_controller_cv.py`：建议新建，而不是把三领域条件继续堆在旧 BANKING77 专用脚本中。
- `scripts/evaluate_budget_frontier.py`：统一 budget profiles、成本容差和 Pareto 计算。
- `results/analysis/`：只提交压缩 JSON/Markdown、manifest 和摘要图；原始/大型缓存和 63 条逐代日志留本地 git-ignored。
- `tests/`：数据标签、同题不跨 split、预算不超门、成本累加、Noul fallback、动作适配器有效性、测试集不可读等结构测试。

## 11. 对当前项目的复用清单

| 已有文件/机制 | 可直接复用 | 需要修改/扩展 |
|---|---|---|
| `scripts/build_action_cache.py` | SINGLE-first 执行、成本累加、错误/耗时/概率存储、Windows 休眠保护 | 加 task adapter；存 forward rows、内层 round 数、token/truncation；选择特定策略不兼容时禁止跑 |
| `scripts/run_three_action.py` | 三策略 cache-only 评测、action mix 和成本读取 | 重构成通用 task schema；保留当前三策略作为 choice adapter |
| `scripts/run_controller_baselines.py` | 特征提取、LR/tree/MLP/threshold 结构 | 加多标签/不平衡指标和任务特征；保留 train-only scaler/encoder |
| `scripts/run_cv_stability.py` | group OOF、折内预处理、配对 bootstrap | 根据 task id/group id 切分；预算档与动作集合折内锁定 |
| `scripts/run_neat_cv.py` | neat-python 真正进化、fixed-topology EA 对照、progress log、线程限制、休眠保护 | 不作为跨域最终脚本直接复制；将 action adapter、budget mask、分组 split 与 Δutility objective 抽成模块 |
| `scripts/analyze_neat_diagnostics.py` | OOF genome/动作混合/特征可预测性诊断 | 改为预测 `Δcorrect`/`Δcost`，加每折样本数与分子分母 |
| `scripts/analyze_cached_errors.py` | 缓存回放、错误重叠分析 | 扩展到 domain-specific labels 和 group/bootstrap |
| `scripts/rerun_anomaly_rows.py` | 保留历史耗时、带休眠保护重跑异常 | 只对真正需要重测的缓存行调用；不可覆盖原始记录 |
| `scripts/sync_checkpoint.ps1`、`STATUS.md` | 提交摘要、报告 Git SHA、数据/模型 provenance | 更新状态与新阶段门槛，不把实验日志上传云端 |

## 12. 首轮执行命令建议（仅用户本地）

**以下是后续建议步骤，不表示已经运行。** 拉取计划并检查工作树后，首先执行轻量测试和数据/schema smoke；不要先做全量缓存。

```powershell
# 在项目根目录
.\.venv\Scripts\python.exe -m pytest tests/ -q

# 之后新增准备脚本时，先验证 ProcessBench 原始 schema 和 label mapping
.\.venv\Scripts\python.exe scripts\prepare_processbench.py --inspect-only

# 然后只做小样本 Laya API 验证（实际参数以新脚本 help 为准）
.\.venv\Scripts\python.exe scripts\laya_task_smoke.py --task processbench --limit 30
```

预期原则：网络下载阶段需要在线；一旦数据与模型进入本地 cache，正式模型/控制器实验设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。每次执行都保存命令、`git rev-parse HEAD`、数据 hash、配置 hash 与资源日志。所有真实模型调用只发生在本地 Windows PC。

## 13. 参考代码与论文（起步清单）

- Laya repo 与 many-option benchmark：<https://github.com/NandhaKishorM/laya>；<https://github.com/NandhaKishorM/laya/tree/main/research/benchmarks/tournament>
- ProcessBench（ACL 2025）：<https://github.com/QwenLM/ProcessBench>；<https://huggingface.co/datasets/Qwen/ProcessBench>
- PRM800K / *Let's Verify Step by Step*：<https://github.com/openai/prm800k>；<https://arxiv.org/abs/2305.20050>
- CodeXGLUE：<https://github.com/microsoft/CodeXGLUE>
- FinDVer：<https://github.com/yilunzhao/FinDVer>；论文 <https://aclanthology.org/2024.emnlp-main.818/>
- FinQA：<https://github.com/czyssrs/FinQA>
- TAT-QA：<https://github.com/NExTplusplus/TAT-QA>
- ConvFinQA：<https://github.com/czyssrs/ConvFinQA>
- NEAT：Stanley & Miikkulainen (2002), DOI <https://doi.org/10.1162/106365602320169811>
- Adaptive computation allocation：Damani et al., *Learning How Hard to Think* <https://arxiv.org/abs/2410.04707>
- Cascade deferral：Jitkrittum et al., *When Does Confidence-Based Cascade Deferral Suffice?* <https://arxiv.org/abs/2307.02764>
- Model routing/cascading：Dekoninck et al., *A Unified Approach to Routing and Cascading for LLMs* <https://arxiv.org/abs/2410.10347>
- Undermind workspace and current deep searches are linked in §4.3; the additional math/code/finance searches should be reviewed when their asynchronous runs complete.

---

## 一句话决策

先把 ProcessBench 做成一个严格的数学 Choice 实验，确认 SINGLE/WIDE/TOURNAMENT 是否真的各有所长；再接入 CodeXGLUE 的二元代码判断和 FinDVer 的财报声明证据验证。NEAT 只有在“策略有互补性、增量收益能被特征预测、预算可被严格匹配”三道门都通过后，才值得训练与做独立确认。否则应把结论写成有边界的负结果，而不是继续增加进化预算。
