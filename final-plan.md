下面进入收尾阶段 下面直接把 \*\*《EvoTIDE 中文审核稿 v0.4》\*\*完整放在对话中。为了方便审核，我保留了六个一级章节、实验表格、公式、参考文献，以及文末单独列出的补充实验计划。

这是一份论文内容审核稿，不是项目进度报告。文末的补充实验计划属于作者内部讨论材料，不属于正式投稿正文。

# 进化式路由为何未能带来稳定收益：结构化决策模型中的动作互补性与效用预测

英文拟题：When Evolutionary Routing Fails: Action Complementarity and Utility Prediction in Structured Decision Models

中文审核稿 v0.4 — 2026 年 11 月 1 日

## 摘要

Jev 等 System One 决策模型将语言模型的输出从自由文本转为可由软件直接调用的类型化概率判断。其代表性训练路线 RLCD（Reinforcement Learning for Calibrated Decisions）以概率报告质量为优化目标；开源模型 Laya 将这一训练思路用于非自回归决策模型，而 Vela 2.0 0.3B 则提供了相近的结构化输出接口，但采用不同的训练配方。概率化输出为推理时路由提供了信号，却没有直接回答一个关键问题：当前动作的概率输出能否识别另一种动作带来的增量收益？

本文在冻结的 Laya 和 Vela 上研究这一问题，覆盖 CodeXGLUE 代码缺陷检测、FinDVer 金融声明核验和 ProcessBench 数学推理错误定位。对于前两项任务，我们以 Noul 概率触发 Choice 第二视图；对于 ProcessBench，则学习基于候选概率和间隔的弃答规则。我们比较遗传算法（GA）、进化策略（ES）和 NSGA-II 与固定策略及预算匹配的随机搜索，并通过动作互补性、效用预测、搜索轨迹和跨任务迁移实验解释搜索结果。

GA 相较随机搜索的 36 个 macro-F1 配对比较均未达到显著；预注册的 GA 对 fixed switch-only confirm 主比较在 12 个种子级单元中也均未达到显著。动作分析发现，部分任务存在毛修复空间，但修复与有害切换抵消后，没有出现统计上可靠的正向净修复；预测模型在训练池中取得的排序信号也未稳定转化为留出策略收益。结果表明，结构化决策模型的概率输出可以支持推理时控制，但有效路由还要求动作之间具有互补性，并要求部署可见信号能够预测替代动作的增量价值。

## 1. Introduction

越来越多的语言模型被部署在需要即时决策的软件流程中：系统可能要判断一项声明是否得到证据支持、一个代码函数是否存在漏洞，或一段推理过程是否已经出现错误。在这些场景里，下游程序需要的是可直接执行的判断，而非一段还要经过解析和二次判读的自然语言回答。Jev 所代表的 System One 模型正是为此类任务设计：给定输入状态和类型化问题，模型直接返回 `Noul`、`Choice` 或 `Score` 等结构化结果及其概率信息。TypeSafe 将 Jev 的训练方法称为 Reinforcement Learning for Calibrated Decisions（RLCD），即通过概率评分目标训练模型报告决策及其不确定性（TypeSafe AI, 2026; Guo et al., 2026）。开源模型 Laya 也将其训练流程定位为 RLCD，并以非自回归方式执行类型化决策；Vela 2.0 则提供相近的决策接口，但使用不同的编码器训练与任务损失。它们共同代表了一类面向机器调用的结构化决策模型。

这种接口改变了推理阶段的控制方式。传统系统通常对每个输入执行同一种推理流程；而在级联或自适应计算系统中，模型可以先给出成本较低的判断，再根据第一步返回的概率决定是否调用额外动作。基于置信度的门控为这一策略提供了直观依据：模型越不确定，系统越可能投入额外计算。然而，当前模型是否容易出错，并不能决定额外计算是否值得。只有当替代动作能够纠正当前错误、且路由器能够识别这些可修复样本时，级联才可能优于固定动作。

Jitkrittum et al.（2023）对置信度级联规则给出了理论分析，并展示了当下游模型具有专门化错误模式、标签存在噪声或数据分布变化时，仅依赖当前模型置信度可能失效。其研究带来一个直接启发：路由器需要理解级联中不同动作的错误结构，而不只是判断当前输出是否可靠。对于 Jev 类模型，这个问题还有另一层含义。RLCD 训练模型表达当前判断的概率，但系统实际要决定的是是否调用另一种问题类型或输出表示。校准当前判断，与预测替代动作带来的增量收益，并不是同一任务。 即使模型能够较好地估计 `P(true)`，也未必能够据此识别 Choice 是否会修正当前 Noul 结果。

本文据此研究冻结结构化决策模型上的推理时路由。与在多个独立模型之间选择不同，我们固定模型参数，只改变同一模型接收的问题类型：在 CodeXGLUE 和 FinDVer 任务上，先通过 Noul 获取二分类概率，再决定是否调用 Choice 作为第二种决策视图；在 ProcessBench 上，则根据候选置信度和 top-1/top-2 间隔决定是否弃答。我们使用 GA、ES 和 NSGA-II 搜索这些策略参数，并与固定动作和预算匹配的随机搜索比较。这样设计能够将“搜索没有提升”拆解为三个可检验的问题：动作之间是否具有净互补性？替代动作的增量效用能否由推理时可见信号预测？进化搜索能否在给定预算内利用这些信号？

这三个问题构成本文的研究主线。RQ1 检验动作互补性，重点区分毛修复机会与有害切换；RQ2 检验增量效用可预测性，重点考察训练池的排序信号能否通过 dev 选择并在 confirm 上保留；RQ3 检验搜索有效性，比较不同进化方法与固定、随机策略在相同候选评估预算下的表现。迁移实验和基于概率不确定性与候选间隔的故障图则用于观察学到的策略能否跨任务使用，以及错误集中出现在哪些概率区域。

实验没有发现 GA、ES 或 NSGA-II 带来稳定的部署优势。GA 相较随机搜索的 macro-F1 比较均未达到显著，预注册的 GA 对 fixed switch-only confirm 主比较也没有显示显著收益。Vela 在 FinDVer 任务上的 GA 结果出现了训练池内显著的正向差值；该组合在 dev 和 confirm 上仍有正向点估计，但置信区间跨越零。动作级分析进一步发现，部分表示组合确实能够修复另一种表示的错误，但相应的有害切换抵消了净收益；训练数据上出现的部分效用排序信号也没有稳定转化为留出策略收益。本文因此将重点从“哪一种优化器更强”转向一个更基础的问题：结构化决策模型的概率输出，是否包含足以支持动作分配的增量效用信息？

本文的贡献是：在冻结的结构化决策模型和固定动作集合上，系统比较 GA、ES、NSGA-II、随机搜索及固定策略；将动作层面的互补性与路由层面的增量效用预测分开测量；并把这两层诊断与搜索轨迹、跨任务迁移及基于概率不确定性与候选间隔的故障分析结合，解释潜在收益为何没有转化为稳定的推理时收益。

## 2. Related Work

### 2.1 Jev 类结构化决策模型与 RLCD

System One 决策模型将模型输出从自由文本转为软件可以直接消费的类型化结果。Jev 是 TypeSafe AI 发布的商业模型：输入一个 state 和一组 typed questions，模型可以在一次调用中返回二元判断 `Noul`、候选选择 `Choice` 和有序评分 `Score` 等结果。Jev 不以自由文本生成为主要输出形式，而是面向分类、路由、审核和评分等机器决策任务（TypeSafe AI, 2026）。Deußer et al.（2026）在 37 个数据集上对 Jev 进行了系统评测；Guo et al.（2026）则研究了 Jev 在多类对齐失败检测任务中的表现。这些工作显示，结构化决策模型正在从单一标签预测扩展到面向不同决策接口的概率化判断。

TypeSafe 将 Jev 的训练方法称为 Reinforcement Learning for Calibrated Decisions（RLCD）。其核心目标是用严格恰当的概率评分规则优化模型报告的分布，使概率不仅用于选择标签，也能表达判断的不确定性。TypeSafe 公布了训练目标的定位，但没有完整公开 Jev 的训练配方与实现细节。Laya 的训练文档进一步给出了可检查的奖励构成：Choice 和 Noul 使用 log score 与 spherical score 的加权组合，Score 任务还引入 ranked probability score（Laya Studio, 2026）。Gao and Wang（2026）提出的 OpenJev-RLCD 为这一方向提供了可复现的独立实现，但不能视为 Jev 原始训练代码的公开版本。需要区分的是，RLCD 也曾被 Yang et al.（2023）用于 Reinforcement Learning from Contrast Distillation，二者名称相同、含义不同；本文中的 RLCD 专指面向校准决策的强化学习。

本文使用的两个开源模型具有相近的类型化决策接口，但训练来源并不相同。Laya 0.4.16 的英文版本以约 395M 参数的 ModernBERT-large 编码器为基础，并在其上加入决策专用层和选项评分头，总参数量约 421M；它以非自回归方式支持 `Choice`、`Noul` 和 `Score`，项目文档将其训练方法描述为基于严格恰当评分规则的 RLCD。Vela 2.0 0.3B 使用 22 层、隐藏维度 768 的双向 ModernBERT 编码器，输入上限为 8,192 tokens，支持 `Choice`、`Noul`、`Score`、`Span` 和 `Set`。其公开训练说明记录了从 Decision-1.0-Kai 的 Choice 路径初始化、任务专用损失和包含合成路由决策的数据配方，并未将其训练方法称为 RLCD（Laya Project, 2026; vllm-sr, 2026）。因此，本文将 Laya 视为开源 RLCD 决策模型，将 Vela 作为接口相近、训练路线不同的比较模型，而不把两者统称为 RLCD 模型。

| 模型            | 结构与输出接口                                                        | 训练路线                                              | 本文中的角色                           |
| ------------- | -------------------------------------------------------------- | ------------------------------------------------- | -------------------------------- |
| Jev           | 商业 System One 模型；一次调用返回 Noul、Choice、Score 等类型化概率决策             | TypeSafe 公布 RLCD 名称及概率评分目标；具体训练实现未完整公开            | RLCD / System One 的代表性参照，未纳入本文实验 |
| Laya 0.4.16   | ModernBERT-large 编码器及决策专用层，总参数量约 421M；支持 Noul、Choice、Score     | 上游项目将其训练流程称为 RLCD，并使用严格恰当评分规则                     | 开源 RLCD 路线的实验模型                  |
| Vela 2.0 0.3B | 22 层、768 hidden 的 ModernBERT 编码器；支持 Noul、Choice、Score、Span、Set | 以 Decision-1.0-Kai 的 Choice 路径初始化，采用任务专用损失与路由决策数据 | 与 Laya 接口相近、训练路线不同的比较模型          |

这一模型背景引出本文的核心区分：概率报告质量与动作分配质量并不相同。模型的 `Noul` 概率描述当前二元判断；路由器要估计的则是切换到 `Choice` 后结果会改善还是恶化。前者是单个动作的预测问题，后者是两个动作之间的增量效用问题。本文因此不把校准概率直接等同于路由分数，而是将它作为待检验的部署信号。

### 2.2 级联与自适应计算：从当前置信度到替代动作价值

级联推理通过按需调用多个模型或计算动作，在不同输入上分配不同的推理成本。FrugalGPT 研究如何在不同价格和能力的生成式模型之间路由，以降低成本并保持或提高质量（Chen et al., 2024）；AutoMix 使用小模型的自验证信号和 POMDP 路由器决定何时升级到更大的模型（Aggarwal et al., 2024）；BEST-Route 则联合决定调用哪个模型以及生成多少个候选，以控制 test-time compute（Ding et al., 2025）。这些工作展示了路由与额外计算分配的实际价值，也说明质量估计、验证信号和动作成本需要共同进入策略设计。

与这些系统直接相关的是 Jitkrittum et al.（2023）对基于置信度的级联延迟（confidence-based cascade deferral）的分析。该研究指出，当前模型的概率输出并未描述下游模型的错误结构；在下游模型具有专门化能力、数据存在标签噪声或分布发生变化时，单纯依据当前置信度进行延迟可能不是最优选择。这一结论让级联问题从“当前模型是否会错”推进到“后续动作是否能修复当前错误”。Damani et al.（2025）关于输入自适应计算分配的研究也强调，应根据额外计算的预期收益决定推理资源，而不是仅依据输入表面难度分配固定计算量。

这些研究启发我们把级联收益拆成两个先后发生的判断。路由之前，动作集合必须先提供足够的互补性：替代动作修复当前错误的次数应超过它破坏正确结果的次数。即便这一机会存在，路由器还必须从部署时可用的信号中预测哪些样本真正值得切换。本文据此设计 E1 和 E2：E1 对逐样本 repair、harmful switch 和 oracle headroom 进行分析；E2 使用不依赖第二动作输出的部署特征，预测切换的增量效用，并将训练排序、dev 策略选择与 confirm 评估分开。这个设计将已有级联理论中的下游错误结构问题，转化为结构化决策模型上的可测量实验。

### 2.3 进化优化：搜索策略的收益取决于可利用的信号

遗传算法通过选择、交叉和变异更新候选策略；进化策略通过连续参数扰动和适应度选择进行优化；NSGA-II 则在多个目标之间维护非支配解集（Deb et al., 2002）。这些方法适用于混合离散和连续参数，且不要求目标函数可微，因此适合在冻结模型输出的缓存上搜索推理时规则。

已有语言模型进化研究主要把提示词或程序作为搜索对象。EvoPrompt 将遗传算法和差分进化用于离散自然语言提示优化（Guo et al., 2024），Promptbreeder 则联合演化任务提示与变异提示（Fernando et al., 2024）。这些工作通过修改模型接收的文本来改变输出；本文保持模型和任务问题不变，搜索的是决定是否调用第二种结构化输出动作的策略参数。研究对象不同，评价问题也不同：提示优化考察新的提示能否提高单次模型输出，而本文考察给定动作之间是否存在能够被规则稳定利用的增量收益。

由此形成本文的实验顺序。级联文献提示我们先检查动作互补性，再判断增量价值是否可预测；进化优化文献提供了搜索这类策略参数的方法，但并不保证动作集合中存在可利用的收益。本文因此将 GA、ES 和 NSGA-II 放在诊断链的最后一环：只有把动作层和预测层的结果放在一起，才能解释搜索表现，而不是仅以优化器之间的分数排名作为研究结论。

## 3. Method and Decision Rules

### 3.1 研究对象与任务动作

本文研究冻结的结构化决策模型在推理阶段如何选择固定动作。每个输入实例记为 \\(x\\)，底层模型参数保持不变，模型调用方式和输出 schema 由任务适配器规定。策略仅读取第一次调用获得的信号，并决定是否执行第二种动作，或是否弃答。进化算法不修改模型权重，也不进化提示词或候选标签集合。

在 CodeXGLUE 代码缺陷检测和 FinDVer 金融声明核验中，第一种动作是 `Noul`：模型回答一个二分类问题并返回 `P(true)`。第二种动作是 `Choice`：模型在两个明确定义的候选标签之间选择。对于 CodeXGLUE，问题分别对应函数是否存在漏洞，以及函数应被判为 `secure` 还是 `vulnerable`；对于 FinDVer，问题分别对应声明是否得到证据支持，以及在 `supported` 与 `not supported` 之间作出选择。级联策略始终先执行 `Noul`，只有当 Noul 返回的正类概率 \\(p(x)\\) 落入指定区间且策略启用第二动作时，才额外调用 `Choice`。

ProcessBench 的任务输出是多类别步骤定位结果，可用策略杠杆不同于二分类任务。该实验不在两种表示之间切换，而是根据最高候选置信度以及前两名候选之间的间隔决定是否弃答。因此，ProcessBench 结果对应选择性预测策略，不与二分类级联的调用成本或准确率差值直接合并。

### 3.2 二分类级联基因组

二分类任务的策略由 `CascadeGenome(lo, hi, use_second)` 参数化，记为：

\\[ g\_{\mathrm{cas}}=(\ell,h,u),\quad \ell,h\in[0,1],\\;u\in\\{0,1\\}. \\]

设第一种 `Noul` 动作返回的正类概率为 \\(p(x)\\)。基因组直接对 \\(p(x)\\) 设置区间门控；这里使用的是正类概率，而不是 \\(\max(p,1-p)\\) 形式的置信度。策略规则为：

\\[ \pi\_{g}(x)= \begin{cases} \mathrm{Choice}(x), & u=1\\;\land\\;\ell\leq p(x)\leq h,\\\ \mathrm{Noul}(x), & \text{otherwise}. \end{cases} \\]

基因修复操作将区间端点裁剪至 [0,1]，并在端点顺序颠倒时交换二者；离散基因 `use_second` 被转换为 0 或 1。该表示包含两个重要固定策略：`use_second=0` 对应 `Noul`-only；`use_second=1, lo=0, hi=1` 对应始终执行第二种 `Choice` 动作。后者即 fixed switch-only 基线，但实际执行时仍包含前置的 `Noul` 调用，因此其每行调用数为两次。基因向量的 `to_vector`/`from_vector` 往返转换由单元测试覆盖。

### 3.3 数学任务的弃答基因组

ProcessBench 使用 `AbstainGenome(t, use_margin, m)`：

\\[ g\_{\mathrm{abs}}=(t,v,m),\quad t,m\in[0,1],\\;v\in\\{0,1\\}. \\]

设 top-1 候选的概率为 \\(s_1(x)\\)，top-1 与 top-2 之间的差值为 \\(\Delta(x)\\)。当满足 \\(s_1(x)\<t\\)，或 \\(v=1\\) 且 \\(\Delta(x)\<m\\) 时，策略弃答；否则返回 top-1 候选。实验报告 coverage（作答比例）与 selective accuracy（已作答样本的准确率）。当必要的置信度字段缺失时，评估实现按 fail-closed 规则处理：系统强制作答并将其计为错误，而不是静默跳过实例。

这两个基因组代表两种不同的推理时策略：二分类任务通过 Noul 正类概率区间触发替代表示，数学任务通过 Choice 输出的 top-1 概率与 margin 控制作答覆盖率。二者共同服务于自适应决策研究，但不共享完全相同的动作空间和评价目标。

### 3.4 适应度、缓存评估与进化算法

二分类任务的策略适应度由任务质量和调用成本构成：

\\[ J(g)=Q(g)-\lambda C(g), \\]

其中 \\(Q(g)\\) 为任务主指标，CodeXGLUE 和 FinDVer 使用 macro-F1；\\(C(g)\\) 为每行模型调用数相对一次基础调用的增量成本，即 `calls_per_row − 1`；\\(\lambda\\) 为预先设定的成本权重，默认值为 0.05。模型输出由冻结模型真实生成并存入不可变缓存；搜索阶段只重放缓存结果，对不同基因组进行离线评估。候选策略即使不触发第二种动作，也仍需承担第一次 `Noul` 调用的成本。

GA 使用群体选择、均匀交叉、连续参数变异和离散位翻转；ES 使用 \\((\mu+\lambda)\\) 策略，对连续区间端点进行高斯变异，并固定 `use_second=1`，以隔离连续阈值搜索的作用；NSGA-II 在二分类任务上优化 accuracy 与调用成本，在 ProcessBench 上优化选择性准确率与覆盖率；其二分类目标与 GA 的 macro-F1 主指标并不相同。主要搜索使用随机种子 17、23、41，唯一基因组评估预算为 24；另在 CodeXGLUE 任务上的 Vela 模型进行预算上限 48 的探测。随机搜索使用相同的唯一候选评估预算。

## 4. Experiment

### 4.1 实验设计总览

实验围绕三个研究问题展开，并增加迁移与故障诊断，以检查结论是否只反映某个数据池或搜索配置。Jev 用于界定 RLCD/System One 研究背景；实际模型比较由能够固定本地权重版本并重放缓存的 Laya 与 Vela 完成。表 1 概括了每项实验希望测量的对象、设计和关键比较。这样安排的目的是把“没有发现进化收益”拆解为可以检验的原因，而不是将最终准确率作为唯一解释。

表 1. 研究问题、实验目标与主要比较

| 实验            | 对应问题 | 测量目标                      | 主要设计与比较                                                                |
| ------------- | ---- | ------------------------- | ---------------------------------------------------------------------- |
| E1：原始动作与互补性   | RQ1  | 第二种表示能否修复第一种表示的错误，净修复是否为正 | 比较 Noul 与 Choice；计算 repair、harmful switch、oracle headroom 和 net repair |
| E2：增量效用预测     | RQ2  | 训练时能否预测替代动作的逐样本增量收益       | group-aware OOF 排序；冻结规则后比较 dev 与 confirm                               |
| E3：进化搜索       | RQ3  | 搜索能否超过固定和随机策略             | GA、ES、NSGA-II 与固定策略、预算匹配随机搜索比较                                         |
| E4：预算与搜索动态    | RQ3  | 搜索是否利用了给定的候选评估预算          | 记录 best-so-far 轨迹、唯一候选数，并对预算 24/48 进行比较                                |
| E5：跨任务迁移与故障诊断 | 综合诊断 | 学到的策略能否迁移；错误是否集中于可修复样本    | 冻结源任务基因组后评估目标任务；构建基于 Noul 概率不确定性与数学候选间隔分层的故障图                          |

### 4.2 模型、任务与数据划分

模型版本与执行环境。 实验使用 Laya 0.4.16 英文检查点（源代码修订号 `b9fe351ce81e2d974bca2bf219c1e4030c0a05ea`；检查点修订号 `7b928d828b7b0e022f929d9bd2e44165aa270148`）和 Vela 2.0 0.3B（模型修订号 `d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409`；权重 SHA-256 纳入版本清单）。所有推理均采用 CPU、Torch FP32 和单进程执行，同一时间只驻留一个模型，设置两个 CPU 线程；记录到的峰值 RSS 不超过约 2.2 GB。

CodeXGLUE。 任务为代码缺陷二分类，标签为 secure/vulnerable。官方数据规模为 train/dev/test = 21,854/2,732/2,732。控制器数据池仅从官方训练集构建，并按函数文本精确哈希分组，以避免完全重复函数跨越内部 search_train、selection_dev 和 confirm 数据池；当前用于策略实验的三个内部池各包含 60 个函数样本。该方案控制重复函数泄漏，但不是 project-level holdout。模型主指标为 macro-F1，accuracy 和 balanced accuracy 为次指标。

FinDVer。 任务为金融声明核验，标签为 supported/not supported。内部数据池从 testmini 构建，并按 report 分组，使同一报告中的相关声明不跨越不同控制器数据池；用于策略实验的 search_train、selection_dev 和 confirm 分别包含 60、59 和 44 条声明，官方 test 标签不参与本研究的控制器搜索。主指标为 macro-F1，accuracy 和 balanced accuracy 为次指标。

ProcessBench。 任务是定位数学推理过程中的最早错误步骤，或判断步骤链全部正确。数据包含 3,400 条标注记录、覆盖 2,842 个数学问题；拆分按 problem 分组。数学策略实验使用三个各 60 条记录的内部池。主指标为 step accuracy，弃答实验另报告 coverage 与 selective accuracy。

### 4.3 E1：基线动作表现与动作互补性

E1 先建立两种表示各自的基线表现，再判断它们是否互相修复错误。表 2 汇总了 selection_dev 上统一原生矩阵的准确率。二分类任务中，Laya 在 CodeXGLUE 任务上的 `Noul` 准确率为 0.45，`Choice` 为 0.65；Vela 对应为 0.633 和 0.667。在 FinDVer 任务上，Laya 的 `Noul` 为 0.712、`Choice` 为 0.644；Vela 两种表示均为 0.458。在 ProcessBench 任务上，Laya 和 Vela 的 Choice 基线准确率均为 0.10。该矩阵显示模型表现随任务和表示而变化：第二种表示在 CodeXGLUE 任务上改善了 Laya 和 Vela 的准确率，但在 FinDVer 任务上并未对两个模型都带来改善；ProcessBench 的低准确率也提示，弃答策略并不能替代缺失的基础预测能力。

表 2. Laya 与 Vela 在三个任务上的原生动作基线（selection_dev accuracy）

| 任务           | 模型   | Noul / Choice 基线表现（accuracy） |
| ------------ | ---- | ---------------------------- |
| CodeXGLUE    | Laya | 0.45 / 0.65                  |
| CodeXGLUE    | Vela | 0.633 / 0.667                |
| FinDVer      | Laya | 0.712 / 0.644                |
| FinDVer      | Vela | 0.458 / 0.458                |
| ProcessBench | Laya | Choice 0.10                  |
| ProcessBench | Vela | Choice 0.10                  |

仅比较两个动作的总体准确率，还不能说明是否值得逐样本路由。接下来的逐样本互补性分析使用 search_train 缓存，以每个任务—模型单元中表现较好的固定动作为参照，计算另一动作能够修复的样本数（repair）、将正确预测变为错误的样本数（harmful switch），以及两者之差（net repair）。oracle headroom 则表示，如果事后能够为每个样本选择正确的动作，准确率相较最佳固定动作最多可以提高多少。它衡量的是动作集合提供的理论机会，并不代表真实路由器能够达到的收益。

Laya 在 CodeXGLUE 任务上的 oracle headroom 最大（0.233）：以 Noul 为基准，Choice 修复了 14 个错误，同时也造成 15 个有害切换，净计数为 −1。Vela 在 CodeXGLUE 任务上的表现以 Choice 为较好的固定基线，Noul 带来 4 次修复和 10 次有害切换，net repair 为 −6；在 FinDVer 任务上，Laya 的 Noul 基线对应 5 次修复和 6 次有害切换，Vela 在 FinDVer 任务上的表现以 Choice 为较好的固定基线，对应的 Noul 带来 6 次修复和 7 次有害切换。两个模型在 ProcessBench 上只有一种可用的步骤选择表示，因此该任务的 oracle headroom 为 0。四个二分类单元的净修复 accuracy 区间均未显示可靠的正向效应。由于 CodeXGLUE 与 FinDVer 的主指标为 macro-F1，E1 的逐样本 repair/harm 分解还需要与主指标下的固定动作及 oracle 比较配套报告；文末将这一项列为优先补充分析。独立 complementarity screen 中的最大观测净值为 +6/60，二项检验 p≈0.09。E1 的结论并不是替代动作从不修复错误，而是现有动作之间的修复和有害切换相互抵消，尚未形成稳定的正向净收益。需要注意，E1 为了计算 oracle headroom，逐单元以准确率更高的固定动作作为基准，所以 Vela 在部分任务上的比较方向是 Choice 到 Noul；E2 则按实际级联策略，统一研究从 Noul 升级到 Choice 的增量效用。

| 任务与模型                      | 较优固定动作 → 替代动作 | n  | Oracle headroom | Repair / Harmful / Net | Net repair 95% CI |
| -------------------------- | ------------- | -- | --------------- | ---------------------- | ----------------- |
| Laya 在 CodeXGLUE 任务上的表现    | Noul → Choice | 60 | 0.233           | 14 / 15 / −1           | [−0.200, 0.167]   |
| Vela 在 CodeXGLUE 任务上的表现    | Choice → Noul | 60 | 0.067           | 4 / 10 / −6            | [−0.217, 0.017]   |
| Laya 在 FinDVer 任务上的表现      | Noul → Choice | 60 | 0.083           | 5 / 6 / −1             | [−0.136, 0.090]   |
| Vela 在 FinDVer 任务上的表现      | Choice → Noul | 60 | 0.100           | 6 / 7 / −1             | [−0.131, 0.087]   |
| Laya 在 ProcessBench 任务上的表现 | 单一 Choice 动作  | 60 | 0.000           | —                      | —                 |
| Vela 在 ProcessBench 任务上的表现 | 单一 Choice 动作  | 60 | 0.000           | —                      | —                 |

### 4.4 E2：增量效用的可预测性

E2 研究路由器能否在执行第二种动作之前，识别它是否值得调用。预测目标是切换到替代动作后是否会修复当前错误，而特征仅使用部署时可以从 Noul 输出获得的信息：`p`、`|p−0.5|` 和由概率计算的熵，不使用 Choice 结果或 gold label。分析比较逻辑回归、浅层决策树、单一置信度阈值以及始终不切换的多数类基线。训练阶段采用按 group 分折的三折 OOF 预测；随后在 selection_dev 上共同选择预测器和工作阈值，并冻结后在 confirm 上评估净效用及额外调用成本。成本调整项与进化适应度一致，额外调用的权重为 λ=0.05。

OOF 排序在部分单元中高于随机水平，但不同模型和任务之间差异明显。表 3 汇总了两种预测器的三折 group-aware OOF AUROC，以及在 dev 上选出的策略在 dev 和 confirm 上的净效用点估计。净效用定义为（修复数−回退数）/样本数；表中的策略先在 dev 选择，之后固定应用于 confirm。

表 3. 增量效用预测的训练排序与冻结策略净效用

| 任务与模型                   | LogReg OOF AUROC | Tree OOF AUROC | Dev 净效用（修复/回退） | Confirm 净效用（修复/回退） | Dev 选中规则                     |
| ----------------------- | ---------------- | -------------- | -------------- | ------------------ | ---------------------------- |
| Laya 在 CodeXGLUE 任务上的表现 | 0.674            | 0.739          | +0.2000（23/11） | +0.0167（17/16）     | LogReg；阈值退化为全量切换（switch-all） |
| Vela 在 CodeXGLUE 任务上的表现 | 0.716            | 0.646          | +0.0333（7/5）   | −0.0167（9/10）      | LogReg                       |
| Laya 在 FinDVer 任务上的表现   | 0.455            | 0.466          | +0.0169（1/0）   | +0.0227（1/0）       | Tree                         |
| Vela 在 FinDVer 任务上的表现   | 0.919            | 0.794          | +0.0339（4/2）   | −0.0227（0/1）       | LogReg                       |

Vela 在 FinDVer 任务上的逻辑回归取得最高 OOF AUROC，但该策略在 confirm 的净效用为负；Laya 在 FinDVer 任务上的效用排序接近随机，dev 和 confirm 的净效用也仅来自极少数修复样本。Laya 在 CodeXGLUE 任务上的 dev 净效用最高，但选中的阈值实际上触发了全量切换，confirm 上的净值降至 +0.0167。Vela 在 CodeXGLUE 任务上的 dev 正向净值也未在 confirm 保留。表中净效用是当前小样本池上的点估计，主要用于比较排序信号与冻结策略的方向和幅度；扩大样本并为净效用差值建立配对区间，是下一轮最直接的统计补充。

这些结果区分了排序指标与策略收益。OOF AUROC 回答模型能否把潜在修复样本排在前面；dev 和 confirm 的净效用则回答冻结阈值后实际切换是否值得。Vela 在 FinDVer 任务上的高 OOF AUROC 并未转化为 confirm 正向净效用，Laya 在 CodeXGLUE 任务上的 dev 收益则来自全量切换，而不是有效区分样本。E2 因此表明，当前 Noul 特征对替代动作增量价值的预测能力尚不稳定。

### 4.5 E3：进化搜索与固定、随机基线比较

E3 在冻结缓存上运行 GA、ES 和 NSGA-II。GA 使用 `CascadeGenome` 搜索二分类任务的 Noul 正类概率区间；ES 搜索连续区间端点并固定启用第二种动作；NSGA-II 对二分类任务优化 accuracy 与调用成本，对数学任务优化选择性准确率与覆盖率；因此其二分类 Pareto 结果与 GA 的 macro-F1 主比较应分开解释。所有主要搜索使用种子 17、23、41 和 24 个唯一基因组评估预算；随机搜索使用匹配预算。固定 Noul-only 与 switch-only 策略提供直接对照。

GA 的 macro-F1 比较。 新增的配对 group-bootstrap 脚本从同一缓存逐样本重建预测，并计算 macro-F1 差值，而不是把逐行准确率差值当作 macro-F1。GA 相较随机搜索的 36 个任务与模型组合 × 数据划分 × 随机种子比较，其置信区间全部包含零。GA 相较固定 switch-only 的唯一显著正向结果出现在 Vela 的 FinDVer 训练池：三个种子上的 macro-F1 差值均为 +0.1202，95% CI 分别为 [0.0043, 0.2271]、[0.0062, 0.2282] 和 [0.0067, 0.2229]。在 dev 和 confirm 上，点估计仍约为 +0.1062 和 +0.1063，但三个种子的置信区间均跨越零。预先指定的部署主比较——train-selected GA 对 fixed switch-only 的 confirm macro-F1 差值——在四个二分类任务—模型组合的全部 12 个种子级单元中均未达到显著。

ES 与 NSGA-II。 ES 在 CodeXGLUE 的两个模型上进行训练池搜索；ES 对随机搜索的六个种子级比较均未显示显著优势。冻结 ES 最优策略后，dev 上 Laya 在 CodeXGLUE 任务上的两个种子级比较显著低于 fixed switch-only，confirm 上没有显著比较。NSGA-II 在 CodeXGLUE 和 ProcessBench 上完成多目标搜索。二分类任务的 Pareto 搜索目标为 accuracy 与负调用成本，ProcessBench 的目标为 selective accuracy 与 coverage；因此，NSGA-II 结果回答的是准确率—成本前沿问题，并非与 GA 的 macro-F1 单目标主比较完全同口径。部署评估使用冻结的训练 Pareto 候选集、dev 上预先规定的点选择规则以及 confirm 报告流程。其 12 个 dev 准确率比较中有 5 个显著为负，12 个 confirm 比较中有 1 个显著为负，没有单元显著支持 NSGA-II 部署策略。ES 和 NSGA-II 的这些部署置信区间以 accuracy 为基础，不能与 GA 的 macro-F1 置信区间混为一谈。表 4 汇总了不同搜索器在各自比较口径下的结果。

表 4. 进化搜索与基线策略的统计比较

| 搜索方法与比较                                        | 比较范围                                                           | 结果                                                        | 主要解释               |
| ---------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------- | ------------------ |
| GA vs. Random（macro-F1）                        | 4 个任务—模型单元 × 3 个划分 × 3 个种子，共 36 个比较                            | 36 个区间均包含 0                                               | 未显示 GA 优于预算匹配随机搜索  |
| GA vs. fixed switch-only（macro-F1）             | 训练池、dev、confirm；每个单元 3 个种子                                     | 唯一显著正向结果为 Vela 在 FinDVer 训练池上的 3 个种子；dev 和 confirm 均未达到显著 | 训练池正向差值未获留出统计确认    |
| GA 主部署比较（macro-F1）                             | train-selected GA vs. fixed switch-only，在 confirm 上共 12 个种子级比较 | 12 个区间均包含 0                                               | 预注册主比较未显示部署收益      |
| ES vs. Random（训练池标量）                           | 两个 CodeXGLUE 模型单元 × 3 个种子，共 6 个比较                              | 6 个比较均未达到显著                                               | ES 未超过随机搜索         |
| ES frozen-best vs. fixed switch-only（accuracy） | dev 与 confirm，共 12 个比较                                         | dev 有 2 个负向显著结果，均来自 Laya 在 CodeXGLUE 任务上的表现；confirm 无显著结果 | 未显示留出正向收益          |
| NSGA-II deploy vs. fixed switch-only（accuracy） | dev 与 confirm，各 12 个比较                                         | dev 有 5 个负向显著结果；confirm 有 1 个负向显著结果；无正向显著结果               | Pareto 部署点未带来准确率优势 |

### 4.6 E4：搜索轨迹与预算利用率

E4 检查搜索是否用尽了分配给它的候选评估预算，以及最优适应度是否随代数改善。在 21 次有日志的 GA/ES 运行中，15 次运行的 best-so-far 适应度从第 0 代开始便未再改善，占 71%。在 Vela 的 CodeXGLUE 预算探测中，GA 的预算上限从 24 提高到 48 后，三个种子实际评估的唯一基因组数分别为 22、17 和 23；随机搜索则使用了全部 48 个候选。GA 与随机搜索的适应度标量差距四舍五入至三位小数为 0。

这组比较说明，当前 GA 配置没有充分利用更高的预算上限，多次运行也在早期便停止取得进展。要进一步判断停滞来自重复候选、适应度噪声还是训练池与留出池之间的排序变化，下一轮搜索需要记录每个子代的适应度、每次变异前后的目标差值，以及每个候选在 dev 上的分数；相应的补充实验列在文末审核计划中。

### 4.7 E5：冻结策略迁移与基于概率不确定性与候选间隔的故障图

迁移实验将源任务 train 上选出的基因组冻结后应用到目标任务的 dev 和 confirm，不在目标任务重新搜索。32 个源—目标 × split 单元中有 6 个达到统计显著，且方向全部不利于迁移策略。对角线单元表示同任务部署检查，单独报告；真正的非对角线迁移单元为 24 个，其中 4 个显著为负，没有显著正向迁移结果。

基于概率不确定性与候选间隔分层的故障图则按任务对应的 Noul 概率不确定性或数学候选间隔划分样本，并汇总各区域的错误严重程度，不执行进化操作，因此它是诊断档案而不是第四种优化器。15 个单元满足最小样本量要求，13 个在 dev 得到确认，10 个在 confirm 得到确认。最稳定的高严重度区域集中在 ProcessBench 的中等 margin 带，说明一部分错误并不是简单由低置信度样本构成；当两个表示在同一类样本上同时失效时，置信度阈值也无法从动作层面创造修复机会。

### 4.8 统计方案与实验复现

二分类任务的主要差值采用 5,000 次 paired group-bootstrap 计算 95% 置信区间。macro-F1 由 `paired_macro_f1_ci.py` 从逐样本预测和标签直接计算；脚本在比较前检查 row ID 顺序和 gold 标签一致。ES 与 NSGA-II 部署分析沿用逐样本正确性差值的 accuracy 置信区间；NSGA-II 的二分类 Pareto 搜索本身也以 accuracy 和调用成本为目标。预先指定的 GA 主部署比较是 confirm 上 GA 对 fixed switch-only 的 macro-F1 差值；其余逐单元比较作为探索性分析报告，并同时给出 effect estimate 和置信区间。

所有搜索结果、派生分析、模型修订号和协议索引均纳入版本控制，并由缓存评估脚本和派生结果 provenance 记录相互关联。最终 locked test 将在模型、问题模板、策略和统计流程完成冻结后单独执行；该结果不参与当前 confirm 分析或策略选择。

## 5. Results

本节不按算法逐项重复数值，而是将实验结果合并起来回答引言提出的研究问题。

RQ1：动作集合是否具有可利用的互补性？ 实验观察到部分表示之间存在逐样本修复机会，但这些机会并未形成稳定的正向净修复效用。Laya 在 CodeXGLUE 任务上的毛 oracle headroom 最大，但 14 次修复与 15 次有害切换几乎抵消；Vela 在 CodeXGLUE 任务上的毛提升空间较小，Laya 和 Vela 在 FinDVer 任务上的毛提升空间也都较小；ProcessBench 在当前动作集合中的 headroom 为零。因而，当前动作组合的主要问题不是第二种表示完全没有不同输出，而是它还没有表现出可以稳定转化为净收益的互补结构。

RQ2：Noul 概率及其派生特征是否能够预测替代动作的增量价值？ 结果显示，训练池中的部分特征可以对修复机会进行排序，但排序指标并没有稳定转化为冻结策略的留出收益。Vela 在 FinDVer 任务上的 OOF AUROC 达到 0.919，但其 confirm 净效用点估计为负；Laya 在 FinDVer 任务上的效用排序接近随机；Laya 在 CodeXGLUE 任务上的 dev 正向结果则由全量切换（switch-all）这一退化策略产生。由此，模型对自身判断的概率与“另一个动作是否更值得执行”的预测之间存在需要单独建模的差异。

RQ3：进化搜索是否把可用信号转化成部署收益？ GA 没有在 macro-F1 下显著超过预算匹配随机搜索，预注册 confirm 主比较也未显示 GA 优于 fixed switch-only。唯一显著的正向 GA 结果发生在 Vela 的 FinDVer 训练池，且 dev 和 confirm 的正向点估计虽然接近，置信区间仍跨越零。ES 与 NSGA-II 的部署结果没有显示进化策略的正向优势，部分 dev 结果甚至显著差于固定策略。搜索轨迹进一步表明，多次运行从初始代便不再改善，扩大预算上限的探测也没有使 GA 用满候选数。

迁移与故障诊断提供了进一步的边界检验。冻结源任务策略后，32 个迁移评估单元中有 6 个显著，方向全部不利于迁移；其中真正的非对角线迁移单元有 24 个，4 个显著为负，没有显著正向迁移。基于 Noul 概率不确定性与数学候选间隔分层的故障图则在 dev 和 confirm 上分别确认了 13 个和 10 个诊断单元，最稳定的高严重度错误集中在 ProcessBench 的中等 margin 区域。它们与前三项实验的解释一致：当前系统的主要困难不是缺少更复杂的搜索器，而是动作互补性与效用信号尚未形成能够跨数据池使用的路由依据。

三个研究问题合在一起，形成本文的核心解释：进化搜索没有建立稳定部署优势，首先是因为当前动作组合未显示统计上可靠的正向净互补性；其次，即使局部存在毛修复机会，现有 Noul 概率特征也不足以稳定预测另一动作的增量效用；搜索早停和预算未用足则进一步解释了当前搜索轨迹，但不是主要的动作层面原因。 这一结论把注意力从“哪种进化算法更强”转向“结构化决策接口提供的概率信息，是否足以支持对替代动作的价值判断”。

## 6. Conclusion

本文从 Jev 所代表的 RLCD / System One 决策范式出发，以开源 RLCD 模型 Laya 和接口相近但训练路线不同的 Vela 作为实验对象，考察结构化概率输出能否支撑推理阶段的动作分配。RLCD 关注模型如何表达当前判断的概率；级联路由则要求系统估计替代动作相较当前动作的增量价值。两者关联紧密，却不是同一个优化目标。本文据此将实验分为三个连续问题：动作能否互相修复、修复机会能否被推理时信号识别，以及进化搜索能否利用这些信号形成稳定收益。

在 CodeXGLUE、FinDVer 和 ProcessBench 上，GA、ES 与 NSGA-II 均未建立相较固定或随机基线的稳定部署优势。动作级结果显示，部分任务确有毛修复空间，但修复与有害切换相抵后，当前动作集合没有呈现统计上可靠的正向净互补性。效用预测实验进一步显示，训练池中较高的排序指标并不自动转化为冻结策略的留出收益；搜索轨迹则记录了多次早期停滞和预算未充分利用。整体证据说明，概率化决策接口是推理时控制的基础，但要让路由真正获益，还需要可互补的动作，以及能够预测动作增量价值的部署信号。

这一发现将后续研究重点从增加搜索复杂度转向检验决策信号本身。对 Jev 类模型而言，重要的问题不只是“模型是否知道自己可能答错”，还包括“模型是否知道另一个动作会不会做得更好”。本文的动作互补性和效用预测分析为这一区分提供了实验化的检验路径，也为之后在更大样本、不同问题模板和更多结构化决策模型上验证路由收益奠定了基础。

## 参考文献

模型与 RLCD

- Deußer, T., Sparrenberg, L., and Sifa, R. (2026). Evaluating and Benchmarking the System One Model Jev. arXiv:2609.37647. [https://arxiv.org/abs/2609.37647](https://arxiv.org/abs/2609.37647?utm_source=chatgpt.com)
- Guo, R., Liu, Y., Deng, G., Li, Y., Zhao, L., Wu, Y., Chen, S., Zhang, Y., and Zhang, L. Y. (2026). Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures. arXiv:2609.29429. [https://arxiv.org/abs/2609.29429](https://arxiv.org/abs/2609.29429?utm_source=chatgpt.com)
- Gao, Z. and Wang, P. (2026). OpenJev-RLCD: A Working RLCD Implementation. arXiv:2609.38850. [https://arxiv.org/abs/2609.38850](https://arxiv.org/abs/2609.38850?utm_source=chatgpt.com)
- TypeSafe AI. (2026). Introducing System One Models & Jev. [https://typesafe.ai/blog/introducing-system-one-models-and-jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev?utm_source=chatgpt.com)
- Laya Project. (2026). Laya: Non-autoregressive System 1 Decision Engine, version 0.4.16; model and training documentation. [https://github.com/NandhaKishorM/laya/tree/b9fe351ce81e2d974bca2bf219c1e4030c0a05ea](https://github.com/NandhaKishorM/laya/tree/b9fe351ce81e2d974bca2bf219c1e4030c0a05ea?utm_source=chatgpt.com)
- Laya Project. (2026). Questions and Answers: Choice, Noul, Score, and Confidence Semantics. [https://github.com/NandhaKishorM/laya/blob/b9fe351ce81e2d974bca2bf219c1e4030c0a05ea/docs/questions-and-answers.md](https://github.com/NandhaKishorM/laya/blob/b9fe351ce81e2d974bca2bf219c1e4030c0a05ea/docs/questions-and-answers.md?utm_source=chatgpt.com)
- Laya Studio. (2026). Proper Scoring Rules: How to Reward Honest Probabilities. [https://laya.studio/learn/proper-scoring-rules](https://laya.studio/learn/proper-scoring-rules?utm_source=chatgpt.com)
- vllm-sr. (2026). Vela 2.0 0.3B: Model Card. [https://huggingface.co/vllm-sr/Vela-2.0-0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B?utm_source=chatgpt.com)
- vllm-sr. (2026). Vela 2.0 0.3B: Training and Model Details. [https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/main/TRAINING.md](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/main/TRAINING.md?utm_source=chatgpt.com)
- Yang, K., Klein, D., Çelikyılmaz, A., Peng, N., and Tian, Y. (2023). RLCD: Reinforcement Learning from Contrast Distillation for Language Model Alignment. arXiv:2307.12950. [https://arxiv.org/abs/2307.12950](https://arxiv.org/abs/2307.12950?utm_source=chatgpt.com)

级联、路由与自适应计算

- Aggarwal, P. et al. (2024). AutoMix: Automatically Mixing Language Models. NeurIPS 2024. [https://proceedings.neurips.cc/paper_files/paper/2024/hash/ecda225cb187b40ea8edc1f46b03ffda-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ecda225cb187b40ea8edc1f46b03ffda-Abstract-Conference.html?utm_source=chatgpt.com)
- Chen, L., Zaharia, M., and Zou, J. (2024). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Transactions on Machine Learning Research. [https://mlanthology.org/tmlr/2024/chen2024tmlr-frugalgpt/](https://mlanthology.org/tmlr/2024/chen2024tmlr-frugalgpt/?utm_source=chatgpt.com)
- Damani, M., Shenfeld, I., Peng, A., Bobu, A., and Andreas, J. (2025). Learning How Hard to Think: Input-Adaptive Allocation of LM Computation. ICLR 2025. [https://proceedings.iclr.cc/paper_files/paper/2025/file/ff414825df833edb8b1839e3d5d495e9-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/ff414825df833edb8b1839e3d5d495e9-Paper-Conference.pdf?utm_source=chatgpt.com)
- Dekoninck, J., Baader, M., and Vechev, M. (2025). A Unified Approach to Routing and Cascading for LLMs. ICML 2025, PMLR 267, 12987–13010. [https://proceedings.mlr.press/v267/dekoninck25a.html](https://proceedings.mlr.press/v267/dekoninck25a.html?utm_source=chatgpt.com)
- Ding, D. et al. (2025). BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute. ICML 2025, PMLR 267, 13870–13884. [https://proceedings.mlr.press/v267/ding25d.html](https://proceedings.mlr.press/v267/ding25d.html?utm_source=chatgpt.com)
- Jitkrittum, W., Gupta, N., Menon, A. K., Narasimhan, H., Rawat, A. S., and Kumar, S. (2023). When Does Confidence-Based Cascade Deferral Suffice? NeurIPS 2023. [https://papers.neurips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html](https://papers.neurips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html?utm_source=chatgpt.com)

进化优化

- Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197. [https://doi.org/10.1109/4235.996017](https://doi.org/10.1109/4235.996017?utm_source=chatgpt.com)
- Fernando, C., Banarse, D. S., Michalewski, H., Osindero, S., and Rocktäschel, T. (2024). Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution. ICML 2024. [https://proceedings.mlr.press/v235/fernando24a.html](https://proceedings.mlr.press/v235/fernando24a.html?utm_source=chatgpt.com)
- Guo, Q. et al. (2024). Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers. ICLR 2024. [https://proceedings.iclr.cc/paper_files/paper/2024/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html?utm_source=chatgpt.com)

任务基准

- Lu, S. et al. (2021). CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation. NeurIPS 2021 Datasets and Benchmarks Track. [https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round1.html](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round1.html?utm_source=chatgpt.com)
- Zhao, Y. et al. (2024). FinDVer: Explainable Claim Verification over Long and Hybrid-content Financial Documents. EMNLP 2024, 14739–14752. [https://aclanthology.org/2024.emnlp-main.818/](https://aclanthology.org/2024.emnlp-main.818/?utm_source=chatgpt.com)
- Zheng, C. et al. (2025). ProcessBench: Identifying Process Errors in Mathematical Reasoning. ACL 2025, 1009–1024. [https://aclanthology.org/2025.acl-long.50/](https://aclanthology.org/2025.acl-long.50/?utm_source=chatgpt.com)

# 作者审核用：建议补充的实验及其研究意义

> 以下是下一轮实验计划，不属于上面的六部分投稿正文。它们不是为了增加实验数量，而是针对目前证据链中会影响主结论的关键问题提出可执行的验证。

## P0：核查 Noul 提问模板与模型语义

为什么要做。 当前 `src/tasks/code.py` 和 `src/tasks/findver.py` 中的 `NOUL_Q` 只包含 `type` 与 `instructions`，没有显式指定 `labels`。Laya 上游文档指出，Noul 的二元标签与问题措辞会影响概率解释；因此需要在项目实际固定的 Laya 0.4.16 检查点上验证模型是否持续依据输入 state 判断，而不是主要响应问题中的词语或默认标签。这个审查也应在 Vela 上执行，以确认现有 Noul/Choice 对照在两个模型中的语义一致。它直接关系到级联策略的输入 \\(p(x)\\)，也是重新解释 H1 和 H2 之前必须完成的模型接口验证。

实验设计。 在 Laya 0.4.16 和 Vela 2.0 0.3B 上，对 CodeXGLUE 与 FinDVer 选取按 group 分层的同一批样本，配对比较：当前仅含 instructions 的 Noul；在 Noul 中显式说明 true/false 语义的版本；在模型接口确认支持时，通过 `labels` 字段使用语义清晰但不诱导答案的替代标签版本；以及含相同正反类含义的二选一 Choice 对照。保持 state 和 gold label 不变，分别改变问题措辞与标签呈现，并记录实际送入模型的渲染文本。

报告指标。 每种模板报告 P(true) 分布、均值与方差、正类预测比例、AUROC、macro-F1、Brier、NLL、ECE、重复调用一致性和错误类别；通过 paired group-bootstrap 比较模板差值，并核对模型修订号、SDK 版本和实际渲染文本。

结果如何改变论文。 若问题模板明显改变同一批样本的输出分布或标签判断，则应采用语义通过审查的模板重建 Noul 缓存，并重算动作互补性、效用预测和 GA 结果；论文的主要解释将相应转向“动作定义和问题呈现影响了路由输入”。若不同模板在两个模型上的判断和排序均稳定，则可以把模型接口语义作为已验证的实验基础，继续解释动作结构与效用预测。

## P0：测量任务内概率校准，并区分校准与增量效用预测

为什么要做。 RLCD 的训练目标是让概率报告诚实，但模型在特定任务、特定问题模板上的概率仍需要任务内验证。并且，即使 `P(true)` 对当前判断校准良好，它也不一定能预测切换到 Choice 后的增量收益。该实验能把“模型概率不准”和“概率准确但不代表替代动作收益”区分开来。

实验设计。 对通过 P0 模板审查的 Noul 概率，在 CodeXGLUE 与 FinDVer 的每个模型上计算 reliability diagram、ECE、Brier score、NLL 和 calibration slope/intercept。用独立 calibration fold 拟合 temperature，不在 selection_dev 或 confirm 上拟合；之后分别用 raw probability 与 calibrated probability 训练/选择同一类阈值策略，并在 confirm 比较。另将当前动作 correctness prediction 与 switch incremental gain prediction 作为两个独立目标评估。

结果如何改变论文。 如果 temperature scaling 改善当前动作的正确性校准，却没有改善增量效用预测，论文就能以直接证据说明“概率校准”与“路由价值预测”并非同一能力；如果校准后路由效用明显改善，H2 的主解释则需要改写为原始概率的校准不足，而非动作价值本身难以预测。

## P0：在留出数据和 macro-F1 主指标下复核动作互补性

为什么要做。 当前 E1 的 repair、harmful switch 和 oracle headroom 来自 search_train，并以 accuracy 定义逐样本正确与错误；但 CodeXGLUE 和 FinDVer 的任务主指标是 macro-F1。H1 是整篇论文的首要解释，因此需要在固定动作选择与主指标两方面补齐证据：一方面，确认训练池观察到的修复—回退关系能否在 dev/confirm 重现；另一方面，直接比较两种固定动作及逐样本 oracle 在 macro-F1 上的表现。

实验设计。 先沿用 search_train 上确定的较优固定动作方向，不在 dev 或 confirm 重新选择基准；将这一方向冻结后，在 selection_dev 和 confirm 缓存中分别计算 repair、harmful switch、net repair、disagreement 与 accuracy headroom，并使用 group-bootstrap 给出区间。对同一批预测再计算 Noul-only 与 Choice-only 的 macro-F1 配对差值；逐样本 oracle 继续只按逐样本正确性构造准确率上界，不将其解释为 macro-F1 上界，也不作为可部署策略。ProcessBench 因只有一个已评估的动作表示，单独报告其 headroom=0，不混入二分类动作比较。

结果如何改变论文。 若 dev 和 confirm 的净 repair 仍无稳定正值，且 macro-F1 的固定动作差异与 oracle headroom 也没有显示可实现的正向空间，H1 将获得跨数据池、主指标一致的支持；若留出池出现持续净修复或 macro-F1 headroom，则需要收窄 H1 的适用范围，并把相应任务—模型单元转入 E2，检验其收益能否被部署信号捕获。

## P0：为增量效用和路由净收益补充 group-bootstrap 区间

为什么要做。 E2 已经把训练池排序、dev 选策略和 confirm 冻结评估分开，但表中的净效用目前是点估计。尤其在 FinDVer 的 confirm 池中，修复和回退计数很少，单个样本就可能改变净值方向。若要用 E2 解释 H2，就需要同时呈现效应大小与不确定性，而不只比较几个点估计。

实验设计。 直接读取现有逐样本缓存和冻结的 dev-selected 路由规则，不重新训练预测器、不重新选择阈值。按 group 对 confirm 样本进行配对重采样，在每个 bootstrap 样本中计算策略相对 Noul-only 的 repair、regression、net repair 和 cost-adjusted net utility；同时对固定 Choice、固定 Noul 及已冻结的路由策略报告一致口径的区间。对 dev 结果使用相同方法，但标记为策略选择数据，不把它当作独立确认结果。

结果如何改变论文。 如果 confirm 区间跨零，H2 应表述为“未获得可确认的净收益”，并报告当前样本允许的效应范围；如果区间稳定为正，说明某些任务—模型组合已经存在可部署的增量价值，需要修订“效用预测不稳定”的主结论，并分析它是否来自单一任务或模板。

## P1：扩大 group 数量并补齐预注册随机种子

为什么要做。 当前 CodeXGLUE 和 FinDVer 的训练池通常约为 60 行，FinDVer 的 confirm 约为 44 行；不同单元的增量修复正例数量较少，特别是 Vela 在 FinDVer 的 confirm 池只有 2 个修复正例。GA、ES 和 NSGA-II 的主要结果使用了 3 个种子，而 v4.1 配置还列出种子 59 和 73。扩大独立 group 数量并补齐预注册种子，可以提高效应估计精度，并检验当前搜索结果是否受少数样本和随机初始化影响。

实验设计。 这项补充分成两个独立步骤，避免把增加随机种子与扩大样本混为一谈。第一步只使用现有缓存，补跑预注册的种子 59、73，不产生新的模型调用，用于检验当前搜索结论对随机初始化的敏感性。第二步才考虑扩大独立样本组：在不触碰 locked test 的前提下，从允许使用的数据范围中增加独立函数组、报告组和问题组，按 group 重新生成 search_train、selection_dev 和 confirm；这一步需要为新增样本构建真实模型输出缓存，因此会产生额外模型调用，应在核查 P0 结果、估算成本并确认预算后执行。两步均保留同一模型版本、问题模板、动作和主指标；所有新旧数据池之间的 row/group overlap 必须在运行前检查。

结果如何改变论文。 若效果估计在扩大样本后收敛到接近零，负结果的统计说服力将提高；若原先宽置信区间收窄后出现稳定正向效应，则需要回到 H1/H2 重新分析其任务和模型适用范围。增加种子可以显示当前结果是否受少数随机初始化驱动。

## P1：测量真实推理延迟，验证调用次数是否能代表成本

为什么要做。 当前进化适应度中的成本主要由每行模型调用数表示，而 Laya 与 Vela 的单次推理延迟并不相同；调用次数是方便且可复现的成本代理，但不必然等价于实际 CPU 延迟。若论文讨论质量—成本权衡，应确认这一代理是否能保持策略之间的成本排序。

实验设计。 对 Noul-only、switch-only、GA、随机搜索、ES 和 NSGA-II 选出的冻结策略，使用同一组输入和 CPU 环境，进行 warm-up 后随机交错重复执行；报告每个策略的实际调用数、总运行时间、mean/median、p50/p95 latency、RSS 和错误数。Laya 与 Vela 分开比较，不用跨模型的单次延迟直接推断策略优劣。

结果如何改变论文。 若调用次数与实测延迟排序一致，当前成本指标可得到经验支持；若二者明显不一致，则重新定义成本项或分别报告调用成本与 wall-clock latency，避免把调用数直接解释为推理时延。

## P1：补全搜索动态诊断（基于现有缓存，不新增模型调用）

为什么要做。 当前搜索日志显示 21 次 GA/ES 运行中有 15 次从第 0 代起不再改善，预算 48 的 GA 又只评估了 17–23 个唯一基因组。这些现象足以说明搜索过程存在早停和预算利用不足，但还不能区分重复候选、变异无效和 train-dev 排序变化分别贡献了多少。

实验设计。 使用现有 GA/ES archive 和固定缓存，对所有已保存候选补算 dev 适应度，计算 train-dev Spearman 排名相关；从已保存基因组生成一代单基因邻域变异，在 train cache 上测量中性变异比例与 fitness delta；对随机搜索重新记录候选序列和 best-so-far 轨迹；对 NSGA-II 保存每代 Pareto front、hypervolume 和 front size。所有适应度计算均从现有缓存读取，不新增模型推理。

报告指标。 train-dev rank correlation、neutral mutation rate、parent-child fitness correlation、duplicate rate、每代唯一候选数、best-so-far 轨迹及 NSGA-II hypervolume trajectory。

结果如何改变论文。 若重复率和中性变异占主导，H3 应解释为候选生成和局部搜索效率问题；若 train-dev 排名相关低，则搜索池内优化与留出表现之间的差距需要作为主要原因讨论；若各指标均显示搜索能找到稳定的 train 改善而留出无收益，则结果会进一步支持 H1/H2 是比优化器效率更前置的约束。

## P1：将 NSGA-II 的 Pareto 目标与任务主指标对齐

为什么要做。 当前 GA 的预注册主比较使用 macro-F1，而 NSGA-II 的二分类 Pareto 搜索目标仍为 accuracy 与调用成本，部署点选择也按 dev accuracy 进行。两种算法的结果因而回答了相关但不完全相同的优化问题。补齐这一实验可以判断 NSGA-II 的负向或无优势结果是否依赖于目标指标，而不是进化搜索本身。

实验设计。 在现有二分类缓存上重跑同一 NSGA-II 实现，将质量目标从 accuracy 替换为 macro-F1，调用成本目标保持不变；使用相同的预算、种子和缓存，冻结 train Pareto 候选集，在 dev 上按预先声明的 macro-F1 / cost 规则选点，再在 confirm 上计算 paired group-bootstrap macro-F1 CI。原 accuracy Pareto 结果保留为对照，不删除旧结果。

结果如何改变论文。 若 macro-F1 对齐后仍无部署优势，NSGA-II 与 GA 的负结果便可在任务主指标上直接比较；若 NSGA-II 的结论改变，则应将算法差异解释为目标函数敏感性，并把当前 accuracy-Pareto 结果作为独立的成本—准确率分析。

## P1：在 Jev 上做同任务外部复现（可选，不作为当前论文启动门槛）

为什么要做。 当前实验覆盖了开源 Laya 和 Vela，但没有直接运行 Jev。若能在相同任务、相同分组样本和相同问题 schema 上评估 Jev，可以判断动作互补性与路由失败是否仅出现在当前两个开源模型，还是也会出现在 Jev 这一 RLCD 代表模型上。

实验设计。 若能获得稳定的 Jev API 和预算，固定 `jev-1.13.0` 或研究开始时可用的精确版本；在独立预留的 group 数据上运行同一 Noul/Choice 问题，并记录 API 版本、延迟、token 和费用。先复现单动作表现与概率校准，再计算 repair/harmful、增量效用和固定阈值路由结果。Jev 不参与现有 GA 参数选择，作为独立模型复现实验。

结果如何改变论文。 Jev 若出现相似的动作互补性或效用预测限制，将支持更一般的结构化决策路由解释；若 Jev 明显不同，则论文应将结论定位为 Laya/Vela 及其当前动作实现的实证研究，而不是外推到整个 RLCD 或 System One 模型家族。

## 最终一次性锁定测试

完成 P0 模板与校准审查、确定最终模型和策略版本、冻结统计脚本及主比较后，再依照协议运行 locked test 一次。此步骤不应用于选择提示模板、重新设定阈值或更换搜索参数；它的目的只是检验冻结方案在最终保留数据上的表现。若此前 P0 审查要求重建缓存，则应先完成全部修订和 confirm 验证，再进行最终冻结。

以上是 v0.4 的完整正文和补充实验计划。

审核时我建议你特别留意三处：

1. 第 2 节模型背景： Jev、Laya、Vela 的模型关系、参数规模、训练方法及 RLCD 定义是否准确。
2. 第 4.3–4.5 节： E1 的 accuracy-based repair 与 macro-F1 主指标之间是否需要重新组织，E2 的点估计是否足以支撑目前的表述，以及 GA/ES/NSGA-II 的比较是否清晰。
3. 文末 P0 实验： Noul 模板审查、概率校准和留出集互补性复核是我认为最优先的三项。它们可能改变论文的核心解释，因此应先于扩大算法搜索规模。 这是论文初稿和需要补充的实验部分，此外你需要注意，这个初稿并不好，不讲人话，可读性差，防御性表述限定等等问题让它无法获得可读性，但是整体架构可以，此外，放弃全部小标题，即不要有二级标题 如2.1 2.2等。请你先完成1.实验的补充请完成 实验的补充可以让我们的结论更加完善 2.论文初稿的重写 根据补充后的实验结果再进行完善，还有整体的文风都需要更改避免使用机械化的连接词 (如 “首先”“其次”“然后”)，改用更具连贯性的自然过渡；通过使用多样化句式，混合简单句、复合句和插入语使表达更富有层次感，同时避免连续短句或过于整齐的句式。在叙述数据或结论时，补充背景信息或个人研究观察，使内容更加具体并贴近实际研究场景，并通过问题引导或总结的方式实现段落之间的自然过渡，避免生硬切换，从而提升整体阅读的流畅性和逻辑性。  并按照1. 用审稿人思维检查

请你用ACL审稿人的标准，批判性检查这段文字。不要只夸我，要指出逻辑漏洞、概念不清、文献支撑不足和可能被审稿人质疑的地方。

2. 用硕导思维检查

请你站在硕导的角度检查这段论文写作，重点看：研究问题是否清楚、理论逻辑是否顺、变量关系是否合理、语言是否像正式论文。

3. 检查段落中心句

请检查每一段的第一句话是否起到了提纲挈领的作用。每段第一句要清晰、言简意赅、易懂，最好能直接概括本段中心思想。不要使用“这一问题”“这种情况”等模糊指代。

4. 检查总分结构

请按照“总—分”结构检查这段文字。每一段先提出中心观点，再展开解释，不要先堆文献、堆背景，最后才让读者猜我要说什么。

5. 检查研究缺口

请检查这段研究缺口是否成立。重点看：已有研究写得是否准确？“但/然而”之后的转折是否和前文对应？本文缺口是否真的没有被已有研究解决？

6. 检查“但/然而”的逻辑

请专门检查文中的“但”“然而”“尽管如此”等转折词。转折前后必须形成清楚的对照关系，不能为了转折而转折。

7. 检查概念定义

请检查这个概念定义是否过窄、过宽或过度情境化。先判断这个概念本身的一般含义，再判断它在本文研究情境中的具体表现。

8. 检查概念和变量是否对应

请检查这段话中的概念、变量和因变量是否对应。不要把文章引向无关结果变量，比如我的因变量是A，就不要过度写B。

9. 检查文献和句子是否匹配

请检查每一句引用的文献是否真的能支撑这句话。不要为了增加文献而硬塞，也不要把文献的结论扩大化。

10. 检查文献适配性

请判断这篇文献是否真的适合引用在我的论文中。请从研究对象、理论视角、变量关系和研究情境四个方面判断，不要只因为关键词相似就建议引用。

11. 检查经典文献是否缺失

请根据我的研究主题，检查我是否遗漏了权威、经典、相关或最新的SSCI/SCI期刊论文。请同时提供DOI，方便我用Zotero抓取。

12. 检查中文表达是否自然

请把这段文字改得更符合中文论文表达，不要有翻译腔。尤其避免“并不必然”“并非在真空中”“依赖方式”等生硬表达。

13. 检查是否提前剧透研究发现

请检查引言中是否提前写了太多后面访谈或实验才会展开的细节。引言只需要提出问题，不要把研究发现提前讲完。

14. 检查是否偏离主线

请检查这段文字是否偏离本文主线。我的主线是：（列出你研究的关键词5个）凡是不能服务这条主线的内容，请建议删减。

15. 检查引言是否“漏斗型但不空泛”

请检查引言是否从相关背景逐步聚焦到具体研究问题。不要从宏大空泛的行业背景开始，要尽快进入本文核心问题。

16. 检查贡献是否足够具体

请检查本文贡献是否具体、可信、有文献推进感。不要写成“丰富了理论、提供了启示”这种空话，要说明具体推进了哪一类研究、推进在哪里。

17. 检查是否给自己挖坑

请检查这段话是否提出了本文没有实证检验的机制、中介或变量。如果没有测量，就不要写得像本文已经检验了。

18. 检查语言是否太绝对

请检查文中是否存在“必然”“本质差异”“不可替代”“完全不同”等过于绝对的表述。请改成更稳妥、更适合学术论文的表达。

19. 检查研究问题是否清楚

请检查我的RQ是否清楚、主次分明，并且和后文研究设计、假设和结果对应。不要让RQ看起来像随便罗列变量。

20. 检查段落之间是否顺
请检查段落之间的衔接是否自然。每一段都要回答：为什么上一段之后必须写这一段？这一段如何把读者推向下一段？ 标准进行审核 3.论文的图表补充，需要补充大量图表 包括数据图、数据表格登等 整体架构图需要请你尝试使用你可以用的opencode里面可以提供的生成图片的模型进行尝试 不行的话生成提示词给我 4.论文初稿排版按照acl的标准写好并自动检查https://github.com/acl-org/aclpubcheck，同时给我一份中文版的供我进行审核 同时需要以下subagents进行审核和辅助，每次代码改动都需要subagent for code-review进行独立审核 每次代码运行后都需要subagent for review进行结果分析，避免出现低级错误或者泄露 论文按照大章节进行逐步完成，每次写作完成都需要 subagent for write进行审核，审核内容、语句和文风，每两段衔接都需要subagent进行连贯性审核，全文写完以及涉及到全文修改的都需要subagent for全文进行故事线的审核 同时还需要subagent对文献进行websearch和对应，在合适的位置加入合适的论文，包括arxiv和scispace等，整体需要40篇参考文献 最后需要subagent for 图片和subagent for 表格的审查，对应图片的字号大小还有排版需要清晰 

