# EvoTIDE v3 协议：Laya + Vela 0.3B 多分支进化研究总设计（冻结版）

- 版本：v3（2026-10-31）。文件性质：研究方案、协议草案与决策树；不是新实验结果。
- 审计基线：`d45f14a1a5880ff6448701479e01ac2a5a8dc7dc`（v2 历史保持不动）。
- 硬件边界：Windows 本地 CPU、最多 2 线程、进程 RSS 目标 ≤3.5 GB，内存预算约 4GB。
- 模型边界：Laya 0.4.16（`b9fe351`，checkpoint `7b928d8`）；Vela-2.0-0.3B（revision `d6f03aa`）。
- 关键原则：所有新研究分支都实际包含一种进化算法。若正向方法假设失败，则切换为进化景观/失败机制研究；绝不把尚未执行的算法写成已完成。

## 0. 总判断

v2 把进化搜索放在了过窄、低敏感度的对象上（max_len/head_max_len 旋钮：h96 零翻转；max384 有害翻转；未复现）。旧 BANKING77 NEAT 路由（3-shot/class，置信度/margin/熵特征）没能稳定优于固定策略。候选菜单、证据片段选择、任务 schema、上下文视图、策略序列是离散/组合结构，与只调整两个数值预算不一样。Laya 与 Vela 不共用虚假内部参数空间。

主问题：在模型权重完全冻结的条件下，能否通过进化计算优化"输入组织、候选选择、证据分配、策略组合"，使质量—成本—鲁棒性 Pareto 前沿得到改善？子问题 RQ1（可进化对象）/ RQ2（搜索算法附加价值）/ RQ3（泛化）/ RQ4（失败机制）。

## 1. 五个进化分支（各有独立 genome / fitness / 失败出口）

| 分支 | 算法 | 进化对象 | 适用模型 |
|---|---|---|---|
| E1 EvoConfig | NSGA-II + DE/CMA-ES 对照 | 原生合法参数 + schema 组合 | Laya、Vela（各自合法参数） |
| E2 EvoMenu（首选方法） | Grammar-Guided GP + NSGA-II | 候选标签检索与菜单生成程序 | Laya、Vela |
| E3 EvoEvidence | Cartesian GP / GP + MAP-Elites | 证据段落/推理步骤/代码片段选择 | Laya、Vela |
| E4 EvoPolicy | NEAT / 进化有限状态机 | 逐样本 STOP/升级/切换/继续 | Laya、Vela 各自策略库 |
| E5 EvoFaultMap（强制失败分析） | NSGA-II + MAP-Elites | 高风险配置/输入变体，失败景观 | Laya、Vela |

E5 与 E1–E4 并行作为诊断侧线：有自己的进化目标，不是"失败后用文字解释"。
优先级：E0 审计与 native baseline → E1 小 DOE → E2 → E3 → E4 → E5（如前面失败，提前转正）。

## 2. 模型与数据 pin（冻结）

- Laya：包 0.4.16，源码 `b9fe351`，`convaiinnovations/laya` rev `7b928d8`，English checkpoint。
- Vela：`vllm-sr/Vela-2.0-0.3B` rev `d6f03aa`，Torch FP32（ONNX 仅 parity/cost 消融）。
- 数据集：ProcessBench（`3bdcd53`，官方 run_eval 映射），CodeXGLUE（官方 21854/2732/2732，60 重复函数已审计），FinDVer（testmini 700 / test 1700，155 报告跨 split），BANKING77（`18072d8`），CLINC150（待准备）。
- MASSIVE 不作为 Vela 确认集（训练来源含 MASSIVE）。Vela 官方 safety/PII/hallucination 指标因 test-informed selection 不得当零样本证据。
- FinDVer Choice parser 已知 bug（`not_supported` 下划线）：修复 + 重跑后才能做 Choice 结论。

## 3. 搜索空间（合法范围，冻结）

- Laya：head_max_len {64,96,128,160,192}（≤192 默认上限）；max_len {128,256,384,512}（>512 为 OOD stress，单独标记）；tournament group_size {4,8,16,（32 视候选）}；option order canonical/reverse/2 固定随机排列；shortlist k {5,10,20,40,80,full}（需 recall@k）；long-window 仅长文本（aggregate="auto" 固定）；min_confidence 为独立 coverage 实验。
- Vela：无 per-call max_len/head_max_len/group_size（禁止虚构）；可搜：Noul semantic criteria、Noul threshold（预注册）、Choice descriptions、typed parts（user/context/answer）、candidate K/order（API 确认后）、target_part、外部证据选择；backend 仅 ONNX parity/cost 消融。
- E2 genome：词/字符/BM25/描述/prototype/先验/冗余惩罚模块 + TopK；NSGA-II 目标 = 下游 macro-F1、Recall@K（硬底线）、延迟/翻转率。必须报告总体/宏平均/长尾 Recall@K 与 retrieval miss。
- E3 genome：段落-句子-section 选择、词面/BM25/实体/日期/数字匹配、去冗余、TopK/knapsack、多样性、排序、预算基因（K∈{1,2,4,8,16}，证据预算 {256,512,1024,2048}）；FinDVer 必须保留 claim；ProcessBench 保留 problem+sentinel；CodeXGLUE 保留签名与控制流。
- 禁止：修改模型权重/config/remote code；把 prompt ID 编码成连续值喂 CMA-ES/DE；用 test 调参/选型；超默认预算不标记 OOD；shortlist+order 不兼容组合。

## 4. Fitness 与公平性（冻结）

- 主目标：max Q，min C（相对 native-single 归一化），min 翻转率；NSGA-II 输出 Pareto。λ/μ 只在 train/dev 拟合。
- 预算：每算法每 seed 相同 unique config/fitness 查询数（起点 400–900）；seeds 17/23/41；重复配置去重不计但报告。
- 成本：C_total = 输入构造 + 检索 + 模型 + 控制器 + 窗口/批处理；如实分开搜索耗时与部署推理成本。
- 统计：grouped 5,000-rep paired cluster bootstrap（seed 17）；macro-F1/balanced accuracy/AUPRC；跨 seed 稳定性；oracle 仅诊断上界。
- 配置缓存：(model, task, split, row, config_hash, backend) 键；surrogate 必须经 full-menu vs restricted-menu vs cached 三方验证后才可用，否则用有限 menu bank + 高保真复核 top 候选。

## 5. 门控（任一失败即停该分支）

- A0：审计修复全过（本文件 §7 清单），否则不做新推理。
- A1：native 矩阵 + 传统基线（majority/random/TF-IDF/检索器），同一 row IDs。
- A2：表示消融；schema bug 必须先修。
- A3：OFAT/交互筛选；无可重复非纯噪声变化 → 只许有限诊断性搜索。
- A4：config bank + 穷举参照，报告 selection regret。
- A5：同预算 EA vs Random；top-config 必须 dev 复核。
- A6：transfer（任务内→任务间→跨模型），confirm→locked test 各一次。
- E4 NEAT 仅当动作互补 + 增量价值信号成立；E5 无论正向成败都执行。

## 6. 论文出口（按证据选择，不预设）

- S1 方法创新：GP/NSGA-II 在同预算下胜 Random + 手工，且 confirm 保持 → EvoTIDE 方法论文。
- S2 简单方法打平：输入优化有效但复杂进化不必要 → 算法分析论文。
- S3 单模型/单域有效 → 模型条件方法论文（必须同时报告失败侧）。
- S4 仅特定任务有效 → 适用边界论文。
- S5 全失败 → E5 失败景观论文（NSGA-II/MAP-Elites 主动搜索失败区域：位置坍缩、schema collapse、上下文丢失、高置信错误、近义竞争、长尾遗漏）。
- S6 模型本身无能力 → 任务-模型失配结论，停止部署优化。

## 7. A0 审计修复清单（本轮必须完成）

1. FinDVer Choice parser：比较归一化 raw_choice 与 supported/not_supported；保存 raw + normalized；单测下划线/空格/大小写/未知键；两个模型重跑。
2. CodeXGLUE：仅用官方 train 行索引做 group 切分；修正 index-vs-group 类型错配；跨 split 零重叠测试。
3. FinDVer 按 report 聚类 bootstrap（已在 group_oof 实现原型；A0 验收）。
4. paired join：唯一 row id、双边集合相等、state/schema/gold hash 相等；test 标签不得进入搜索缓存（加测试）。
5. 数据 manifest：记录所有文件 revision、行数、组数、标签比、重叠元数据。
6. 纠正文档测试计数与本地测试日志；不得声称 GitHub CI 验证过本地运行（无 CI workflow）。
7. max_len=1024/head_max_len=512 标为 OOD stress，不与原生范围混排。
8. 冻结修订、种子、预算（见 results/v2/audit/protocol_freeze.json，A0 验收时更新本轮新增项）。
