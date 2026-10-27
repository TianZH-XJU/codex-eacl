# NEAT 驱动的 Laya 多策略自适应推理与预算控制（研究设计稿 v2026-10-27）

> 版本：2026 年 10 月 27 日｜研究设计稿｜不代表本计划中的新实验已经运行。
> 暂定英文标题：Beyond One-Pass Decisions: Budget-Aware Neuroevolution for Multi-Strategy Inference
> 执行约束：所有模型推理、数据处理和实验只在本地 Windows PC 执行；GitHub 只做版本控制与备份。
> 注：原文中的 mermaid 样式代码已略去，保留流程语义；完整讨论以本文档为准。

## 核心思想

Laya 先执行一次低成本 SINGLE 初始推理。之后 NEAT 不仅决定要不要升级，还要决定升级到什么策略、
是否继续追加预算、是否转用另一种策略，以及什么时候停止。每一次策略动作结束后，都重新评估
当前答案、策略间的分歧和剩余预算。

不再局限于 SINGLE/WIDE/TOURNAMENT，而是建立可逐步扩展、经实证验证的策略库
（Shortlist、选项顺序聚合、长文本分窗、任务表示切换等）。不同任务用不同的有效动作集合。

## 一、研究问题、目标与贡献

冻结 Laya 下，NEAT 能否根据当前预测、策略历史和剩余预算，逐轮选择是否停止、下一步执行哪种
策略、是否继续追加计算，从而在相同总计算预算下改善准确率—成本权衡？子问题：

1. 不同 Laya 策略是否犯不同的错误（互补性）？
2. SINGLE 结果及廉价特征能否预测哪些样本值得升级？
3. 第一次升级后，NEAT 能否利用新结果决定第二步策略？
4. NEAT 拓扑进化是否真优于阈值/LR/浅层树/固定拓扑网络？

方法名（暂定）：BLSC（Budgeted Laya Strategy Controller）。Laya 冻结；NEAT 在离线缓存和训练
数据上进化；预算管理器屏蔽不适用/超预算/重复动作并统计累计成本。

## 二、核心架构：每次策略动作后重新决策

```text
输入样本 + 任务适配器
  → Laya SINGLE 初始推理（保存预测、概率、特征和成本）
  → NEAT 第 1 次决策：STOP 还是升级？
  → 执行一个适用且预算可行的策略 → 更新结果历史和累计成本
  → 重新计算特征和剩余预算 → NEAT 第 2 次决策
  → 停止条件触发？选择下一策略或更高预算级别
  → 记录最终答案、成本和所选路径（STOP / 升级 / 是否继续）
```

允许路径示例：SINGLE→STOP；SINGLE→WIDE→STOP；SINGLE→SHORTLIST→TOURNAMENT→STOP；
SINGLE→WIDE→OPTION-ORDER→STOP；SINGLE→TOURNAMENT→WIDE→STOP。
NEAT 学习的是当前状态下下一步动作的选择，不是固定顺序。

状态 s_t：初始/当前预测、各次概率/置信度/margin/熵、策略一致性、答案是否改变、
候选数量、输入长度、截断、窗口数、已执行动作、累计成本、剩余预算/动作次数。

停止条件（按顺序执行）：任务适用性 → 预算约束（P90/P95 保守估计超预算则移除动作）→
重复动作（同策略同参数默认一次）→ NEAT 决策（STOP 最高分则终止）→ 增量收益门槛 δ
（开发集预定）→ 最大动作数（SINGLE 后最多两次外层策略动作）→ STOP 时从所有已保存
预测中选最终答案（不默认最后一次最好）。

区分三种“迭代”：外层迭代（本研究核心）、策略内部多轮计算、模型继续思考
（Laya 不会沿隐藏状态继续生成；重新调用≠继续推理）。

## 三、策略库（逐步扩展、实证验证）

- 策略 0 SINGLE：初始推理 + 最低成本基线，不属于升级动作。
- 策略 1 WIDE：同题重跑，更大 head_max_len/max_len；不是“继续思考”。
- 策略 2 TOURNAMENT：候选分组淘汰；成本按分组数/阶段/前向量。
- 策略 3 SHORTLIST：predict_shortlist 缩减到 top-k（预注册 k=10/20，仅候选多时启用），报告 recall@k。
- 策略 4 OPTION-ORDER AGGREGATION：本项目构建的包装（非免费内置推理）；先试 2 个顺序。
- 策略 5 LONG-WINDOW：predict_long（如 pinned 版可用），报告窗口数/覆盖率/截断率/总延迟。
- 策略 6 COARSE-TO-FINE：新增任务适配器（非原生 API），须分析错误传播。
- 策略 7 二元任务表示切换（Noul vs 二选一 Choice）：表示变体，重新运行须计成本。

校准是辅助决策模块，不与策略动作平级：训练/开发拟合 temperature scaling，
校准置信度/margin/entropy 作 NEAT 特征，test 不拟合，报告 ECE/Brier。
shortlist 与 option_order 不可假设可组合，须在固定版本上先测（shortlist 会丢弃不适用的 option_order）。

## 四、三大任务领域

- 数学 ProcessBench（优先）：错误步骤定位 Choice；动作 SINGLE/WIDE/TOURNAMENT/SHORTLIST/小规模顺序聚合/LONG-WINDOW；按题分组；指标最早错误准确率、macro-F1、全正确识别率、定位偏差。
- 代码 CodeXGLUE Defect/Devign：二元判断；先比 noul vs 二选一 choice；动作以校准/表示切换为主；防项目/函数泄漏；指标 macro-F1、balanced accuracy、AUROC/AUPRC。
- 金融 FinDVer：claim entailment；noul vs choice；长文本分窗；按报告分组（155 报告跨 testmini/test 已确认）；oracle-evidence 须标注非端到端检索；指标 balanced accuracy、macro-F1、AUROC/AUPRC。

## 五、策略互补性门（第一道科学门槛）

每策略回答：A 能否修正 SINGLE 错误？B 是否引入新错误？C 收益是否可预测？
Δ_correct(a) = correct(a) − correct(current) ∈ {−1,0,+1}；ΔC(a) 为增量成本。
报告：各策略准确率、两两错误重叠、仅 A/B 可修正比例、全对/全错比例、增量收益-成本分布、
特征预测 Δ_correct 的 OOF 表现。无互补性或不可预测 → 暂停该任务 NEAT。

## 六、NEAT 训练：输入/输出/fitness/缓存分离

特征：置信度（原始/校准/margin/entropy）、历史变化、任务结构、长文本、策略历史、预算状态。
仅部署可见特征。输出 STOP + 可行下一动作（任务 adapter 屏蔽不适用动作）。
主 fitness（独立版本，与 BANKING77 旧版区分登记）：U = R_task − λ·(C_total/C_SINGLE)，
λ 训练/开发确定；另设硬预算版本。旧版 run_neat_cv.py 用 raw ms、无复杂度惩罚——按实际
实现解释，不事后修改。训练缓存 genome 模拟；锁定测试真实执行、真实成本。

## 七、预算与成本

档位 1.00×/1.50×/2.00×/3.00×（相对 SINGLE，候选值，以 smoke 实测为准）。
C_total = C_SINGLE + ΣC_action + C_controller；策略内部轮次/窗口/排列全部计入。
运行期硬预算：P90（初期）成本估计超剩余预算则禁用动作；报告超预算率。
报告：均值/P50/P95 累计延迟、调用数/前向量、内部轮数、动作比例、超预算率、控制器耗时、
缓存构建与搜索耗时、峰值 RSS、异常延迟标记（原始保留）。

## 八、基线

固定策略：SINGLE/WIDE/TOURNAMENT/SHORTLIST/2-排列聚合/LONG-WINDOW/最强单策略/
固定序列/随机路由/Oracle（仅离线）。控制器：confidence 阈值、margin/entropy、LR、
浅层树、小 MLP、固定拓扑 EA、NEAT、直接预测 Δcorrect/ΔC 的控制器；代码/金融加
TF-IDF+LR/SVM 任务参考基线。预算匹配基于累计成本，开发集选工作点，test 只评冻结方法；
无法精确匹配则报告真实成本 + 完整 Pareto，不插值。

## 九、阶段与停止门

阶段 0 环境与 Laya 接口验证（20–50 条 smoke）→ 阶段 1 策略互补性 pilot（train/dev 小样本缓存；
无互补性/不可预测则不进 NEAT）→ 阶段 2 基础控制器与预算协议 → 阶段 3 NEAT 多轮决策与 OOF
（SINGLE 后最多两次策略动作）→ 阶段 4 冻结 + 独立测试（test 最后用一次）。
顺序：数学 → 代码 → 金融；跨域不强行合并。

## 十、统计与消融

主指标见 §四；均报告 Pareto、延迟、调用、超预算率、动作比例、RSS。
消融：动作/控制器/预算目标/样本量/特征/迭代/任务表示/稳健性。
统计：≥3 seeds；配对 bootstrap（按题/函数/报告分组采样）；test 冻结后评估；报告差异与区间。

## 十一、本地执行与资源

CPU-only，~4GB，单进程，2 线程起步。环境：Win + Py3.12.12 + CPU torch + Laya 0.4.16
（b9fe351）+ checkpoint 7b928d8 + neat-python 2.0.0。离线仅在缓存就绪后启用。
记录 revision/SHA/配置 hash/线程/长度/调用/成本；异常延迟保留原始。

## 十二、代码实现计划

复用缓存/统计/OOF/配对比较/休眠保护基础设施。新增/扩展：
configs/tasks.yaml；prepare_processbench/codexglue_defect/findver；laya_task_smoke；
build_action_cache（+task adapter）；evaluate_strategy_complementarity（新增）；
run_budgeted_controller_cv（新增）；evaluate_budget_frontier（新增）；results/analysis；
tests。旧 run_neat_cv.py 保留为旧实验记录，不堆新逻辑。

## 十三、与旧实验关系

BANKING77 3-shot NEAT OOF 已完成、未过门；不再加代数、不重跑。旧结果按实际实现保留；
新方法独立版本运行。已诊断：预测动作正确性 ≠ 预测动作能否修正当前错误。

## 十四、叙事分支

A NEAT 多域稳定胜出；B 自适应有收益但简单方法一样好；C 互补存在但收益不可预测；
D 领域差异大；E 升级提质量但无预算优势；F 某任务原生策略均不够好（转任务适配）。
