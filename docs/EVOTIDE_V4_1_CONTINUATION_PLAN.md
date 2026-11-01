# EvoTIDE v4.1 后续完整计划（v0.4）

> 基线：`dd52fe9`（2026-11-01）。本文档是计划正文 + 逐项落地状态。
> 状态图例：✅ 已完成（有产物+测试+自检） / ⏳ 进行中 / ⬜ 按计划冻结（neat/gp/locked test）/ ❌ 未启动且计划内不启动。

## Phase-1 修复与校准

| # | 事项 | 状态 | 证据 |
|---|---|---|---|
| 1 | C-01..C-10 修复验证 | ✅ | `docs/V4_1_PHASE1_COMPLETION.md` §C-01..C-10；全套件绿 |
| 2 | C-11 标签 | ✅ | `test_evo_significance.py` f-string tag；`es_significance.json` 已改 `es_minus_random`；测试覆盖 |
| 3 | C-12 gens_effective | ✅ | `ga_gens_effective()` + ES 按实际 log；7 个搜索文件重跑逐位一致，仅元数据变化 |
| 4 | C-13 dev/confirm 动态键 | ✅ | `entry[args.algo]`；合成缓存端到端测试；泄漏守卫负向测试 |
| 5 | C-14 NSGA-II dev 选择 | ✅ | `scripts/eval_nsga2_deploy.py` + `nsga2_deploy_devconfirm.json`（dev 5/12 SIG 负向，confirm 1/12，全无正向） |
| 6 | E-01 GA 分层 | ✅ | 证据等级表 + `train_search_{dev,confirm}_eval.json`（split/metric/n/算法/baseline 齐全） |
| 7 | E-02 ES dev/confirm | ✅ | `scripts/eval_es_deploy.py` + `es_deploy_devconfirm.json`（dev 2/3 SIG 负向，confirm 全 ns） |
| 8 | E-03 NSGA-II 证据等级 | ✅ | 旧 in-sample 保留标注；新结论以 `nsga2_deploy_devconfirm.json` 为准 |
| 9 | E-04 transfer 降级/A-B | ✅ | 方案 A（in-sample 标注）+ 已做方案 B 冻结 held-out（`transfer_heldout.json`，6/32 SIG 全负向） |
| 10 | E-05 诊断定位 | ✅ | 全文统一 "diagnostic fault map (MAP-Elites-inspired, not an evolutionary optimizer)" |
| 11 | E-06 b48 解释 | ✅ | 计划原句已入论文；cap 未用尽 22/17/23/48 |
| 12 | E-07 +6/60 来源链 | ✅ | `complementarity_screen.json` + 生成脚本 + 引用 |
| 13 | 论文声明 8 项 | ✅ | v0.3 开篇 + §6/§7/§8；无矛盾声明残留 |
| 14 | 交付物 | ✅ | 修复清单/测试日志/口径记录/结果索引/声明清单见 `docs/V4_1_PHASE1_COMPLETION.md`、`results/v4_1/manifests/` |

Phase-1 出口门槛：✅ 无已知 P0/P1；✅ 无 train/held-out 混用；✅ 无矛盾声明。locked test 按协议未跑（无 locked cache + pipeline 未冻结），属正常 defer。

## Phase-2 机制证据

| # | 实验 | 状态 | 结论 |
|---|---|---|---|
| EXP-01 | headroom | ✅ | H1：math 0；其余 net ≤ 0；code/Laya 毛 headroom 0.23 但 repair≈harmful |
| EXP-02 | predictability | ✅ | OOF 最高 0.92 → confirm ~0；零可转移效用 |
| EXP-03 | landscape | ✅ | 71% 首代峰值；b48 用 22/17/23；不可用项已标记（rank 相关、parent-child、neutral mutation、NSGA-II 逐代） |
| EXP-04 | group fault CI | ✅ | math 高危 cell 组 CI 下界 ≥ 0.5，三 split 稳定 |
| EXP-05 | held-out transfer | ✅ | 冻结 genome → dev/confirm：6/32 SIG 全负向；对角线与 eval 文件 est 一致 |
| S-06 | 汇总索引 | ✅ | `protocol_metrics_index.json`（27 文件，transfer_heldout 有 provenance 条目） |
| 决策表 | 2.3 映射 | ✅ | §6：math=行1；code/Laya=行2（高毛headroom/低可预测/平坦）；findver 居间；无"持续收益"行 |

Phase-2 出口：✅ 解释链完整（H1 绑定约束 + H2 次要 + H3 一致）；"尚不能区分"仅用于 neutrality 等未记录项（已标 unavailable）。

## Phase-3 整理

| # | 事项 | 状态 |
|---|---|---|
| 1 | 总结果矩阵 | ✅ §6 证据等级表（12 行：方法/pool/metric/结论/等级） |
| 2 | 主解释选定 | ✅ H1 绑定约束，H2 次要，H3 一致（§6 末段） |
| 3 | What/Why/How | ✅ §1/§6/§4-5 结构对应 |
| 4 | 主文/Appendix | ✅ `paper/appendix.md`（A–K 全表，脚本生成）+ 主文引用 |
| 5 | 图表引用 | ✅ 3 图 + 生成脚本 + 双自检（作图单元） |
| 6 | 版本冻结 | ✅ run_manifest + git SHA + derived_provenance + S-06 index |
| 7 | 锁定候选 | ✅ 冻结 genome/point 均有来源文件；confirm 只读 |
| 8 | locked test | ⬜ 按协议未执行（无 locked cache；pipeline 未冻结前跑即违规） |
| 9 | 最终核对 | ✅ 审计 #4 全修 + 本表 |

## 中心结论（与计划一致，未加强）

> 在本文评估的冻结模型、结构化任务和有限 inference action space 下，GA、ES 和 NSGA-II 没有带来稳定、可重复的性能优势。诊断分析区分：动作之间缺乏足够的互补性（H1，绑定约束）、替代动作的增量效用难以预测（H2，次要）、搜索过程未能有效利用存在的机会（H3，一致但次要）。

## 不启动项（计划内冻结）

NEAT、grammar-GP、新 optimizer、扩大 population、盲目扩大 budget、为正向结果新增搜索分支。

## P0 补充实验（Phase-2 优先）

1. **P0-1 Noul 模板审查**：验证模型是否依据输入 state 判断而非响应问题词语或默认标签。Laya/Vela × CodeXGLUE/FinDVer，配对比较 4 种模板变体。
2. **P0-2 概率校准**：reliability diagram、ECE、Brier、NLL、calibration slope/intercept；temperature 用独立 calibration fold 拟合；raw vs calibrated 在 confirm 比较。
3. **P0-3 留出数据 + macro-F1 复核**：dev/confirm 上计算 repair/harmful/net/disagreement + accuracy headroom；Noul-only vs Choice-only macro-F1 配对差值。
4. **P0-4 group-bootstrap 区间**：按 group 对 confirm 样本重采样，计算 router vs fixed 的 net utility + cost-adjusted 区间。

## P1 补充实验（可选）

- 扩大 group 数量 + 补齐预注册种子 59/73
- 真实推理延迟测量（验证调用次数代理）
- 搜索动态诊断（基于现有缓存，不新增模型调用）
- NSGA-II Pareto 目标与 macro-F1 对齐
- Jev 同任务外部复现（可选）

## 最终一次性锁定测试

完成 P0 模板与校准审查、确定最终模型和策略版本、冻结统计脚本及主比较后，再依照协议运行 locked test 一次。此步骤不应用于选择提示模板、重新设定阈值或更换搜索参数；它的目的只是检验冻结方案在最终保留数据上的表现。若此前 P0 审查要求重建缓存，则应先完成全部修订和 confirm 验证，再进行最终冻结。

## 审核要求（1-20）

1. 用 ACL 审稿人思维检查：逻辑漏洞、概念不清、文献支撑不足
2. 用硕导思维检查：研究问题清楚、理论逻辑顺、变量关系合理、语言像正式论文
3. 检查段落中心句：每段第一句清晰、言简意赅、概括本段中心思想
4. 检查总分结构：先中心观点，再展开解释
5. 检查研究缺口：已有研究准确？转折对应？缺口未被解决？
6. 检查"但/然而"逻辑：转折前后形成清楚对照关系
7. 检查概念定义：过窄、过宽或过度情境化
8. 检查概念与变量对应：不要引导向无关结果变量
9. 检查文献和句子匹配：不要硬塞，不要扩大化
10. 检查文献适配性：从研究对象、理论视角、变量关系、研究情境判断
11. 检查经典文献是否缺失：提供 SSCI/SCI 论文 + DOI
12. 检查中文表达自然：避免翻译腔
13. 检查是否提前剧透：引言只提问题，不讲发现
14. 检查是否偏离主线：关键词 5 个；不服务主线的内容建议删减
15. 检查引言"漏斗型但不空泛"：从背景聚焦到具体问题
16. 检查贡献具体性：具体推进了哪类研究、推进在哪里
17. 检查是否给自己挖坑：未测量的机制不要写得像已检验
18. 检查语言绝对性：避免"必然""本质差异""不可替代"
19. 检查 RQ 清楚、主次分明、与设计/假设/结果对应
20. 检查段落衔接：为什么上一段后必须写这一段？如何把读者推向下一段？

## 论文图表补充

- 需要补充大量图表：数据图、数据表格等
- 整体架构图需要生成（可用 opencode 图片生成模型）
- 论文初稿排版按照 ACL 标准
- 自动检查 https://github.com/acl-org/aclpubcheck
- 中文版供审核

## Subagent 审核要求

- 每次代码改动：subagent for code-review 独立审核
- 每次代码运行：subagent for review 结果分析，避免低级错误/泄露
- 论文按大章节逐步完成，每次写作完成：subagent for write 审核内容/语句/文风
- 每两段衔接：subagent 进行连贯性审核
- 全文写完及全文修改：subagent for 全文故事线审核
- 文献：subagent 对文献进行 websearch，合适位置加入合适论文（arxiv/scispace 等），整体 40 篇参考文献
- 图片/表格审查：对应图片字号大小和排版清晰
