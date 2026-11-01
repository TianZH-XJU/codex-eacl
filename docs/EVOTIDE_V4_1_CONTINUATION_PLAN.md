# EvoTIDE v4.1 执行计划 · 落库版（含执行状态映射）

> 基线：`dd52fe9`（2026-11-01）。本文档是计划正文 + 逐项落地状态。
> 状态图例：✅ 已完成（有产物+测试+自检） / ⏳ 进行中 / ⬜ 按计划冻结（neat/gp/locked test）/ ❌ 未启动且计划内不启动。

## Phase-1 修 calibrations

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
| 13 | 论文声明 8 项 | ✅ | v0.3 开篇 + §6/§7/§8；无矛盾声明残留（审计 #4 全修） |
| 14 | 交付物 | ✅ | 修复清单/测试日志/口径记录/结果索引/声明清单见 `docs/V4_1_PHASE1_COMPLETION.md`、`results/v4_1/manifests/` |

Phase-1 出口门槛：✅ 无已知 P0/P1；✅ 无 train/held-out 混用；✅ 无矛盾声明。locked test 按协议未跑（无 locked cache + pipeline 未冻结），属正常 defer。

## Phase-2 机制证据

| # | 实验 | 状态 | 结论 |
|---|---|---|---|
| EXP-01 | headroom | ✅ | H1：math 0；其余 net ≤ 0；code/Laya 毛 headroom 0.23 但 repair≈harmful |
| EXP-02 | predictability | ✅ | OOF 最高 0.92 → confirm ~0；零可转移效用 |
| EXP-03 | landscape | ✅ | 71% 首代峰值；b48 用 22/17/23；不可用项已标记（rank 相关、parent-child、neutral mutation、NSGA-II 逐代） |
| EXP-04 | group fault CI | ✅ | math 高危 cell 组 CI 下界 ≥ 0.5，三 split 稳定 |
| EXP-05 | held-out transfer | ✅（待形式自检） | 冻结 genome → dev/confirm：6/32 SIG 全负向；对角线与 eval 文件 est 一致 |
| S-06 | 汇总索引 | ✅ | `protocol_metrics_index.json`（27 文件 → 随 S-05 更新） |
| 决策表 | 2.3 映射 | ✅ | §6：math=行1；code/Laya=行2（高毛headroom/低可预测/平坦）；findver 居间；无"持续收益"行 |

Phase-2 出口：✅ 解释链完整（H1 绑定约束 + H2 次要 + H3 一致）；"尚不能区分"仅用于 neutrality 等未记录项（已标 unavailable）。

## Phase-3 整理

| # | 事项 | 状态 |
|---|---|---|
| 1 | 总结果矩阵 | ✅ §6 证据等级表（12 行：方法/pool/metric/结论/等级） |
| 2 | 主解释选定 | ✅ H1 绑定约束，H2 次要，H3 一致（§6 末段） |
| 3 | What/Why/How | ✅ §1/§6/§4-5 结构对应 |
| 4 | 主文/Appendix | ✅ `paper/appendix.md`（A–I 全表，脚本生成）+ 主文引用 |
| 5 | 图表引用 | ✅ 3 图 + 生成脚本 + 双自检（作图单元） |
| 6 | 版本冻结 | ✅ run_manifest + git SHA + derived_provenance + S-06 index |
| 7 | 锁定候选 | ✅ 冻结 genome/point 均有来源文件；confirm 只读 |
| 8 | locked test | ⬜ 按协议未执行（无 locked cache；pipeline 未冻结前跑即违规） |
| 9 | 最终核对 | ✅ 审计 #4 全修 + 本表 |

## 中心结论（与计划一致，未加强）

> 在本文评估的冻结模型、结构化任务和有限 inference action space 下，GA、ES 和 NSGA-II 没有带来稳定、可重复的性能优势。诊断分析区分：动作之间缺乏足够的互补性（H1，绑定约束）、替代动作的增量效用难以预测（H2，次要）、搜索过程未能有效利用存在的机会（H3，一致但次要）。

## 不启动项（计划内冻结）

NEAT、grammar-GP、新 optimizer、扩大 population、盲目扩大 budget、为正向结果新增搜索分支。
