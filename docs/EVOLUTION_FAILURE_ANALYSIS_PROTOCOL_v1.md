# EVOLUTION_FAILURE_ANALYSIS_PROTOCOL_v1

**轨道名称：** `post_gate_evolutionary_diagnostic`（与旧 `NEAT 路由`/`Gate 1 pilot` 轨道并列，不覆盖旧结果）
**冻结日期：** 2026-10-31 · **审计基线：** `73ac20a`（不可变历史）
**运行约束：** Windows 本地 CPU，单模型单进程，2 线程，RSS 硬停止 3.5 GB；GitHub 只存代码/配置/manifest/小结。

## 1. 本轨道回答的问题

旧实验（BANKING77 NEAT 路由、Gate 1 三域 pilot、Gate 4 配置敏感性）一致失败。
本轨道不重复旧搜索，而是按 A0→A10 逐阶段定位失败机制：
平坦地形 / schema 不匹配 / 模型能力上限 / 搜索过拟合 / 算法探索不足 / 成本冲突。

## 2. 有效性修复清单（A0，全部完成后才允许 A1 之后的新推理）

| # | 修复 | 验收 |
|---|---|---|
| F1 | group-aware OOF：GroupKFold 按 group 切分，缓存/输出写 group IDs | `test_group_folds_have_zero_group_overlap` 通过 |
| F2 | majority 基线只用训练折标签 | `test_majority_baseline_uses_train_labels_only` 通过（改 held-out 标签不影响预测） |
| F3 | cluster bootstrap：同一 group 多行一起进样本 | `test_cluster_bootstrap_resamples_groups` 通过 |
| F4 | 稳定 ID 配对 join + state/schema/gold hash 校验 | `test_paired_join_*` 通过；hash 不一致即失败 |
| F5 | semantic Noul schema（true/false 显式语义）+ Choice 对照 | `test_semantic_noul_schema_validates` 通过 |
| F6 | CodeXGLUE function-group manifest（跨 split 零重叠） | 生成器测试通过 |
| F7 | FinDVer test-report-disjoint 池（排除与官方 test 共享报告） | manifest 记录排除数 |
| F8 | ProcessBench 新 search/dev/confirm 排除历史 pilot problem groups | manifest 记录排除数 |
| F9 | Gate 4 配置拆分为 native-range vs OOD-stress | 文档标注，不混排 |

## 3. 数据分组键（冻结）

| 数据集 | group_id 定义 | 规范化规则 |
|---|---|---|
| ProcessBench | SHA256(Unicode NFKC + 空白塌缩 + 换行统一后的 problem 文本) | 同一 problem 文本 = 同组 |
| CodeXGLUE | SHA256(规范化后的 func 文本：去行尾空格，统一换行) | exact hash；AST hash 仅作额外审计 |
| FinDVer | report 文件名（官方 report ID） | 同一报告 = 同组 |

## 4. 统计规则（冻结）

- OOF：GroupKFold（group 数允许时用 StratifiedGroupKFold）；scaler/encoder/prior 只在 fold-train 拟合。
- 主区间：cluster bootstrap（整组重采样），seed 17，5000 reps，95%。
- 辅助区间：普通 row bootstrap，仅作对照。
- 报告 group 数、样本数、组大小分布；group 数少时标注区间不稳定。

## 5. 缓存字段（新增 runs 必须遵守；旧缓存用离线映射补 group_id）

run_id, git_sha, model_id, model_revision, adapter_version, optimizer, seed,
dataset, source_revision, split, row_id, group_id,
state_text_sha256, question_schema_sha256, gold_sha256,
representation, config_json, config_hash,
raw_prediction, normalized_prediction, probabilities, confidence, abstain,
input_tokens, forward_calls, tournament_rounds, shortlist_k, windows,
latency_ms, load_time_ms, peak_rss_mb, errors, truncation_flags.
Vela 另记 usage.input_tokens/output_tokens/sequences/windows（如接口返回）。

## 6. 阶段门（任一失败即停该分支）

- A0：分组零交叉、join 校验、schema smoke、config validation 测试全过。
- A1：只记录原生性能，不做选择。
- A2：表示消融；legacy 结果保留为历史对照。
- A3：OFAT；无可重复非噪声变化 → 只允许受限 bank 诊断，不宣称可学信号。
- A4/A5：bank 穷举参照 + 有限预算搜索；regret = bank_best − optimizer_best。
- A6：同 bank、同 budget、同 seeds 比较 Random/GA/DE/CMA-ES/NSGA-II(+TPE)。
- A7：train best 必须在 group-dev 上复评；rank 相关 + optimism 一起报告。
- A8：跨任务/跨模型迁移；ModernBERT-family 边界如实写。
- A9：每个根因必须有可证伪的最小对照实验。
- A10：冻结 FREEZE_MANIFEST 后才允许 locked test，且只跑一次。

## 7. 明确禁止

- 用 test 标签/分数/逐样本预测选模型、策略、提示、阈值、参数、seed。
- 把 OOD stress（如 max_len=1024）与原生范围结果混排。
- 把缺省 Noul 与语义 Noul 混为一类报告。
- 未运行算法标 completed；把 48/48 单元测试当作模型研究成功。
