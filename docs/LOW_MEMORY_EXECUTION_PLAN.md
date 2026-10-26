# 本机低内存执行计划：AMD Ryzen 5 / 项目可用 RAM 约 4 GB

## 目标与运行原则

所有环境安装、数据下载、模型推理和后续实验都在用户自己的 Windows 电脑上完成，不在 ChatGPT 执行环境或 GitHub Actions runner 上运行。电脑虽有 16 GB RAM，但本项目必须控制在约 4 GB 可用内存内，以免影响日常工作。所有任务串行、CPU-only、可恢复、有日志。

## 根据本次提供的本地终端日志，已经完成的步骤

日期：2026-10-26

- Windows PowerShell；目录为 C:\Users\32085\Desktop\codex-eacl-main\codex-eacl-main。
- Python 3.12.12，虚拟环境 .venv。
- PyTorch 2.14.1+cpu，CUDA 不可用，3 个 CPU 线程。
- Laya 0.4.16，本地源码 commit b9fe351ce81e2d974bca2bf219c1e4030c0a05ea。
- neat-python 2.0.0、datasets 5.1.0、pyarrow 26.0.0 等已安装。
- Laya 英文模型 revision 7b928d828b7b0e022f929d9bd2e44165aa270148 已下载并缓存，模型文件约 846 MB。
- 三个数据集的 8 个固定版本文件下载完成，并写出 SHA-256 清单。
- 对 BANKING77、CLINC150 和 MASSIVE en-US 各跑 5 条样本的 smoke test，三个进程均返回代码 0。
- tests/ 下的结构测试在修正后为 3 passed。
- 本地项目目录没有 .git，因此目前运行目录还没有连接到 GitHub。此前的 GitHub 归档工作目录不能等同于用户本地这份环境。

以上来自用户粘贴的本地日志；日志摘录没有包含每个策略的具体准确率、延迟及峰值 RSS，需要读取本地 JSON 报告后再记录。5 条样本只证明安装、API 和模型加载流程基本可用，不足以作为研究结果。

## 内存与 CPU 设置

每次运行前在 PowerShell 设置：

    $env:CUDA_VISIBLE_DEVICES = ""
    $env:OMP_NUM_THREADS = "2"
    $env:MKL_NUM_THREADS = "2"
    $env:OPENBLAS_NUM_THREADS = "2"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"

一次只运行一个推理进程。不要并行跑三个数据集，不要同时加载多个 checkpoint，不要立即对所有测试集做全量推理。将项目进程的目标工作集控制在约 3.5 GB 以下，为 Windows 和工作应用预留余量；若内存逼近该阈值或电脑明显变慢，停止当前实验并记录状态。

先使用默认模型精度复现基线；暂时不要量化或改精度，以免混淆第一轮比较。保持 Laya 参数冻结。

## 第 0 步：把现有目录安全地接入 GitHub

现有目录是压缩包解压后的项目，而不是 Git clone。不能直接依赖之前创建的 GitHub 仓库自动保存本地改动。

先在本地保留一份代码、配置、文档和测试文件备份，不要复制 .venv、external/laya、原始数据和模型缓存，以免产生几 GB 的额外副本。然后在当前项目目录中运行：

    Set-Location 'C:\Users\32085\Desktop\codex-eacl-main\codex-eacl-main'
    git init -b main
    git remote add origin https://github.com/TianZH-XJU/codex-eacl.git
    git fetch origin main
    git reset --mixed origin/main
    git ls-tree -r --name-only origin/main | ForEach-Object { if (-not (Test-Path $_)) { git checkout origin/main -- $_ } }
    Add-Content .gitignore ([Environment]::NewLine + 'external/laya/')
    git branch --set-upstream-to=origin/main main
    git status --short

关键：必须使用 git reset --mixed，而不是 --hard。mixed reset 会把 HEAD 和索引同步到远端提交，但保留工作区文件，从而让本地差异显示为待检查的修改。先检查 git status，再逐项确认后提交。确保 .venv、external/laya、原始数据和模型权重都被忽略。不要盲目执行 git add .，也不要 force push。

首次提交前检查：

    git diff --stat
    git status --short
    git log -1 --oneline

只提交代码、文档、配置、测试和小型报告/清单。确认无误后：

    git add AGENTS.md README.md STATUS.md docs configs scripts tests .gitignore requirements-cpu.txt results
    git status --short
    git commit -m "chore: sync local setup and baseline smoke"
    git push -u origin main

如果 Git 不可用，安装 Git for Windows 或 GitHub Desktop。若 fetch/reset 出错，不要强制覆盖，保留原目录并提供完整错误信息。

## 第 1 步：查看已经存在的报告，不要重新下载

在当前目录读取：

    Get-Content results\smoke\baseline_smoke_report.json -Raw
    Get-Content results\dataset_manifest.json -Raw
    Get-Content results\environment.json -Raw

记录 status、每个数据集的返回码、策略准确率、平均/p50/p95 延迟、调用次数。没有记录的指标保留为空，不推算为真实数据。复用已下载的模型缓存与原始数据，不要重新下载重复文件。

## 第 2 步：只跑一个数据集的 20 条训练样本

先从 BANKING77 开始，因为标签数比 CLINC150 少。该步骤是资源和代码试运行，不是统计结果，不读最终测试集。

    New-Item -ItemType Directory -Force results\pilot | Out-Null
    $env:CUDA_VISIBLE_DEVICES = ""
    $env:OMP_NUM_THREADS = "2"
    $env:MKL_NUM_THREADS = "2"
    $env:OPENBLAS_NUM_THREADS = "2"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    .\.venv\Scripts\python.exe external\laya\research\benchmarks\tournament\benchmark.py --dataset banking77 --split train --limit 20 --seed 17 --strategies single,tournament --model english --device cpu --out results\pilot\banking77_train20.json

启动前确认 results\pilot 已存在。运行时在任务管理器观察 Python 的内存占用。若稳定且内存距离 3.5 GB 阈值有充足余量，再对相同规模运行 single,wide,tournament，输出到另一个 JSON 文件。不要并行执行这两条命令。

## 第 3 步：逐步扩大规模

20 条样本稳定后，先在 BANKING77 上扩到 50 条，再依次在 CLINC150 validation 和 MASSIVE validation 上各运行 20–30 条。每次只运行一个数据集。先测量每条样本耗时再估算全量成本，暂时不要跑全测试集，也不要构造全量动作缓存。

对比策略与超参数只使用 train/validation 数据。测试集在实验协议锁定之前不使用。基线的作用是了解 API、延迟、内存和错误情况；小样本准确率不可用于宣称某策略更好。

## 第 4 步：先建动作缓存，再决定是否值得进化 NEAT

当前上游基准主要输出汇总指标，不能直接替代逐样本策略缓存。下一步应编写单独缓存程序，每条样本只计算每种动作一次，并记录：
- 样本 ID、输入特征及 gold label（分开存储）。
- SINGLE/WIDE/TOURNAMENT 每个动作的预测标签和可用分数。
- 每个动作的实际延迟、Laya 调用次数和失败信息。
- 数据 revision、模型 revision、代码 commit 和随机种子。

NEAT 的每一代只从缓存查结果计算 fitness，不能对每个个体重新运行 Laya。最终测试时仍要实际执行选出的动作，报告真正端到端成本。

在开发集上先算 best fixed action、动作间错误重叠，以及使用 gold label 事后挑动作的 oracle 上限。oracle 只用于测量理论空间，不可部署。若 oracle 相比最佳固定策略提升很小，暂停 NEAT，先调整研究问题或动作集合。

## 第 5 步：资源受限的最小 NEAT 实验

不要从三个数据集、三个种子和大种群开始。

1. 首先只做 BANKING77。
2. 控制器训练采用每类 1-shot 或 3-shot，而不是简单随机抽 50 条，因为需要覆盖多个标签。
3. 缓存构建只跑一次，进化期间只读文件。
4. 起步搜索预算用 population 20、10 generations；将每代最好 fitness、耗时、节点数和连接数写入报告。
5. 先对比最佳固定策略、置信度阈值和浅层决策树；只有显示出潜力才增加进化预算。
6. 流程稳定后扩展到 CLINC150，再到 MASSIVE。
7. 初期只跑一个 NEAT seed；通过 go/no-go 后再增加到至少三个 seed。

按行或小批次读取数据，避免把整个数据集、全部模型输出和模型张量同时保存在 RAM 中。

## 第 6 步：停止条件与每次运行的报告

每次运行记录：时间、Git SHA、Laya commit、Python/PyTorch 版本、模型与数据 revision、device、线程数、样本数、accuracy（只有在完整评测后）、mean/p50/p95 latency、调用次数、耗时、峰值 RSS 和 status。状态只能从 not_run、blocked、failed、partial_failure、passed 等准确标签中选择。

遇到以下任一情况就停止并记录：
- 进程内存逼近 3.5 GB。
- Windows 或工作应用开始严重卡顿/换页。
- 模型在每个样本上重复加载。
- 动作报错或超出上下文的比例过高。
- 缓存成本和实际推理成本无法对账。

## 现在的下一步

1. 先把本地目录安全接入 GitHub，保留现有虚拟环境和数据缓存。
2. 读取已有 JSON 报告，补全 STATUS 中真实测得的数字。
3. 只运行 BANKING77 train 的 20 条 SINGLE+TOURNAMENT pilot。
4. 根据耗时和峰值内存决定是否扩到 50 条。
5. 完成逐样本缓存和 oracle gap 检查后，才开始 NEAT。
