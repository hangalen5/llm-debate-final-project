# 自动评分与结果保存

在共享仓库目录运行下列命令。这里沿用电脑上位于仓库上一级的 `.venv`；其他电脑可使用自己虚拟环境中的 `python`。

## 1. 离线测试（不调用 API、不下载数据、不收费）

```bash
../.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_evaluation.py
```

检查负数、小数、千位逗号、数值相等、非法答案、CSV 保存、中断后保留记录，以及模拟 API 的完整流程。

## 2. 实际试跑（需要 DEEPSEEK_API_KEY，会产生 API 费用）

```bash
../.venv/bin/python src/run_pilot.py
```

默认运行 test split 的前 3 道题，正常情况下共 30 次模型调用，失败重试可能增加请求次数。之前终端中的结果不会自动导入，新命令会重新调用模型。

也可显式选择题数和 split，例如 `--n 3 --split train`。调整设置用的题目应与最终评价题目分开。此程序仍是基础试跑，不构成相同成本下的正式比较。

## 3. 查看结果

终端最后显示结果文件夹的完整路径。每次创建一个新的 `results/pilot_<timestamp>/`，包含：

- `predictions.csv`：每道题、每种方法一行，3 道题对应 9 行数据。
- `summary.csv`：每种方法一行，记录正确数、总题数、无有效答案数和正确率（0–1）。
- `run.json`：题目来源、split、采样方式、方法设置及运行完成标记。

逐题结果字段：

| 字段 | 含义 |
| --- | --- |
| question_id | `gsm8k:test:0` 表示 test split 的第 1 道题（索引从 0 开始） |
| question | 原始题目 |
| method | direct / self_consistency / debate |
| gold_answer | 标准数字答案 |
| prediction | 提取并规范化后的最终答案 |
| correct | True 为正确，False 为错误或无有效答案 |
| status | ok 表示数字有效（不保证正确）；no_valid_prediction 表示没有有效预测 |

正确率 = 正确题数 / 全部题数；无有效答案仍计入分母。`no_valid_prediction` 可能来自 API 失败或输出格式问题，当前不能区分原因，需要结合终端信息检查。

数字按十进制精确比较，`3`、`3.0` 相等，`-3`、`3` 不相等。优先读取最后一个 `Answer:` 或 `####` 标记，或接受整段仅为数字的回复。没有最终答案时，不从推理中猜测最后一个数字。分数、百分比、表达式暂不支持，会标为无效。

每产生一条结果就写入文件。若中途异常，已完成记录保留，`run.json` 中的 `complete` 仍为 false；不会生成完整汇总。`--output-dir` 可指定新的目录，已有目录会被拒绝以避免覆盖记录。

API 每次调用的原文、token 和费用持久化由另一个模块补齐，本次 CSV 仅保存最终答案与评分。不要把密钥放进结果文件或 Git。
