# 第 3 轮作业：Marathon 长上下文评测与分段编排对比

马照新（智能体编排组 · 论文方向）的第 3 轮实验代码与数据记录，目标数据集为 `Marathon`（ACL 2024）。

## 目录说明

- `data/`
  - `eval_split_main.jsonl`：从还原真值后的 1,182 题里按 `seed=42` 分层抽出的 120 题评测集（6 类任务各 20 题）
  - `eval_split_pilot.jsonl`：前期 10 题小批量试跑集
- `docs/`
  - `第3轮实验报告_马照新.md`：本轮完整实验记录、错题分析与对比结论
  - `rsi_flow_familiarity.md`：`ymxiong/rsi-flow` 结构梳理与编排 JSON 蓝图
- `results/`
  - `baseline_6sol_*` / `baseline_6sol_trunc36k_*`：`gpt-6-sol` 全量与 36K 截断基线逐题日志和汇总
  - `baseline_*` / `baseline_trunc36k_*`：`gemini-3.8-flash` 全量与 36K 截断基线逐题日志和汇总
  - `improved_6sol_*`：`gpt-6-sol` 分块取证 + 汇总校验改进版逐题日志和汇总
  - `error_analysis*.json`、`comparison_6sol.json`：错误分类统计与配对对比结果
- `src/`
  - `thingo_client.py`：Thingo API 调用、重试、预算熔断和 CNY 计费统计
  - `build_marathon_eval_set.py`：从上游 `LooGLE` 和 `LongBench` 还原标准答案与 `evidence`
  - `run_baseline.py`：跑全上下文和 36K 截断基线
  - `analyze_errors.py`：按错误类型、上下文长度、证据原文位置统计失败分布
  - `run_improved.py`：跑分块并行取证 + 汇总校验改进版
  - `compare_experiments.py`：生成三组实验的配对对比表

## 复现命令

在根目录建一个 `.env`（已加进 `.gitignore`，不要提交）：

```dotenv
THINGO_INTL_API_KEY=你的Gemini分组Key
THINGO_GPT_API_KEY=你的GPT特惠分组Key
THINGO_INTL_BASE_URL=https://intlapi.thingo.com.cn/v1
```

如果上游源文件放在 `upstream_cache/`，可按下面顺序重跑全流程：

```powershell
python src/build_marathon_eval_set.py
python src/run_baseline.py --use-existing-split --model gpt-6-sol --trace-out results/baseline_6sol_traces.jsonl --summary-out results/baseline_6sol_summary.json
python src/run_baseline.py --use-existing-split --model gpt-6-sol --max-context-chars 36000 --trace-out results/baseline_6sol_trunc36k_traces.jsonl --summary-out results/baseline_6sol_trunc36k_summary.json
python src/run_improved.py --model gpt-6-sol --trace-out results/improved_6sol_traces.jsonl --summary-out results/improved_6sol_summary.json
python src/analyze_errors.py --traces results/baseline_6sol_trunc36k_traces.jsonl --out results/error_analysis_6sol_trunc36k.json
python src/compare_experiments.py --base-full results/baseline_6sol_traces.jsonl --base-trunc results/baseline_6sol_trunc36k_traces.jsonl --improved results/improved_6sol_traces.jsonl --out results/comparison_6sol.json
```
