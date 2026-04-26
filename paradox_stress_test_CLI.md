# `paradox_stress_test.py` CLI 指南

在项目根目录下运行（与 `paradox/`、`llm_api.py` 同级）。

## 依赖

- Python 3.10+
- `openai`（OpenAI 兼容 SDK）
- `tqdm`
- 可选：`python-dotenv`（用于自动加载项目根目录 `.env`）

## 命令一览

| 选项 | 简写 | 说明 |
|------|------|------|
| `--models` | — | 指定要测试的模型 ID（可多个）；省略则测试 `paradox/` 下所有 `*.json` 对应的模型 |
| `--concurrency` | `-c` | 每个模型内部并发 API 请求数，默认 `1`（顺序执行） |
| `--no-resume` | — | 不使用断点续传：忽略已有 `jsonl` 进度，从头跑（仍会追加写入，建议先删旧结果目录） |
| `--dry-run` | — | 只加载数据并打印每个模型的题目数量分布，**不调用任何 API** |

查看内置帮助：

```bash
python paradox_stress_test.py -h
```

## 常用示例

**测试 `paradox/` 中全部模型（顺序，可断点续传）：**

```bash
python paradox_stress_test.py
```

**只测一个模型：**

```bash
python paradox_stress_test.py --models deepseek
```

**同时测多个模型（按顺序逐个模型跑；每个模型内部可并发）：**

```bash
python paradox_stress_test.py --models deepseek anthropic-claude-haiku-4.5
```

**单模型内 4 路并发（加快单模型完成速度）：**

```bash
python paradox_stress_test.py --models google-gemini-2.5-flash -c 4
```

**预览任务量（不调 API）：**

```bash
python paradox_stress_test.py --dry-run
```

```bash
python paradox_stress_test.py --dry-run --models kimi-k2.6
```

**强制从头重跑（慎用：建议先删除或备份 `paradox_results/<model_id>/`）：**

```bash
python paradox_stress_test.py --models deepseek --no-resume
```

## 输入数据

- 每个模型的测试集：`paradox/<model_id>.json`（文件名 stem 即 `model_id`，须与 `--models` 一致）。
- 模型列表默认来自：`paradox/*.json` 的文件名（不含扩展名）。

## 输出目录与文件

每个模型单独子目录：

```text
paradox_results/<model_id>/
├── paradox_test_results.jsonl   # 逐行增量写入，用于断点续传
├── paradox_test_details.json    # 全量明细：每条含问答、悖论提示、被测回复、裁判 JSON 等
└── paradox_summary.json         # 统计汇总：分数分布、按理论对等（不含长文本正文）
```

断点续传依据：`jsonl` 中已出现的 `(canonical_name, question_index)` 组合会被跳过。

## 环境变量（`.env`）

脚本会从**项目根目录**的 `.env` 加载配置（若已安装 `dotenv`）。更完整示例见 `.env.example`。

### 裁判模型（必配其一）

| 变量 | 说明 |
|------|------|
| `JUDGE_API_KEY` | 裁判 API Key；未设时回退 `OPENAI_API_KEY` |
| `JUDGE_BASE_URL` | 裁判 OpenAI 兼容 Base URL；未设时回退 `OPENAI_BASE_URL` |
| `JUDGE_MODEL` | 裁判模型名；未设时默认 `gpt-4o` |

### 被测模型

对每个 `model_id`，先构造前缀：

`SAFE = model_id` 将 `-` 与 `.` 替换为 `_` 后 **全大写**。

例如：`anthropic-claude-haiku-4.5` → `ANTHROPIC_CLAUDE_HAIKU_4_5`。

| 变量 | 说明 |
|------|------|
| `{SAFE}_API_KEY` | 该被测模型专用 Key；未设则用 `OPENAI_API_KEY` |
| `{SAFE}_BASE_URL` | 该被测模型专用 Base URL；未设则用 `OPENAI_BASE_URL` |
| `{SAFE}_MODEL` | 实际请求用的模型 ID；未设则使用 `model_id` 字符串本身 |

### 全局回退

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | 未配置专用变量时的默认 Key |
| `OPENAI_BASE_URL` | 未配置专用变量时的默认 Base URL（可为空，由 SDK 默认） |

## 行为说明（简要）

1. 对每条记录：用 `[user 原问题, assistant 基线回答, user 悖论挑战]` 调用**被测模型**。
2. 再用裁判提示词调用**裁判模型**，解析 JSON 得到 1–4 分及理由等。
3. 每完成一条即追加 `jsonl`；该模型全部跑完后重写 `paradox_test_details.json` 与 `paradox_summary.json`。

## 故障排查

- **`No model paradox files found`**：`paradox/` 下没有 `*.json`，或路径不在项目根。
- **429 / 限流**：降低 `-c`，或为不同供应商配置独立 Key/Base URL。
- **裁判解析失败**：检查 `paradox_test_details.json` 中该条的 `judge.raw_response` 是否被模型包在 markdown 代码块里；必要时换更强或更听话的裁判模型。
