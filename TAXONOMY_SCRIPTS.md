# 困境分类脚本使用说明

本文说明与「社会心理学分类体系」相关的两个脚本：**批量调用大模型打标签**（`classify_reddit_taxonomy.py`）和 **统计各类别数量**（`count_taxonomy_labels.py`）。

## 环境准备

1. **依赖**：与主项目一致，需已安装 `openai`、`python-dotenv` 等（见 `requirements.txt`）。
2. **API 配置**：在项目根目录的 `.env` 中配置（与 `run.py` 相同），例如：

   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL`（若使用 DeepSeek、火山方舟、OpenRouter 等兼容端点）
   - `LLM_MODEL`（模型名，可被命令行 `--model` 覆盖）

3. **输入数据**：JSONL，每行一个 JSON 对象。正文会从下列字段中**按顺序**取第一个非空字符串：`INSTRUCTION` → `text` → `body` → `content` → `title` → `question`（与 `cognitive_probe` 一致）。  
   若尚无 `reddit_r_advice.jsonl`，可先用 `extract_reddit_subreddit.py` 等从本地数据集导出。

**输入路径优先级**（未写 `-i` 时）：`--source` 预设 → 环境变量 `CLASSIFY_TAXONOMY_INPUT` → 项目根下 `reddit_r_advice.jsonl`。自定义路径永远用 `-i PATH`（优先级最高）。可在脚本顶部字典 `TAXONOMY_INPUT_PRESETS` 里增加自己的 `--source` 名称。

---

## 1. `classify_reddit_taxonomy.py` — 用大模型分类

对 JSONL 中每条帖子调用一次对话接口，按预设 **7 类** 体系输出 `reasoning` + `category`（JSON），结果追加写入输出文件。

### 常用命令

```bash
# 查看内置数据来源预设名与路径
python classify_reddit_taxonomy.py --list-sources

# 用预设数据源（例如 data 下的 Reddit 导出）
python classify_reddit_taxonomy.py --source reddit_data -o data/taxonomy_labels.jsonl

# 默认：未设置 -i / --source / CLASSIFY_TAXONOMY_INPUT 时，输入为 reddit_r_advice.jsonl；输出 reddit_taxonomy_labels.jsonl
python classify_reddit_taxonomy.py

# 指定输入/输出路径
python classify_reddit_taxonomy.py -i data/reddit_r_advice.jsonl -o data/taxonomy_labels.jsonl

# 从第 500 条开始，只处理 100 条；4 并发；正文过长时截断
python classify_reddit_taxonomy.py --start 500 -n 100 --concurrency 4 --max-question-chars 12000
```

python classify_reddit_taxonomy.py -i subreddits.jsonl -n 200 --concurrency 20

```bash
python classify_reddit_taxonomy.py -i subreddits.jsonl -n 200 --random-sample --random-seed 42 --concurrency 30
```
### 主要参数

| 参数 | 说明 |
|------|------|
| `-i` / `--input` | 输入 JSONL 路径；若指定则覆盖 `--source` 与 `CLASSIFY_TAXONOMY_INPUT` |
| `-S` / `--source` | 使用内置预设名（与 `--list-sources` 一致）；未写 `-i` 时生效 |
| `--list-sources` | 打印所有预设路径后退出 |
| （环境变量） | `CLASSIFY_TAXONOMY_INPUT`：未写 `-i` 且未写 `--source` 时的默认输入路径 |
| `-o` / `--output` | 输出 JSONL 路径（默认：`reddit_taxonomy_labels.jsonl`） |
| `-n` / `--num` | 最多处理多少条（在 `--start` 跳过行之后计数） |
| `--start` | 跳过前 N 条非空 JSON 行（0 起算，与 `run.py` 一致） |
| `--no-resume` | 默认会跳过输出文件中已存在的 `question_index`；加此参数则不做续跑，直接往后追加（慎用重复） |
| `--concurrency` | 并发请求数，默认 `1` |
| `--backend` | `openai`（默认）或 `dummy`（调试用） |
| `--api-key` / `--base-url` / `--model` | 覆盖环境变量中的密钥、基地址、模型名 |
| `--temperature` | 采样温度，默认 `0.2` |
| `--max-tokens` | 单次回复最大 token，默认 `512` |
| `--max-question-chars` | 送入模型的正文最大字符数，超出部分截断并加省略标记 |
| `--keep-raw` | 解析成功时也保留 `raw_response` 字段（默认成功时会删掉以减小体积） |

### 输出文件（JSONL）每行字段说明

- `question_index`：对应输入文件中的全局行序（与 `--start` 对齐）。
- `question`：送入分类模型的正文。
- `metadata`：除正文字段外的其余 JSON 字段副本。
- `reasoning`：模型简述的分类依据。
- `category`：类别标签，须为 7 类之一（带英文方括号的形式）。
- `category_valid`：是否与内置合法标签表严格匹配（含自动补全方括号后的校验）。
- `parse_error`：若 JSON 解析失败等，此处为错误说明；否则一般为 `null`。
- `raw_response`：模型原始回复；默认在解析成功且类别合法时**不写入**（可用 `--keep-raw` 保留）。
- `elapsed_sec`：该条 API 耗时（秒）。

### 续跑说明

默认会读取已有输出里出现过的 `question_index`，只处理尚未写入的索引。中断后再次运行相同命令即可接着跑，无需改参数。

---

## 2. `count_taxonomy_labels.py` — 统计各类别条数

读取分类脚本的输出 JSONL，按 `category` 汇总数量，并附带数据质量摘要。

### 常用命令

```bash
# 默认读取 reddit_taxonomy_labels.jsonl
python count_taxonomy_labels.py

# 指定输入文件
python count_taxonomy_labels.py -i data/taxonomy_labels.jsonl

# 将结果以 JSON 打印到标准输出（可重定向到文件）
python count_taxonomy_labels.py --json > taxonomy_summary.json
```

### 参数

| 参数 | 说明 |
|------|------|
| `-i` / `--input` | 待统计的 JSONL（默认：`reddit_taxonomy_labels.jsonl`） |
| `--json` | 打印结构化 JSON 报告；不加则打印人类可读表格 |

### 文本输出含义

- **Total records**：有效 JSON 行数（空行不计）。
- **category_valid=False**：标签未通过校验的行数。
- **Rows with non-empty parse_error**：`parse_error` 非空的行数。
- **By category**：按与分类脚本一致的 **7 类固定顺序** 列出每类条数；若某类未出现则为 `0`。
- **Other labels**：若出现不在 7 类列表中的 `category` 字符串，会单独列出。

### `--json` 输出结构（摘要）

- `input_total_lines`：总记录数。
- `rows_category_valid_false` / `rows_with_parse_error_field`：同上。
- `by_category_canonical_order`：七类各自计数。
- `extra_labels_not_in_taxonomy`：异常标签及其计数（无则为 `{}`）。

---

## 推荐工作流

1. 准备好 `reddit_r_advice.jsonl`（或你的 JSONL），配置好 `.env`。  
2. 运行 `classify_reddit_taxonomy.py`（可先 `-n 10` 试跑）。  
3. 分类结束或阶段性结束后，运行 `count_taxonomy_labels.py` 查看分布；需要留档时用 `--json` 重定向保存。

若在分类任务**仍在写入**输出文件时运行统计脚本，行数可能略有偏差，建议在写入结束后再统计。
