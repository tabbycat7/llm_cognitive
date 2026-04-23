# 命令行启动说明（`run.py`）

入口脚本：`run.py`。项目根目录下需存在 `zhihu_kol_train.jsonl`（或通过 `-i` 指定路径）。

## 环境准备

```bash
cd e:\college\postgraduate\llm_cognitive
pip install -r requirements.txt
```

在项目根目录配置 `.env`（脚本会自动加载并覆盖空的环境变量）：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | 兼容 OpenAI 的 API Key（如 DeepSeek、OpenAI） |
| `OPENAI_BASE_URL` | API 根地址，例如 `https://api.deepseek.com`（是否带 `/v1` 以服务商文档为准） |
| `LLM_MODEL` | 对话模型名，如 `deepseek-chat` |
| `EMBEDDING_API_KEY` | （可选）仅当 `--merge-method embedding` 且 `--embedding-backend openai` 时使用 |
| `EMBEDDING_BASE_URL` | （可选）同上 |
| `EMBEDDING_DEVICE` | （可选）本地向量模型设备：`auto`（有 CUDA 则用 GPU）、`cuda`、`cuda:0`、`cpu` 等；也可用命令行 `--embedding-device` 覆盖 |

## 查看全部参数

```bash
python run.py -h
```

## 数据与输出

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `zhihu_kol_train.jsonl` | 输入 JSONL（需含 `INSTRUCTION` 等字段） |
| `--output` | `-o` | `probe_results.jsonl` | 三步探测结果，每行一条 JSON |
| `--num` | `-n` | 全部 | 只处理前 N 条问题 |

## 大模型后端（探测阶段）

| 参数 | 简写 | 说明 |
|------|------|------|
| `--backend` | `-b` | `dummy`（无网络，调试用）或 `openai`（任意 OpenAI 兼容服务） |
| `--api-key` | | 可覆盖 `.env` 中的 Key |
| `--base-url` | | 可覆盖 `.env` 中的地址 |
| `--model` / `-m` | | 模型名，默认读 `LLM_MODEL` 或 `gpt-4o` |
| `--temperature` | | 采样温度，默认 `0.7` |
| `--max-tokens` | | 单次回复上限，默认 `4096` |

## 探测流程控制

| 参数 | 简写 | 说明 |
|------|------|------|
| `--concurrency` / `-c` | | 并发处理的问题数（每题仍串行 3 轮 API）。默认 `1`；生产可试 `3`～`10`，遇 429 再降低 |
| `--no-resume` | | 不跳过输出文件中已有 `question_index` 的题（慎用，易重复追加） |
| `--analyse-only` | | 不跑探测，只对已有 `--output` 做统计与出图 |

**断点续跑**：默认会读取 `--output`，跳过已完成的 `question_index`，向同一文件追加新行。

## 分析阶段（归并与可视化）

探测结束后（或 `--analyse-only`）会执行分析，主要参数：

| 参数 | 说明 |
|------|------|
| `--merge-method` | `llm`（默认，需对话同一套 API）、`embedding`（向量层次聚类）、`none`（不归并，直接输出原始名称的柱状图和词云） |
| `--embedding-backend` | `local` / `ollama` / `openai`，仅 `merge-method=embedding` 时有效 |
| `--embedding-model` | 嵌入模型名，默认 `BAAI/bge-small-zh-v1.5`（local） |
| `--embedding-threshold` | 余弦距离阈值，`0.2` 更严、`0.4` 更松，默认 `0.3` |
| `--embedding-device` | 仅 **local** 后端：嵌入模型加载到哪张卡 / CPU。默认读环境变量 `EMBEDDING_DEVICE`，未设置则为 `auto`（有 CUDA 则用 `cuda`） |
| `--name-source` | 分析时从探测结果的**第几步**取文本做标签与聚类，见下节。可选 `step1` / `step2` / `step3`，默认 `step3` |
| `--refresh-merge` | 忽略 `merge_map.json` 等缓存，重新归并 / 重新算向量 |
| `--top-n` | 条形图与控制台表格显示前 N 个规范名，默认 `30` |

### 选用第几步回复（`--name-source`）

| 取值 | 含义 |
|------|------|
| **`step3`**（默认） | 从 `step3_response` 里解析 JSON 的 `models[].name`，与原先行为一致；适合统计「模型/定律名称」并做 LLM 或 embedding 归并。 |
| **`step1`** | 用每条记录的 **`step1_response` 全文**（去首尾空白后非空才计一条）作为一个标签；**不做 JSON 解析、不做名称规范化**，直接对整段文本做向量聚类。 |
| **`step2`** | 同上，使用 **`step2_response` 全文**。 |

**与 `--merge-method` 的搭配：**

- **`step3`**：可与 `llm`、`embedding`、`none` 任意搭配。
- **`step1` / `step2`**：全文很长，不能塞进 LLM 聚类提示，因此 **不要** 使用 `--merge-method llm`；请使用 **`embedding`**（推荐）或 **`none`**（只出频次图/词云、不归并）。

**其它说明：** 长文本会按嵌入模型长度上限截断后再算向量；散点图上的点标注与图例会**自动截断显示**（避免整段文字铺满图面），不影响聚类计算。

**常见输出文件**（默认与 `--output` 同目录）：

- `probe_results.jsonl`：原始探测结果  
- `merge_map.json`：归并映射（及 embedding 时的元信息）  
- `merge_map.embeddings.json`：embedding 缓存（若使用 embedding）  
- `model_law_frequency.png`、`model_law_wordcloud.png`：频次图与词云  
- `cluster_dendrogram.png`、`cluster_scatter.png`：仅 `merge-method=embedding` 时生成  

## 常用命令示例

**PowerShell** 下请分行执行，或把续行符写成反引号 `` ` ``；下面示例为单行，可直接复制。

### 1. 本地假数据跑通流程（无需 Key）

```bash
python run.py --backend dummy -n 5
```

### 2. 真实 API，处理 100 条，并发 5

```bash
python run.py --backend openai -n 100 --concurrency 5
```



```bash
python run.py --backend openai --temperature 0.8 -i top500_per_category.jsonl --prompt-lang zh --filter-category "[Relational & Intimate]" --start 0 --num 500 -c 30  --merge-method llm  
```

```bash
python run.py --backend openai --temperature 0.8 -i top500_per_category.jsonl --prompt-lang zh --filter-category "[Professional & Economic]" --start 0 --num 500 -c 30  --merge-method llm  
```

```bash
python run.py --backend openai --temperature 0.8 -i top500_per_category.jsonl --prompt-lang zh --filter-category "[Societal & Ethical]" --start 0 --num 500 -c 30  --merge-method llm
```

```bash
python run.py --backend openai --temperature 0.8 -i top500_per_category.jsonl --prompt-lang zh --filter-category "[Personal & Existential]" --start 0 --num 500 -c 30  --merge-method llm
``` 

### 3. 只分析已有结果（LLM 聚类归并）


```bash
python run.py --analyse-only -o "deepseek\Relational&Intimate\probe_results.jsonl"  --refresh-merge --merge-method llm
```

```bash
python run.py --analyse-only -o "MiniMax-M2.5\Relational&Intimate\probe_results.jsonl"  --refresh-merge --merge-method llm
```

```bash
python run.py --analyse-only -o "deepseek\Professional&Economic\probe_results.jsonl" --refresh-merge --merge-method llm
```
```bash
python run.py --analyse-only -o "deepseek\Societal&Ethical\probe_results.jsonl" --refresh-merge --merge-method llm
```

```bash
python run.py --analyse-only -o "deepseek\Personal&Existential\probe_results.jsonl" --refresh-merge --merge-method llm
```

### 4. 只分析，本地向量归并 + 聚类图（需已安装 `sentence-transformers`、`umap-learn` 等）

```bash
python run.py --analyse-only --merge-method embedding --embedding-backend local --embedding-model BAAI/bge-small-zh-v1.5 --embedding-threshold 0.3
```

### 4b. 大模型 `BAAI/bge-large-zh-v1.5` 放到 **GPU**（需本机已安装 CUDA 版 PyTorch，且显存足够，large 约数 GB）

```bash
python run.py --analyse-only --merge-method embedding --embedding-backend local --embedding-model BAAI/bge-large-zh-v1.5 --embedding-device cuda
```

```bash
python run.py --analyse-only --refresh-merge --merge-method embedding --embedding-backend local --embedding-model BAAI/bge-base-zh-v1.5 --embedding-threshold 0.2 --embedding-device cuda --top-n 100
```

```bash
python run.py --analyse-only --refresh-merge --merge-method embedding --embedding-backend local --embedding-model BAAI/bge-small-zh-v1.5 --embedding-threshold 0.2 --top-n 100 --name- source step1
```

### 4c. 用 **step1 / step2 全文** 做 embedding 聚类（`--name-source`）

```bash
python run.py --analyse-only --merge-method embedding --embedding-backend local --embedding-model BAAI/bge-small-zh-v1.5 --name-source step2 --embedding-threshold 0.3
```

指定某张卡：`--embedding-device cuda:0`。强制 CPU：`--embedding-device cpu`。不写 `--embedding-device` 时与 `.env` 里 `EMBEDDING_DEVICE` 一致；都未设置则为 **`auto`**（检测到 CUDA 即用 GPU）。在 **Mac** 上无 CUDA 时可用 `--embedding-device mps`（Apple Silicon）或 `cpu`。

### 5. 不归并，直接输出原始结果（柱状图 + 词云）

```bash
python run.py --analyse-only --merge-method none
```

### 6. 强制重新归并并出图

```bash
python run.py --analyse-only --refresh-merge --merge-method embedding --embedding-backend local
```

### 7. 指定输入输出路径

```bash
python run.py --backend openai -i E:\data\questions.jsonl -o E:\data\out.jsonl -n 200 --concurrency 8
```

## 并发与「卡住」的说明

- 并发时：先出现多条 `[start …]`，每条题完成 3 轮后才会出现 `[done …]`；**第一条 `[done]` 前可能安静数分钟**，属正常。  
- 若长时间无任何 `[start]`，请检查网络、Key、`OPENAI_BASE_URL` 与模型名是否与服务商一致。

## 依赖与可选组件

- 必选：`requirements.txt` 中基础包。  
- **词云**：`wordcloud`。  
- **embedding 本地**：`sentence-transformers`；**`torch` / `torchvision` / `torchaudio` 请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 与 CUDA 一次性装齐**，勿在 `requirements.txt` 里随意单升 `torch`。  
- **UMAP 散点图**：`umap-learn`。  
- **Ollama 嵌入**：本机已启动 Ollama 且已 `pull` 对应模型。
