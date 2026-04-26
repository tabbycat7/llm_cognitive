# `batch_llm_merge.py` CLI 使用说明

在仓库根目录下执行（或把 `python` 指向可解释器 / 已激活的虚拟环境）。  
脚本会扫描 `model/`（或 `--root`）下所有 `probe_results.jsonl`，对每个文件依次执行与 `run.py --analyse-only` 相同的流程：`analyse` → `visualise` → `write_analysis_summary_log`。

---

## 1. 基本命令

```text
python batch_llm_merge.py [选项]
```

查看全部参数与模块内嵌示例：

```text
python batch_llm_merge.py -h
```

**默认（未加 `--verbose`）**：会设置 `COGNITIVE_BATCH_QUIET=1`，子任务里 `[merge]` / `[plots]` 等行不刷屏；主进程用 **tqdm 进度条** 显示已完成的 `probe_results.jsonl 数量`（单线程为 `file`，多线程为 `job`），postfix 显示当前/刚完成的任务路径与耗时。加 **`-v` / `--verbose`** 则关闭该安静模式，并**不再使用**主进度条，恢复逐任务横幅与详尽情输出。

**Windows PowerShell**：链式命令请用分号 `;` 而不是 `&&`。

---

## 2. 主要选项一览

| 选项 | 说明 |
|------|------|
| `--root PATH` | 扫描根目录，默认：脚本同级的 `model/`。 |
| `--models DIR [DIR ...]` | 只处理 `root` 下第一级子目录名匹配的模型（如 `deepseek`）。 |
| `--categories CAT [CAT ...]` | 只处理指定分类子目录，可用 taxonomy 形式如 `"[Personal & Existential]"`，或与磁盘名一致如 `Relational&Intimate`。 |
| `--include-root-file` | 若 `root/probe_results.jsonl` 存在也处理（在指定了 `--models`/`--categories` 时无效）。 |
| `--exclude-dir NAME` | 路径中任一段含该目录名则跳过，可重复。与内置排除列表（含 `.git`、`__pycache__` 等）取并集。 |
| `--merge-method METHOD` | 同义归并方式，见下节。默认 `llm`。 |
| `--refresh-merge` | 忽略已有 `merge_map` 等缓存，强制重新归并/请求。 |
| `--top-n N` | 图表与日志中展示的 Top-N，默认 30。 |
| `--name-source` | `step1` / `step2` / `step3` / `step4`，与 `run.py` 一致，默认 `step3`。多数 LLM 类 merge 不可与 `step1`/`step2` 同用。 |
| `--embedding-backend` | `local` / `ollama` / `openai`，默认读环境变量 `EMBEDDING_BACKEND`，否则 `local`。 |
| `--embedding-model` | 嵌入模型名，默认 `EMBEDDING_MODEL` 或 `BAAI/bge-small-zh-v1.5`。 |
| `--embedding-device` | 仅本地 embedding：如 `auto`、`cuda`、`cpu`；默认 `EMBEDDING_DEVICE` 或 `auto`。 |
| `--embedding-threshold` | 余弦聚类距离阈值 (0–1)，默认 0.3。 |
| `--llm-naming-workers N` | `embedding_llm` / `embedding_llm_llm` 下，为每簇取名的**并发** LLM 调用数，默认 8。 |
| `-j`, `--jobs N` | **同时处理多少个** `probe_results.jsonl` 任务（线程池），默认 1。大并发易触发限流。 |
| `--dry-run` | 只打印将处理的文件路径，不调用 API、不写输出。 |
| `-v`, `--verbose` | 详细控制台输出；默认较安静。 |
| `--continue-on-error` / `--no-continue-on-error` | 单文件失败后是否继续（默认可继续）。 |
| `--export-models-summary-json PATH` | 批处理结束后，额外写一个汇总 JSON（每模型×分类的 Top-K 名称与频次）。 |
| `--export-models-summary-phase` | `after_merge`（归并后）或 `before_merge`（归并前 raw），默认 `after_merge`。 |
| `--export-models-summary-top-k K` | 汇总中每个条目的前 K 个名，默认 50。 |
| `--export-models-summary-merge` | 若汇总文件已存在，按 `probe_results` 键合并/更新，而不是整盘重写。 |
| `--export-models-summary-only` | 与导出 JSON 联用：只跑 `analyse` 计数并写汇总，**不**画图、**不**写 `analysis_summary.log`（更快）。 |

---

## 3. `--merge-method` 说明

| 值 | 含义 |
|----|------|
| `llm` | 仅用 LLM 对高频名称做同义归并（默认 `merge_map.json` 为缓存目标）。 |
| `embedding` | 向量聚类，无 per-cluster 取名 LLM。 |
| `embedding_llm` | 先 embedding 聚类，再**并发**为每个簇调用 LLM 命名；会生成 `merge_map_llm_stage1` 等产物及 `embedding_llm_cluster.log` 等。 |
| `embedding_llm_llm` | 在 `embedding_llm` 基础上再做**第二轮** LLM 把 Stage1 的 canonical 再归并；需 Stage1+Stage2+最终 `merge_map.json` 的完整流程。 |
| `embedding_llm_llm_stage2` | 不跑 embedding/Stage1；**读**已有 `merge_map_llm_stage1.json`，只跑 Stage2 LLM，并写 `merge_map_llm_stage2.json` 与最终 `merge_map.json`。 |
| `embedding_llm_llm_stage1_final` | 不跑 Stage2；**只**用 `merge_map_llm_stage1.json` 写入最终 `merge_map.json`（无 Stage2 API）。与下项等价。 |
| `embedding_llm_stage1_final` | **同上**的短名/别名，解析后与 `embedding_llm_llm_stage1_final` 相同。 |
| `none` | 不做归并，原标签计数。 |

> **Stage2 专用模型**（仅影响 `embedding_llm_llm` / `embedding_llm_llm_stage2` 的第二次 LLM）：在 `.env` 中可设 `LLM_MODEL_MERGE_STAGE2`、`OPENAI_BASE_URL_MERGE_STAGE2`、`OPENAI_API_MERGE_STAGE2_KEY`；若都不设，与合并阶段默认的 `LLM_MODEL_MERGE` 等相同。

---

## 4. 与 `.env` 的常见关系

归并用 OpenAI 兼容端点（与探测阶段可分离），典型变量：

- `OPENAI_API_MERGE_KEY` / `OPENAI_BASE_URL_MERGE` / `LLM_MODEL_MERGE`  
- 回退：`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL`

嵌入用：

- `EMBEDDING_BACKEND` / `EMBEDDING_MODEL` / `EMBEDDING_DEVICE`（及 API 用 key、base 等，见项目内 `analyze_results` 与 `llm_api`）

脚本启动时会加载**仓库根目录**下的 `.env`。

---

## 5. 每个 `probe_results.jsonl` 同目录下常见输出

与 `merge-method` 相关，可能包括：

- `merge_map.json` — 最终 **raw → canonical** 映射（三阶段方案中为合成结果）。
- `merge_map_llm_stage1.json` — embedding+簇命名后的 Stage1。
- `merge_map_llm_stage2.json` — 第二轮 LLM 归并映射（canonical→canonical 侧，见 `analyze_results` 逻辑）。
- `merge_map_llm_stage1.embeddings.json` — 嵌入缓存。
- `cluster_dendrogram.png`、`cluster_scatter.png` — 聚类可视化（若对应流程生成）。
- `embedding_llm_cluster.log` — 混合 `embedding_llm` 系方法的聚类/簇详情日志（若已生成该流程）。
- `analysis_summary.log`、`model_law_frequency.png`、`model_law_wordcloud.png` 等分析输出。

---

## 6. 命令示例

```text
# 全库顺序执行，默认 LLM 归并
python batch_llm_merge.py

# 28 个任务并行；混合归并，刷新缓存，12 个并发簇命名
python batch_llm_merge.py --merge-method embedding_llm --refresh-merge -j 28 --llm-naming-workers 12

# 三阶段归并，阈值 0.35
python batch_llm_merge.py --embedding-threshold 0.35 --merge-method embedding_llm_llm --refresh-merge -j 2 --llm-naming-workers 20

# 只重跑 Stage2（需已有 merge_map_llm_stage1.json）
python batch_llm_merge.py --merge-method embedding_llm_llm_stage2 --refresh-merge -j 4

# 以 Stage1 为最终表，不调 Stage2（下两行等价）
python batch_llm_merge.py --merge-method embedding_llm_llm_stage1_final -j 4
python batch_llm_merge.py --merge-method embedding_llm_stage1_final -j 4

# 指定模型 + 分类 + 写 merged_top50
python batch_llm_merge.py --models anthropic-claude-haiku-4.5 --categories "Professional&Economic" --export-models-summary-json merged_top50.json --export-models-summary-phase after_merge
```

在 PowerShell 中若要拆行，用反引号 `` ` `` 作行续符；或保持单行命令最省事。

---

## 7. 退出与失败

- 默认遇单文件错误会记录并继续，最后报告失败列表。  
- `--no-continue-on-error` 在首次失败时停止。  
- 具体退出码以脚本 `main()` 为准，可用 `python batch_llm_merge.py; echo $LASTEXITCODE` 在 PowerShell 中查看。

若需**仅**频率汇总、跳过绘图与长日志，使用 `--export-models-summary-json` 与 `--export-models-summary-only` 组合，并按需选 `--merge-method none`。

---

## 8. 配套脚本 `analyze_merge_stages.py`（Stage1/Stage2 归并分析）

在含 `merge_map_llm_stage1.json` 与 `merge_map_llm_stage2.json` 的目录下生成 `merge_stages_analysis.log`（`analyse` 在批处理中也会自动写，本脚本可事后全量重跑）：

```text
python analyze_merge_stages.py --all
python analyze_merge_stages.py --all --root model --use-probe
```

`--all` 会扫描与 `batch_llm_merge` 相同的 `model` 根目录，并复用其路径排除规则；`--use-probe` 会在每个子目录有 `probe_results.jsonl` 时加入按频次加权的统计。

---

*文档与 `batch_llm_merge.py` 行为以仓库内实现为准；参数若有增减，以 `python batch_llm_merge.py -h` 输出为准。*
