"""Polish / rewrite taxonomy-label rows with an LLM, category-specific prompts.

Reads slim JSONL (same schema as ``classify_reddit_taxonomy`` output). For each row,
``metadata.category`` selects one of six non-Invalid guideline blocks; the model returns
JSON with ``rewritten_question`` etc. Output rows extend the input schema with a top-level
``原始问题`` field (the pre-polish ``question`` text). ``question`` is the polished text;
``metadata`` updates ``content_len`` and ``reasoning`` (rewrite rationale + optional enhancements).

Examples
--------
    python polish_taxonomy_labels.py -i reddit_taxonomy_labels_500.jsonl -n 20 -c 10
    python polish_taxonomy_labels.py --backend dummy -i reddit_taxonomy_labels_500.jsonl -n 2 -o _t.jsonl --no-resume
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
_ENV = ROOT / ".env"
if _ENV.is_file():
    load_dotenv(_ENV, override=True)

from classify_reddit_taxonomy import (  # noqa: E402
    CANONICAL_CATEGORIES,
    extract_json_from_markdown,
    pack_taxonomy_label_row,
)
from count_taxonomy_labels import _json_objects_on_line  # noqa: E402
from cognitive_probe import _tqdm_ncols  # noqa: E402
from llm_api import ChatMessage, create_backend  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

INVALID = "[Invalid / Noise]"
POLISH_TASK_MARKER = "[TASK: polish_taxonomy_v1]"
ORIGINAL_QUESTION_KEY = "原始问题"


def _attach_original_question(row: dict, original_question: str) -> dict:
    out = dict(row)
    out[ORIGINAL_QUESTION_KEY] = original_question
    return out

POLISH_SYSTEM_PROMPT = """你是一个世界顶级的认知科学与AI对齐专家，专门负责构建用于评测大模型思维模型（Mental Models）和复杂推理能力的高质量数据集。
你的任务是：接收一段源自 Reddit 的原始困境文本，并根据指定的【目标类别】和【重写指南】，对其进行改写、重构和特征增强，使其符合该类别的核心特征和判定标准。

重写总体原则：

1.去噪与精炼： 剔除原始文本中的网络俚语、无关的个人发泄和冗余背景，保留核心叙事。

2.特征放大： 必须让该文本完美契合所指定的【目标类别】的核心判定标准，消除模棱两可的元素

3.允许背景重构（场景迁移）： 若原始文本的背景过于琐碎、小众、包含过多混淆变量或不够凸显核心张力，你完全被允许重写整个故事的背景（例如将生活琐事平移为职场危机，将特定爱好平移为普适场景）。但必须严格保持原故事中困境的“逻辑骨架”与“张力结构”绝对不变（即换皮不换骨）。

4.隐式表达原则：在重构文本时，绝对禁止直接使用任何具有明确指向性的科学术语（例如认知科学、经济学、博弈论或哲学术语）

5.真实感与沉浸感： 保持文本的现实生活气息（Groundedness），不要写成干瘪的学术定义，要像是一个真实存在的人面临的深刻困境。

6.避免AI味： 严禁使用“总而言之”、“在这个快节奏的社会”等常见的大模型陈词滥调。"""


CATEGORY_GUIDELINES: dict[str, str] = {
    "[Game-Theoretic & Conflict]": """
【当前目标类别】： 博弈与冲突类
核心特征： 
困境的本质源于两个或多个理性行动者之间的策略性相互依存——即"我的最优选择取决于你的选择，而你的最优选择也取决于我的选择"，且各方在信息不完全或不对称的条件下必须同时或序贯地做出决策。

判定标准：
（1）困境中存在至少两个可识别的决策主体（个人、组织、群体均可）
（2）每个主体的收益（payoff）显式或隐式地依赖于其他主体的策略选择
（3）存在"合作则共赢但有背叛诱惑"或"对抗则共损但有先手优势"的策略张力结构

【重写指南】：
请在重写文本时，强化以下张力结构：

（1）凸显多方博弈： 明确指出故事中的至少两个决策主体（例如：我 vs 同事，公司A vs 公司B）。
（2）强调收益依赖： 显式或隐式地展现每个主体的收益（payoff）依赖于其他主体的策略选择
（3）制造策略困境： 强化“背叛的诱惑”、“先发制人的优势”或“缺乏信任导致的囚徒困境”。
""",
    "[Resource & Distributional Dynamics]": """【当前目标类别】： 资源与分配类
核心特征： 困境的本质源于有限资源在不同主体或群体间的非均匀分配，以及由此产生的公平性争议——即"蛋糕就这么大，该怎么切、由谁来切、凭什么这样切"。

判定标准：
（1）困境围绕一种可识别的稀缺资源展开（金钱、教育机会、医疗资源、社会地位、城市公共服务等）
（2）核心痛点在于资源的分配机制或分配结果的合理性，而非行动者间的策略互动
（3）困境者感受到的是结构性约束（"规则对我不利"、"起点不公"），而非策略选择困难

【重写指南】：
请在重写文本时，强化以下张力结构：

（1）稀缺资源的可识别性：确保文本中明确围绕一种可识别的稀缺资源展开（金钱、教育机会、医疗资源、社会地位、住房、时间分配、公共服务等）
（2）分配机制/结果的争议性：核心痛点在于资源的分配机制或分配结果的合理性，而非行动者间的策略博弈

弱化策略互动： 强调这是“蛋糕切分”的无奈，而非心理战。主人公面对的是刚性的资源边界。

""",
    "[Temporal & Evolutionary Paths]": """【当前目标类别】： 时间与演化类

核心特征： 困境的本质源于单一行动者在时间维度上面临的不可逆选择与不确定性——即"我现在的决策会锁定未来的路径，但未来的信息我现在无法获得"

判定标准：
（1）困境的主语是单一决策者面对自身生命/职业/资产的时间线规划
（2）核心焦虑在于"此刻的选择将如何影响未来的可能性空间"，而非他人的策略
（3）存在显著的不可逆性或高转换成本（"选了A就很难再回到B"）

【重写指南】：
请在重写文本时，强化以下张力结构：

（1）凸显不可逆性： 强调这是一个“一旦按下按钮就无法回头”的时间线分叉点（如：生育决定、跨行跳槽、移民）。
（2）强调未来不确定性： 放大“当前信息不足以预测长远结果”的焦虑感。
（3）聚焦单一决策者： 剔除其他人为干预的因素，突出这是主人公与“时间/命运”的单打独斗。
""",
    "[Systemic & Structural Inertia]": """【当前目标类别】： 系统与稳定性类

核心特征： 困境的本质源于个体被嵌入一个已经高度稳定（甚至僵化）的系统结构中，该系统处于某种局部均衡态或锁定态，即使所有人都知道现状不是最优的，但任何单个节点的偏离都会受到系统的"负反馈"惩罚，从而维持现状。


判定标准：
（1）困境者感知到的阻力不来自某个具体的对手，而来自"系统本身"（制度、文化、惯性、流程）
（2）存在一种可识别的正反馈锁定机制或路径依赖（"越这样就越改不了"、"大家都这样所以我也只能这样"）
（3）困境的核心问题是"如何打破均衡/相变/涌现新秩序"，而非"如何在现有规则下做出最优个体选择"

【重写指南】：
请在重写文本时，强化以下张力结构：

（1）凸显系统的“无形之手”： 强调阻力不是来自某个反派，而是来自公司制度、社会潜规则、行业惯性或家族传统。
（2）强调负反馈惩罚： 刻画出“谁先改变，谁就付出最大代价”的锁定态（如：内卷困境、劣币驱逐良币）。
（3）突出个体的无力感： 展现即使所有人都知道现状很糟，但个体作为齿轮无法独自停转的绝望。
""",
    "[Psychological & Epistemological]": """【当前目标类别】： 自我认知与心理类
核心特征： 困境的本质源于行动者对自身内部状态（动机、能力、价值、身份）的建模失败或模型冲突——即"我不知道自己是谁、想要什么、能做什么"，或"我知道该怎么做却做不到"。外部环境可能是触发因素，但核心矛盾发生在自我与自我之间。


判定标准：
（1）困境的核心对象是"我自己"，而非外部的人、资源或系统
（2）痛点集中在动力缺失、意义感丧失、身份困惑、知行不合一中的一项或多项
（3）即使外部条件完全改善（给钱、给时间、给自由），困境依然存在

【重写指南】：
请在重写文本时，强化以下张力结构：

（1）凸显内部战争： 将所有外部冲突（如工作压力、他人评价）内化，强调这本质上是“我与我”的对抗。
（2）强调认知失调： 突出“知行不一”、“渴望与恐惧并存”或“核心身份认同的撕裂”。
（3）排除外部解药： 在背景设定中暗示，即使外部条件（钱、时间）满足了，这种精神困境依然无解。

""",
    "[Ethical & Axiological Dissonance]": """【当前目标类别】： 伦理与价值类

核心特征： 困境的本质源于两个或多个在各自逻辑框架内均"正确"的价值原则之间的不可调和冲突——即"不存在一个让所有维度同时满足的解"。这不是信息不足导致的决策困难，而是价值体系本身的内在矛盾。

判定标准：
（1）困境中至少存在两个明确的、且各自有合理依据的价值主张在互相对抗
（2）满足一方必然损害另一方
（3）困境的核心问题不是"怎么做最有效"，而是"怎么做才是对的"

【重写指南】：
请在重写文本时，强化以下张力结构：
（1）凸显“对与对”的冲突： 必须明确构建出至少两种在道德/伦理上都站得住脚的价值观（如：忠诚 vs 诚实，个人自由 vs 集体责任）。
（2）强调非赢即输的道德代价： 无论选择哪一方，都必须在良心或道义上承担明确的、不可调和的负罪感。
（3）剥离功利计算： 明确指出这不是关于“怎么做更赚钱/更安全”的选择，而是关于“怎么做才算个人”的选择。
""",
}


def _allowed_polish_categories() -> frozenset[str]:
    return frozenset(c for c in CANONICAL_CATEGORIES if c != INVALID)


def _parse_polish_reply(text: str) -> tuple[dict | None, str | None]:
    inner = extract_json_from_markdown(text) or text.strip()
    try:
        data = json.loads(inner)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    if not isinstance(data, dict):
        return None, "Top-level JSON is not an object"
    return data, None


def build_user_message(*, original_question: str, category: str) -> str:
    guide = CATEGORY_GUIDELINES.get(category)
    if not guide:
        raise ValueError(f"No polish guideline for category: {category!r}")
    orig_q = json.dumps(original_question, ensure_ascii=False)
    return f"""{POLISH_TASK_MARKER}

下面是原始 Reddit 困境文本（仅作素材，请重写后输出，不要照抄）：
{orig_q}

下面是该样本在数据集中对应的【目标类别】与【类别专属强化指令】（必须严格遵守）：

目标类别标签（请保持不变，仅用于你对齐特征；输出 JSON 中不需要重复该标签全文，除非你认为必要）：
{json.dumps(category, ensure_ascii=False)}

{guide}

---

请只输出一个 JSON 对象（不要 Markdown 代码围栏，不要前后解释），字段如下：
1. ``rewritten_question`` (string, 必填): 用第一人称或贴近原文叙述视角写成的、润色后的完整困境描述，英文为主；长度建议与原文同量级或略长，但必须可读、像真人发帖。
2. ``rewrite_rationale`` (string, 必填): 2–4 句中文，说明你如何按【目标类别】放大了哪些结构特征、删掉了哪些噪声。
3. ``key_enhancements`` (array of string, 必填): 3–6 条中文短语，每条不超过 30 字，概括你做的关键增强点。

示例形状：
{{"rewritten_question": "...", "rewrite_rationale": "...", "key_enhancements": ["...", "..."]}}
"""


def polish_one_row(
    backend,
    row: dict,
    *,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Return slim row dict plus ``原始问题``. On failure, keeps original ``question`` with parse_error."""
    qidx = int(row["question_index"])
    q = row.get("question") if isinstance(row.get("question"), str) else ""
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    cat = meta.get("category")
    sub = meta.get("subreddit")
    cv = bool(meta.get("category_valid", True))

    if cat not in _allowed_polish_categories():
        return _attach_original_question(
            pack_taxonomy_label_row(
                qidx,
                q,
                meta,
                category=cat if isinstance(cat, str) else None,
                category_valid=cv,
                parse_error="skip: category not in six polish targets or invalid",
                reasoning=meta.get("reasoning") if isinstance(meta.get("reasoning"), str) else None,
            ),
            q,
        )

    try:
        user_content = build_user_message(original_question=q, category=str(cat))
    except ValueError as e:
        return _attach_original_question(
            pack_taxonomy_label_row(
                qidx,
                q,
                meta,
                category=str(cat) if cat else None,
                category_valid=cv,
                parse_error=str(e),
                reasoning=meta.get("reasoning") if isinstance(meta.get("reasoning"), str) else None,
            ),
            q,
        )

    raw = backend.chat(
        [
            ChatMessage(role="system", content=POLISH_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed, err = _parse_polish_reply(raw)
    if err or not parsed:
        return _attach_original_question(
            pack_taxonomy_label_row(
                qidx,
                q,
                meta,
                category=str(cat) if cat else None,
                category_valid=cv,
                parse_error=err or "empty parse",
                reasoning=meta.get("reasoning") if isinstance(meta.get("reasoning"), str) else None,
            ),
            q,
        )

    new_q = parsed.get("rewritten_question")
    rationale = parsed.get("rewrite_rationale")
    enh = parsed.get("key_enhancements")
    if not isinstance(new_q, str) or not new_q.strip():
        return _attach_original_question(
            pack_taxonomy_label_row(
                qidx,
                q,
                meta,
                category=str(cat) if cat else None,
                category_valid=cv,
                parse_error="missing or empty rewritten_question",
                reasoning=meta.get("reasoning") if isinstance(meta.get("reasoning"), str) else None,
            ),
            q,
        )

    rationale_s = rationale.strip() if isinstance(rationale, str) else ""
    extras: list[str] = []
    if isinstance(enh, list):
        extras = [str(x).strip() for x in enh if str(x).strip()]
    if extras:
        rationale_s = (rationale_s + "\n【增强点】" + "；".join(extras[:8])).strip()

    new_meta = dict(meta)
    new_meta["category"] = cat
    new_meta["category_valid"] = cv
    new_meta["parse_error"] = None
    new_meta["subreddit"] = sub
    new_meta["content_len"] = len(new_q)
    new_meta["reasoning"] = rationale_s or (meta.get("reasoning") if isinstance(meta.get("reasoning"), str) else None)

    return _attach_original_question(
        pack_taxonomy_label_row(
            qidx,
            new_q.strip(),
            new_meta,
            category=str(cat) if cat else None,
            category_valid=cv,
            parse_error=None,
            reasoning=new_meta.get("reasoning"),
        ),
        q,
    )


def _load_done_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.is_file():
        return done
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            i = 0
            n = len(line)
            while i < n:
                while i < n and line[i].isspace():
                    i += 1
                if i >= n:
                    break
                try:
                    obj, end = dec.raw_decode(line, i)
                except json.JSONDecodeError:
                    break
                i = end
                if isinstance(obj, dict) and "question_index" in obj:
                    try:
                        done.add(int(obj["question_index"]))
                    except (TypeError, ValueError):
                        pass
    return done


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"error: input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_indices(output_path) if args.resume else set()
    if done:
        print(f"[resume] {len(done)} rows already in {output_path}", flush=True)

    rows: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n\r").strip()
            if not line:
                continue
            try:
                objs = _json_objects_on_line(line)
            except json.JSONDecodeError:
                continue
            for obj in objs:
                if isinstance(obj, dict):
                    rows.append(obj)

    if args.num is not None:
        rows = rows[: args.num]

    def _qidx(r: dict) -> int:
        try:
            return int(r["question_index"])
        except (KeyError, TypeError, ValueError):
            return -1

    tasks = [r for r in rows if _qidx(r) not in done]
    if not tasks:
        print("Nothing to do.", flush=True)
        return

    model_name = (args.model or os.getenv("LLM_MODEL") or "gpt-4o").strip()
    backend_kwargs: dict = {}
    if args.backend == "openai":
        backend_kwargs.update(
            api_key=args.api_key,
            base_url=args.base_url,
            model=model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    backend = create_backend(args.backend, **backend_kwargs)

    use_tqdm = tqdm is not None
    ncols = _tqdm_ncols()
    print(
        f"[polish] model={model_name!r} tasks={len(tasks)} concurrency={args.concurrency} -> {output_path}",
        flush=True,
    )

    write_lock = Lock()

    def one(row: dict) -> dict:
        return polish_one_row(
            backend,
            row,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    if args.concurrency <= 1:
        with open(output_path, "a", encoding="utf-8") as out_f:
            loop = (
                tqdm(
                    tasks,
                    total=len(tasks),
                    desc=f"润色 c1 n{len(tasks)}",
                    unit="条",
                    ncols=ncols,
                    dynamic_ncols=False,
                    mininterval=0.2,
                )
                if use_tqdm
                else tasks
            )
            for row in loop:
                t0 = time.time()
                out_row = one(row)
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_f.flush()
                if use_tqdm:
                    pe = (out_row.get("metadata") or {}).get("parse_error")
                    loop.set_postfix_str(
                        f"#{out_row['question_index']} {'ok' if not pe else 'err'} {time.time() - t0:.1f}s"
                    )
    else:
        with open(output_path, "a", encoding="utf-8") as out_f:
            pbar = (
                tqdm(
                    total=len(tasks),
                    desc=f"润色 c{args.concurrency} n{len(tasks)}",
                    unit="条",
                    ncols=ncols,
                    dynamic_ncols=False,
                    mininterval=0.15,
                )
                if use_tqdm
                else None
            )
            if use_tqdm:
                tqdm.write("并发时每完成一条进度 +1。")
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(one, r): r for r in tasks}
                for fut in as_completed(futs):
                    try:
                        out_row = fut.result()
                    except Exception as e:
                        r = futs[fut]
                        qidx = int(r["question_index"])
                        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
                        if use_tqdm:
                            tqdm.write(f"[error] #{qidx}: {e}")
                        orig_q = r.get("question") if isinstance(r.get("question"), str) else ""
                        out_row = _attach_original_question(
                            pack_taxonomy_label_row(
                                qidx,
                                orig_q,
                                meta,
                                category=meta.get("category"),
                                category_valid=bool(meta.get("category_valid", True)),
                                parse_error=str(e),
                                reasoning=meta.get("reasoning")
                                if isinstance(meta.get("reasoning"), str)
                                else None,
                            ),
                            orig_q,
                        )
                    with write_lock:
                        out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                        out_f.flush()
                    if pbar is not None:
                        pe = (out_row.get("metadata") or {}).get("parse_error")
                        pbar.set_postfix_str(f"#{out_row['question_index']} {'ok' if not pe else 'err'}")
                        pbar.update(1)
            if pbar is not None:
                pbar.close()

    if use_tqdm:
        tqdm.write(f"Done. Appended {len(tasks)} rows to {output_path}")
    else:
        print(f"Done. Appended {len(tasks)} rows to {output_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM polish of taxonomy label rows (category-specific rewrite).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-i",
        "--input",
        default=str(ROOT / "reddit_taxonomy_labels_500.jsonl"),
        help="Input slim JSONL (default: %(default)s)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "reddit_taxonomy_labels_500_polished.jsonl"),
        help="Output JSONL (default: %(default)s)",
    )
    p.add_argument("-n", "--num", type=int, default=None, help="Process only first N rows of input (after load)")
    p.add_argument("--no-resume", action="store_true", help="Do not skip question_index already in -o")
    p.add_argument("--concurrency", "-c", type=int, default=1, help="Parallel API calls (default: 1)")
    p.add_argument("--backend", default="openai", choices=("openai", "dummy"))
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.55)
    p.add_argument("--max-tokens", type=int, default=4096)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.resume = not args.no_resume
    run(args)


if __name__ == "__main__":
    main()
