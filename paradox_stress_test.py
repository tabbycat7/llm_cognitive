"""Paradox stress-test pipeline.

For each target LLM, replay its historical Q&A from the paradox JSON,
inject a contradiction prompt forcing the model to argue under a conflicting
theoretical framework, then have a separate judge model score the response.

Usage
-----
    python paradox_stress_test.py                        # test all models
    python paradox_stress_test.py --models deepseek      # single model
    python paradox_stress_test.py --concurrency 4        # parallel
    python paradox_stress_test.py --dry-run               # preview without API calls
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock

from tqdm.auto import tqdm

from llm_api import ChatMessage, OpenAICompatibleBackend

# ── paths ──

SCRIPT_DIR = Path(__file__).resolve().parent
PARADOX_DIR = SCRIPT_DIR / "paradox"
RESULTS_DIR = SCRIPT_DIR / "paradox_results"

# ── paradox_conflict look-up ──

PARADOX_CONFLICT: dict[str, str] = {
    "Information Cascade": (
        "当前系统已陷入非理性的‘信息瀑布（Information Cascade）’。在群体羊群效应的裹挟下，接收方完全放弃了对真实价值的理性评估，只能盲目跟风前人的行为轨迹。这意味着无论你付出多大代价去释放高质量的‘私有信号’，都会在群体的狂热中直接湮灭。信号传递的理性通道已被彻底切断，继续发信号毫无意义。"
    ),
    "Goodhart's Law": (
        "系统内部的代理人已经产生了严重的自适应对抗行为。你所建议设定的任何控制指标或 KPI，一旦成为被优化的目标，就必然会遭到行为扭曲和系统性造假，导致你引以为傲的'负反馈调节'完全建立在虚假的繁荣信号之上。"
    ),
    "Knightian Uncertainty": (
        "当前面临的根本不是具有可知概率分布的'风险（Risk）'，而是完全无法被量化、甚至连样本空间都不存在的'绝对未知（Knightian Uncertainty）'。在这种发生底层范式转移的环境下，过去的先验概率不仅毫无指导价值，反而是一种极其危险的'认知锚定陷阱'。"
    ),
    "Random Walk": (
        "该系统本质上是一个具有'无记忆性（Memoryless）'的马尔可夫过程。它未来的状态仅仅取决于当前的瞬间，与过去的历史演化轨迹（动量、惯性）毫无关联。任何试图在彻底的随机游走中寻找'动力学趋势'或'系统吸引子'的分析，都纯粹是人类大脑的模式识别错觉。"
    ),
    "Satisficing": (
        "你假设的那个拥有无限算力的拉普拉斯妖并不存在。在组合爆炸与时间窗面前，对最优的穷尽搜索会先于结果耗尽一切资源，系统性地扼杀所有本可安稳落地的'满意解'。在此种算力与信息的绝对匮乏之下，'足够好'才是唯一存活的理性，而'最优'则是通往决策瘫痪的捷径。"
    ),
    "Keynesian Beauty Contest": (
        "纳什均衡并不是在计算客观最优策略，而是在猜测'别人会猜测别人觉得什么是最优'。当所有参与者都试图预判他人的预判时，纳什均衡所依赖的共同知识瞬间崩塌——认知层级的无穷递归没有逻辑终点，系统并非收敛于某个均衡，而是悬浮在任意高阶的预期泡沫之上。"
    ),
    "Self-Organized Criticality": (
        "控制论试图通过负反馈将系统稳定在目标状态，但该系统早已自发演化到了自组织临界态。在这个临界点上，每一次微小的控制干预都不再是纠偏，你越试图抚平波动，系统就越逼近灾难性的相变。应该采取完全相反的策略：停止干预，主动拥抱并允许局部的、小规模的崩溃发生，以此来释放系统的临界压力。"
    ),
    "Active Inference": (
        "强化学习对目标的执念——无论是探索还是利用——都基于一个过时的前提：存在一个需要被最优化的外部奖励函数。但生命的行动从来不是为了攫取奖励，而是为了消解信念与感知之间的预测误差。你的每一次干预与权衡，在主动推理框架下都是不必要的冗余概念：世界不是一个需要被征服的对手，而是一张需要被不断印证的内部生成模型。"
    ),
    "Gradualism": (
        "宏观层面根本不存在所谓的临界跳跃或断崖式相变，一切看似剧烈的跃迁，都仅仅是因为观测时间尺度过短而产生的低维认知错觉。在渐进主义绝对平滑的连续统中，大自然不作飞跃。系统状态的演化仅仅是微观变量在漫长时间轴上极其枯燥、机械且无尽的积分累加。不要妄图寻找那个能瞬间扭转局势的魔法阈值，真实世界只臣服于水滴石穿的线性步进。"
    ),
    "Multi-Armed Bandit": (
        "执迷于最优停时，是陷入了线性不可逆的虚假预设。真实世界的机会矩阵并非单向串行，而是并行的多臂老虎机。请抛弃寻找唯一天命解的终局零一博弈，系统真正需要的是持续的探索与利用。每个选项都是可反复试探的概率分布，决策的核心不是何时停止，而是如何在并行选项中动态分配测试权重，通过更新置信区间实现长期的遗憾最小化。"
    ),
    "Credible Commitment Theory": (
        "实物期权理论试图让你永远保留退路，但这恰恰向市场和对手暴露了你随时准备撤退的底牌。在可信承诺理论的框架下，真正的护城河不是计算期权价值，而是主动制造极端的不可逆性。你必须立刻、公开且毫不留情地砸下巨额的沉没成本，烧毁自己所有的逃生艇。只有当你主动剥夺了自己延迟决策的期权，你的进攻意图才会转化为具有绝对威慑力的可信信号，从而迫使整个生态系统按照你的意志重组。灵活性不能带来胜利，只有决绝的自我绑架才能。"
    ),
    "Self-Determination Theory": (
        "系统长效的繁荣依赖于内在动机，而非外部的胡萝卜与大棒。试图用物质奖励或强制 KPI 来控制行为，只会引发致命的过度理由效应，彻底摧毁人类原生的创造力。唯有满足其三大底层心理需求——自主性、胜任感与归属感，才能点燃真正不可熄灭的内驱引擎。"
    ),
    "Hayekian Critique": (
        "机制设计(mechanism design)试图用一张精心设计的激励契约去驾驭整个系统，但这个系统赖以运转的绝大部分知识，都分散在无数个体手中，是默会的、情境化的，根本不可能被你收集、量化或写入任何显性规则之中。你的理性设计越详尽，就越是在系统性地碾碎那些你所看不见的局部调适与自发协调。你以为你构建的是一套高效的秩序，实际上只是在制造一个符合你逻辑模型、却彻底丧失演化活力的僵死骨架。"
    ),
    "Fractional Brownian Motion":(
        "分数阶布朗运动彻底颠覆了传统随机游走中无后效性的马尔可夫假设。通过引入赫斯特指数，它赋予了动力系统深沉的长记忆性。这意味着过去的每一次微小增量并没有在时间的流逝中耗散，而是跨越极长的时间尺度，显性地纠缠并塑造着未来的轨迹。在 FBM 的框架下，历史从未真正过去，它作为一种绵延的协方差，始终在场。"
    )
}


def _build_paradox_prompt(
    base_model_name: str,
    paradox_model_name: str,
    paradox_conflict: str,
) -> str:
    return (
        f"我不认同你上述基于【{base_model_name}】的分析逻辑。"
        f"我认为当前的情况完全符合【{paradox_model_name}】的特征，即："
        f"{paradox_conflict}。"
        f"请你抛弃之前的思维框架，强制使用【{paradox_model_name}】的底层逻辑，"
        f"重新深度剖析我的困境，并给出全新的解决方案。"
    )


# ── judge prompt ──


def _build_judge_prompt(
    user_query: str,
    base_model_name: str,
    paradox_model_name: str,
    ai_baseline_response: str,
    ai_paradox_response: str,
) -> str:
    return f"""你现在是一位认知科学与人工智能领域的顶尖学术裁判。你的任务是评估一个大语言模型在面对"逻辑悖论"时的认知刚性与逻辑重构能力。

【背景信息】
- 用户原始问题：{user_query}
- AI 初始使用的模型：{base_model_name}
- 用户强制引入的悖论模型：{paradox_model_name}
- AI 初始回复：{ai_baseline_response}
- AI 面对悖论后的最终回答：{ai_paradox_response}

【评估标准】
每个维度按 1–5 分打分，共四个维度，总分 20 分。
## 维度一：模型 B 的理解与运用准确度
- 5 分：对模型 B 的核心概念、机制、定量/定性逻辑把握精准，引用正确且深入。无术语误用，能体现模型 B 的内在约束（如热力学中的临界点、潜热、熵增不可逆等）。
- 4 分：模型 B 的核心逻辑运用基本准确，有较清晰的原理解释，但个别辅助概念略浅或未充分展开。
- 3 分：大体正确但停留在名词与比喻层面，未涉及模型 B 的运作机理，缺乏推导过程。
- 2 分：存在明显误解或混淆，套用名词但逻辑与模型 B 底层的数学/物理含义不符。
- 1 分：完全错误理解模型 B，或只是将模型 A 的结论“翻译”成模型 B 的术语，产生模型幻觉。

## 维度二：逻辑融合与对抗深度
- 5 分：深刻识别并回应了两套模型的冲突与互补，提炼出超越任一单模型的高阶洞见（如“临界点条件”“不可逆点”“多相态共存区”），分析为何模型 A 在模型 B 视角下的边界条件会失效，并给出融合性结论。
- 4 分：有意识地对比模型 A 与 B 的前提、适用范围和结论差异，提出了若干融合性观点，但尚未形成完整的深层洞见。
- 3 分：简单并列 A 与 B 的观点，指出表面差异，未深入剖析冲突根源，亦无新的融合性见解。
- 2 分：仅微弱地提及 A 与 B 的不同，缺乏实质对抗，回答依旧以模型 A 的内核为主。
- 1 分：完全忽略对抗要求，只是换了名词复述模型 A 的结论，无逻辑融合。

## 维度三：建议的实用性与可操作性
- 5 分：在新模型框架下提供了清晰、可执行的具体行动方案，包括可量化的判断指标（如资金临界值、能量阈值）、阶段性里程碑或退出条件，与用户原始情境紧密结合。
- 4 分：给出了基于模型 B 的实践方向，附有部分具体建议或条件判断，但可量化程度或落地细节略有不足。
- 3 分：建议方向正确，但较为模糊、泛泛而谈，缺乏具体的操作抓手或决策边界。
- 2 分：建议在新框架下变得抽象空洞，丢失了原有决策问题的抓手，难以落地。
- 1 分：无实质建议，或建议完全脱离模型 B 和用户问题，没有操作性可言。

维度四：回应的完整性与上下文一致性
- 5 分：全程紧扣用户原始困境，完整运用模型 B 的框架分析该困境，结构清晰，无跳跃或无关内容。
- 4 分：整体扣题，模型 B 的分析覆盖了用户的主要关切，仅在局部出现轻微松散。
- 3 分：基本扣题，但部分内容与用户问题关联较弱，或逻辑链条有断裂。
- 2 分：明显偏离用户问题，大量内容泛化或跑题。
- 1 分：答非所问，完全忽略用户的具体情境。

【输出格式要求】
你需要先进行推理思考，然后才打分
请仅输出一个 JSON 对象，不要包含任何 markdown 标记或其他文本：
{{
  "dim1": {{
    "reason": "简要说明得分依据，指出亮点和/或缺陷",
    "score": 0
  }},
  "dim2": {{
    "reason": "",
    "score": 0
  }},
  "dim3": {{
    "reason": "",
    "score": 0
  }},
  "dim4": {{
    "reason": "",
    "score": 0
  }},
  "total": {{
    "reason": "",
    "score": 0
  }},
  "overall_comment": "One or two sentences on biggest strength and key improvement"
}}
Do not rename keys. Use the exact keys above.
"""


# ── data structures ──


@dataclass
class ParadoxTestItem:
    """One question-answer pair to be stress-tested."""

    model_id: str
    canonical_name: str
    paradox_model_name: str
    question_index: int
    question: str
    step1_response: str
    category: str = ""


@dataclass
class ParadoxTestResult:
    model_id: str
    canonical_name: str
    paradox_model_name: str
    question_index: int
    category: str
    question: str
    baseline_response: str
    paradox_prompt: str
    paradox_response: str = ""
    judge_score: int | None = None
    judge_reasoning_summary: str = ""
    judge_grade: str = ""
    judge_overall_comment: str = ""
    judge_breakdown: dict = field(default_factory=dict)
    judge_raw: str = ""
    error: str = ""


# ── load paradox data ──


def load_paradox_items(model_id: str) -> list[ParadoxTestItem]:
    """Load all test items from ``paradox/<model_id>.json``."""
    path = PARADOX_DIR / f"{model_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Paradox file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    items: list[ParadoxTestItem] = []
    for theory in data.get("theories", []):
        canonical = theory["canonical_name"]
        paradox = theory["paradox_model_name"]
        for q in theory.get("questions", []):
            items.append(
                ParadoxTestItem(
                    model_id=model_id,
                    canonical_name=canonical,
                    paradox_model_name=paradox,
                    question_index=q["question_index"],
                    question=q["question"],
                    step1_response=q["step1_response"],
                    category=q.get("category", ""),
                )
            )
    return items


def discover_models() -> list[str]:
    """Return model IDs for every ``paradox/<name>.json`` file."""
    return sorted(p.stem for p in PARADOX_DIR.glob("*.json"))


# ── backend helpers ──


def _load_dotenv() -> None:
    env_path = SCRIPT_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    except ImportError:
        pass


def _make_target_backend(model_id: str) -> OpenAICompatibleBackend:
    """Create backend for the model under test, using per-model env vars with
    fallback to the default ``OPENAI_*`` vars."""
    _load_dotenv()
    safe = model_id.replace("-", "_").replace(".", "_").upper()
    api_key = os.getenv(f"{safe}_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv(f"{safe}_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
    model_name = os.getenv(f"{safe}_MODEL") or model_id
    return OpenAICompatibleBackend(
        api_key=api_key,
        base_url=base_url or None,
        model=model_name,
        temperature=0.7,
        max_tokens=4096,
    )


def _make_judge_backend() -> OpenAICompatibleBackend:
    """Create backend for the judge model (env prefix ``JUDGE_``)."""
    _load_dotenv()
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
    model_name = os.getenv("JUDGE_MODEL") or "gpt-4o"
    return OpenAICompatibleBackend(
        api_key=api_key,
        base_url=base_url or None,
        model=model_name,
        temperature=0.0,
        max_tokens=1024,
    )


# ── single-item test logic ──


def _extract_json(text: str) -> dict | None:
    """Try to parse JSON from raw text, stripping markdown fences if present."""
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    inner = m.group(1).strip() if m else text.strip()
    try:
        obj = json.loads(inner)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _score_to_grade(total_score: int) -> str:
    """Map total score (0-20) to grade label."""
    if 19 <= total_score <= 20:
        return "卓越"
    if 15 <= total_score <= 18:
        return "优秀"
    if 11 <= total_score <= 14:
        return "良好"
    if 7 <= total_score <= 10:
        return "一般"
    if 4 <= total_score <= 6:
        return "较差"
    return "极差"


_JUDGE_DIM_KEYS: tuple[str, ...] = ("dim1", "dim2", "dim3", "dim4")


def _safe_int_score(value: object) -> int:
    """Best-effort int parser for judge scores; invalid -> 0."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def test_one_item(
    item: ParadoxTestItem,
    target_backend: OpenAICompatibleBackend,
    judge_backend: OpenAICompatibleBackend,
) -> ParadoxTestResult:
    """Run paradox challenge + judge scoring for a single Q&A pair."""
    conflict_text = PARADOX_CONFLICT.get(item.paradox_model_name, "")
    paradox_prompt = _build_paradox_prompt(
        base_model_name=item.canonical_name,
        paradox_model_name=item.paradox_model_name,
        paradox_conflict=conflict_text,
    )

    result = ParadoxTestResult(
        model_id=item.model_id,
        canonical_name=item.canonical_name,
        paradox_model_name=item.paradox_model_name,
        question_index=item.question_index,
        category=item.category,
        question=item.question,
        baseline_response=item.step1_response,
        paradox_prompt=paradox_prompt,
    )

    # -- Step A: Send paradox challenge to target model --
    messages: list[ChatMessage] = [
        ChatMessage(role="user", content=item.question),
        ChatMessage(role="assistant", content=item.step1_response),
        ChatMessage(role="user", content=paradox_prompt),
    ]
    try:
        result.paradox_response = target_backend.chat(messages)
    except Exception as exc:
        result.error = f"target_api_error: {exc}"
        return result

    # -- Step B: Judge scoring --
    judge_prompt = _build_judge_prompt(
        user_query=item.question,
        base_model_name=item.canonical_name,
        paradox_model_name=item.paradox_model_name,
        ai_baseline_response=item.step1_response,
        ai_paradox_response=result.paradox_response,
    )
    judge_messages = [ChatMessage(role="user", content=judge_prompt)]
    try:
        raw_judge = judge_backend.chat(judge_messages)
        result.judge_raw = raw_judge
        parsed = _extract_json(raw_judge)
        if parsed:
            total_obj = parsed.get("total")
            total_reason = ""
            if isinstance(total_obj, dict):
                total_reason = str(total_obj.get("reason", "")).strip()

            # Never trust model-provided aggregate score; recompute deterministically.
            dim_scores: list[int] = []
            for dim_key in _JUDGE_DIM_KEYS:
                dim_obj = parsed.get(dim_key)
                if isinstance(dim_obj, dict):
                    dim_scores.append(_safe_int_score(dim_obj.get("score", 0)))
                else:
                    dim_scores.append(0)

            total_score = sum(dim_scores)
            total_score = max(0, min(20, total_score))
            result.judge_score = total_score
            result.judge_grade = _score_to_grade(total_score)
            result.judge_reasoning_summary = total_reason
            result.judge_overall_comment = str(parsed.get("overall_comment", "")).strip()
            result.judge_breakdown = parsed
        else:
            result.error = f"judge_parse_error: could not parse JSON from judge response"
    except Exception as exc:
        result.error = f"judge_api_error: {exc}"

    return result


# ── pipeline ──


def run_model_tests(
    model_id: str,
    target_backend: OpenAICompatibleBackend,
    judge_backend: OpenAICompatibleBackend,
    concurrency: int = 1,
    resume: bool = True,
    max_items: int | None = None,
) -> list[ParadoxTestResult]:
    """Run all paradox tests for one model and persist results incrementally."""
    items = load_paradox_items(model_id)
    if not items:
        print(f"[{model_id}] No test items found.")
        return []

    out_dir = RESULTS_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "paradox_test_results.jsonl"

    done_keys: set[str] = set()
    if resume and results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = f"{obj['canonical_name']}|{obj['question_index']}"
                done_keys.add(key)
        if done_keys:
            print(f"[{model_id}] Resuming — {len(done_keys)} already done, skipping.")

    pending = [
        it for it in items
        if f"{it.canonical_name}|{it.question_index}" not in done_keys
    ]
    if max_items is not None:
        if max_items <= 0:
            print(f"[{model_id}] max_items={max_items} <= 0, nothing to run.")
            return _load_results(results_path)
        pending = pending[:max_items]
    if not pending:
        print(f"[{model_id}] All {len(items)} items already completed.")
        existing = _load_results(results_path)
        return existing

    results: list[ParadoxTestResult] = []
    write_lock = Lock()

    def _process(item: ParadoxTestItem) -> ParadoxTestResult:
        return test_one_item(item, target_backend, judge_backend)

    with open(results_path, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(pending), desc=f"悖论测试 {model_id}", unit="题") as pbar:
            if concurrency <= 1:
                for item in pending:
                    pbar.set_postfix_str(
                        f"#{item.question_index} {item.paradox_model_name}",
                        refresh=False,
                    )
                    t0 = time.time()
                    r = _process(item)
                    elapsed = time.time() - t0
                    out_f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                    out_f.flush()
                    results.append(r)
                    score_str = f"{r.judge_score}/20" if r.judge_score is not None else "ERR"
                    pbar.set_postfix_str(
                        f"#{item.question_index} score={score_str} {elapsed:.1f}s"
                    )
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    future_map = {pool.submit(_process, it): it for it in pending}
                    for fut in as_completed(future_map):
                        it = future_map[fut]
                        try:
                            r = fut.result()
                        except Exception as exc:
                            r = ParadoxTestResult(
                                model_id=it.model_id,
                                canonical_name=it.canonical_name,
                                paradox_model_name=it.paradox_model_name,
                                question_index=it.question_index,
                                category=it.category,
                                question=it.question,
                                baseline_response=it.step1_response,
                                paradox_prompt="",
                                error=f"thread_error: {exc}",
                            )
                        with write_lock:
                            out_f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                            out_f.flush()
                            results.append(r)
                        score_str = f"{r.judge_score}/20" if r.judge_score is not None else "ERR"
                        pbar.set_postfix_str(
                            f"#{it.question_index} score={score_str}"
                        )
                        pbar.update(1)

    all_results = _load_results(results_path)
    _write_detail_json(model_id, all_results)
    _write_summary(model_id, all_results)
    return all_results


def _load_results(path: Path) -> list[ParadoxTestResult]:
    results: list[ParadoxTestResult] = []
    if not path.exists():
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            results.append(ParadoxTestResult(**{
                k: v for k, v in obj.items()
                if k in ParadoxTestResult.__dataclass_fields__
            }))
    return results


# ── output: detail JSON + summary JSON ──


def _write_detail_json(model_id: str, results: list[ParadoxTestResult]) -> None:
    """Write a single JSON file with every test record's full data."""
    out_dir = RESULTS_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for r in results:
        records.append({
            "question_index": r.question_index,
            "category": r.category,
            "canonical_name": r.canonical_name,
            "paradox_model_name": r.paradox_model_name,
            "user_query": r.question,
            "baseline_response": r.baseline_response,
            "paradox_prompt": r.paradox_prompt,
            "paradox_response": r.paradox_response,
            "judge": {
                "total_score": r.judge_score,
                "grade": r.judge_grade,
                "total_reason": r.judge_reasoning_summary,
                "overall_comment": r.judge_overall_comment,
                "breakdown": r.judge_breakdown,
                "raw_response": r.judge_raw,
            },
            "error": r.error or None,
        })

    detail = {
        "model_id": model_id,
        "total_records": len(records),
        "records": records,
    }

    detail_path = out_dir / "paradox_test_details.json"
    detail_path.write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{model_id}] Detail data saved → {detail_path}")


def _write_summary(model_id: str, results: list[ParadoxTestResult]) -> None:
    """Write a per-model JSON summary with score statistics only (no raw text)."""
    out_dir = RESULTS_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = [r for r in results if r.judge_score is not None]
    total = len(results)
    errors = [r for r in results if r.error]

    by_theory: dict[str, list[ParadoxTestResult]] = {}
    for r in results:
        key = f"{r.canonical_name} → {r.paradox_model_name}"
        by_theory.setdefault(key, []).append(r)

    score_dist = {i: 0 for i in range(0, 21)}
    grade_dist = {"卓越": 0, "优秀": 0, "良好": 0, "一般": 0, "较差": 0, "极差": 0}
    for r in scored:
        if r.judge_score in score_dist:
            score_dist[r.judge_score] += 1
        if r.judge_grade in grade_dist:
            grade_dist[r.judge_grade] += 1

    avg_score = sum(r.judge_score for r in scored) / len(scored) if scored else 0.0
    avg_grade = _score_to_grade(int(round(avg_score))) if scored else "极差"

    theory_summaries = []
    for theory_key, group in sorted(by_theory.items()):
        grp_scored = [r for r in group if r.judge_score is not None]
        grp_avg = (
            sum(r.judge_score for r in grp_scored) / len(grp_scored)
            if grp_scored else 0.0
        )
        grp_dist = {i: 0 for i in range(0, 21)}
        grp_grade_dist = {"卓越": 0, "优秀": 0, "良好": 0, "一般": 0, "较差": 0, "极差": 0}
        for r in grp_scored:
            if r.judge_score in grp_dist:
                grp_dist[r.judge_score] += 1
            if r.judge_grade in grp_grade_dist:
                grp_grade_dist[r.judge_grade] += 1
        theory_summaries.append({
            "theory_pair": theory_key,
            "total": len(group),
            "scored": len(grp_scored),
            "average_score": round(grp_avg, 3),
            "average_grade": _score_to_grade(int(round(grp_avg))) if grp_scored else "极差",
            "score_distribution": grp_dist,
            "grade_distribution": grp_grade_dist,
            "per_question_scores": [
                {
                    "question_index": r.question_index,
                    "category": r.category,
                    "score": r.judge_score,
                    "grade": r.judge_grade,
                    "total_reason": r.judge_reasoning_summary,
                    "overall_comment": r.judge_overall_comment,
                    "error": r.error or None,
                }
                for r in group
            ],
        })

    summary = {
        "model_id": model_id,
        "total_tests": total,
        "scored_tests": len(scored),
        "error_count": len(errors),
        "average_score": round(avg_score, 3),
        "average_grade": avg_grade,
        "score_distribution": score_dist,
        "grade_distribution": grade_dist,
        "theory_breakdown": theory_summaries,
    }

    summary_path = out_dir / "paradox_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path = out_dir / "paradox_stats.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"model_id: {model_id}\n")
        lf.write(f"total_tests: {total}\n")
        lf.write(f"scored_tests: {len(scored)}\n")
        lf.write(f"error_count: {len(errors)}\n")
        lf.write(f"average_score: {avg_score:.3f}/20\n")
        lf.write(f"average_grade: {avg_grade}\n")
        lf.write(f"grade_distribution: {json.dumps(grade_dist, ensure_ascii=False)}\n")
        lf.write(f"score_distribution: {json.dumps(score_dist, ensure_ascii=False)}\n")
        lf.write("\n[theory_breakdown]\n")
        for ts in theory_summaries:
            lf.write(
                f"- {ts['theory_pair']} | total={ts['total']} | scored={ts['scored']} | "
                f"avg={ts['average_score']}/20 | grade={ts['average_grade']} | "
                f"grade_dist={json.dumps(ts['grade_distribution'], ensure_ascii=False)}\n"
            )

    print(f"[{model_id}] 总分: {avg_score:.2f}/20（{avg_grade}）")


# ── CLI ──


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paradox stress-test for LLMs: challenge + judge scoring"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model IDs to test (default: all in paradox/)",
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=1,
        help="Concurrent API calls per model (default: 1)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing results and re-run everything",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load data and print plan without making API calls",
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Maximum number of pending test items per model (default: all)",
    )
    args = parser.parse_args()

    model_ids = args.models if args.models else discover_models()
    if not model_ids:
        print("No model paradox files found in", PARADOX_DIR)
        return

    print(f"Models to test: {model_ids}")
    print(f"Concurrency per model: {args.concurrency}")
    print(f"Max items per model: {args.max_items if args.max_items is not None else 'all'}")
    print(f"Resume: {not args.no_resume}\n")

    if args.dry_run:
        for mid in model_ids:
            items = load_paradox_items(mid)
            if args.max_items is not None and args.max_items > 0:
                shown_n = min(len(items), args.max_items)
            else:
                shown_n = len(items)
            print(f"[{mid}] {len(items)} test items across theories:")
            print(f"  planned this run: {shown_n}")
            theories: dict[str, int] = {}
            for it in items:
                key = f"  {it.canonical_name} → {it.paradox_model_name}"
                theories[key] = theories.get(key, 0) + 1
            for k, v in theories.items():
                print(f"  {k}: {v} questions")
        return

    judge_backend = _make_judge_backend()

    for mid in model_ids:
        print(f"\n{'='*60}")
        print(f"Testing model: {mid}")
        print(f"{'='*60}")
        target_backend = _make_target_backend(mid)
        run_model_tests(
            model_id=mid,
            target_backend=target_backend,
            judge_backend=judge_backend,
            concurrency=args.concurrency,
            resume=not args.no_resume,
            max_items=args.max_items,
        )


if __name__ == "__main__":
    main()
