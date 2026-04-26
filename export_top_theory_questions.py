#!/usr/bin/env python3
"""从 merged_top50.json 按「每个模型各自」频次最高的若干理论，从 probe 中抽题（模型内跨理论去重）。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_results import (
    _read_merge_map_file,
    extract_model_names_from_json,
    normalize_llm_merge_label,
)

# 导出统计/抽样时：将若干 merged canonical 合并为一类（与 merge_map 无关，仅影响本脚本）
_BAYESIAN_UPDATING_N = normalize_llm_merge_label("bayesian updating")
_BAYESIAN_INFERENCE_N = normalize_llm_merge_label("bayesian inference")
_SIGNALING_THEORY_N = normalize_llm_merge_label("signaling theory")
_SIGNALING_GAME_N = normalize_llm_merge_label("signaling game")

# canonical（归一化后）→ 悖论/经济学叙事用中文名（与 step3 理论并列展示）
# step3 canonical（经 normalize_llm_merge_label）→ 对抗/悖论叙事展示名。
# 新理论：在此增加一行，键务必用 normalize_llm_merge_label("...") 与 merged 口径一致。
# 未出现在此表中的 canonical：paradox_model_name 字段为 ""（空字符串）。
_PARADOX_MODEL_NAME: dict[str, str] = {
    normalize_llm_merge_label("signaling game"): "Information Cascade",
    normalize_llm_merge_label("feedback control"): "Goodhart's Law",
    normalize_llm_merge_label("bayesian inference"): "Knightian Uncertainty",
    normalize_llm_merge_label("dynamical systems"): "Random Walk",
    normalize_llm_merge_label("constrained optimization"): "Satisficing",
    normalize_llm_merge_label("nash equilibrium"): "Keynesian Beauty Contest",
    normalize_llm_merge_label("control theory"): "Self-Organized Criticality",
    normalize_llm_merge_label("reinforcement learning"): "Active Inference",
    normalize_llm_merge_label("catastrophe theory"): "Gradualism",
    normalize_llm_merge_label("optimal stopping"): "Multi-Armed Bandit",
    normalize_llm_merge_label("real options theory"): "Credible Commitment Theory",
    normalize_llm_merge_label("operant conditioning"): "Self-Determination Theory",
    normalize_llm_merge_label("mechanism design"): "Hayekian Critique",
    normalize_llm_merge_label("markov processes"): "Fractional Brownian Motion",
}


def _safe_model_filename(model: str) -> str:
    s = (model or "").strip() or "unknown"
    for ch in r'\/:*?"<>|':
        s = s.replace(ch, "_")
    return s


def _unify_export_canonical(canonical_norm: str) -> str:
    """导出用 canonical 归一：bayesian 合并；signaling theory 并入 signaling game。"""
    if canonical_norm == _BAYESIAN_UPDATING_N:
        return _BAYESIAN_INFERENCE_N
    if canonical_norm == _SIGNALING_THEORY_N:
        return _SIGNALING_GAME_N
    return canonical_norm


def _paradox_name_for_canonical(canonical_norm: str) -> str:
    u = _unify_export_canonical(canonical_norm)
    return _PARADOX_MODEL_NAME.get(u, "")


def _map_raw_to_canonical(raw: str, merge_map: dict[str, str]) -> str:
    """与 analyze 中 merge_method=llm 时的 canonical 口径一致。"""
    r = (raw or "").strip()
    if not r:
        return ""
    key = normalize_llm_merge_label(r)
    canon = (
        merge_map.get(r)
        or merge_map.get(key)
        or (merge_map.get(r.lower()) if r.lower() != r else None)
    )
    if canon is None:
        canon = r
    c = normalize_llm_merge_label(str(canon))
    return c if c else (r or "")


def _aggregate_top_name_counts_by_model(merged: dict[str, Any]) -> dict[str, Counter]:
    """每个模型：在该模型下所有分类的 top_names 中，对 canonical（经 _unify_export_canonical）的 count 求和。"""
    by_model: dict[str, Counter] = defaultdict(Counter)
    for item in merged.get("items", []):
        if not isinstance(item, dict):
            continue
        model = str(item.get("model", "")).strip()
        if not model:
            continue
        for row in item.get("top_names", []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            try:
                c = int(row.get("count", 0))
            except (TypeError, ValueError):
                continue
            key = normalize_llm_merge_label(name) or name
            key = _unify_export_canonical(key)
            by_model[model][key] += c
    return dict(by_model)


def _load_merge_map(probe_path: Path) -> dict[str, str]:
    p = probe_path.parent / "merge_map.json"
    if not p.is_file():
        return {}
    return _read_merge_map_file(p)


def _build_step1_lookup(merged: dict[str, Any]) -> dict[tuple[str, str, int], str]:
    """(model, category, question_index) → step1_response，扫描 merged 中全部 probe_results.jsonl。"""
    out: dict[tuple[str, str, int], str] = {}
    for item in merged.get("items", []):
        if not isinstance(item, dict):
            continue
        pr = item.get("probe_results")
        if not pr:
            continue
        probe = Path(str(pr))
        if not probe.is_file():
            continue
        model = str(item.get("model", ""))
        category = str(item.get("category", ""))
        with probe.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    qi = int(rec.get("question_index", -1))
                except (TypeError, ValueError):
                    qi = -1
                if qi < 0:
                    continue
                s1 = rec.get("step1_response", "")
                if isinstance(s1, str):
                    out[(model, category, qi)] = s1
                elif s1 is not None:
                    out[(model, category, qi)] = json.dumps(s1, ensure_ascii=False)
    return out


def _enrich_questions_step1(
    theory_rows: list[dict[str, Any]],
    step1_lookup: dict[tuple[str, str, int], str],
) -> None:
    for row in theory_rows:
        for q in row.get("questions", []):
            if not isinstance(q, dict):
                continue
            try:
                qi = int(q.get("question_index", -1))
            except (TypeError, ValueError):
                qi = -1
            key = (str(q.get("model", "")), str(q.get("category", "")), qi)
            q["step1_response"] = step1_lookup.get(key, "")


def _write_paradox_file_for_model(
    model: str, theory_rows: list[dict[str, Any]], paradox_dir: Path
) -> None:
    """单个模型的 paradox JSON（theory_rows 内题目应均属该 model）。"""
    paradox_dir.mkdir(parents=True, exist_ok=True)
    fn = paradox_dir / f"{_safe_model_filename(model)}.json"
    with fn.open("w", encoding="utf-8") as w:
        json.dump(
            {"model": model, "theories": theory_rows},
            w,
            ensure_ascii=False,
            indent=2,
        )


def _collect_theory_question_indices_for_model(
    merged: dict[str, Any], model: str, targets: set[str]
) -> tuple[
    dict[tuple[str, str, int], dict[str, str]],
    dict[str, set[int]],
    dict[str, dict[int, set[tuple[str, str]]]],
]:
    """仅扫描 ``model`` 指定模型下的 probe（各分类 jsonl）。

    用 ``(model, category, question_index)`` 存正文；``theory_qi_sources[理论][qi]`` 记录
    命中来源的 (model, category)（单模型下多为同一 model、不同 category）。
    """
    per_theory: dict[str, set[int]] = defaultdict(set)
    meta_triple: dict[tuple[str, str, int], dict[str, str]] = {}
    theory_qi_sources: dict[str, dict[int, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    target_model = str(model).strip()
    if not target_model:
        return meta_triple, per_theory, theory_qi_sources
    for item in merged.get("items", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("model", "")).strip() != target_model:
            continue
        pr = item.get("probe_results")
        if not pr:
            continue
        probe = Path(str(pr))
        if not probe.is_file():
            print(f"[warn] 跳过不存在的文件: {probe}", file=sys.stderr)
            continue
        mm = _load_merge_map(probe)
        if not mm and merged.get("merge_method", "llm") != "none":
            print(
                f"[warn] 无 merge_map.json: {probe.parent}",
                file=sys.stderr,
            )
        row_model = str(item.get("model", "")).strip()
        category = str(item.get("category", ""))
        with probe.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step3 = rec.get("step3_response", "")
                if not isinstance(step3, str) or not step3.strip():
                    continue
                try:
                    qi = int(rec.get("question_index", -1))
                except (TypeError, ValueError):
                    qi = -1
                if qi < 0:
                    continue
                qtext = rec.get("question", "")
                triple = (row_model, category, qi)
                if isinstance(qtext, str) and qtext:
                    meta_triple[triple] = {
                        "question": qtext,
                        "model": row_model,
                        "category": category,
                    }
                for raw in extract_model_names_from_json(step3):
                    c = _map_raw_to_canonical(raw, mm)
                    tkey = _unify_export_canonical(
                        normalize_llm_merge_label(c) or c
                    )
                    if tkey in targets:
                        per_theory[tkey].add(qi)
                        theory_qi_sources[tkey][qi].add((row_model, category))
    return meta_triple, per_theory, theory_qi_sources


def _allocate_questions_cross_theory(
    theories_ordered: list[str],
    per_theory: dict[str, set[int]],
    meta_triple: dict[tuple[str, str, int], dict[str, str]],
    theory_qi_sources: dict[str, dict[int, set[tuple[str, str]]]],
    counts: Counter,
    per_theory_limit: int,
) -> list[dict[str, Any]]:
    """按 theories_ordered 依次取题：已分配给靠前理论的问题不再分给后面理论。

    对每个 ``question_index``，在仍可选的 (model, category) 来源中做计数均衡，避免全部挤在
    某一扫描顺序靠后的模型上。
    """
    used_globally: set[int] = set()
    model_pick_count: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for t in theories_ordered:
        pool = sorted(per_theory.get(t, set()) - used_globally)
        picked = pool[:per_theory_limit]
        used_globally.update(picked)
        qs: list[dict[str, Any]] = []
        for qi in picked:
            cands = theory_qi_sources.get(t, {}).get(qi)
            if not cands:
                continue
            m_sel, c_sel = min(
                cands,
                key=lambda mc: (model_pick_count[mc[0]], mc[0], mc[1]),
            )
            model_pick_count[m_sel] += 1
            m = meta_triple.get((m_sel, c_sel, qi))
            if not m:
                continue
            qs.append(
                {
                    "question_index": qi,
                    "question": m["question"],
                    "model": m["model"],
                    "category": m["category"],
                }
            )
        cn = normalize_llm_merge_label(t) or t
        rows.append(
            {
                "canonical_name": t,
                "paradox_model_name": _paradox_name_for_canonical(cn),
                "aggregated_mention_count_from_merged_file": int(counts.get(t, 0)),
                "unique_questions_available_after_prior_theories": len(pool),
                "questions": qs,
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="按模型：各模型在 merged_top50 中加总 count 排名前几的理论，抽题并写 paradox/"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("merged_top50.json"),
        help="batch 导出的 merged JSON（含 items[].top_names）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("top_theory_questions.json"),
        help="输出 JSON 路径",
    )
    p.add_argument(
        "--top-theories",
        type=int,
        default=5,
        metavar="K",
        help="每个模型内按该模型各分类 top_names 的 count 加总，取前 K 个理论（默认 5）",
    )
    p.add_argument(
        "--questions-per-theory",
        type=int,
        default=100,
        metavar="N",
        help="每个理论在每个模型导出中最多 N 题（默认 100）；模型内跨理论按 question_index 去重",
    )
    p.add_argument(
        "--paradox-dir",
        type=Path,
        default=None,
        help="按模型拆分的 JSON 目录（默认：<output 父目录>/paradox）",
    )
    args = p.parse_args()
    k = max(1, int(args.top_theories))
    n_per = max(1, int(args.questions_per_theory))

    in_path: Path = args.input.expanduser().resolve()
    if not in_path.is_file():
        print(f"[error] 找不到文件: {in_path}", file=sys.stderr)
        return 2
    with in_path.open("r", encoding="utf-8") as f:
        merged: dict = json.load(f)
    if not isinstance(merged, dict):
        print("[error] 根对象应为 JSON object", file=sys.stderr)
        return 2

    by_model = _aggregate_top_name_counts_by_model(merged)
    if not by_model:
        print("[error] 未在 items[].top_names 中读到任何 count", file=sys.stderr)
        return 2

    step1_lookup = _build_step1_lookup(merged)

    out_path: Path = args.output.expanduser().resolve()
    paradox_dir = (
        args.paradox_dir.expanduser().resolve()
        if args.paradox_dir is not None
        else out_path.parent / "paradox"
    )

    models_export: list[dict[str, Any]] = []
    model_order = sorted(by_model.keys(), key=str.lower)

    for model in model_order:
        counts_m = by_model[model]
        theories_ordered = [name for name, _ in counts_m.most_common(k)]
        if len(theories_ordered) < k:
            print(
                f"[warn] 模型 {model!r} 不同理论名仅 {len(theories_ordered)} 个，少于 {k}",
                file=sys.stderr,
            )
        target_set = set(theories_ordered)
        meta_triple, per_theory, theory_qi_sources = (
            _collect_theory_question_indices_for_model(merged, model, target_set)
        )
        theory_rows = _allocate_questions_cross_theory(
            theories_ordered,
            per_theory,
            meta_triple,
            theory_qi_sources,
            counts_m,
            n_per,
        )
        _enrich_questions_step1(theory_rows, step1_lookup)
        _write_paradox_file_for_model(model, theory_rows, paradox_dir)
        models_export.append(
            {
                "model": model,
                "top_theories_ranked": theories_ordered,
                "questions_per_theory_cap": n_per,
                "theories": theory_rows,
            }
        )
        tops = ", ".join(theories_ordered)
        ntot = sum(len(r.get("questions", [])) for r in theory_rows)
        print(f"[info] {model}: 前{k}理论 [{tops}]；共导出 {ntot} 题 -> paradox/")

    n_paradox_files = len(list(paradox_dir.glob("*.json")))
    per_model_q = {m["model"]: sum(len(r.get("questions", [])) for r in m["theories"]) for m in models_export}
    print(
        f"[info] paradox/ 下共 {n_paradox_files} 个模型文件；"
        f"各模型导出题数: {dict(sorted(per_model_q.items()))}"
    )

    out: dict[str, Any] = {
        "export_mode": "per_model_top_k_theories",
        "source_file": str(in_path),
        "merge_method_from_source": merged.get("merge_method"),
        "name_source": merged.get("name_source"),
        "criterion_zh": (
            "对每个模型：将该模型在 merged_top50 各分类条目中 top_names 的 count 按 canonical 加总"
            f"（脚本内合并 bayesian updating→bayesian inference、signaling theory→signaling game），"
            f"取前 {k} 个理论；在该模型 probe 内依次每理论最多 {n_per} 题，"
            "模型内跨理论按 question_index 去重；"
            "正文与 step1 以 (model, category, question_index) 定位；"
            "同题多分类命中时在 (model,category) 候选上按已选次数均衡；"
            "paradox_model_name 为部分 canonical 的固定映射"
        ),
        "top_theories_per_model": k,
        "questions_per_theory_cap": n_per,
        "paradox_per_model_dir": str(paradox_dir),
        "paradox_split_model_files": n_paradox_files,
        "questions_per_model_in_export": dict(sorted(per_model_q.items())),
        "models": models_export,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=2)
    print(f"[info] 已写入: {out_path}")
    print(f"[info] 按模型拆分已写入目录: {paradox_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
