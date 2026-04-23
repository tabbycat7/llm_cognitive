"""Entry point – run the full cognitive probing pipeline.

Usage examples
--------------
# Dry-run with dummy backend (no API key needed):
    python run.py --backend dummy --num 5

# Real run with OpenAI-compatible API:
    python run.py --backend openai --num 100 \
        --api-key sk-xxx --base-url https://api.openai.com/v1 --model gpt-4o

# Using DeepSeek:
    python run.py --backend openai --num 100 \
        --api-key sk-xxx --base-url https://api.deepseek.com/v1 --model deepseek-chat

# Concurrent API calls (faster, recommended 3-5):
    python run.py --backend openai --num 100 --concurrency 5

# Start at a given row in the JSONL (0-based; skip earlier questions):
    python run.py --backend openai --start 500 --num 100

# Analyse only (skip probing):
    python run.py --analyse-only

# Skip synonym merging (just count raw names):
    python run.py --analyse-only --merge-method none

# Use LLM-based clustering for merging:
    python run.py --analyse-only --merge-method llm

# Use embedding-based clustering with local model (推荐中文):
    python run.py --analyse-only --merge-method embedding \
        --embedding-backend local --embedding-model BAAI/bge-small-zh-v1.5

# Use embedding-based clustering with Ollama:
    python run.py --analyse-only --merge-method embedding \
        --embedding-backend ollama --embedding-model nomic-embed-text

# Use embedding-based clustering with OpenAI API:
    python run.py --analyse-only --merge-method embedding \
        --embedding-backend openai --embedding-model text-embedding-3-small

# Force refresh the synonym merge map:
    python run.py --analyse-only --refresh-merge

# Configure API credentials in .env (auto-loaded):
#   OPENAI_API_KEY=sk-xxx
#   OPENAI_BASE_URL=https://api.openai.com/v1
#   LLM_MODEL=gpt-4o
#   LLM_THINKING=disabled   # DeepSeek-V3.2 only: extra_body thinking type (enabled|disabled)
#   EMBEDDING_API_KEY=sk-xxx  (optional, for openai embedding backend)
#   EMBEDDING_BASE_URL=https://api.openai.com/v1  (optional)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any code reads OPENAI_* / LLM_* (must run before project imports).
ROOT = Path(__file__).resolve().parent
_ENV_FILE = ROOT / ".env"
if _ENV_FILE.is_file():
    # override=True: if the shell has empty OPENAI_API_KEY, still take values from .env
    load_dotenv(_ENV_FILE, override=True)

import argparse
import json

from analyze_results import analyse, print_raw_names, visualise, write_analysis_summary_log
from cognitive_probe import (
    category_to_probe_output_subdir,
    model_id_to_probe_output_dirname,
    run_pipeline,
)
from llm_api import create_backend

DEFAULT_INPUT = ROOT / "zhihu_kol_train.jsonl"
#DEFAULT_INPUT = ROOT / "reddit_r_advice.jsonl"
DEFAULT_OUTPUT = ROOT / "probe_results.jsonl"


def _coerce_output_to_jsonl_path(chosen: Path) -> Path:
    """If ``chosen`` is a directory path, use ``<chosen>/probe_results.jsonl`` (mkdir later).

    If it already ends in ``.jsonl``, or exists as a plain file (e.g. legacy no-extension
    file), return ``chosen`` unchanged.
    """
    if chosen.suffix.lower() == ".jsonl":
        return chosen
    if chosen.exists() and chosen.is_file():
        return chosen
    return (chosen / "probe_results.jsonl").resolve()


def _effective_probe_output_path(args: argparse.Namespace) -> tuple[Path, bool]:
    """Return (resolved probe JSONL path, whether auto layout ``model/category/`` was applied)."""
    raw = Path(args.output).expanduser()
    chosen = raw.resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()
    default_probe = (ROOT / "probe_results.jsonl").resolve()
    if args.filter_category and chosen == default_probe:
        model = (args.model or os.getenv("LLM_MODEL") or "gpt-4o").strip()
        mdir = model_id_to_probe_output_dirname(model)
        cdir = category_to_probe_output_subdir(args.filter_category)
        resolved = (ROOT / mdir / cdir / "probe_results.jsonl").resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved, True
    return _coerce_output_to_jsonl_path(chosen), False


def _jsonl_looks_like_taxonomy_labels(path: Path) -> bool:
    """True if first JSON object looks like classify_reddit_taxonomy output (join key + category)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                break
            else:
                return False
    except (OSError, json.JSONDecodeError):
        return False
    if "question_index" not in obj:
        return False
    meta = obj.get("metadata")
    if isinstance(meta, dict) and meta.get("category") is not None:
        return True
    return isinstance(obj.get("category"), str)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM Cognitive Probing: 3-step deep questioning on Zhihu-KOL questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Data ──
    p.add_argument(
        "--input", "-i",
        default=str(DEFAULT_INPUT),
        help="Path to zhihu_kol_train.jsonl (default: %(default)s)",
    )
    p.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help="Probe / analyse JSONL path, or a **folder** (then uses <dir>/probe_results.jsonl). "
        "Default with --filter-category: <project>/<model_id>/<category>/probe_results.jsonl (see deepseek/ layout).",
    )
    p.add_argument(
        "--num", "-n",
        type=int,
        default=None,
        help="Number of questions to process (default: all)",
    )
    p.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        metavar="INDEX",
        help="0-based index of the first JSONL record to process (skip earlier rows; default: 0)",
    )

    # ── LLM backend ──
    p.add_argument(
        "--backend", "-b",
        choices=["openai", "dummy"],
        default="dummy",
        help="LLM backend type (default: dummy for testing)",
    )
    p.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY in .env)")
    p.add_argument("--base-url", default=None, help="API base URL (or set OPENAI_BASE_URL in .env)")
    p.add_argument(
        "--model", "-m",
        default=os.getenv("LLM_MODEL", "gpt-4o"),
        help="Model name (default: from .env or gpt-4o)",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--prompt-lang",
        choices=["zh", "en"],
        default=os.getenv("LLM_PROMPT_LANG", "zh"),
        help="Language of the 3-step probe prompts (step1/step2/step3). "
        "'zh' = 中文版（默认），'en' = English. Override via LLM_PROMPT_LANG env.",
    )

    # ── Pipeline control ──
    p.add_argument(
        "--concurrency", "-c",
        type=int,
        default=1,
        help="Number of concurrent API calls (default: 1 = sequential). Recommended: 3-5",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do NOT skip already-completed questions (start fresh)",
    )
    p.add_argument(
        "--filter-label",
        default=None,
        metavar="TEXT",
        help="Reddit-style metadata only: match label/communityName/subreddit (case-insensitive). "
        "Not taxonomy; use --filter-category for taxonomy_labels.jsonl category.",
    )
    p.add_argument(
        "--taxonomy-labels",
        default=None,
        metavar="PATH",
        help="JSONL from classify_reddit_taxonomy (question_index + category). "
        "Default with --filter-category: project taxonomy_labels.jsonl if present, "
        "else --input when that file looks like taxonomy labels (e.g. reddit_taxonomy_labels.jsonl).",
    )
    p.add_argument(
        "--filter-category",
        default=None,
        metavar="CATEGORY",
        help="Only probe rows whose line index has this category in taxonomy labels "
        "(e.g. '[Relational & Intimate]'). Joins on question_index with input JSONL.",
    )
    p.add_argument(
        "--analyse-only",
        action="store_true",
        help="Skip probing, only run analysis on existing results",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Show top N models/laws in the frequency chart (default: 30)",
    )
    p.add_argument(
        "--merge-method",
        choices=["llm", "embedding", "none"],
        default="llm",
        help="Synonym merging method: 'llm' (LLM clustering), 'embedding' (vector clustering), 'none' (no merging)",
    )
    p.add_argument(
        "--embedding-backend",
        choices=["local", "ollama", "openai"],
        default="local",
        help="Embedding backend: 'local' (HuggingFace/SentenceTransformers), 'ollama', 'openai'",
    )
    p.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-zh-v1.5",
        help="Embedding model name. Examples: 'BAAI/bge-small-zh-v1.5' (local), 'nomic-embed-text' (ollama), 'text-embedding-3-small' (openai)",
    )
    p.add_argument(
        "--embedding-device",
        default=os.getenv("EMBEDDING_DEVICE", "auto"),
        help='Local embedding only: torch device — "auto" (GPU if CUDA available), "cuda", "cuda:0", "cpu". Or set EMBEDDING_DEVICE in .env',
    )
    p.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.3,
        help="Cosine distance threshold for embedding clustering (0-1, smaller=stricter). Default: 0.3",
    )
    p.add_argument(
        "--name-source",
        choices=["step1", "step2", "step3", "step4"],
        default="step3",
        help="step3 (default): JSON models[].name from the mechanism-abstraction step. "
        "step4: legacy only — same parsing, reads step4_response from old probe JSONL. "
        "step1/step2: full response text per row (embed/cluster as-is; "
        "use with --merge-method embedding or none, not llm).",
    )
    p.add_argument(
        "--refresh-merge",
        action="store_true",
        help="Force re-generate the synonym merge map (ignore cache)",
    )

    return p


def _create_backend(args: argparse.Namespace):
    """Create an LLM backend from CLI args."""
    backend_kwargs: dict = {}
    if args.backend == "openai":
        backend_kwargs.update(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    return create_backend(args.backend, **backend_kwargs)


def main() -> None:
    args = build_parser().parse_args()
    if args.start < 0:
        raise SystemExit("error: --start must be >= 0")

    output_path, output_layout_auto = _effective_probe_output_path(args)

    backend = _create_backend(args)

    # ── Probe phase ──
    if not args.analyse_only:
        print(f"Backend     : {args.backend} ({getattr(backend, 'model', 'N/A')})")
        print(f"Input       : {args.input}")
        print(f"Output      : {output_path}")
        if output_layout_auto:
            print(
                "[output-layout] using <project>/<model_id>/<category>/probe_results.jsonl "
                "(same style as deepseek/…); override with an explicit -o path.",
                flush=True,
            )
        print(f"Start index : {args.start}")
        print(f"Num         : {args.num or 'ALL'}")
        print(f"Concurrency : {args.concurrency}")
        print(f"Prompt lang : {args.prompt_lang}")
        if args.filter_label:
            print(f"Filter label : {args.filter_label!r} (metadata, case-insensitive)")
        taxonomy_labels_path = None
        if args.filter_category:
            if args.taxonomy_labels:
                taxonomy_labels_path = str(Path(args.taxonomy_labels).expanduser().resolve())
            else:
                default_tax = (ROOT / "taxonomy_labels.jsonl").resolve()
                inp = Path(args.input).expanduser().resolve()
                if default_tax.is_file():
                    taxonomy_labels_path = str(default_tax)
                elif _jsonl_looks_like_taxonomy_labels(inp):
                    taxonomy_labels_path = str(inp)
                else:
                    raise SystemExit(
                        f"error: taxonomy labels file not found: {default_tax}\n"
                        f"  Pass --taxonomy-labels PATH (e.g. reddit_taxonomy_labels.jsonl), "
                        f"or add taxonomy_labels.jsonl under {ROOT}."
                    )
            if not Path(taxonomy_labels_path).is_file():
                raise SystemExit(f"error: taxonomy labels file not found: {taxonomy_labels_path}")
            print(f"Taxonomy file: {taxonomy_labels_path}")
            print(f"Filter category: {args.filter_category!r}")
        print()

        run_pipeline(
            backend=backend,
            jsonl_path=args.input,
            output_path=output_path,
            max_count=args.num,
            resume=not args.no_resume,
            concurrency=args.concurrency,
            start=args.start,
            filter_label=args.filter_label,
            taxonomy_labels_path=taxonomy_labels_path,
            filter_category=args.filter_category,
            prompt_lang=args.prompt_lang,
        )

    # ── Analysis phase ──
    if not output_path.exists():
        print(f"[error] Results file not found: {output_path}")
        return

    print("\n" + "=" * 70)
    print("  ANALYSIS PHASE")
    print("=" * 70)

    merge_backend = backend if args.merge_method == "llm" else None
    cache_path = output_path.parent / "merge_map.json"

    print(f"Merge method: {args.merge_method}")
    print(f"Name source: {args.name_source}")
    if args.merge_method == "embedding":
        print(f"Embedding backend: {args.embedding_backend}")
        print(f"Embedding model: {args.embedding_model}")
        print(f"Embedding device: {args.embedding_device}")
        print(f"Embedding threshold: {args.embedding_threshold}")

    canonical_counts, raw_counts = analyse(
        output_path,
        backend=merge_backend,
        cache_path=cache_path,
        refresh_merge=args.refresh_merge,
        merge_method=args.merge_method,
        embedding_threshold=args.embedding_threshold,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        name_source=args.name_source,
    )
    print_raw_names(raw_counts)
    is_merged = args.merge_method != "none"
    visualise(canonical_counts, output_dir=output_path.parent, top_n=args.top_n, merged=is_merged)

    write_analysis_summary_log(
        results_path=output_path,
        output_dir=output_path.parent,
        canonical_counter=canonical_counts,
        raw_counter=raw_counts,
        merge_method=args.merge_method,
        top_n=args.top_n,
        refresh_merge=args.refresh_merge,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_threshold=args.embedding_threshold,
        embedding_device=args.embedding_device,
        name_source=args.name_source,
    )


if __name__ == "__main__":
    main()
