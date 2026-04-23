"""LLM API interface module.

Provides a pluggable interface for different LLM backends.
Users should implement their own adapter or configure the built-in ones.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_from_project() -> None:
    """Load project-root .env so API keys work even when cwd or import order differs."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=True)


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str


class LLMBackend(ABC):
    """Abstract base class – implement `chat` for any LLM provider."""

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        """Send a multi-turn conversation and return the assistant reply."""
        ...


# ── OpenAI-compatible backend (works with OpenAI, DeepSeek, vLLM, etc.) ──


class OpenAICompatibleBackend(LLMBackend):
    """Works with any OpenAI-compatible API (OpenAI, DeepSeek, Moonshot, local vLLM …).

    DeepSeek-V3.2 chat models accept ``thinking`` (via ``extra_body`` on the OpenAI SDK).
    When the configured ``model`` or ``base_url`` contains ``deepseek`` (case-insensitive),
    requests default to ``thinking: {type: disabled}``. Override with env ``LLM_THINKING``
    (``enabled`` / ``disabled``), constructor ``thinking=...``, or per-call
    ``backend.chat(..., thinking={"type": "enabled"})``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking: dict | str | bool | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai  to use OpenAICompatibleBackend") from exc

        _load_dotenv_from_project()

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        key = (api_key or os.getenv("OPENAI_API_KEY", "") or "").strip()
        url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self.client = OpenAI(api_key=key, base_url=url)
        self._thinking_body: dict | None = self._resolve_default_thinking(thinking, model, url)

    @staticmethod
    def _resolve_default_thinking(
        thinking: dict | str | bool | None,
        model: str,
        base_url: str | None,
    ) -> dict | None:
        """Body fragment for ``extra_body['thinking']``; ``None`` = omit on requests."""
        if thinking is False:
            return None
        if isinstance(thinking, dict):
            return dict(thinking)
        if thinking is True:
            return {"type": "enabled"}
        if isinstance(thinking, str):
            t = thinking.strip().lower()
            if t in ("enabled", "disabled"):
                return {"type": t}
            raise ValueError("thinking= str must be 'enabled' or 'disabled'")
        # thinking is None → auto for DeepSeek endpoints only
        ml = (model or "").lower()
        ul = (base_url or "").lower()
        if "deepseek" not in ml and "deepseek" not in ul:
            return None
        env_t = (os.getenv("LLM_THINKING") or "disabled").strip().lower()
        return {"type": "enabled" if env_t == "enabled" else "disabled"}

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        extra_body: dict = dict(kwargs.get("extra_body") or {})
        if "thinking" in kwargs:
            th = kwargs["thinking"]
            if th is False:
                extra_body.pop("thinking", None)
            elif isinstance(th, dict):
                extra_body["thinking"] = dict(th)
            elif th is True:
                extra_body["thinking"] = {"type": "enabled"}
            elif isinstance(th, str) and th.strip().lower() in ("enabled", "disabled"):
                extra_body["thinking"] = {"type": th.strip().lower()}
            else:
                raise TypeError("thinking= must be False, dict, bool, or 'enabled'/'disabled'")
        elif self._thinking_body is not None:
            extra_body.setdefault("thinking", dict(self._thinking_body))

        payload: dict = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if extra_body:
            payload["extra_body"] = extra_body
        resp = self.client.chat.completions.create(**payload)
        return resp.choices[0].message.content.strip()


# ── Dummy backend for testing without real API ──


class DummyBackend(LLMBackend):
    """Returns canned responses – useful for pipeline debugging."""

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        if "聚类" in last_user or ("clusters" in last_user and "canonical" in last_user):
            return self._handle_cluster_prompt(last_user)

        if "[TASK: audit_taxonomy_labels_v1]" in last_user:
            return (
                '{"verdict": "agree", "rationale": "dummy backend: skip real audit", '
                '"suggested_category": null, "confidence": 0.5}'
            )

        if "[TASK: polish_taxonomy_v1]" in last_user:
            return (
                '{"rewritten_question": "dummy polished text for pipeline test.", '
                '"rewrite_rationale": "占位说明。", '
                '"key_enhancements": ["增强1", "增强2"]}'
            )

        step = len([m for m in messages if m.role == "assistant"])
        if step == 0:
            return (
                "这是第一步的分析回复（测试模式）。\n"
                "建议的核心在于边际效用递减原理。"
            )
        if step == 1:
            return (
                "这是第二步的深层洞察回复（测试模式）。\n"
                "底层规律是信息不对称导致的委托-代理问题。"
            )
        return (
            '```json\n'
            '{\n'
            '  "models": [\n'
            '    {\n'
            '      "name": "边际效用递减",\n'
            '      "variables": "U(x) 表示效用函数，x 为消费量",\n'
            '      "explanation": "随着投入增加，每单位额外收益递减",\n'
            '      "confidence": 7,\n'
            '      "confidence_reason": "问题涉及主观偏好，模型适用但非完美"\n'
            '    },\n'
            '    {\n'
            '      "name": "纳什均衡",\n'
            '      "variables": "参与人策略集合 S_i，收益函数 u_i",\n'
            '      "explanation": "双方陷入非合作博弈的稳态",\n'
            '      "confidence": 6,\n'
            '      "confidence_reason": "博弈框架适用但参与人有限"\n'
            '    }\n'
            '  ]\n'
            '}\n'
            '```'
        )

    @staticmethod
    def _handle_cluster_prompt(prompt: str) -> str:
        """Parse the numbered name list from the cluster prompt and return
        canned cluster results that group similar dummy model names."""
        names: list[str] = re.findall(r"^\d+\.\s*(.+)$", prompt, re.MULTILINE)
        
        cluster_rules = {
            "边际效用递减": ["边际效用", "marginal utility", "递减效用"],
            "纳什均衡": ["nash", "均衡", "博弈"],
            "信息不对称": ["信息不对称", "information asymmetry"],
            "帕累托最优": ["帕累托", "pareto"],
        }
        
        clusters: dict[str, list[str]] = {}
        assigned: set[str] = set()
        
        for name in names:
            name_lower = name.lower()
            matched = False
            for canonical, keywords in cluster_rules.items():
                if any(kw in name_lower or kw in name for kw in keywords):
                    if canonical not in clusters:
                        clusters[canonical] = []
                    clusters[canonical].append(name)
                    assigned.add(name)
                    matched = True
                    break
            if not matched and name not in assigned:
                clusters[name] = [name]
                assigned.add(name)
        
        result = {
            "clusters": [
                {"canonical": canonical, "members": members}
                for canonical, members in clusters.items()
            ]
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


# ── Factory ──


def create_backend(backend_type: str = "openai", **kwargs) -> LLMBackend:
    """Create an LLM backend by name.

    Supported types:
        - "openai"  : OpenAI-compatible API (default)
        - "dummy"   : Canned test responses
    """
    registry: dict[str, type[LLMBackend]] = {
        "openai": OpenAICompatibleBackend,
        "dummy": DummyBackend,
    }
    cls = registry.get(backend_type)
    if cls is None:
        raise ValueError(f"Unknown backend '{backend_type}'. Choose from: {list(registry)}")
    return cls(**kwargs)
