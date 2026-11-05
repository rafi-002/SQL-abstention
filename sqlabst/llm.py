"""Chat clients. All return a ChatReply with tool calls and usage; cost is in USD.

OpenRouterClient  - real, paid. Key from env OPENROUTER_API_KEY (never stored in files).
ScriptedClient    - free; replays a fixed list of replies (unit tests).
GoldClient        - free; explores once, then submits the released gold SQL (smoke test of the whole pipeline).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""


@dataclass
class ChatReply:
    content: str | None
    tool_calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_source: str = "none"          # 'provider' | 'price_table' | 'none'
    model: str = ""
    finish_reason: str | None = None
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)


class ScriptedClient:
    name = "scripted"

    def __init__(self, replies: list[ChatReply]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools, **kw) -> ChatReply:
        self.calls.append(json.loads(json.dumps(messages)))
        if not self.replies:
            return ChatReply("(script exhausted)", [], model="scripted")
        return self.replies.pop(0)


class GoldClient:
    """Free pipeline check: one exploration query, then submit the gold SQL given at construction."""
    name = "gold"

    def __init__(self, gold_sql: str):
        self.gold_sql = gold_sql
        self.step = 0

    def chat(self, messages, tools, **kw) -> ChatReply:
        self.step += 1
        if self.step == 1:
            return ChatReply("Looking at the tables first.", [ToolCall("c1", "execute_sql",
                             {"sql": "SELECT name FROM sqlite_master WHERE type='table' LIMIT 5"})],
                             input_tokens=0, output_tokens=0, model="gold")
        return ChatReply("Submitting.", [ToolCall("c2", "submit_sql", {"sql": self.gold_sql})], model="gold")


class OpenRouterClient:
    name = "openrouter"
    URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, price_in_per_m: float, price_out_per_m: float, temperature: float = 0.7,
                 max_output_tokens: int = 2048, retries: int = 5, backoff_cap_s: float = 60.0,
                 provider_order: list[str] | None = None, request_timeout_s: float = 180.0):
        from openai import OpenAI  # imported lazily so free paths never need a key
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in this shell")
        self.client = OpenAI(base_url=self.URL, api_key=key, timeout=request_timeout_s, max_retries=0)
        self.model, self.temperature, self.max_out = model, temperature, max_output_tokens
        self.p_in, self.p_out = price_in_per_m, price_out_per_m
        self.retries, self.backoff_cap = retries, backoff_cap_s
        self.provider_order = provider_order

    def chat(self, messages, tools, seed: int | None = None, **kw) -> ChatReply:
        import openai
        extra = {"usage": {"include": True}}
        if self.provider_order:
            extra["provider"] = {"order": self.provider_order, "allow_fallbacks": False}
        last = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, tool_choice="auto",
                    temperature=self.temperature, max_tokens=self.max_out, seed=seed, extra_body=extra)
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as e:
                last = e
                ra = getattr(getattr(e, "response", None), "headers", {}) or {}
                wait = float(ra.get("retry-after", 0) or 0) or min(self.backoff_cap, 2 ** attempt)
                time.sleep(min(wait, self.backoff_cap))
        else:
            raise RuntimeError(f"model call failed after {self.retries} retries: {last!r}")
        msg = r.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args, tc.function.arguments or ""))
        u = r.usage
        it, ot = (u.prompt_tokens or 0, u.completion_tokens or 0) if u else (0, 0)
        cost = getattr(u, "cost", None) if u else None
        if cost is None and u is not None:
            cost = (u.model_extra or {}).get("cost")
        src = "provider"
        if cost is None:
            cost, src = it * self.p_in / 1e6 + ot * self.p_out / 1e6, "price_table"
        return ChatReply(msg.content, calls, it, ot, float(cost), src, r.model, r.choices[0].finish_reason,
                         time.time() - t0)
