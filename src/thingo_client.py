"""Thingo API 客户端与精确计费/重试封装（遵循实验增量记录与预算熔断原则）。"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# 单价来源：research/llm/src/research_llm/prices.toml (CNY / 1M tokens)
MODEL_PRICES_CNY_PER_1M: dict[str, dict[str, Any]] = {
    "gemini-3.8-flash": {"input": 0.2556, "output": 1.278, "cache_read": 0.02556},
    "gemini-3.7-flash": {"input": 0.2556, "output": 1.278, "cache_read": 0.02556},
    "gemini-3.6-flash": {"input": 0.2556, "output": 1.278, "cache_read": 0.02556},
    "gemini-3.5-flash": {"input": 0.5112, "output": 3.0672, "cache_read": 0.05112},
    "gemini-3.1-flash-lite": {"input": 0.0852, "output": 0.5112, "cache_read": 0.00852},
    "gpt-6-sol": {
        "input": 0.6816,
        "output": 2.556,
        "cache_read": 0.06816,
        "tiers": [{"max_input_tokens": 272000, "input": 0.3408, "output": 1.704, "cache_read": 0.03408}],
    },
    "gpt-6.1-sol": {
        "input": 0.6816,
        "output": 2.556,
        "cache_read": 0.06816,
        "tiers": [{"max_input_tokens": 272000, "input": 0.3408, "output": 1.704, "cache_read": 0.03408}],
    },
    "gpt-6.2-sol": {
        "input": 0.6816,
        "output": 2.556,
        "cache_read": 0.03408,
        "tiers": [{"max_input_tokens": 272000, "input": 0.3408, "output": 1.704, "cache_read": 0.01704}],
    },
    "gpt-6-astra": {
        "input": 3.408,
        "output": 12.78,
        "cache_read": 0.3408,
        "tiers": [{"max_input_tokens": 272000, "input": 1.704, "output": 8.52, "cache_read": 0.1704}],
    },
    "gpt-6.1-terra": {
        "input": 0.3408,
        "output": 1.278,
        "cache_read": 0.01704,
        "tiers": [{"max_input_tokens": 272000, "input": 0.1704, "output": 0.852, "cache_read": 0.00864}],
    },
}


def load_dotenv(env_path: str | Path = ".env") -> dict[str, str]:
    """从本地 .env 读取环境变量（不覆盖进程已有环境变量）。"""
    loaded: dict[str, str] = {}
    p = Path(env_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            loaded[k] = v
            os.environ.setdefault(k, v)
    return loaded


def estimate_cost_cny(model: str, usage: dict[str, Any] | None) -> float:
    """根据 Thingo 返回的 usage 字段估算单次请求费用（人民币元）。"""
    if not usage:
        return 0.0
    price = MODEL_PRICES_CNY_PER_1M.get(model)
    if not price:
        return 0.0
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    uncached_prompt = max(0, prompt_tokens - cached_tokens)
    selected = price
    for tier in price.get("tiers") or []:
        if prompt_tokens <= int(tier["max_input_tokens"]):
            selected = tier
            break
    cost = (
        uncached_prompt * selected["input"]
        + cached_tokens * selected["cache_read"]
        + completion_tokens * selected["output"]
    ) / 1_000_000.0
    return cost


@dataclass
class ChatResult:
    ok: bool
    content: str
    usage: dict[str, Any]
    cost_cny: float
    latency_sec: float
    finish_reason: str | None = None
    error: str | None = None


class BudgetExceededError(RuntimeError):
    """超出本次实验设定的人民币预算上限时触发。"""


class ThingoClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gemini-3.8-flash",
        max_budget_cny: float = 8.0,
        timeout_sec: int = 120,
        api_key_env: str | None = None,
    ) -> None:
        load_dotenv()
        env_name = api_key_env or ("THINGO_GPT_API_KEY" if model.startswith("gpt-") and os.environ.get("THINGO_GPT_API_KEY") else "THINGO_INTL_API_KEY")
        self.api_key = api_key or os.environ.get(env_name, "")
        self.base_url = (base_url or os.environ.get("THINGO_INTL_BASE_URL") or "https://intlapi.thingo.com.cn/v1").rstrip("/")
        if not self.api_key:
            raise ValueError(f"缺少 {env_name}，请在 .env 或环境变量中配置。")
        self.model = model
        self.max_budget_cny = max_budget_cny
        self.timeout_sec = timeout_sec
        self.spent_cny = 0.0
        self._lock = threading.Lock()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 2,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        with self._lock:
            if self.spent_cny >= self.max_budget_cny:
                raise BudgetExceededError(
                    f"已达到预算上限 ¥{self.max_budget_cny:.4f}（当前累计 ¥{self.spent_cny:.4f}）"
                )

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last_err = ""
        for attempt in range(max_retries + 1):
            t0 = time.time()
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                dt = time.time() - t0
                choice = (raw.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                usage = raw.get("usage") or {}
                cost = estimate_cost_cny(self.model, usage)
                with self._lock:
                    self.spent_cny += cost
                return ChatResult(
                    ok=True,
                    content=content,
                    usage=usage,
                    cost_cny=cost,
                    latency_sec=dt,
                    finish_reason=choice.get("finish_reason"),
                )
            except urllib.error.HTTPError as e:
                dt = time.time() - t0
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                last_err = f"HTTP {e.code}: {err_body}"
                if e.code in (400, 401, 403):
                    return ChatResult(ok=False, content="", usage={}, cost_cny=0.0, latency_sec=dt, error=last_err)
            except Exception as e:
                dt = time.time() - t0
                last_err = f"{type(e).__name__}: {e}"

            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))

        return ChatResult(ok=False, content="", usage={}, cost_cny=0.0, latency_sec=0.0, error=last_err)
