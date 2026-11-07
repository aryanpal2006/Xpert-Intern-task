"""LLM client for OpenAI-compatible APIs with disk caching, token tracking, retry, and concurrency."""
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

CACHE_DIR = Path(".cache_llm")


class LLMClient:
    def __init__(self, cache_dir: Optional[Path] = None, no_llm: bool = False):
        self.no_llm = no_llm or os.environ.get("NO_LLM", "").lower() in ("1", "true", "yes")
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.default_model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        self.strong_model = os.environ.get("LLM_MODEL_STRONG", self.default_model)
        try:
            self.max_workers = max(1, int(os.environ.get("LLM_WORKERS", "4")))
        except ValueError:
            self.max_workers = 4

        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.cached_calls = 0

    @property
    def is_available(self) -> bool:
        return bool(not self.no_llm and self.base_url and self.api_key and self.default_model)

    def _cache_key(self, model: str, messages: List[Dict[str, Any]], temperature: float, response_format: Any) -> str:
        payload = json.dumps({"model": model, "messages": messages, "temperature": temperature, "format": response_format}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        strong: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: int = 90,
        max_retries: int = 5,
    ) -> Optional[str]:
        """Send chat completion with disk caching, retry, and usage tracking."""
        if not self.is_available:
            return None

        chosen_model = model or (self.strong_model if strong else self.default_model)
        ckey = self._cache_key(chosen_model, messages, temperature, response_format)
        cache_file = self.cache_dir / f"{ckey}.json"

        # Check disk cache
        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                with self._lock:
                    self.cached_calls += 1
                return cached_data.get("content")
            except Exception:
                pass

        body: Dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if response_format:
            body["response_format"] = response_format

        endpoint = f"{self.base_url}/chat/completions"

        def encode_body():
            return json.dumps(body).encode("utf-8")

        req_data = encode_body()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Some gateways (e.g. Cloudflare in front of Groq) block the default urllib UA.
            "User-Agent": "coldstart-challenge/1.0",
        }

        backoff = 2.0
        for attempt in range(max_retries):
            started = time.time()
            try:
                req = urllib.request.Request(endpoint, data=req_data, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)

                usage = data.get("usage") or {}
                p_toks = usage.get("prompt_tokens", 0) or 0
                c_toks = usage.get("completion_tokens", 0) or 0
                elapsed = round(time.time() - started, 2)
                with self._lock:
                    self.total_prompt_tokens += p_toks
                    self.total_completion_tokens += c_toks
                    self.total_calls += 1
                    # Log usage to usage.jsonl
                    try:
                        with open("usage.jsonl", "a", encoding="utf-8") as uf:
                            uf.write(json.dumps({
                                "model": chosen_model,
                                "prompt_tokens": p_toks,
                                "completion_tokens": c_toks,
                                "seconds": elapsed,
                            }) + "\n")
                    except Exception:
                        pass

                content = data["choices"][0]["message"]["content"]
                # Save to cache
                try:
                    cache_file.write_text(json.dumps({"content": content, "usage": usage}, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

                return content

            except urllib.error.HTTPError as e:
                # Some models (notably reasoning models like gpt-oss on Groq) can fail strict
                # json_object mode on large outputs. Drop response_format and retry as free text;
                # chat_json still parses the JSON out of the reply.
                if e.code == 400 and "response_format" in body:
                    try:
                        detail = e.read().decode("utf-8", "replace")
                    except Exception:
                        detail = ""
                    if "json_validate_failed" in detail or "response_format" in detail:
                        body.pop("response_format", None)
                        req_data = encode_body()
                        continue
                # Handle rate limits or temporary server errors.
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    wait = backoff
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    if retry_after:
                        try:
                            wait = max(wait, float(retry_after))
                        except ValueError:
                            pass
                    time.sleep(min(wait, 30.0))
                    backoff *= 2
                    continue
                return None
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(min(backoff, 30.0))
                    backoff *= 2
                    continue
                return None
        return None

    def chat_json(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        strong: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
    ) -> Optional[Any]:
        """Request JSON object and parse it safely with retry."""
        resp = self.chat(
            messages=messages,
            model=model,
            strong=strong,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        if not resp:
            return None
        # Clean markdown fences if any
        cleaned = resp.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            # Last resort: grab the outermost JSON object/array if the model wrapped it in prose.
            for open_ch, close_ch in (("{", "}"), ("[", "]")):
                i, j = cleaned.find(open_ch), cleaned.rfind(close_ch)
                if 0 <= i < j:
                    try:
                        return json.loads(cleaned[i:j + 1])
                    except Exception:
                        continue
            return None

    def map_json(
        self,
        payloads: List[List[Dict[str, Any]]],
        strong: bool = False,
        max_tokens: Optional[int] = None,
        timeout: int = 90,
    ) -> List[Optional[Any]]:
        """Run many chat_json calls concurrently, preserving input order."""
        if not self.is_available or not payloads:
            return [None] * len(payloads)
        results: List[Optional[Any]] = [None] * len(payloads)
        workers = min(self.max_workers, len(payloads))

        def _run(idx_msgs):
            idx, msgs = idx_msgs
            return idx, self.chat_json(msgs, strong=strong, max_tokens=max_tokens, timeout=timeout)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for idx, res in ex.map(_run, list(enumerate(payloads))):
                results[idx] = res
        return results

    def cost_summary(self) -> Dict[str, Any]:
        return {
            "calls": self.total_calls,
            "cached": self.cached_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }

