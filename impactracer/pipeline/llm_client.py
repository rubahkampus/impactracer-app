"""OpenRouter LLM client.

All five LLM invocations (interpret, validate_sis, validate_trace,
validate_propagation, synthesize) flow through :meth:`LLMClient.call`.
All requests are routed exclusively through the OpenRouter API via
``httpx``. No other LLM SDK is used or imported.

The client enforces:

  - ``temperature=0`` and configured ``seed`` for determinism.
  - Pydantic v2 ``response_schema`` constraint on every call via JSON mode.
  - Retry with exponential backoff for transient 429/5xx errors.
  - Session-scoped ``config_hash`` logged per call for NFR-05 audit.
  - JSONL audit entry appended to ``settings.llm_audit_log_path`` on every
    successful call.

Transport:
  HTTP POST to ``https://openrouter.ai/api/v1/chat/completions`` with
  ``Authorization: Bearer <openrouter_api_key>``. JSON mode is enforced via
  ``response_format: {"type": "json_object"}``; the response content is
  parsed against ``response_schema`` using ``model_validate_json``.

Reference: master_blueprint.md §7.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel

from impactracer.shared.config import Settings

T = TypeVar("T", bound=BaseModel)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMClient:
    """Single entry point for every LLM invocation in the pipeline."""

    def __init__(self, settings: Settings) -> None:
        """Construct the client and compute the session config hash."""
        self.settings = settings
        self.call_counter: int = 0
        self.session_config_hash: str = self._compute_config_hash()
        self._http_client = httpx.Client(timeout=120.0)

        Path(settings.llm_audit_log_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "LLMClient initialized: model={}, config_hash={}",
            settings.llm_model,
            self.session_config_hash,
        )

    def _compute_config_hash(self) -> str:
        """SHA-256 digest of model + temperature + seed; used by NFR-05."""
        payload = json.dumps(
            {
                "model": self.settings.llm_model,
                "temperature": self.settings.llm_temperature,
                "seed": self.settings.llm_seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def call(
        self,
        system: str,
        user: str,
        response_schema: type[T],
        call_name: str,
    ) -> T:
        """Invoke OpenRouter with structured output.

        Args:
            system: System-level instruction string.
            user: User message content.
            response_schema: Pydantic v2 class constraining the response.
            call_name: Audit label (e.g. ``"interpret"``, ``"validate_sis"``).

        Returns:
            A parsed instance of ``response_schema``.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        self.call_counter += 1
        logger.info(
            "LLM call #{} [{}] model={} config_hash={}",
            self.call_counter,
            call_name,
            self.settings.llm_model,
            self.session_config_hash,
        )

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.llm_temperature,
            "seed": self.settings.llm_seed,
            "max_tokens": self.settings.llm_max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        attempts = 0
        while True:
            try:
                response = self._http_client.post(
                    _OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise _TransientHTTPError(response.status_code)
                response.raise_for_status()
                raw_json = response.json()["choices"][0]["message"]["content"]
                result = response_schema.model_validate_json(raw_json)
                self._append_audit_entry(call_name)
                return result
            except Exception as exc:
                attempts += 1
                if attempts >= self.settings.llm_retry_max_attempts:
                    raise
                if not _is_transient(exc):
                    raise
                backoff = self.settings.llm_retry_base_backoff * (2 ** (attempts - 1))
                logger.warning(
                    "Transient OpenRouter error on call #{} [{}] attempt {}; "
                    "retrying in {:.1f}s: {}",
                    self.call_counter,
                    call_name,
                    attempts,
                    backoff,
                    exc,
                )
                time.sleep(backoff)

    def _append_audit_entry(self, call_name: str) -> None:
        """Append one JSONL line to the audit log (NFR-05)."""
        entry = json.dumps(
            {
                "call_index": self.call_counter,
                "call_name": call_name,
                "config_hash": self.session_config_hash,
                "model": self.settings.llm_model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        with open(self.settings.llm_audit_log_path, "a", encoding="utf-8") as fh:
            fh.write(entry + "\n")


class _TransientHTTPError(Exception):
    """Raised internally for retriable HTTP status codes."""

    def __init__(self, status_code: int) -> None:
        super().__init__(str(status_code))
        self.status_code = status_code


def _is_transient(exc: BaseException) -> bool:
    """True for 429/5xx HTTP errors and network-level timeouts."""
    if isinstance(exc, _TransientHTTPError):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "500", "502", "503", "504", "timeout", "connection", "unavailable")
    )
