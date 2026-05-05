"""OpenRouter LLM client.

All five LLM invocations (interpret, validate_sis, validate_trace,
validate_propagation, synthesize) flow through :meth:`LLMClient.call`.
All requests are routed exclusively through the OpenRouter API via
``httpx``. No other LLM SDK is used or imported.

The client enforces:

  - ``temperature=0`` and configured ``seed`` for determinism.
  - Pydantic v2 ``response_schema`` constraint on every call via JSON mode.
  - Retry with exponential backoff for transient 429/5xx errors.
  - Retry-After header respected for 429 responses (Phase 2.10 / E-NEW-5).
  - Session-scoped ``config_hash`` logged per call for NFR-05 audit.
  - JSONL audit entry appended to ``settings.llm_audit_log_path`` on every
    call (success OR failure), with prompt_hash, response_hash, retry_count
    (Phase 2.11 / F-NEW-3 / E-NEW-7).

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

# Per-call-name synthesis timeout override (seconds).
# LLM #5 synthesis receives full context (up to ~60K tokens) and requires
# significantly more time than the shorter validation calls.
_SYNTHESIS_TIMEOUT_S: float = 300.0
_DEFAULT_TIMEOUT_S: float = 120.0
_SYNTHESIS_CALL_NAMES: frozenset[str] = frozenset({"synthesize"})


class LLMClient:
    """Single entry point for every LLM invocation in the pipeline."""

    def __init__(self, settings: Settings) -> None:
        """Construct the client and compute the session config hash."""
        self.settings = settings
        # Phase 2.11 (E-NEW-7): call_counter now increments AFTER a successful
        # call. A separate _attempt_counter tracks pre-call attempts for logging.
        self.call_counter: int = 0
        self.session_config_hash: str = self._compute_config_hash()
        # Keep two clients: one with standard timeout, one with extended timeout
        # for synthesis calls.
        self._http_client = httpx.Client(timeout=_DEFAULT_TIMEOUT_S)
        self._synthesis_client = httpx.Client(timeout=_SYNTHESIS_TIMEOUT_S)

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
        # Phase 2.11 (E-NEW-7): prompt_hash for audit reproducibility.
        prompt_content = system + "\n\n" + user
        prompt_hash = hashlib.sha256(prompt_content.encode("utf-8")).hexdigest()[:16]

        # Use extended timeout for synthesis calls (Phase 2.10 / F-4).
        http_client = (
            self._synthesis_client
            if call_name in _SYNTHESIS_CALL_NAMES
            else self._http_client
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
        last_exc: BaseException | None = None

        # Log the call attempt BEFORE making the request (for debugging).
        logger.info(
            "LLM call [{}] model={} config_hash={} prompt_hash={}",
            call_name,
            self.settings.llm_model,
            self.session_config_hash,
            prompt_hash,
        )

        while True:
            try:
                response = http_client.post(
                    _OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                )

                # Phase 2.10 (E-NEW-5): honour Retry-After header on 429.
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            wait_s = float(retry_after)
                            logger.warning(
                                "429 rate-limit on [{}]; honouring Retry-After={:.0f}s",
                                call_name, wait_s,
                            )
                            time.sleep(wait_s)
                            attempts += 1
                            continue
                        except ValueError:
                            pass
                    raise _TransientHTTPError(response.status_code, response)

                if response.status_code >= 500:
                    raise _TransientHTTPError(response.status_code, response)

                response.raise_for_status()
                raw_json = response.json()["choices"][0]["message"]["content"]

                # Phase 2.11 (F-NEW-3): response_hash for audit.
                response_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:16]

                # Extract token usage if provided by OpenRouter.
                usage = response.json().get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")

                result = response_schema.model_validate_json(raw_json)

                # Phase 2.11 (E-NEW-7): increment call_counter AFTER success.
                self.call_counter += 1
                self._append_audit_entry(
                    call_name=call_name,
                    status="success",
                    prompt_hash=prompt_hash,
                    response_hash=response_hash,
                    retry_count=attempts,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                return result

            except Exception as exc:
                last_exc = exc
                attempts += 1
                if attempts >= self.settings.llm_retry_max_attempts:
                    # Log failed call to audit before raising.
                    self._append_audit_entry(
                        call_name=call_name,
                        status="failed",
                        prompt_hash=prompt_hash,
                        response_hash=None,
                        retry_count=attempts,
                        error=str(exc),
                    )
                    raise
                if not _is_transient(exc):
                    self._append_audit_entry(
                        call_name=call_name,
                        status="failed",
                        prompt_hash=prompt_hash,
                        response_hash=None,
                        retry_count=attempts,
                        error=str(exc),
                    )
                    raise
                backoff = self.settings.llm_retry_base_backoff * (2 ** (attempts - 1))
                logger.warning(
                    "Transient OpenRouter error on [{}] attempt {}; "
                    "retrying in {:.1f}s: {}",
                    call_name,
                    attempts,
                    backoff,
                    exc,
                )
                time.sleep(backoff)

    def close(self) -> None:
        """Release the underlying httpx connection pool."""
        self._http_client.close()
        self._synthesis_client.close()

    def _append_audit_entry(
        self,
        call_name: str,
        status: str,
        prompt_hash: str,
        response_hash: str | None,
        retry_count: int,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        """Append one JSONL line to the audit log (NFR-05).

        Phase 2.11 (F-NEW-3/E-NEW-7): extended fields — prompt_hash,
        response_hash, retry_count, prompt_tokens, completion_tokens, error.
        Failed calls are now logged (status='failed') so the audit trail is
        complete even when the pipeline raises an exception.
        """
        record: dict = {
            "call_index": self.call_counter,
            "call_name": call_name,
            "status": status,
            "config_hash": self.session_config_hash,
            "model": self.settings.llm_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "retry_count": retry_count,
        }
        if prompt_tokens is not None:
            record["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            record["completion_tokens"] = completion_tokens
        if error is not None:
            record["error"] = error[:500]  # cap to prevent bloated log entries

        entry = json.dumps(record)
        with open(self.settings.llm_audit_log_path, "a", encoding="utf-8") as fh:
            fh.write(entry + "\n")


class _TransientHTTPError(Exception):
    """Raised internally for retriable HTTP status codes."""

    def __init__(self, status_code: int, response: httpx.Response | None = None) -> None:
        super().__init__(str(status_code))
        self.status_code = status_code
        self.response = response


def _is_transient(exc: BaseException) -> bool:
    """True for 429/5xx HTTP errors and network-level timeouts."""
    if isinstance(exc, _TransientHTTPError):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "500", "502", "503", "504", "timeout", "connection", "unavailable")
    )
