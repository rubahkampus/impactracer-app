"""Gemini LLM client wrapper (google-genai SDK).

All five LLM invocations (interpret, validate-SIS, validate-trace,
validate-propagation, synthesize) flow through :meth:`LLMClient.call`.
The client enforces:

  - ``temperature=0`` and configured ``seed`` for determinism.
  - Pydantic v2 ``response_schema`` via ``GenerateContentConfig``.
  - Retry with exponential backoff for transient 429/5xx errors.
  - Session-scoped ``config_hash`` logged on every call for NFR-05 audit.
  - Permissive safety settings for technical content (code/CR text).

The google-genai SDK natively supports Pydantic v2 classes as
``response_schema`` and returns a parsed object via ``response.parsed``.

Reference: 07_online_pipeline.md §14.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TypeVar

from google import genai
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel

from impactracer.shared.config import Settings


T = TypeVar("T", bound=BaseModel)


_DEFAULT_SAFETY_SETTINGS: list[genai_types.SafetySetting] = [
    genai_types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_NONE",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_NONE",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_NONE",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_NONE",
    ),
]


class LLMClient:
    """Single entry point for every LLM invocation in the pipeline."""

    def __init__(self, settings: Settings) -> None:
        """Construct the client, compute and cache the session config hash."""
        self.settings = settings
        self._client = genai.Client(api_key=settings.google_api_key)
        self.call_counter: int = 0
        self.session_config_hash: str = self._compute_config_hash()
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
        """Invoke Gemini with structured output.

        Args:
            system: System-level instruction string.
            user: User message content.
            response_schema: Pydantic v2 class used as ``response_schema``.
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

        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=self.settings.llm_temperature,
            seed=self.settings.llm_seed,
            max_output_tokens=self.settings.llm_max_output_tokens,
            safety_settings=_DEFAULT_SAFETY_SETTINGS,
        )

        attempts = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self.settings.llm_model,
                    contents=user,
                    config=config,
                )
                # google-genai populates ``response.parsed`` when
                # response_schema is a Pydantic class. Fall back to
                # JSON text parsing if ``.parsed`` is absent.
                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    return parsed  # type: ignore[return-value]
                return response_schema.model_validate_json(response.text)
            except Exception as exc:
                attempts += 1
                if attempts >= self.settings.llm_retry_max_attempts:
                    raise
                if not _is_transient(exc):
                    raise
                backoff = self.settings.llm_retry_base_backoff * (2 ** (attempts - 1))
                logger.warning(
                    "Transient Gemini error on call #{} [{}] attempt {}; "
                    "retrying in {:.1f}s: {}",
                    self.call_counter, call_name, attempts, backoff, exc,
                )
                time.sleep(backoff)


def _is_transient(exc: BaseException) -> bool:
    """True for 429/5xx and network timeouts."""
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "500", "502", "503", "504", "timeout", "connection", "unavailable")
    )
