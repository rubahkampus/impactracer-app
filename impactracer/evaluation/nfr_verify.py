"""NFR-01..NFR-05 verification procedures.

Reference: 10_evaluation_protocol.md §5.
"""

from __future__ import annotations

from pathlib import Path

from impactracer.shared.config import Settings


def verify_nfr_01(cr_text: str, settings: Settings) -> bool:
    """Determinism of structural components.

    Run V7 twice, compare the validated SIS code node set after Step 5b.
    """
    raise NotImplementedError("Sprint 12")


def verify_nfr_02(settings: Settings) -> bool:
    """Local execution: index + non-LLM stages work with network disabled."""
    raise NotImplementedError("Sprint 12")


def verify_nfr_03(cr_dataset: list[dict], settings: Settings) -> dict[str, float]:
    """Latency distribution: median and p95 across 20 test CRs."""
    raise NotImplementedError("Sprint 12")


def verify_nfr_04(cr_text_id: str, settings: Settings) -> bool:
    """Cross-lingual: Indonesian CR retrieves English-identifier code nodes."""
    raise NotImplementedError("Sprint 12")


def verify_nfr_05(audit_path: Path) -> bool:
    """LLM configuration determinism: all calls in a run share config_hash."""
    raise NotImplementedError("Sprint 12")


def verify_all_nfrs(
    cr_dataset: list[dict],
    settings: Settings,
    output_dir: Path,
) -> dict[str, bool | dict]:
    """Execute NFR-01 through NFR-05 and return a pass/fail dict."""
    raise NotImplementedError("Sprint 12")
