"""Ground Truth schema for dual-granularity evaluation.

Two granularity levels are evaluated independently:
  - File-level: baseline routing accuracy (does the system surface the right files?).
  - Entity-level: AST precision (does it pinpoint the right functions/interfaces/etc.?).

Superset rule: every file path that appears in ``impacted_entities`` must also appear
in ``impacted_files``.  ``impacted_files`` may additionally contain file-only impacts
(e.g. a configuration file with no named entity of interest).
"""

from __future__ import annotations

from pydantic import BaseModel


class ImpactedFile(BaseModel):
    file_path: str
    justification: str


class ImpactedEntity(BaseModel):
    node: str
    justification: str


class GTEntry(BaseModel):
    """One annotated Change Request in the Ground Truth dataset."""

    cr_id: str
    cr_description: str
    impacted_files: list[ImpactedFile]
    impacted_entities: list[ImpactedEntity]

    def file_paths(self) -> set[str]:
        return {f.file_path for f in self.impacted_files}

    def entity_node_ids(self) -> set[str]:
        return {e.node for e in self.impacted_entities}
