"""Offline knowledge representation construction (FR-A1..A7).

Modules:
    doc_indexer       - Markdown chunking and classification (FR-A1, A2)
    code_indexer      - Tree-sitter AST Pass 1 + Pass 2 (FR-A3, A4)
    skeletonizer      - Internal logic abstraction (FR-A6)
    embedder          - BGE-M3 wrapper (FR-A5)
    reranker          - BGE-Reranker-v2-M3 wrapper (FR-C3)
    traceability      - Layer-weighted doc<->code pair precompute (FR-A7)
    runner            - Orchestrator invoked by ``impactracer index``
"""
