"""Online analysis pipeline (FR-B1..E3) with five LLM invocations.

Modules:
    llm_client                 - Gemini google-genai wrapper, cross-cutting
    interpreter                - LLM #1 (FR-B1, FR-B2)
    retriever                  - FR-C1, FR-C2 (Adaptive RRF)
    prevalidation_filter       - FR-C4 (Steps 3.5, 3.6, 3.7)
    validator                  - LLM #2 (FR-C5)
    seed_resolver              - FR-C6
    traceability_validator     - LLM #3 (FR-C7)
    graph_bfs                  - FR-D1
    traversal_validator        - LLM #4 (FR-D2)
    context_builder            - FR-E1, FR-E2
    synthesizer                - LLM #5 (FR-E3)
    runner                     - Nine-step orchestrator
"""
