"""ImpacTracer v4.0 - Change Impact Analysis via RAG + Structural Graph.

Package structure mirrors the Bab III architectural packages:
- shared: cross-cutting data contracts and configuration
- indexer: offline knowledge representation construction
- pipeline: online analysis pipeline with OpenRouter LLM client
- persistence: SQLite and ChromaDB client modules
- evaluation: ablation harness, metrics, NFR verification
"""

__version__ = "4.0.0"
