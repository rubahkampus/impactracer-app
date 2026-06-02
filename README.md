# ImpacTracer v4.0

Change Impact Analysis tool that traces impact from natural-language Change
Requests down to source code via a RAG framework over SRS, SDD, and code.

## Status

Implemented and in the evaluation/calibration phase. The authoritative
specification is `master_blueprint.md` (repo root), with operational detail in
`index_implementation.md` (offline indexer) and `analysis_implementation.md`
(online pipeline). `pipeline.md` is a three-layer explainer for the thesis
committee, and `INDEX_REPORT.md` compares the indexed target repositories
(citrakara vs NOVA) for the Bab V evaluation.

## Quick Start

```bash
# 1. Clone and enter
git clone <repo>
cd impactracer

# 2. Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt
pip install -e .

# 4. Configure
cp .env.template .env
# Edit .env and set OPENROUTER_API_KEY

# 5. Index a target repository (default profile "citrakara" -> ./data/citrakara/)
impactracer index /path/to/citrakara

#    A second repo gets its own profile/index — no re-indexing between them:
impactracer index /path/to/NOVA --profile nova       # -> ./data/nova/

# 6. Analyze a Change Request (picks the index by --profile; default citrakara)
impactracer analyze "Tambahkan fitur duplikasi komisi pada dashboard"
impactracer analyze "Add audit log to checkout" --profile nova

# 7. Run full evaluation (after GT dataset prepared)
#    --output defaults to ./eval/results/<profile>/
impactracer evaluate --dataset ./ground_truth/citrakara_final
impactracer evaluate --dataset ./ground_truth/nova_final --profile nova
```

## LLM Provider

OpenRouter is the exclusive transport, accessed via a raw `httpx` POST to
`https://openrouter.ai/api/v1/chat/completions` (JSON mode). No direct
Google/Gemini or OpenAI SDK is used. Every call is constrained by a Pydantic v2
`response_schema` and validated with `model_validate_json`. See
`pipeline/llm_client.py`.

Recommended models (OpenRouter `provider/model` IDs):
- `google/gemini-2.5-flash` (default: fast, cost-efficient)
- `google/gemini-2.5-pro` (quality: use for final evaluation runs)

## Index Profiles

Each target repo gets its own on-disk index under `./data/<profile>/` (SQLite +
ChromaDB + project skeleton), so you can keep e.g. `citrakara` and `nova` indexed
simultaneously and switch between them without re-indexing.

- Select per command with `--profile NAME` (or `-p`), or per shell with the
  `IMPACTRACER_PROFILE` env var. Precedence: `--profile` > `IMPACTRACER_PROFILE`
  > default (`citrakara`).
- `get_settings(profile)` derives all store paths from the profile, so any
  `DB_PATH`/`CHROMA_PATH`/`LLM_AUDIT_LOG_PATH`/`LOCKED_PARAMETERS_PATH` in `.env`
  are ignored.
- `analyze`/`evaluate` fail fast with a clear message if the chosen profile has
  not been indexed yet.

## Architectural Packages

Follows the Bab III package structure:
- `shared/` - Cross-cutting contracts (models, config, constants)
- `indexer/` - Offline KR construction (FR-A1..A7)
- `pipeline/` - Online analysis (FR-B1..E3) + LLM client
- `persistence/` - SQLite and ChromaDB clients
- `evaluation/` - Ablation harness, metrics, NFR verification

## Sprint Plan

Twelve sprints from foundation to NFR verification. See
`12_project_structure_and_sprints.md` for acceptance criteria.

## License

See LICENSE file.
