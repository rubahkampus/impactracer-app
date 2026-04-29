# ImpacTracer v4.0

Change Impact Analysis tool that traces impact from natural-language Change
Requests down to source code via a RAG framework over SRS, SDD, and code.

## Status

Greenfield scaffold. Implementation follows the twelve-file Master Blueprint
(paste the Master Blueprint into the repository root; they are the authoritative build specification).

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
# Edit .env and set GOOGLE_API_KEY

# 5. Index a target repository
impactracer index /path/to/target/repo

# 6. Analyze a Change Request
impactracer analyze "Tambahkan fitur duplikasi komisi pada dashboard"

# 7. Run full evaluation (after GT dataset prepared)
impactracer evaluate --dataset ./eval/gt/ --output ./eval/results/
```

## LLM Provider

Gemini via `google-genai` SDK. Pydantic v2 schemas are passed natively as
`response_schema` in `GenerateContentConfig`. See `pipeline/llm_client.py`.

Recommended models:
- `gemini-2.5-flash` (default: fast, cost-efficient, structured-output native)
- `gemini-2.5-pro` (quality: use for final evaluation runs)

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
