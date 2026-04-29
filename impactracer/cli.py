"""Typer CLI: the three user-facing commands.

Commands:
    index     - Build the knowledge representation for a target repository.
    analyze   - Analyze a Change Request and emit an ImpactReport.
    evaluate  - Run the ablation harness over a Ground Truth dataset.

See 11_configuration_and_cli.md for the full command contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(
    name="impactracer",
    help="Change Impact Analysis via RAG with multilingual embeddings and structural graph.",
    no_args_is_help=True,
)


def _configure_logging(verbose: bool = False) -> None:
    """Configure loguru sinks. Called once per CLI invocation."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")
    log_path = Path("./data/impactracer.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="DEBUG", rotation="10 MB")


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    _configure_logging(verbose=verbose)


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to repository root."),
    force: bool = typer.Option(False, "--force", help="Reindex all files regardless of hash."),
) -> None:
    """Build or update the knowledge representation for a repository.

    Orchestrates the offline indexing pipeline:
      1. Scan repository for .md and .ts/.tsx files.
      2. Check file_hashes for changed/new files (unless --force).
      3. Chunk Markdown (FR-A1, FR-A2).
      4. Parse TypeScript AST (Pass 1: nodes; Pass 2: edges).
      5. Embed all pending chunks and nodes via BGE-M3.
      6. Compute layer-weighted traceability pairs (FR-A7).
      7. Write index_metadata and print statistics.

    Implementation entry point:
        impactracer.indexer.runner.run_indexing(repo_path, settings, force)
    """
    raise NotImplementedError("Sprint 7: offline indexer integration")


@app.command()
def analyze(
    cr_text: str = typer.Argument(..., help="Change Request text (Indonesian or English)."),
    output: Path = typer.Option(Path("./impact_report.json"), "--output", "-o"),
    variant: str = typer.Option("V7", "--variant", help="Ablation variant V0-V7 (default V7)."),
) -> None:
    """Analyze a CR against the indexed repository and emit an ImpactReport.

    Invokes the nine-step online pipeline. V7 runs all five LLM invocations;
    lower variants disable subsets as specified in 09_ablation_harness.md.

    Implementation entry point:
        impactracer.pipeline.runner.run_analysis(cr_text, settings, flags)
    """
    raise NotImplementedError("Sprint 8-10: online pipeline")


@app.command()
def evaluate(
    dataset: Path = typer.Option(..., "--dataset", help="GT dataset JSON file."),
    output_dir: Path = typer.Option(Path("./eval/results/"), "--output"),
    run_full_ablation: bool = typer.Option(True, "--run-full-ablation"),
    verify_nfr: bool = typer.Option(False, "--verify-nfr"),
) -> None:
    """Run the full evaluation protocol on a GT dataset.

    Executes:
      1. V0..V7 ablation across every CR in the dataset.
      2. Metric computation (Precision@10, Recall@10, F1@10).
      3. One pre-registered Wilcoxon signed-rank test (V7 vs V5, F1@10).
      4. Optional NFR-01..NFR-05 verification.

    Implementation entry point:
        impactracer.evaluation.ablation.run_full_evaluation(...)
    """
    raise NotImplementedError("Sprint 11-12: evaluation")


if __name__ == "__main__":
    app()
