"""Ablation harness, metrics, statistical tests, NFR verification.

Modules:
    variant_flags     - V0..V7 variant definitions
    ablation          - Harness: run all variants per CR
    metrics           - Precision@K, Recall@K, F1@K
    statistical       - Wilcoxon paired test (V7 vs V5) + Cliff's delta
    report_builder    - CSV / JSON aggregation outputs
    nfr_verify        - NFR-01..NFR-05 verification procedures
    annotator_tool    - Ground Truth construction CLI helper
"""
