"""Extraction evaluation: synthetic labelled invoices, scan degradation, accuracy benchmarks."""

from multimodal_vision_pipeline.evals.dataset import (
    SyntheticDocument,
    generate_dataset,
    generate_document,
)
from multimodal_vision_pipeline.evals.evaluator import (
    AccuracySummary,
    DocumentEvaluation,
    MatchCounts,
    empty_extraction,
    evaluate,
    iou,
    levenshtein,
    summarize,
)
from multimodal_vision_pipeline.evals.perturbation import (
    PROFILES,
    ContrastLoss,
    DegradationProfile,
    GaussianBlur,
    JpegCompression,
    SaltAndPepper,
    Skew,
)
from multimodal_vision_pipeline.evals.runner import (
    BenchmarkExtractor,
    BenchmarkReport,
    VisionPipelineExtractor,
    run_benchmark,
)

__all__ = [
    "PROFILES",
    "AccuracySummary",
    "BenchmarkExtractor",
    "BenchmarkReport",
    "ContrastLoss",
    "DegradationProfile",
    "DocumentEvaluation",
    "GaussianBlur",
    "JpegCompression",
    "MatchCounts",
    "SaltAndPepper",
    "Skew",
    "SyntheticDocument",
    "VisionPipelineExtractor",
    "empty_extraction",
    "evaluate",
    "generate_dataset",
    "generate_document",
    "iou",
    "levenshtein",
    "run_benchmark",
    "summarize",
]
