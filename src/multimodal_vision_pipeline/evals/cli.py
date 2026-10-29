"""``python -m multimodal_vision_pipeline.evals``: benchmark a vision model on synthetic invoices.

Calls the real Messages API (``MVP_ANTHROPIC_API_KEY`` and the other ``MVP_*`` settings
apply), once per document per profile, and writes ``benchmark.json`` and ``benchmark.md``.
"""

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from multimodal_vision_pipeline.evals.dataset import generate_dataset
from multimodal_vision_pipeline.evals.perturbation import PROFILES
from multimodal_vision_pipeline.evals.runner import (
    BenchmarkExtractor,
    BenchmarkReport,
    VisionPipelineExtractor,
    run_benchmark,
)
from multimodal_vision_pipeline.schemas import VisionModelId, VisionSettings
from multimodal_vision_pipeline.tools.vision_client import VisionClient

type ExtractorFactory = Callable[
    [VisionSettings, VisionModelId | None], AbstractAsyncContextManager[BenchmarkExtractor]
]

DEFAULT_OUTPUT_DIR = Path(".cache/evals")


@asynccontextmanager
async def vision_extractor(
    settings: VisionSettings, model: VisionModelId | None
) -> AsyncIterator[BenchmarkExtractor]:
    """The production client, closed when the benchmark ends."""
    config = settings.extraction_config()
    if model is not None:
        config = config.model_copy(update={"model": model})
    async with VisionClient.from_settings(settings) as client:
        yield VisionPipelineExtractor(client, config=config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m multimodal_vision_pipeline.evals",
        description="Benchmark extraction accuracy on synthetic, degraded invoices.",
    )
    parser.add_argument("--documents", type=int, default=10, help="documents per profile")
    parser.add_argument("--seed", type=int, default=0, help="dataset and noise seed")
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILES),
        help=f"comma-separated degradation profiles (available: {', '.join(PROFILES)})",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="documents in flight")
    parser.add_argument(
        "--model",
        choices=[model.value for model in VisionModelId],
        help="override MVP_VISION_MODEL",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where reports are written"
    )
    return parser


async def _benchmark(
    args: argparse.Namespace, settings: VisionSettings, factory: ExtractorFactory
) -> BenchmarkReport:
    model = None if args.model is None else VisionModelId(args.model)
    async with factory(settings, model) as extractor:
        return await run_benchmark(
            extractor,
            generate_dataset(args.documents, seed=args.seed),
            profiles=[PROFILES[name] for name in args.profiles],
            seed=args.seed,
            max_concurrency=args.concurrency,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: VisionSettings | None = None,
    extractor_factory: ExtractorFactory = vision_extractor,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.profiles = [name.strip() for name in args.profiles.split(",") if name.strip()]
    if unknown := [name for name in args.profiles if name not in PROFILES]:
        parser.error(f"unknown profile(s): {', '.join(unknown)}")
    if not args.profiles:
        parser.error("no profiles selected")
    if args.documents < 1 or args.concurrency < 1:
        parser.error("--documents and --concurrency must be >= 1")

    report = asyncio.run(
        _benchmark(args, settings if settings is not None else VisionSettings(), extractor_factory)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(report.to_json(), encoding="utf-8")
    markdown = report.to_markdown()
    (args.output_dir / "benchmark.md").write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    sys.stderr.write(f"reports written to {args.output_dir}\n")
    return 0
