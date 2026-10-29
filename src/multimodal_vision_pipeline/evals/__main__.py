"""Entry point: ``python -m multimodal_vision_pipeline.evals --help``."""

import sys

from multimodal_vision_pipeline.evals.cli import main

if __name__ == "__main__":
    sys.exit(main())
