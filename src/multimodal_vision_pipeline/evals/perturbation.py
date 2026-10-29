"""Scan-defect simulation: deterministic, in-memory Pillow transforms of page images.

Every perturbation keeps the image size and mode. Geometric ones (``Skew``) also map
ground-truth bounding boxes into the degraded page, so anchors stay comparable.
"""

import io
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal, override

from PIL import Image, ImageEnhance, ImageFilter
from pydantic import Field

from multimodal_vision_pipeline.schemas import BoundingBox, PipelineModel

MAX_SKEW_DEGREES: Final = 15.0
SUPPORTED_MODES: Final = frozenset({"RGB", "L"})

type BoxTransform = Callable[[BoundingBox], BoundingBox | None]
"""Maps a normalized box into the degraded page; ``None`` if it left the page."""


def _check_mode(image: Image.Image) -> None:
    if image.mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported image mode {image.mode!r}; expected RGB or L")


def _identity(box: BoundingBox) -> BoundingBox | None:
    return box


class PerturbationStep(PipelineModel, ABC):
    """One transform. Same size and mode out as in; randomness only from ``rng``."""

    @abstractmethod
    def apply(self, image: Image.Image, rng: random.Random) -> Image.Image: ...

    def box_transform(self, size: tuple[int, int]) -> BoxTransform:  # noqa: ARG002
        """Photometric steps leave geometry alone."""
        return _identity


class GaussianBlur(PerturbationStep):
    """Out-of-focus optics or a dirty scanner glass."""

    kind: Literal["gaussian_blur"] = "gaussian_blur"
    radius: float = Field(gt=0, le=10, allow_inf_nan=False)

    @override
    def apply(self, image: Image.Image, rng: random.Random) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.radius))


class SaltAndPepper(PerturbationStep):
    """Impulse noise: ``amount`` of all pixels flipped to pure black or white (half each)."""

    kind: Literal["salt_and_pepper"] = "salt_and_pepper"
    amount: float = Field(gt=0, le=0.5, allow_inf_nan=False)

    @override
    def apply(self, image: Image.Image, rng: random.Random) -> Image.Image:
        # One random byte per pixel; the lowest and highest bands become pepper and salt.
        noise = Image.frombytes("L", image.size, rng.randbytes(image.width * image.height))
        band = round(self.amount * 128)
        pepper = noise.point([255 if value < band else 0 for value in range(256)])
        salt = noise.point([255 if value >= 256 - band else 0 for value in range(256)])
        black = Image.new(image.mode, image.size, 0)
        white = Image.new(image.mode, image.size, "white")
        return Image.composite(white, Image.composite(black, image, pepper), salt)


class Skew(PerturbationStep):
    """Page placed at an angle: counter-clockwise rotation about the centre, white fill."""

    kind: Literal["skew"] = "skew"
    degrees: float = Field(ge=-MAX_SKEW_DEGREES, le=MAX_SKEW_DEGREES, allow_inf_nan=False)

    @override
    def apply(self, image: Image.Image, rng: random.Random) -> Image.Image:
        fill: int | tuple[int, int, int] = 255 if image.mode == "L" else (255, 255, 255)
        return image.rotate(
            self.degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill
        )

    @override
    def box_transform(self, size: tuple[int, int]) -> BoxTransform:
        width, height = size
        cx, cy = width / 2, height / 2
        theta = math.radians(self.degrees)
        cos, sin = math.cos(theta), math.sin(theta)

        def rotate(box: BoundingBox) -> BoundingBox | None:
            corners = [
                (box.x0 * width, box.y0 * height),
                (box.x1 * width, box.y0 * height),
                (box.x0 * width, box.y1 * height),
                (box.x1 * width, box.y1 * height),
            ]
            # Image y grows downwards, so a visually counter-clockwise turn is +sin on x.
            xs = [cx + (x - cx) * cos + (y - cy) * sin for x, y in corners]
            ys = [cy - (x - cx) * sin + (y - cy) * cos for x, y in corners]
            x0, x1 = max(0.0, min(xs) / width), min(1.0, max(xs) / width)
            y0, y1 = max(0.0, min(ys) / height), min(1.0, max(ys) / height)
            if x0 >= x1 or y0 >= y1:
                return None
            return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

        return rotate


class ContrastLoss(PerturbationStep):
    """Faded toner or over-exposure: ``factor`` 1.0 is unchanged, lower is flatter."""

    kind: Literal["contrast_loss"] = "contrast_loss"
    factor: float = Field(gt=0, le=1, allow_inf_nan=False)

    @override
    def apply(self, image: Image.Image, rng: random.Random) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(self.factor)


class JpegCompression(PerturbationStep):
    """Lossy re-encoding round trip; low ``quality`` gives blocking and ringing artifacts."""

    kind: Literal["jpeg_compression"] = "jpeg_compression"
    quality: int = Field(ge=1, le=95)

    @override
    def apply(self, image: Image.Image, rng: random.Random) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert(image.mode)


Perturbation = Annotated[
    GaussianBlur | SaltAndPepper | Skew | ContrastLoss | JpegCompression,
    Field(discriminator="kind"),
]


@dataclass(frozen=True, slots=True)
class DegradedPage:
    image: Image.Image
    transform_box: BoxTransform


class DegradationProfile(PipelineModel):
    """A named, ordered chain of perturbations, e.g. ``office_scan``."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    steps: tuple[Perturbation, ...] = ()

    def apply(self, image: Image.Image, *, seed: int | str) -> DegradedPage:
        """Degrade ``image``; the same seed always yields the same pixels."""
        _check_mode(image)
        # Seeded so a benchmark run is reproducible; not used for anything secret.
        rng = random.Random(seed)  # noqa: S311  # nosec B311
        transforms: list[BoxTransform] = []
        current = image
        for step in self.steps:
            current = step.apply(current, rng)
            transforms.append(step.box_transform(image.size))
        return DegradedPage(image=current, transform_box=_compose(transforms))


def _compose(transforms: Sequence[BoxTransform]) -> BoxTransform:
    def apply(box: BoundingBox) -> BoundingBox | None:
        current: BoundingBox | None = box
        for transform in transforms:
            if current is None:
                return None
            current = transform(current)
        return current

    return apply


PROFILES: Final[dict[str, DegradationProfile]] = {
    profile.name: profile
    for profile in (
        DegradationProfile(name="clean"),
        DegradationProfile(
            name="office_scan",
            steps=(Skew(degrees=1.5), GaussianBlur(radius=0.6), JpegCompression(quality=75)),
        ),
        DegradationProfile(
            name="low_quality_scan",
            steps=(
                Skew(degrees=-4.0),
                GaussianBlur(radius=1.2),
                ContrastLoss(factor=0.7),
                SaltAndPepper(amount=0.01),
                JpegCompression(quality=35),
            ),
        ),
        DegradationProfile(
            name="fax",
            steps=(
                Skew(degrees=7.0),
                ContrastLoss(factor=0.5),
                SaltAndPepper(amount=0.03),
                GaussianBlur(radius=1.0),
                JpegCompression(quality=15),
            ),
        ),
        DegradationProfile(
            name="worst_case",
            steps=(
                Skew(degrees=15.0),
                GaussianBlur(radius=2.0),
                ContrastLoss(factor=0.4),
                SaltAndPepper(amount=0.05),
                JpegCompression(quality=5),
            ),
        ),
    )
}
"""Built-in profiles, from pristine to barely legible."""
