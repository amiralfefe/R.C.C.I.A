"""Bounded decoding for images received by public Streamlit uploaders."""

from __future__ import annotations

import io
import warnings
from collections.abc import Iterable
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 8_000_000
ESTIMATED_DECODED_BYTES_PER_PIXEL = 4
MAX_DECODED_BYTES = MAX_IMAGE_PIXELS * ESTIMATED_DECODED_BYTES_PER_PIXEL
MAX_IMAGE_FRAMES = 1

HUB_IMAGE_FORMATS = frozenset({"JPEG", "PNG"})
LEUKEMIA_IMAGE_FORMATS = frozenset({"BMP", "JPEG", "PNG"})
STANDALONE_IMAGE_FORMATS = frozenset({"BMP", "JPEG", "PNG", "TIFF", "WEBP"})

UploadSource = bytes | bytearray | memoryview | BinaryIO


class ImageUploadError(ValueError):
    """Raised when an uploaded image cannot be decoded within the public budget."""


def _read_bounded(source: UploadSource, max_upload_bytes: int) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        if len(source) > max_upload_bytes:
            limit_mib = max_upload_bytes // (1024 * 1024)
            raise ImageUploadError(
                f"Image refusee : le fichier depasse la limite de {limit_mib} Mio."
            )
        content = bytes(source)
    else:
        declared_size = getattr(source, "size", None)
        if isinstance(declared_size, int) and declared_size > max_upload_bytes:
            limit_mib = max_upload_bytes // (1024 * 1024)
            raise ImageUploadError(
                f"Image refusee : le fichier depasse la limite de {limit_mib} Mio."
            )

        try:
            source.seek(0)
            content = source.read(max_upload_bytes + 1)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise ImageUploadError("Le fichier image ne peut pas etre lu.") from exc

        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise ImageUploadError(
                "Le fichier image ne contient pas de donnees binaires valides."
            )
        content = bytes(content)

    if not content:
        raise ImageUploadError("Le fichier image est vide.")
    if len(content) > max_upload_bytes:
        limit_mib = max_upload_bytes // (1024 * 1024)
        raise ImageUploadError(
            f"Image refusee : le fichier depasse la limite de {limit_mib} Mio."
        )
    return content


def _validate_header(
    candidate: Image.Image,
    *,
    allowed_formats: frozenset[str],
    max_width: int,
    max_height: int,
    max_pixels: int,
    max_decoded_bytes: int,
    max_frames: int,
) -> None:
    detected_format = (candidate.format or "").upper()
    if detected_format not in allowed_formats:
        supported = ", ".join(sorted(allowed_formats))
        raise ImageUploadError(
            f"Format image non supporte : {detected_format or 'inconnu'}. "
            f"Formats acceptes : {supported}."
        )

    width, height = candidate.size
    if width <= 0 or height <= 0:
        raise ImageUploadError("L'image possede des dimensions invalides.")
    if width > max_width or height > max_height:
        raise ImageUploadError(
            f"Image refusee : dimensions maximales {max_width}x{max_height} pixels."
        )

    pixels = width * height
    estimated_decoded_bytes = pixels * ESTIMATED_DECODED_BYTES_PER_PIXEL
    if pixels > max_pixels or estimated_decoded_bytes > max_decoded_bytes:
        raise ImageUploadError(
            f"Image refusee : resolution limitee a {max_pixels:,} pixels.".replace(",", " ")
        )

    frames = int(getattr(candidate, "n_frames", 1))
    if frames < 1 or frames > max_frames:
        raise ImageUploadError(
            "Image refusee : les images animees ou multi-pages ne sont pas acceptees."
        )


def decode_uploaded_image(
    source: UploadSource,
    *,
    allowed_formats: Iterable[str],
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_decoded_bytes: int = MAX_DECODED_BYTES,
    max_frames: int = MAX_IMAGE_FRAMES,
) -> Image.Image:
    """Return a fully decoded RGB image only when every resource limit is satisfied."""

    normalized_formats = frozenset(value.upper() for value in allowed_formats)
    if not normalized_formats:
        raise ValueError("allowed_formats must not be empty")

    content = _read_bounded(source, max_upload_bytes)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content), formats=sorted(normalized_formats)) as candidate:
                _validate_header(
                    candidate,
                    allowed_formats=normalized_formats,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                    max_decoded_bytes=max_decoded_bytes,
                    max_frames=max_frames,
                )
                candidate.verify()

            with Image.open(io.BytesIO(content), formats=sorted(normalized_formats)) as candidate:
                _validate_header(
                    candidate,
                    allowed_formats=normalized_formats,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                    max_decoded_bytes=max_decoded_bytes,
                    max_frames=max_frames,
                )
                image = candidate.convert("RGB")
                image.load()
    except ImageUploadError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageUploadError(
            "Image refusee : les dimensions compressees depassent la limite de securite."
        ) from exc
    except MemoryError as exc:
        raise ImageUploadError(
            "Image refusee : memoire insuffisante pour un decodage sur."
        ) from exc
    except (UnidentifiedImageError, OSError, TypeError, ValueError) as exc:
        raise ImageUploadError(
            "Le fichier ne contient pas une image valide et supportee."
        ) from exc

    return image
