from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from rccia_common.image_uploads import (
    ESTIMATED_DECODED_BYTES_PER_PIXEL,
    HUB_IMAGE_FORMATS,
    MAX_DECODED_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    ImageUploadError,
    decode_uploaded_image,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_APP_PATHS = (
    REPO_ROOT / "projects" / "multicancer" / "app.py",
    REPO_ROOT / "projects" / "leukemia" / "app.py",
    REPO_ROOT / "projects" / "breast" / "app.py",
    REPO_ROOT / "projects" / "metastasis" / "app.py",
    REPO_ROOT / "projects" / "lung_colon" / "app.py",
)


def image_bytes(image_format: str, size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(32, 96, 160)).save(buffer, format=image_format)
    return buffer.getvalue()


class DeclaredOversizedUpload:
    size = MAX_UPLOAD_BYTES + 1

    def seek(self, _position: int) -> None:
        raise AssertionError("oversized uploads must be rejected before seek")

    def read(self, _size: int) -> bytes:
        raise AssertionError("oversized uploads must be rejected before read")


def test_declared_oversized_upload_is_rejected_before_read() -> None:
    with pytest.raises(ImageUploadError, match="depasse la limite"):
        decode_uploaded_image(
            DeclaredOversizedUpload(),
            allowed_formats=HUB_IMAGE_FORMATS,
        )


def test_actual_encoded_byte_limit_is_enforced() -> None:
    with pytest.raises(ImageUploadError, match="depasse la limite"):
        decode_uploaded_image(
            b"12345",
            allowed_formats=HUB_IMAGE_FORMATS,
            max_upload_bytes=4,
        )


def test_dimensions_are_rejected_before_full_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    content = image_bytes("PNG", size=(11, 1))

    def fail_convert(*_args, **_kwargs):
        raise AssertionError("convert must not run after a header budget rejection")

    monkeypatch.setattr(Image.Image, "convert", fail_convert)
    with pytest.raises(ImageUploadError, match="dimensions maximales"):
        decode_uploaded_image(
            content,
            allowed_formats={"PNG"},
            max_width=10,
        )


def test_pixel_limit_is_enforced() -> None:
    with pytest.raises(ImageUploadError, match="resolution limitee"):
        decode_uploaded_image(
            image_bytes("PNG", size=(11, 10)),
            allowed_formats={"PNG"},
            max_pixels=100,
        )


def test_estimated_decoded_memory_limit_is_enforced() -> None:
    with pytest.raises(ImageUploadError, match="resolution limitee"):
        decode_uploaded_image(
            image_bytes("PNG", size=(10, 10)),
            allowed_formats={"PNG"},
            max_pixels=1_000,
            max_decoded_bytes=399,
        )


def test_multiframe_image_is_rejected() -> None:
    buffer = io.BytesIO()
    first = Image.new("RGB", (16, 16), color="red")
    second = Image.new("RGB", (16, 16), color="blue")
    first.save(buffer, format="TIFF", save_all=True, append_images=[second])

    with pytest.raises(ImageUploadError, match="multi-pages"):
        decode_uploaded_image(buffer.getvalue(), allowed_formats={"TIFF"})


def test_detected_format_wins_over_filename_or_mime_claims() -> None:
    # Rejected before entering an unapproved parser, even if the extension claims PNG.
    with pytest.raises(ImageUploadError, match="image valide et supportee"):
        decode_uploaded_image(image_bytes("BMP"), allowed_formats=HUB_IMAGE_FORMATS)


def test_corrupted_image_has_a_controlled_error() -> None:
    with pytest.raises(ImageUploadError, match="image valide"):
        decode_uploaded_image(b"not-an-image", allowed_formats=HUB_IMAGE_FORMATS)


def test_pillow_decompression_warning_is_treated_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = image_bytes("PNG", size=(11, 10))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ImageUploadError, match="dimensions compressees"):
        decode_uploaded_image(content, allowed_formats={"PNG"})


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "BMP", "TIFF", "WEBP"])
def test_normal_single_frame_formats_remain_compatible(image_format: str) -> None:
    image = decode_uploaded_image(
        image_bytes(image_format),
        allowed_formats={image_format},
    )

    assert image.mode == "RGB"
    assert image.size == (32, 24)


def test_default_decoded_budget_is_explicit_and_conservative() -> None:
    assert MAX_DECODED_BYTES == MAX_IMAGE_PIXELS * ESTIMATED_DECODED_BYTES_PER_PIXEL
    assert ESTIMATED_DECODED_BYTES_PER_PIXEL == 4


@pytest.mark.parametrize("app_path", PUBLIC_APP_PATHS)
def test_every_public_app_uses_the_bounded_decoder(app_path: Path) -> None:
    source = app_path.read_text(encoding="utf-8")

    assert "decode_uploaded_image(" in source
    assert "Image.open(uploaded_file)" not in source
    assert "uploaded_file.getvalue()" not in source
