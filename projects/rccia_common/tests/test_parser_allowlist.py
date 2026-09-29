import io
from unittest.mock import patch

import pytest
from PIL import Image

from rccia_common.image_uploads import HUB_IMAGE_FORMATS, ImageUploadError, decode_uploaded_image


def test_unapproved_parser_is_never_called():
    Image.init()
    original, accept = Image.OPEN["EPS"]
    with patch.dict(Image.OPEN, {"EPS": (lambda *_: pytest.fail("EPS parser was reached"), accept)}):
        with pytest.raises(ImageUploadError, match="image valide"):
            decode_uploaded_image(b"%!PS-Adobe-3.0 EPSF-3.0\n", allowed_formats=HUB_IMAGE_FORMATS)


@pytest.mark.parametrize("format_name", ["PNG", "JPEG"])
def test_both_image_opens_have_parser_allowlist(format_name):
    content = io.BytesIO()
    Image.new("RGB", (12, 12), "red").save(content, format=format_name)
    with patch.object(Image, "open", wraps=Image.open) as opens:
        result = decode_uploaded_image(content.getvalue(), allowed_formats=HUB_IMAGE_FORMATS)
    assert result.size == (12, 12)
    assert opens.call_count == 2
    assert all(set(call.kwargs["formats"]) == HUB_IMAGE_FORMATS for call in opens.call_args_list)
