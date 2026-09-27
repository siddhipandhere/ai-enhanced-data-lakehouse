"""
Small helpers for image data stored in tables (image uploads keep the raw
file bytes in a `content` column, from Spark's binaryFile source).
"""

from __future__ import annotations

import base64
import hashlib
from collections import OrderedDict
from io import BytesIO

import pandas as pd

THUMBNAIL_PX = 160
_THUMB_CACHE_SIZE = 512
_thumb_cache: "OrderedDict[str, str | None]" = OrderedDict()

_BYTES_TYPES = (bytes, bytearray, memoryview)


def is_bytes_column(series: pd.Series) -> bool:
    """True if the column's (non-null) values are raw bytes."""
    sample = series.dropna().head(20).tolist()
    return bool(sample) and all(isinstance(v, _BYTES_TYPES) for v in sample)


def open_image(blob):
    """Bytes -> an RGB PIL image, or None if the bytes aren't a readable image."""
    if not isinstance(blob, _BYTES_TYPES):
        return None
    try:
        from PIL import Image
        im = Image.open(BytesIO(bytes(blob)))
        im.load()
        if im.mode != "RGB":
            # RGBA/P (PNG transparency) -> composite onto white, not black.
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
        return im
    except Exception:
        return None


def image_column(df: pd.DataFrame) -> str | None:
    """The column holding image bytes, if this table is an image collection
    (refine.describe_images found at least one readable image)."""
    if "content" in df.columns and "width_px" in df.columns \
            and df["width_px"].notna().any() and is_bytes_column(df["content"]):
        return "content"
    return None


def thumbnail_data_uri(blob, size: int = THUMBNAIL_PX) -> str | None:
    """Image bytes -> a small 'data:image/jpeg;base64,...' thumbnail that the
    dashboard can show directly in a results table. None if not an image.
    Cached, since the same pictures are shown again on every search."""
    if not isinstance(blob, _BYTES_TYPES):
        return None
    raw = bytes(blob)
    key = hashlib.sha1(raw).hexdigest()
    if key in _thumb_cache:
        _thumb_cache.move_to_end(key)
        return _thumb_cache[key]

    uri = None
    im = open_image(raw)
    if im is not None:
        im.thumbnail((size, size))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=80)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    _thumb_cache[key] = uri
    if len(_thumb_cache) > _THUMB_CACHE_SIZE:
        _thumb_cache.popitem(last=False)
    return uri
