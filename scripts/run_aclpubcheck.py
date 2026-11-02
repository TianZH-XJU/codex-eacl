"""Run aclpubcheck on a PDF using PyMuPDF instead of ImageMagick.

aclpubcheck renders margin-violation pages through pdfplumber's
``Page.to_image``, which depends on ``wand``/ImageMagick.  ImageMagick is not
installed on this Windows box, so this wrapper monkeypatches
``pdfplumber.page.Page.to_image`` with a small PyMuPDF-backed renderer that
exposes the same surface used by aclpubcheck (``.original``, ``.draw_rect``
and ``.save``).

Usage:
    python scripts/run_aclpubcheck.py paper/acl/main.pdf --paper_type long
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pymupdf
import pdfplumber
from PIL import Image, ImageDraw
from pdfplumber.page import Page


class _RenderedPage:
    """Minimal stand-in for ``pdfplumber.display.PageImage``."""

    def __init__(self, pdf_path: str, page_number: int, bbox, resolution: int):
        self.resolution = int(resolution)
        self.bbox = tuple(float(v) for v in bbox)
        doc = pymupdf.open(pdf_path)
        try:
            page = doc[page_number - 1]
            clip = pymupdf.Rect(*self.bbox)
            pix = page.get_pixmap(dpi=self.resolution, clip=clip, alpha=False)
            self.original = Image.frombytes(
                "RGB", (pix.width, pix.height), pix.samples
            )
        finally:
            doc.close()

    def _to_pixels(self, bbox):
        scale = self.resolution / 72.0
        x0, y0 = self.bbox[0], self.bbox[1]
        return [
            (bbox[0] - x0) * scale,
            (bbox[1] - y0) * scale,
            (bbox[2] - x0) * scale,
            (bbox[3] - y0) * scale,
        ]

    def draw_rect(self, bbox, fill=None, stroke="red", stroke_width=5):
        draw = ImageDraw.Draw(self.original)
        draw.rectangle(
            self._to_pixels(bbox), fill=fill, outline=stroke, width=stroke_width
        )
        return self

    def save(self, path, format="PNG"):
        self.original.save(path, format=format)
        return self


def _patched_to_image(self, resolution=72, **kwargs):
    """Replacement for pdfplumber's ImageMagick-backed renderer."""
    stream = getattr(self.pdf, "stream", None)
    pdf_path = getattr(stream, "name", None)
    if not pdf_path or not os.path.isfile(pdf_path):
        raise RuntimeError("cannot resolve PDF path for PyMuPDF rendering")
    bbox = self.bbox if self.bbox is not None else (
        0,
        0,
        self.width,
        self.height,
    )
    return _RenderedPage(pdf_path, self.page_number, bbox, resolution)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="path to the compiled paper PDF")
    parser.add_argument(
        "--paper_type", choices={"short", "long", "other"}, default="long"
    )
    parser.add_argument(
        "--output_dir", default="paper/acl", help="where error JSON/PNGs go"
    )
    args = parser.parse_args()

    Page.to_image = _patched_to_image

    from aclpubcheck import formatchecker

    # aclpubcheck's main() parses sys.argv; call the checker directly so we
    # keep control over the output directory and the return value.
    formatter = formatchecker.Formatter()
    logs = formatter.format_check(
        submission=args.pdf,
        paper_type=args.paper_type,
        output_dir=args.output_dir,
    )
    if logs:
        print("\naclpubcheck reported issues:")
        for key, messages in logs.items():
            for message in messages:
                print(f"  [{key}] {message}")
        return 1
    print("\naclpubcheck: no errors or warnings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
