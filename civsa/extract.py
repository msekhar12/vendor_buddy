"""
Text extraction from PDF (native + scanned), images, DOCX, XLSX, plain text.

Uses PyMuPDF (fitz) for PDFs. Scanned PDF pages (almost no native text) and
image files go through Tesseract via `ocr_image`, which also returns
quality metrics (word confidence, sharpness, size) so callers can decide
whether the text is trustworthy — see civsa/ocr_quality.py.

`extract_text(path)` keeps its old contract (returns a string);
`extract_with_quality(path)` returns text + metrics.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import docx
import fitz  # type: ignore[import-untyped] # PyMuPDF
import numpy as np
import openpyxl  # type: ignore
import pytesseract
from PIL import Image, ImageOps

from .config import OCR_PSM, OCR_TIMEOUT_S

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTS = {".pdf", ".docx", ".xlsx", ".txt", ".md"} | IMAGE_EXTS


@dataclass
class OcrMetrics:
    mean_conf: float = 0.0       # character-weighted mean word confidence, 0-100
    n_words: int = 0
    junk_ratio: float = 0.0      # share of words that are mostly symbols
    sharpness: float = 0.0       # Laplacian variance at 1000 px width
    width: int = 0               # original pixel size (before upscaling)
    height: int = 0
    timed_out: bool = False
    rotated_deg: int = 0         # orientation correction applied

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class ExtractionResult:
    text: str
    method: str                                   # native | ocr | mixed
    ocr: OcrMetrics | None = None                 # None when no OCR ran
    ocr_pages: list[int] = field(default_factory=list)
    # Single-image OCR only: preprocessed image + word boxes per line, so
    # identity checks can re-read one field (see reread_value_after_label)
    layout: OcrLayout | None = None


@dataclass
class OcrLayout:
    image: Image.Image
    # one entry per text line: (line text, [(word, (x0, y0, x1, y1)), ...])
    lines: list[tuple[str, list[tuple[str, tuple[int, int, int, int]]]]]


def extract_text(path: Path) -> str:
    """Dispatch to the right extractor based on file extension."""
    return extract_with_quality(path).text


def extract_with_quality(path: Path) -> ExtractionResult:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in IMAGE_EXTS:
        with Image.open(path) as img:
            text, m, layout = ocr_image(img, keep_layout=True)
        return ExtractionResult(text=text, method="ocr", ocr=m, ocr_pages=[1],
                                layout=layout)
    if ext == ".docx":
        return ExtractionResult(_extract_docx(path), "native")
    if ext == ".xlsx":
        return ExtractionResult(_extract_xlsx(path), "native")
    if ext in (".txt", ".md"):
        return ExtractionResult(path.read_text(errors="ignore"), "native")
    raise ValueError(f"Unsupported file type: {ext}")


# ------------------------------------------------------------------
# OCR
# ------------------------------------------------------------------

def _sharpness(gray: Image.Image) -> float:
    """Variance of the Laplacian, measured at a fixed width so it is
    comparable across image sizes. Low = blurry."""
    g = gray.resize((1000, max(1, int(1000 * gray.height / gray.width))))
    a = np.asarray(g, dtype=float)
    if a.shape[0] < 3:
        return 0.0
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1]
           + a[1:-1, :-2] + a[1:-1, 2:])
    return float(lap.var())


def _fix_orientation(gray: Image.Image) -> tuple[Image.Image, int]:
    """Use Tesseract OSD to undo 90/180/270 degree rotations. Silent on
    failure (OSD needs a fair amount of text)."""
    try:
        osd = pytesseract.image_to_osd(gray, config="--psm 0", timeout=10)
    except Exception:  # noqa: BLE001
        return gray, 0
    rot = re.search(r"Rotate:\s*(\d+)", osd)
    conf = re.search(r"Orientation confidence:\s*([\d.]+)", osd)
    deg = int(rot.group(1)) if rot else 0
    if deg in (90, 180, 270) and conf and float(conf.group(1)) >= 2.0:
        return gray.rotate(-deg, expand=True, fillcolor=255), deg
    return gray, 0


def _normalise_ocr_text(text: str) -> str:
    # Rupee sign is often absent from fonts / misread; downstream regexes
    # and the extraction prompt understand "INR".
    text = text.replace("₹", "INR ")
    text = re.sub(r"\bR[s5]\s?[.,]\s?(?=\d)", "Rs. ", text)
    return text


def ocr_image(img: Image.Image, keep_layout: bool = False
              ) -> tuple[str, OcrMetrics, OcrLayout | None]:
    """OCR one image. Returns text (line layout preserved), metrics and,
    with keep_layout, the preprocessed image plus word boxes."""
    img = ImageOps.exif_transpose(img)
    gray = img.convert("L")
    m = OcrMetrics(width=gray.width, height=gray.height)

    # Small images: upscale so glyphs reach Tesseract's preferred size
    if gray.width < 1600:
        scale = min(3.0, 2000 / gray.width)
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)),
                           Image.LANCZOS)
    gray = ImageOps.autocontrast(gray)
    gray, m.rotated_deg = _fix_orientation(gray)
    m.sharpness = _sharpness(gray)

    try:
        d = pytesseract.image_to_data(
            gray, config=f"--psm {OCR_PSM}", timeout=OCR_TIMEOUT_S,
            output_type=pytesseract.Output.DICT)
    except RuntimeError:                       # pytesseract timeout
        m.timed_out = True
        return "", m, None

    lines: dict[tuple[int, int, int], list[str]] = {}
    boxes: dict[tuple[int, int, int], list] = {}
    total_chars = conf_chars = junk = 0
    for w, c, b, p, ln, x, y, bw, bh in zip(
            d["text"], d["conf"], d["block_num"], d["par_num"], d["line_num"],
            d["left"], d["top"], d["width"], d["height"]):
        w = w.strip()
        c = float(c)
        if not w or c < 0:
            continue
        lines.setdefault((b, p, ln), []).append(w)
        boxes.setdefault((b, p, ln), []).append((w, (x, y, x + bw, y + bh)))
        m.n_words += 1
        total_chars += len(w)
        conf_chars += c * len(w)
        if sum(ch.isalnum() for ch in w) / len(w) < 0.5:
            junk += 1
    m.mean_conf = conf_chars / total_chars if total_chars else 0.0
    m.junk_ratio = junk / m.n_words if m.n_words else 0.0

    out, prev_block = [], None
    for (b, _p, _l), words in lines.items():
        if prev_block is not None and b != prev_block:
            out.append("")
        out.append(" ".join(words))
        prev_block = b
    layout = (OcrLayout(image=gray,
                        lines=[(" ".join(lines[k]), boxes[k]) for k in lines])
              if keep_layout else None)
    return _normalise_ocr_text("\n".join(out)), m, layout


_ID_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def reread_value_after_label(layout: OcrLayout, label_re: re.Pattern) -> list[str]:
    """
    Second OCR pass for one field: on each line matching `label_re`, crop the
    words after the colon, enlarge them and re-read with an A-Z/0-9
    whitelist. Single-word mode first (best at separating 0 from O), then
    single-line mode. Returns the raw readings; callers validate them.
    """
    readings = []
    for text, words in layout.lines:
        if not label_re.search(text):
            continue
        colon = next((i for i, (w, _b) in enumerate(words) if ":" in w), None)
        if colon is None:
            continue
        value = words[colon + 1:]
        if not value:
            continue
        x0 = min(b[0] for _w, b in value) - 10
        y0 = min(b[1] for _w, b in value) - 10
        x1 = max(b[2] for _w, b in value) + 10
        y1 = max(b[3] for _w, b in value) + 10
        crop = layout.image.crop((max(0, x0), max(0, y0), x1, y1))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        for psm in (8, 7):
            try:
                r = pytesseract.image_to_string(
                    crop, timeout=10,
                    config=f"--psm {psm} -c tessedit_char_whitelist={_ID_WHITELIST}")
            except RuntimeError:
                continue
            if r.strip():
                readings.append(r.strip())
    return readings


def _merge_metrics(ms: list[OcrMetrics]) -> OcrMetrics:
    """Combine per-page metrics: word-weighted confidence, worst sharpness."""
    words = sum(m.n_words for m in ms)
    out = OcrMetrics(
        n_words=words,
        mean_conf=(sum(m.mean_conf * m.n_words for m in ms) / words) if words else 0.0,
        junk_ratio=(sum(m.junk_ratio * m.n_words for m in ms) / words) if words else 0.0,
        sharpness=min(m.sharpness for m in ms),
        width=min(m.width for m in ms), height=min(m.height for m in ms),
        timed_out=any(m.timed_out for m in ms),
    )
    return out


# ------------------------------------------------------------------
# Other formats
# ------------------------------------------------------------------

def _extract_pdf(path: Path) -> ExtractionResult:
    """Page-by-page PDF text with OCR fallback for scanned pages."""
    doc = fitz.open(path)
    out, metrics, ocr_pages = [], [], []
    for i, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if len(text) < 40:
            # Very little native text → probably a scanned page. OCR it.
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text, m, _ = ocr_image(img)
            metrics.append(m)
            ocr_pages.append(i)
        out.append(f"--- Page {i} ---\n{text}")
    n_pages = len(doc)
    doc.close()
    if not metrics:
        method = "native"
    elif len(ocr_pages) == n_pages:
        method = "ocr"
    else:
        method = "mixed"
    return ExtractionResult(
        text="\n\n".join(out), method=method,
        ocr=_merge_metrics(metrics) if metrics else None, ocr_pages=ocr_pages)


def _extract_docx(path: Path) -> str:
    """Grab all non-empty paragraphs from a Word document."""
    d = docx.Document(str(path))
    return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())


def _extract_xlsx(path: Path) -> str:
    """Flatten every sheet's populated cells into a text stream."""
    wb = openpyxl.load_workbook(path, data_only=True)
    parts = []
    for sheet in wb.worksheets:
        parts.append(f"=== Sheet: {sheet.title} ===")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)
