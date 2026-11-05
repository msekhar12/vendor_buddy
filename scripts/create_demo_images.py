"""
Generate SPECIMEN image test data for the OCR pipeline.

Writes to TEST-DATA/images/:
  uploads/   quote, PAN and GST-certificate images for fictional vendors,
             one folder per scenario (upload each folder under its vendor)
  quality/   one quote rendered clean and with increasing degradation,
             plus irrelevant images (menu, landscape, blank, scribble)
  expected.json  ground truth for every file (used by tests and eval)

Every vendor, name, address and ID number is fictional. PAN / GSTIN values
have a valid *format and check character* so the validator is exercised,
but documents are plain layouts with a SPECIMEN watermark, not replicas of
any government design.

Usage:  PYTHONPATH=. python scripts/create_demo_images.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from civsa.identity import gstin_check_char

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "TEST-DATA" / "images"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
W, H = 1654, 2339                       # A4 @ 200 dpi

rng = random.Random(42)
np_rng = np.random.default_rng(42)


def font(size: int, bold: bool = False, name: str | None = None):
    candidates = [name] if name else (
        ["Arial Bold.ttf"] if bold else ["Arial.ttf"])
    for c in candidates:
        p = FONT_DIR / c
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def gstin(state: str, pan: str, entity: str = "1") -> str:
    first14 = f"{state}{pan}{entity}Z"
    return first14 + gstin_check_char(first14)


def watermark(img: Image.Image, text: str = "SPECIMEN") -> Image.Image:
    layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(layer)
    f = font(int(img.width / 7), bold=True)
    tw = d.textlength(text, font=f)
    d.text(((img.width - tw) / 2, img.height / 2 - img.width / 14), text,
           font=f, fill=(200, 200, 200, 70))
    layer = layer.rotate(30, resample=Image.BICUBIC,
                         center=(img.width / 2, img.height / 2))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


# ------------------------------------------------------------------
# Document renderers
# ------------------------------------------------------------------

BUYER = ["To: Indian Institute of Information Technology Dharwad",
         "Procurement Cell, Kurubarahalli, Dharwad - 580009, Karnataka"]


def render_quote(v: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    x, y = 110, 100
    d.text((x, y), "SPECIMEN - NOT A REAL DOCUMENT", font=font(26), fill=(120, 0, 0))
    y += 60
    d.text((x, y), v["name"], font=font(52, bold=True), fill="black"); y += 75
    d.text((x, y), v["address"], font=font(28), fill="black"); y += 45
    ids = []
    if v.get("quote_gstin"):
        ids.append(f"GSTIN: {v['quote_gstin']}")
    if v.get("quote_pan"):
        ids.append(f"PAN: {v['quote_pan']}")
    ids.append(f"Email: {v['email']}")
    d.text((x, y), "   |   ".join(ids), font=font(28), fill="black"); y += 70
    d.line((x, y, W - x, y), fill="black", width=3); y += 30
    d.text((x, y), "QUOTATION", font=font(44, bold=True), fill="black"); y += 70
    for label, val in [("Quotation No.", v["quote_no"]), ("Date", v["date"]),
                       ("Validity", v["validity"])]:
        d.text((x, y), f"{label}: {val}", font=font(30), fill="black"); y += 46
    y += 10
    for line in BUYER:
        d.text((x, y), line, font=font(28), fill="black"); y += 42
    y += 30

    cols = [x, x + 170, x + 790, x + 910, x + 1010, x + 1240]
    heads = ["SKU", "Description", "Unit", "Qty", "Unit Price (INR)", "Line Total (INR)"]
    d.line((x, y - 8, W - x, y - 8), fill="black", width=2)
    for c, h in zip(cols, heads):
        d.text((c, y), h, font=font(26, bold=True), fill="black")
    y += 48
    d.line((x, y - 8, W - x, y - 8), fill="black", width=2)
    subtotal = 0
    for sku, desc, unit, qty, price in v["items"]:
        total = qty * price
        subtotal += total
        for c, val in zip(cols, [sku, desc, unit, str(qty), f"{price:,}", f"{total:,}"]):
            d.text((c, y), val, font=font(27), fill="black")
        y += 46
    d.line((x, y, W - x, y), fill="black", width=2); y += 20
    gst = round(subtotal * 0.18)
    for label, val in [("Subtotal", subtotal), ("GST @ 18%", gst),
                       ("Grand Total", subtotal + gst)]:
        d.text((cols[4], y), label, font=font(28, bold=True), fill="black")
        d.text((cols[5], y), f"{val:,}", font=font(28), fill="black")
        y += 46
    y += 30
    d.text((x, y), "Commercial & Delivery Terms", font=font(34, bold=True), fill="black"); y += 58
    for label, val in v["terms"]:
        d.text((x, y), f"{label}: {val}", font=font(28), fill="black"); y += 44
    y += 20
    d.text((x, y), f"Certifications: {v['certs']}", font=font(28), fill="black"); y += 80
    d.text((x, y), f"For {v['name']}", font=font(28), fill="black"); y += 44
    d.text((x, y), f"{v['signatory']}, Authorised Signatory", font=font(28), fill="black")
    return watermark(img)


def render_pan(name: str, pan: str, doi: str) -> Image.Image:
    img = Image.new("RGB", (1400, 880), (250, 248, 240))
    d = ImageDraw.Draw(img)
    d.rectangle((20, 20, 1380, 860), outline="black", width=4)
    d.text((70, 60), "SPECIMEN - NOT A REAL DOCUMENT", font=font(28), fill=(120, 0, 0))
    d.text((70, 120), "Permanent Account Number (PAN) - Specimen Record",
           font=font(42, bold=True), fill="black")
    rows = [("Name", name), ("Date of Incorporation", doi),
            ("Permanent Account Number", pan), ("Status", "Company")]
    y = 260
    for label, val in rows:
        d.text((70, y), f"{label}: {val}", font=font(40), fill="black")
        y += 110
    return watermark(img)


def render_gst(legal: str, trade: str, g: str, address: str, state: str) -> Image.Image:
    img = Image.new("RGB", (W, 1500), "white")
    d = ImageDraw.Draw(img)
    x, y = 110, 90
    d.text((x, y), "SPECIMEN - NOT A REAL DOCUMENT", font=font(26), fill=(120, 0, 0)); y += 60
    d.text((x, y), "Goods and Services Tax - Registration Certificate",
           font=font(44, bold=True), fill="black"); y += 70
    d.text((x, y), "Form GST REG-06 (specimen layout)", font=font(30), fill="black"); y += 90
    rows = [("Registration Number", g), ("Legal Name", legal), ("Trade Name, if any", trade),
            ("Constitution of Business", "Private Limited Company"),
            ("Address of Principal Place of Business", address),
            ("State", state), ("Date of Liability", "01/07/2017"),
            ("Type of Registration", "Regular")]
    for label, val in rows:
        d.text((x, y), f"{label}: {val}", font=font(34), fill="black"); y += 90
    return watermark(img)


# ------------------------------------------------------------------
# Degradations
# ------------------------------------------------------------------

def _perspective_coeffs(src, dst):
    m = []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(m, dtype=float)
    b = np.array(src, dtype=float).reshape(8)
    return np.linalg.solve(a, b).tolist()


def rotated(img, deg=3.0):
    return img.rotate(deg, expand=True, fillcolor="white", resample=Image.BICUBIC)


def photo(img):
    w, h = img.size
    dst = [(40, 60), (w - 20, 0), (w, h), (0, h - 40)]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    out = img.transform(img.size, Image.PERSPECTIVE, _perspective_coeffs(dst, src),
                        Image.BICUBIC, fillcolor=(90, 90, 90))
    arr = np.asarray(out).astype(float)
    grad = np.linspace(1.0, 0.72, w)[None, :, None]       # uneven lighting
    arr = np.clip(arr * grad, 0, 255).astype("uint8")
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1))


def blurred(img, r=7):
    return img.filter(ImageFilter.GaussianBlur(r))


def low_res(img, scale=0.18):
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)


def noisy(img, sigma=110):
    # Half size, grayscale: random noise does not compress, and a full-size
    # colour version was 8 MB. Still unreadable, which is the point.
    g = img.convert("L").resize((img.width // 2, img.height // 2), Image.BILINEAR)
    arr = np.asarray(g).astype(float)
    arr += np_rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def jpeg_crushed(img, path: Path):
    small = img.resize((img.width // 3, img.height // 3), Image.BILINEAR)
    small.save(path, "JPEG", quality=4)


# ------------------------------------------------------------------
# Irrelevant images
# ------------------------------------------------------------------

def menu():
    img = Image.new("RGB", (1200, 1600), (255, 250, 235))
    d = ImageDraw.Draw(img)
    d.text((300, 80), "Annapoorna Tiffin House", font=font(56, bold=True), fill=(90, 30, 0))
    d.text((420, 160), "Breakfast Menu", font=font(40), fill=(90, 30, 0))
    items = [("Masala Dosa", 80), ("Idli Vada (2+1)", 60), ("Rava Upma", 50),
             ("Poori Sagu", 70), ("Filter Coffee", 25), ("Masala Chai", 20),
             ("Kesari Bath", 45), ("Onion Uttapam", 75)]
    y = 280
    for name, price in items:
        d.text((150, y), name, font=font(40), fill="black")
        d.text((900, y), f"Rs. {price}", font=font(40), fill="black")
        y += 90
    d.text((150, y + 60), "Open 6:30 am to 11:00 am. Sundays closed.", font=font(32), fill="black")
    return img


def landscape():
    w, h = 1600, 1000
    sky = np.linspace([120, 170, 230], [235, 240, 250], h // 2)
    ground = np.linspace([60, 140, 60], [30, 90, 30], h - h // 2)
    arr = np.concatenate([sky, ground])[:, None, :].repeat(w, axis=1)
    img = Image.fromarray(arr.astype("uint8"))
    d = ImageDraw.Draw(img)
    d.ellipse((1200, 80, 1380, 260), fill=(250, 220, 80))
    for i in range(5):
        cx = 150 + i * 330
        d.polygon([(cx - 260, 500), (cx, 220 + (i % 2) * 60), (cx + 260, 500)],
                  fill=(90 + i * 10, 100, 110))
    return img


def blank():
    # Off-white page with a faint shading gradient (no per-pixel grain, which
    # made the file 6 MB without changing the result: there is no text).
    shade = np.linspace(249, 238, 2339)[:, None].repeat(1654, axis=1)
    return Image.fromarray(shade.astype("uint8"), mode="L")


def scribble():
    img = Image.new("RGB", (1400, 1000), "white")
    d = ImageDraw.Draw(img)
    for _ in range(14):
        pts = [(rng.randint(50, 1350), rng.randint(50, 950)) for _ in range(6)]
        d.line(pts, fill=(20, 20, 120), width=rng.randint(3, 8), joint="curve")
    d.text((120, 420), "ring Ravi re: sat??", font=font(70, name="Bradley Hand Bold.ttf"),
           fill=(20, 20, 120))
    return img


# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------

def vendor(name, city_addr, email, quote_no, items, terms, certs, signatory,
           quote_gstin=None, quote_pan=None):
    return dict(name=name, address=city_addr, email=email, quote_no=quote_no,
                date="18 October 2026", validity="30 days from date of issue",
                items=items, terms=terms, certs=certs, signatory=signatory,
                quote_gstin=quote_gstin, quote_pan=quote_pan)


TERMS_STD = [("Payment Terms", "30 days net from date of invoice"),
             ("Delivery Lead Time", "7 to 10 working days from receipt of PO"),
             ("Freight", "FOR destination, Dharwad"),
             ("Packaging", "HDPE bottles in corrugated cartons"),
             ("Quality Warranty", "CoA with every batch")]


def main() -> None:
    import shutil
    for sub in ("uploads", "quality"):              # no stale files from older runs
        shutil.rmtree(OUT / sub, ignore_errors=True)
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    uploads: list[dict] = []
    quality: list[dict] = []

    def save(img, rel, gray=True):
        # Grayscale + optimised PNG (or JPEG for .jpg): OCR converts to
        # grayscale anyway, and it keeps the committed test data small.
        p = OUT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        out = img.convert("L") if gray else img
        if p.suffix == ".jpg":
            out.save(p, "JPEG", quality=70)
        else:
            out.save(p, optimize=True)
        return rel

    # 1. Coromandel: quote has GSTIN but no PAN; PAN image fills the gap
    pan1 = "AABCC4821K"
    v1 = vendor("Coromandel Solvents Pvt Ltd", "14 Ambattur Industrial Estate, Chennai - 600058, Tamil Nadu",
                "sales@coromandel-solvents.example", "CSP/Q/26-27/0612",
                [("CS-110", "Methanol, HPLC grade, 2.5 L", "bottle", 8, 2390),
                 ("CS-120", "Acetone, AR grade, 2.5 L", "bottle", 15, 1290),
                 ("CS-135", "Ethanol Absolute, AR grade, 2.5 L", "bottle", 12, 1720)],
                TERMS_STD, "ISO 9001:2015 certified", "V. Lakshmi",
                quote_gstin=gstin("33", pan1))
    quote1 = render_quote(v1)
    uploads.append({"scenario": "pan_fills_gap", "vendor": v1["name"],
                    "file": save(quote1, "uploads/coromandel/coromandel_quote.png"),
                    "labels": ["quote"], "expect": {"status": "ok", "doc_type": "quote"}})
    uploads.append({"scenario": "pan_fills_gap", "vendor": v1["name"],
                    "file": save(render_pan("COROMANDEL SOLVENTS PRIVATE LIMITED", pan1, "11/03/2009"),
                                 "uploads/coromandel/coromandel_pan.png"),
                    "labels": ["pan_card"],
                    "expect": {"status": "ok", "doc_type": "pan_card", "pan": pan1}})

    # 2. Hooghly: quote has no GSTIN/PAN; GST certificate supplies both
    pan2 = "AADCH7310M"
    g2 = gstin("19", pan2)
    v2 = vendor("Hooghly Glassworks Pvt Ltd", "22 Strand Road, Kolkata - 700001, West Bengal",
                "orders@hooghlyglass.example", "HGW/QT/1188",
                [("HG-250", "Borosilicate Beaker 250 ml", "piece", 40, 145),
                 ("HG-500", "Volumetric Flask 500 ml, Class A", "piece", 20, 610),
                 ("HG-010", "Measuring Cylinder 10 ml", "piece", 30, 95)],
                [("Payment Terms", "45 days net"), ("Delivery Lead Time", "12 to 15 working days"),
                 ("Freight", "Ex-works Kolkata"), ("Packaging", "Foam-lined cartons"),
                 ("Quality Warranty", "Free replacement of breakage in transit")],
                "Not ISO certified", "Arindam Bose")
    uploads.append({"scenario": "gst_fills_gap", "vendor": v2["name"],
                    "file": save(render_quote(v2), "uploads/hooghly/hooghly_quote.png"),
                    "labels": ["quote"], "expect": {"status": "ok", "doc_type": "quote"}})
    uploads.append({"scenario": "gst_fills_gap", "vendor": v2["name"],
                    "file": save(render_gst("HOOGHLY GLASSWORKS PRIVATE LIMITED", "Hooghly Glassworks",
                                            g2, "22 Strand Road, Kolkata 700001", "West Bengal"),
                                 "uploads/hooghly/hooghly_gst_certificate.png"),
                    "labels": ["gst_certificate"],
                    "expect": {"status": "ok", "doc_type": "gst_certificate", "gstin": g2, "pan": pan2}})

    # 3. Pink City: two different PAN images -> both flagged
    pan3 = "AAFCP5562Q"
    v3 = vendor("Pink City Fasteners Pvt Ltd", "B-41 Vishwakarma Industrial Area, Jaipur - 302013, Rajasthan",
                "quotes@pinkcityfasteners.example", "PCF/2026/0907",
                [("PCF-M10", "SS Hex Bolt M10 x 40, Grade 8.8", "nos", 500, 198),
                 ("PCF-N10", "SS Hex Nut M10", "nos", 500, 22)],
                TERMS_STD, "ISO 9001:2015 certified", "Kavita Sharma", quote_gstin=gstin("08", pan3))
    uploads.append({"scenario": "multiple_pan", "vendor": v3["name"],
                    "file": save(render_quote(v3), "uploads/pinkcity/pinkcity_quote.png"),
                    "labels": ["quote"], "expect": {"status": "ok", "doc_type": "quote"}})
    for i, p in enumerate([pan3, "AAFCP5526Q"], 1):
        uploads.append({"scenario": "multiple_pan", "vendor": v3["name"],
                        "file": save(render_pan("PINK CITY FASTENERS PRIVATE LIMITED", p, "02/08/2012"),
                                     f"uploads/pinkcity/pinkcity_pan_{i}.png"),
                        "labels": ["pan_card"],
                        "expect": {"status": "needs_review", "doc_type": "pan_card", "pan": p,
                                   "reasons_any": ["multiple_pan_docs"]}})

    # 4. Malabar: two GST certificates (Kerala + Karnataka branch, same PAN) -> both flagged
    pan4 = "AAJCM9034E"
    v4 = vendor("Malabar Lab Supplies Pvt Ltd", "Kakkanad, Kochi - 682030, Kerala",
                "sales@malabarlab.example", "MLS/Q/26/331",
                [("MLS-07", "Sulphuric Acid 98%, AR grade, 2.5 L", "bottle", 6, 760),
                 ("MLS-12", "Methanol, HPLC grade, 2.5 L", "bottle", 8, 2550)],
                TERMS_STD, "ISO 9001:2015 certified", "Joseph Mathew")
    uploads.append({"scenario": "multiple_gst", "vendor": v4["name"],
                    "file": save(render_quote(v4), "uploads/malabar/malabar_quote.png"),
                    "labels": ["quote"], "expect": {"status": "ok", "doc_type": "quote"}})
    for st, state, addr in [("32", "Kerala", "Kakkanad, Kochi 682030"),
                            ("29", "Karnataka", "Peenya, Bengaluru 560058")]:
        g = gstin(st, pan4)
        uploads.append({"scenario": "multiple_gst", "vendor": v4["name"],
                        "file": save(render_gst("MALABAR LAB SUPPLIES PRIVATE LIMITED", "Malabar Lab Supplies",
                                                g, addr, state),
                                     f"uploads/malabar/malabar_gst_{state.lower()}.png"),
                        "labels": ["gst_certificate"],
                        "expect": {"status": "needs_review", "doc_type": "gst_certificate", "gstin": g,
                                   "reasons_any": ["multiple_gst_docs"]}})

    # 5. Narmada: PAN card disagrees with the PAN inside the quote's GSTIN -> conflict
    pan5 = "AAKCN6618B"
    v5 = vendor("Narmada Scientific Pvt Ltd", "GIDC Makarpura, Vadodara - 390010, Gujarat",
                "info@narmadasci.example", "NSP/QT/0458",
                [("NS-301", "Hydrochloric Acid 37%, AR grade, 2.5 L", "bottle", 10, 655),
                 ("NS-322", "Sodium Hydroxide pellets, AR grade, 500 g", "bottle", 20, 790)],
                TERMS_STD, "ISO 14001:2015 certified", "Hetal Patel", quote_gstin=gstin("24", pan5))
    uploads.append({"scenario": "pan_conflict", "vendor": v5["name"],
                    "file": save(render_quote(v5), "uploads/narmada/narmada_quote.png"),
                    "labels": ["quote"], "expect": {"status": "ok", "doc_type": "quote"}})
    uploads.append({"scenario": "pan_conflict", "vendor": v5["name"],
                    "file": save(render_pan("NARMADA SCIENTIFIC PRIVATE LIMITED", "AAKCN6681B", "19/01/2015"),
                                 "uploads/narmada/narmada_pan.png"),
                    "labels": ["pan_card"],
                    "expect": {"status": "needs_review", "doc_type": "pan_card", "pan": "AAKCN6681B",
                               "reasons_any": ["conflicts_with_quote"]}})

    # 6. Konkan: PAN card belongs to a different company -> name mismatch
    v6 = vendor("Konkan Chemicals Pvt Ltd", "MIDC Lote Parshuram, Ratnagiri - 415722, Maharashtra",
                "sales@konkanchem.example", "KCP/26/1120",
                [("KC-01", "Acetone, AR grade, 2.5 L", "bottle", 15, 1350)],
                TERMS_STD, "Not ISO certified", "Sameer Joshi")
    uploads.append({"scenario": "name_mismatch", "vendor": v6["name"],
                    "file": save(render_quote(v6), "uploads/konkan/konkan_quote.png"),
                    "labels": ["quote"], "expect": {"status": "ok", "doc_type": "quote"}})
    uploads.append({"scenario": "name_mismatch", "vendor": v6["name"],
                    "file": save(render_pan("KONKAN MARINE EXPORTS PRIVATE LIMITED", "AALCK2290F", "05/05/2011"),
                                 "uploads/konkan/konkan_pan.png"),
                    "labels": ["pan_card"],
                    "expect": {"status": "needs_review", "doc_type": "pan_card",
                               "reasons_any": ["name_mismatch"]}})

    # 7. Sahyadri: unreadable PAN photo
    v7 = vendor("Sahyadri Reagents Pvt Ltd", "Bhosari MIDC, Pune - 411026, Maharashtra",
                "sales@sahyadrireagents.example", "SRP/Q/2026/077",
                [("SR-15", "Ethanol Absolute, AR grade, 2.5 L", "bottle", 12, 1610)],
                TERMS_STD, "ISO 9001:2015 certified", "Neha Kulkarni")
    uploads.append({"scenario": "unreadable_pan", "vendor": v7["name"],
                    "file": save(render_quote(v7), "uploads/sahyadri/sahyadri_quote.png"),
                    "labels": ["quote"], "expect": {"status": "ok", "doc_type": "quote"}})
    uploads.append({"scenario": "unreadable_pan", "vendor": v7["name"],
                    "file": save(blurred(render_pan("SAHYADRI REAGENTS PRIVATE LIMITED", "AAMCS3307H",
                                                    "14/09/2016"), r=6),
                                 "uploads/sahyadri/sahyadri_pan_blurry.png"),
                    "labels": ["pan_card"],
                    "expect": {"status": "needs_review", "doc_type": "pan_card",
                               "reasons_any": ["id_unreadable", "low_confidence", "blurry"]}})

    # 8. Rajshree (existing specimen vendor): PAN image fills its missing PAN
    uploads.append({"scenario": "existing_vendor_pan", "vendor": "Rajshree Fastners",
                    "file": save(render_pan("RAJSHREE FASTENERS PRIVATE LIMITED", "AAAAA0000A", "10/10/2005"),
                                 "uploads/rajshree/rajshree_pan.png"),
                    "labels": ["pan_card"],
                    "expect": {"status": "ok", "doc_type": "pan_card", "pan": "AAAAA0000A"}})

    # 9. Label mismatch: PAN image uploaded labelled as a quote
    uploads.append({"scenario": "label_mismatch", "vendor": v7["name"],
                    "file": save(render_pan("SAHYADRI REAGENTS PRIVATE LIMITED", "AAMCS3307H", "14/09/2016"),
                                 "uploads/sahyadri/sahyadri_pan_labelled_quote.png"),
                    "labels": ["quote"],
                    "expect": {"status": "needs_review", "doc_type": "pan_card",
                               "reasons_any": ["label_mismatch"]}})

    # ---- quality ladder (standalone OCR assessment) ----
    q = [("clean", quote1, "ok", []),
         ("rotated_3deg", rotated(quote1), "ok", []),
         ("phone_photo", photo(quote1), "ok", []),
         ("blur_heavy", blurred(quote1), "needs_review", ["no_text", "low_confidence", "blurry", "too_little_text", "garbled"]),
         ("low_res", low_res(quote1), "needs_review", ["low_resolution", "low_confidence", "too_little_text"]),
         ("noise_heavy", noisy(quote1), "needs_review", ["ocr_timeout", "no_text", "low_confidence", "garbled", "too_little_text"])]
    for tag, img, status, reasons in q:
        ext = "jpg" if tag == "noise_heavy" else "png"
        quality.append({"file": save(img, f"quality/quote_{tag}.{ext}"), "kind": "quote",
                        "expect_status": status, "expect_reasons_any": reasons})
    jp = OUT / "quality" / "quote_jpeg_crushed.jpg"
    jpeg_crushed(quote1, jp)
    quality.append({"file": "quality/quote_jpeg_crushed.jpg", "kind": "quote",
                    "expect_status": "needs_review",
                    "expect_reasons_any": ["low_confidence", "low_resolution", "garbled", "too_little_text"]})
    for tag, img, reasons in [("menu", menu(), ["not_procurement"]),
                              ("landscape", landscape(), ["no_text", "too_little_text"]),
                              ("blank", blank(), ["no_text"]),
                              ("scribble", scribble(), ["no_text", "too_little_text", "garbled", "low_confidence"])]:
        quality.append({"file": save(img, f"quality/irrelevant_{tag}.png", gray=(tag != "landscape")),
                        "kind": "irrelevant",
                        "expect_status": "needs_review", "expect_reasons_any": reasons})

    (OUT / "expected.json").write_text(json.dumps(
        {"uploads": uploads, "quality": quality}, indent=2))
    print(f"Wrote {len(uploads)} upload files and {len(quality)} quality files under {OUT}")


if __name__ == "__main__":
    main()
