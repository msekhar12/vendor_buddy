"""
Build 6 demo PDFs under documents/ so the UI has content to demonstrate.

All documents are watermarked SPECIMEN — they must never be mistaken for
real vendor documents.
"""
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path("documents")
TODAY = date.today().isoformat()  # noqa: DTZ011


def make_pdf(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in lines:
        c.drawString(50, y, line[:120])
        y -= 14
        if y < 50:
            c.showPage()
            y = 800
    c.save()


# -------- Vendor A: Rajshree Fasteners (fasteners supplier) --------
make_pdf(ROOT / "Rajshree_Fasteners" / TODAY / "quote_hex_bolt.pdf", [
    "SPECIMEN — NOT A REAL DOCUMENT",
    "RAJSHREE FASTENERS PVT LTD  |  Pune, Maharashtra",
    "GSTIN: 27AAAAA0000A1Z5   HSN: 7318 15",
    "",
    "QUOTATION Ref: RF/Q/2026/00841   Date: " + TODAY,
    "Buyer: Acme Manufacturing Pvt Ltd",
    "",
    "Item: SS Hex Bolt M10 x 40  Grade 8.8   ISO 9001 compliant",
    "Quantity: 500 nos    Unit Price: INR 210.00    Total: INR 1,05,000",
    "MOQ: 100 nos    Lead time: 18 days    Payment: Net 30 days",
    "Valid until: 2026-10-31",
    "Certifications held: ISO 9001:2015, ISO 14001:2015",
])
make_pdf(ROOT / "Rajshree_Fasteners" / TODAY / "iso_9001_cert.pdf", [
    "SPECIMEN — NOT A REAL CERTIFICATE",
    "Certificate of Registration — ISO 9001:2015",
    "Issued to: Rajshree Fasteners Pvt Ltd, Pune",
    "Certificate No: DEMO-9001-2026-0042",
    "Valid from: 2024-01-15   Expires: 2027-01-14",
    "Scope: Manufacture and supply of stainless steel fasteners.",
])
make_pdf(ROOT / "Rajshree_Fasteners" / TODAY / "mtc_ss304.pdf", [
    "SPECIMEN — Mill Test Certificate",
    "Rajshree Fasteners Pvt Ltd — Heat No. HT-2026-1187",
    "Material: Stainless Steel 304    Grade: SS304",
    "Chemical composition: C 0.05, Si 0.6, Mn 1.7, Cr 18.2, Ni 8.4",
    "Tensile strength: 620 MPa   Yield strength: 285 MPa",
])

# -------- Vendor B: Nirmala Chemicals --------
make_pdf(ROOT / "Nirmala_Chemicals" / TODAY / "quote_acetone.pdf", [
    "SPECIMEN — NOT A REAL DOCUMENT",
    "NIRMALA CHEMICALS LLP  |  Ahmedabad, Gujarat",
    "GSTIN: 24BBBBB0000B1Z5   HSN: 2914 11",
    "",
    "QUOTATION Ref: NC/Q/2026/00307   Date: " + TODAY,
    "Item: Acetone 99.5% pure, 200 L drums",
    "Quantity: 10 drums    Unit Price: INR 12,500   Total: INR 1,25,000",
    "Lead time: 7 days    Payment: Net 15 days",
    "Valid until: 2026-10-15",
    "Certifications: ISO 14001:2015, BIS IS 170:1997",
])
make_pdf(ROOT / "Nirmala_Chemicals" / TODAY / "iso_14001_cert.pdf", [
    "SPECIMEN — Certificate of Registration — ISO 14001:2015",
    "Issued to: Nirmala Chemicals LLP, Ahmedabad",
    "Certificate No: DEMO-14001-2026-0198",
    "Valid from: 2025-03-01   Expires: 2028-02-29",
    "Scope: Manufacture, storage and supply of industrial solvents.",
])
make_pdf(ROOT / "Nirmala_Chemicals" / TODAY / "msds_acetone.pdf", [
    "SPECIMEN — Material Safety Data Sheet (MSDS)",
    "Product: Acetone (Propan-2-one)   CAS: 67-64-1",
    "Hazard: Highly flammable liquid and vapour (H225).",
    "Storage: Cool, well-ventilated area away from ignition sources.",
    "Chemical composition: Acetone >= 99.5%, water <= 0.5%",
])
print("Created 6 demo documents under documents/")