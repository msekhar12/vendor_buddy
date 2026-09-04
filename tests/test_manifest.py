import tempfile
from pathlib import Path

from civsa.manifest import append_doc, find_duplicate, load_manifest, sha256_of
from civsa.storage import LocalDriver


def test_hash_stable():
    assert sha256_of(b"hello") == sha256_of(b"hello")


def test_duplicate_detected_across_dates():
    with tempfile.TemporaryDirectory() as tmp:
        drv = LocalDriver(Path(tmp))
        content = b"same bytes"
        h = sha256_of(content)

        p = drv.put("V1", "2026-09-02", "a.pdf", content)
        append_doc(p.parent, {"filename": "a.pdf", "sha256": h,
                              "labels": ["quote"]})

        dup = find_duplicate(drv, "V1", h)
        assert dup is not None
        assert dup.name == "a.pdf"


def test_new_file_not_duplicate():
    with tempfile.TemporaryDirectory() as tmp:
        drv = LocalDriver(Path(tmp))
        assert find_duplicate(drv, "V1", "0" * 64) is None