import tempfile
from pathlib import Path

import pytest

from civsa.storage import LocalDriver, _safe


def test_put_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        drv = LocalDriver(Path(tmp))
        p = drv.put("VendorA", "2026-09-02", "quote.pdf", b"hello")
        assert p.exists()
        assert drv.get(p) == b"hello"


def test_safe_path_traversal():
    with pytest.raises(ValueError):
        _safe("../../etc/passwd")


def test_list_folders_empty():
    with tempfile.TemporaryDirectory() as tmp:
        drv = LocalDriver(Path(tmp))
        assert drv.list_folders("Nobody") == []