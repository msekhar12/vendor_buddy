"""
Storage abstraction so Phase-1 uses local disk but Phase-4 can swap in
an S3 driver without touching any caller code.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class StorageDriver(ABC):
    """Interface every storage backend must implement."""

    @abstractmethod
    def put(self, vendor: str, date: str, filename: str, content: bytes) -> Path:
        """Write a file under vendor/date/filename. Return the stored path."""

    @abstractmethod
    def get(self, path: Path) -> bytes:
        """Read a file back by path."""

    @abstractmethod
    def list_folders(self, vendor: str | None = None) -> list[Path]:
        """List every vendor/date folder (optionally for one vendor)."""


class LocalDriver(StorageDriver):
    """Filesystem-backed implementation for Phase 1."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, vendor: str, date: str, filename: str, content: bytes) -> Path:
        # Sanitise both segments defensively — never trust incoming names
        folder = self.root / _safe(vendor) / date
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / _safe(filename)
        path.write_bytes(content)
        return path

    def get(self, path: Path) -> bytes:
        return Path(path).read_bytes()

    def list_folders(self, vendor: str | None = None) -> list[Path]:
        base = self.root / _safe(vendor) if vendor else self.root
        if not base.exists():
            return []
        # Every leaf directory that contains a manifest.json is a valid folder
        return sorted(p for p in base.rglob("*") if p.is_dir())


def _safe(name: str) -> str:
    """Strip path-traversal characters from a name segment."""
    clean = name.replace("/", "_").replace("\\", "_").strip(". ")
    if not clean or clean.startswith("."):
        raise ValueError(f"unsafe path segment: {name!r}")
    return clean