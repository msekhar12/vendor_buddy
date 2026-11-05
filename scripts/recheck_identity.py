"""
Re-check the holder name on every stored PAN / GST document against its
vendor with the current matching rule, and re-apply the vendor rules.
Prints the documents whose review status changed. Safe to run repeatedly.

    PYTHONPATH=. python scripts/recheck_identity.py
"""
from pathlib import Path

from civsa.review import recheck_identity_names


def main() -> None:
    changes = recheck_identity_names()
    if not changes:
        print("No review status changes.")
    for c in changes:
        print(f"{c['status']:13s} {Path(c['source']).parent.parent.name} / "
              f"{Path(c['source']).name}  reasons={c['reasons']}")


if __name__ == "__main__":
    main()
