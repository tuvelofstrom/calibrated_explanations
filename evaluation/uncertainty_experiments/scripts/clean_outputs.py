from __future__ import annotations

"""Clean generated outputs for the uncertainty experiments.

Removes all files/subfolders under these directories (if they exist):
- evaluation/uncertainty_experiments/artifacts
- evaluation/uncertainty_experiments/derived
- evaluation/uncertainty_experiments/figures

By default, keeps the top-level directories in place. Use --hard to also
remove and recreate the directories themselves.
"""

import argparse
from pathlib import Path
import shutil


# Resolve base folder: evaluation/uncertainty_experiments
# __file__ = .../evaluation/uncertainty_experiments/scripts/clean_outputs.py
# parents[1] -> .../evaluation/uncertainty_experiments
BASE = Path(__file__).resolve().parents[1]
TARGETS = [
    BASE / "artifacts",
    BASE / "derived",
    BASE / "figures",
]


def _safe_clean_dir(path: Path, *, hard: bool = False) -> int:
    if not path.exists():
        return 0
    removed = 0
    if hard:
        try:
            shutil.rmtree(path)
            removed += 1
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return removed

    # Soft: remove contents only
    for p in list(path.glob("*")):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1
        except Exception:
            # best-effort
            pass
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean outputs (artifacts/derived/figures)")
    ap.add_argument("--hard", action="store_true", help="Remove and recreate the directories themselves")
    args = ap.parse_args()

    total = 0
    for t in TARGETS:
        removed = _safe_clean_dir(t, hard=args.hard)
        print(f"Cleaned {t}: removed {removed} items")
        total += removed
    print(f"Done. Total removed items: {total}")


if __name__ == "__main__":
    main()
