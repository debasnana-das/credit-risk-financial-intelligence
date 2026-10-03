from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_credit import main as train_credit
from scripts.train_eps import main as train_eps


if __name__ == "__main__":
    train_credit()
    train_eps()
