"""Cấu hình pytest: bảo đảm import được các module ở thư mục gốc dự án."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
