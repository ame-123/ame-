"""intent-finetune 路径：数据、产出，以及旅行客服 backend（只读复用）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
OUTPUTS_DIR = ROOT / "outputs"
EVAL_REPORTS_DIR = OUTPUTS_DIR / "eval_reports"
TRAVEL_BACKEND = ROOT.parent / "travel_agent" / "backend"
TRAVEL_ENV = ROOT.parent / "travel_agent" / "course.env"
SFT_PROMPT_PATH = PROMPTS_DIR / "router_sft.md"


def ensure_travel_backend_on_path() -> Path:
    """让评测/造数可以 import 现有 planning，不改旅行客服代码。"""
    backend = str(TRAVEL_BACKEND.resolve())
    if backend not in sys.path:
        sys.path.insert(0, backend)
    return TRAVEL_BACKEND


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
