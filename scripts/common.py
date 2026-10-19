"""Shared settings: which UCF101 categories we use and where the videos live."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UCF101_DIR = ROOT / "archive (3)" / "UCF101" / "UCF-101"

# The original 4 folders, plus UCF101 categories that contain jump landings.
# Candidates are screened with suggest_frames.py; drop any that give few usable clips.
CATEGORIES = [
    "HighJump", "JumpingJack", "JumpRope", "LongJump",
    "PoleVault", "TrampolineJumping", "BasketballDunk", "VolleyballSpiking",
    "FloorGymnastics", "BalanceBeam", "ParallelBars", "UnevenBars", "Basketball",
]


def category_dir(category):
    """Folder holding a category's clips: the copy in the project root if present,
    otherwise the full UCF101 archive."""
    local = ROOT / category
    return local if local.is_dir() else UCF101_DIR / category


def clip_path(category, clip):
    return category_dir(category) / clip
