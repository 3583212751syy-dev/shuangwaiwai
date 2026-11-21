"""Shared paths for the independent upgrade; never import the legacy directory."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get('ML701_STATE_DIR', str(BASE / '.workbuddy' / 'tmp'))).expanduser().resolve()


def state_file(name):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return str(STATE_DIR / name)
