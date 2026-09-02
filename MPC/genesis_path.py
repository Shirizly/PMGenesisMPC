"""Put the simulator's directory on sys.path.

`Genesis/` is not a Python package - it has no `__init__.py`, and its modules
import each other flatly (`from utilities.materials import *`,
`from sandbox_manipulation import ...`). That is upstream's convention and this
port keeps it, so reaching the simulator from here means putting that directory
on the path rather than importing `Genesis.something`.

The path has to STAY on sys.path, not be added and removed around one import:
`sandbox_manipulation` imports `placement_sampling`, `action_sampling` and
`state_library` lazily, inside the methods that use them, so it needs the
directory to still be there long after the class is first imported.

That is also why this package's own model-training code lives in
`model_training/` rather than `training/`: `Genesis/training/` exists too (it
holds the dataset and the DINO/LeWM exporters), and with both directories on
sys.path a plain `import training` would resolve to whichever came first.
"""
import sys
from pathlib import Path

GENESIS_DIR = Path(__file__).resolve().parent.parent / "Genesis"
MPC_DIR = Path(__file__).resolve().parent


def ensure() -> Path:
    """Idempotently put Genesis/ (and this directory) on sys.path."""
    for p in (str(GENESIS_DIR), str(MPC_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return GENESIS_DIR


ensure()
