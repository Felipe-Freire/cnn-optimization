"""Core modules for ATLAS L0 Ringer CNN optimization."""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from src.device import select_device

# Must happen before any module of this package imports TensorFlow.
select_device(verbose=False)
