import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from src.device import select_device

select_device(verbose=True)
