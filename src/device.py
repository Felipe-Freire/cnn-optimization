"""Compute device selection: use the CUDA GPU when TensorFlow can run on it, otherwise the CPU.

TensorFlow may list a GPU whose kernels it cannot launch, e.g. when the installed wheel has no
binaries for the card's compute capability (CUDA_ERROR_INVALID_PTX / INVALID_HANDLE at the first
op). Listing devices is therefore not enough: ``select_device`` runs a few kernels on the GPU in a
subprocess and, if they fail, hides the GPU through CUDA_VISIBLE_DEVICES. The choice must be made
before TensorFlow is imported in this process; it is kept in the environment, so later calls and
child processes reuse it.
"""

import os
import subprocess
import sys

_CHOICE_VAR = "RINGER_COMPUTE_DEVICE"

_PROBE = r"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print("NO_GPU")
    raise SystemExit(0)
tf.config.experimental.set_memory_growth(gpus[0], True)
details = tf.config.experimental.get_device_details(gpus[0])
print("GPU", details.get("device_name", "GPU"), details.get("compute_capability"))
with tf.device("/GPU:0"):
    x = tf.random.uniform((256, 256))
    y = tf.reduce_sum(tf.nn.relu(tf.matmul(x, x) - 0.5) * 2.0)
    float(y.numpy())
print("OK")
"""


def _probe_gpu(timeout_s: float) -> tuple:
    """(usable, description) after running the probe in a fresh interpreter."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return False, f"GPU probe timed out after {timeout_s:.0f} s"
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    if not lines or lines[0] == "NO_GPU":
        return False, "TensorFlow sees no CUDA GPU"
    gpu = lines[0][4:] if lines[0].startswith("GPU ") else "GPU"
    if lines[-1] == "OK" and out.returncode == 0:
        return True, gpu
    return False, f"{gpu} is visible but TensorFlow cannot run kernels on it"


def select_device(prefer_gpu: bool = True, verbose: bool = True, timeout_s: float = 600.0) -> str:
    """Choose GPU or CPU for this process and its children; returns a description of the choice.

    Call it before importing TensorFlow. If TensorFlow is already imported the device set can no
    longer be changed, and the function only reports what is visible.
    """
    choice = os.environ.get(_CHOICE_VAR)
    if choice is None:
        if "tensorflow" in sys.modules:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            choice = "GPU (already initialized)" if gpus else "CPU (no GPU visible)"
        elif not prefer_gpu or os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            choice = "CPU (GPU disabled)"
        else:
            usable, description = _probe_gpu(timeout_s)
            if usable:
                choice = f"GPU: {description}"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                choice = f"CPU ({description})"
        os.environ[_CHOICE_VAR] = choice
        if verbose:
            print(f"[device] {choice}")
    return choice
