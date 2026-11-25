"""Device selection: a GPU is used only if TensorFlow can run kernels on it."""

import subprocess
import sys
from pathlib import Path

import pytest

from src import device


class _Run:
    def __init__(self, stdout, returncode=0):
        self.stdout, self.returncode = stdout, returncode


@pytest.mark.parametrize(
    "stdout, returncode, usable, text",
    [
        ("GPU NVIDIA GeForce RTX 5060 Ti (12, 0)\nOK\n", 0, True, "NVIDIA GeForce RTX 5060 Ti (12, 0)"),
        ("GPU NVIDIA GeForce RTX 5060 Ti (12, 0)\n", 1, False, "cannot run kernels"),
        ("NO_GPU\n", 0, False, "no CUDA GPU"),
    ],
)
def test_probe_uses_the_gpu_only_if_kernels_run(monkeypatch, stdout, returncode, usable, text):
    monkeypatch.setattr(device.subprocess, "run", lambda *a, **k: _Run(stdout, returncode))
    ok, description = device._probe_gpu(timeout_s=1.0)
    assert ok is usable
    assert text in description


def test_unusable_gpu_is_hidden_before_tensorflow_starts():
    code = (
        "import os, sys\n"
        "os.environ.pop('RINGER_COMPUTE_DEVICE', None)\n"
        "from src import device\n"
        "device._probe_gpu = lambda timeout_s: (False, 'GPU X is visible but TensorFlow cannot run kernels on it')\n"
        "os.environ.pop('RINGER_COMPUTE_DEVICE', None)\n"
        "os.environ.pop('CUDA_VISIBLE_DEVICES', None)\n"
        "sys.modules.pop('tensorflow', None)\n"
        "print(device.select_device(verbose=False))\n"
        "print(os.environ['CUDA_VISIBLE_DEVICES'])\n"
    )
    repo = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, cwd=repo)
    lines = out.stdout.strip().splitlines()
    assert lines[-2].startswith("CPU (GPU X is visible")
    assert lines[-1] == "-1"
