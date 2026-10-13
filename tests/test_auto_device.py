"""Runtime selection keeps GPU dependencies optional."""

import sys
from types import SimpleNamespace

import pytest

from sort.api import _resolve_auto_device


@pytest.mark.parametrize("cuda_available,cupy_devices,expected", [
    (False, None, "numpy"),
    (True, None, "numpy"),
    (True, 0, "numpy"),
    (True, 1, "cuda"),
])
def test_auto_device_with_optional_cupy(monkeypatch, cuda_available, cupy_devices, expected):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available)))
    cupy = None if cupy_devices is None else SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: cupy_devices)))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    assert _resolve_auto_device() == expected
