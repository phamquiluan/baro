"""Tests."""
from os import path

import numpy as np
import pytest


from baro.anomaly_detection import nsigma

def test_basic():
    """Test basic."""
    time_col = np.arange(0, 1000, 1)
    x_col = np.random.normal(0, 1, 1000)

