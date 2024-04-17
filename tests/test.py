"""Tests."""
from os import path

import numpy as np
import pandas as pd
import pytest


from baro.anomaly_detection import nsigma

def test_nsigma_basic():
    """Test basic."""
    time_col = np.arange(0, 1000, 1)
    normal_latency = np.random.normal(3, 1, 500)
    normal_latency = np.clip(normal_latency, 1, 5)    
    
    abnormal_latency = np.random.normal(50, 1, 500)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # make df from time_col and latency
    df = pd.DataFrame({'time': time_col, 'latency': latency})
    anomalies = nsigma(df, startsfrom=300)
    assert abs(anomalies[0] - 500) < 10, anomalies

     

test_basic()