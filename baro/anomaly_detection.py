import pandas 

def nsigma(data, k=3):
    pass

def bocpd(data):
    # TODO: Implement Bayesian Online Change Point Detection
    raise NotImplementedError

def anomaly_detector(data, method="nsigma"):
    # assert data is dataframe
    assert isinstance(data, pandas.DataFrame)

    # retain all `latency` and `error` columns
    data = data[['latency', 'error']]
    
