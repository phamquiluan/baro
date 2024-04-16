import pandas 

def nsigma(data, k=3, startsfrom=100):
    """For each time series (column) in the data,
    detect anomalies using the n-sigma rule.
    Return the timestamps of the anomalies.
    """
    anomalies = []
    for col in data.columns:
        if col == "time":
            continue
        # for each timestep starts from `startsfrom`,
        # calculate the mean and standard deviation
        # of the all past timesteps
        for i in range(startsfrom, len(data)):
            mean = data[col].iloc[:i].mean()
            std = data[col].iloc[:i].std()
            if abs(data[col].iloc[i] - mean) > k * std:
                anomalies.append(data['time'].iloc[i])
    return anomalies




def bocpd(data):
    # TODO: Implement Bayesian Online Change Point Detection
    raise NotImplementedError

def anomaly_detector(data, method="nsigma"):
    # assert data is dataframe
    assert isinstance(data, pandas.DataFrame)

    # retain all `latency` and `error` columns
    data = data[['latency', 'error']]
    
