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


def find_anomalies(data, threshold=0.01):
    anomalies = []
    for i in range(1, len(data)):
        if data[i] > threshold:
            anomalies.append(i)

    # re-try if threshold doesn't work
    if len(anomalies) == 0:
        head = 5
        data = data[head:]
        anomalies = [np.argmax(data) + head]

    # merge continuous anomalies if the distance are shorter than 5 steps
    merged_anomalies = [] if len(anomalies) == 0 else [anomalies[0]]
    for i in range(1, len(anomalies)):
        if anomalies[i] - anomalies[i-1] > 5:
            merged_anomalies.append(anomalies[i])
    
    return merged_anomalies, anomalies


def bocpd(data):
    from functools import partial
    from baro._bocpd import online_changepoint_detection, constant_hazard
    R, maxes = online_changepoint_detection(
        data, hazard_function, online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0)
    )

def anomaly_detector(data, method="nsigma"):
    # assert data is dataframe
    assert isinstance(data, pandas.DataFrame)

    # retain all `latency` and `error` columns
    data = data[['latency', 'error']]
    
