import warnings
warnings.filterwarnings("ignore")
import pandas 
import numpy as np

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


def find_anomalies(data, time_col=None,threshold=0.01):
    anomalies = []
    for i in range(1, len(data)):
        if data[i] > threshold:
            # anomalies.append(i)
            anomalies.append(time_col.iloc[i])

    # re-try if threshold doesn't work
    if len(anomalies) == 0:
        head = 5
        data = data[head:]
        # anomalies = [np.argmax(data) + head]
        anomalies = [time_col.iloc[np.argmax(data) + head]]

    # merge continuous anomalies if the distance are shorter than 5 steps
    merged_anomalies = [] if len(anomalies) == 0 else [anomalies[0]]
    for i in range(1, len(anomalies)):
        if anomalies[i] - anomalies[i-1] > 5:
            merged_anomalies.append(anomalies[i])
    
    return merged_anomalies, anomalies


def bocpd(data):
    from functools import partial
    from baro._bocpd import online_changepoint_detection, constant_hazard, MultivariateT
    data = data.copy()
    # "    data = pd.read_csv(data_path)   \n",
    # "    selected_cols = [c for c in data.columns if \"latency-50\" in c]\n",
    # "    data = data[selected_cols]\n",
    # "    data = data.fillna(method=\"ffill\")\n",
    # "    data = data.fillna(0)\n",
    # "\n",
    time_col = data['time']
    data.drop(columns=['time'], inplace=True)
    for c in data.columns:
        data[c] = (data[c] - np.min(data[c])) / (np.max(data[c]) - np.min(data[c]))
    data = data.ffill()
    data = data.fillna(0)
    data = data.to_numpy()

    R, maxes = online_changepoint_detection(
        data,
        partial(constant_hazard, 50),
        MultivariateT(dims=data.shape[1])
    )
    Nw = 10
    anomalies, merged_anomalies = find_anomalies(data=R[Nw,Nw:-1].tolist(), time_col=time_col)
    return anomalies
    

def anomaly_detector(data, method="nsigma"):
    # assert data is dataframe
    assert isinstance(data, pandas.DataFrame)

    # retain all `latency` and `error` columns
    data = data[['latency', 'error']]
    
