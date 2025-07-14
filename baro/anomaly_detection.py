import warnings
warnings.filterwarnings("ignore")
import pandas 
import numpy as np
from baro.utility import drop_constant, find_cps

def nsigma(data, k=3, startsfrom=100):
    """For each time series (column) in the data,
    detect anomalies using the n-sigma rule.
    
    Parameters:
    - data : pandas DataFrame
        The input data containing time series columns.
    - k : int, optional
        The number of standard deviations from the mean to consider as an anomaly. Default is 3.
    - startsfrom : int, optional
        The index from which to start calculating mean and standard deviation. Default is 100.
        
    Returns:
    - anomalies : list
        List of timestamps where anomalies were detected.
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
    """Find anomalies in the data based on a given threshold.
    
    Parameters:
    - data : list or numpy array
        The input data to search for anomalies.
    - time_col : pandas Series, optional
        The timestamps corresponding to the data. Default is None.
    - threshold : float, optional
        The threshold value above which a data point is considered an anomaly. Default is 0.01.
        
    Returns:
    - merged_anomalies : list
        List of merged timestamps where anomalies were detected.
    - anomalies : list
        List of timestamps where anomalies were detected.
    """
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
    """Perform Multivariate Bayesian Online Change Point Detection (BOCPD) on the input data.
    
    Parameters:
    - data : pandas DataFrame
        The input data containing metrics from microservices.
        
    Returns:
    - anomalies : list
        List of timestamps where anomalies were detected.
    """
    from functools import partial
    from baro._bocpd import online_changepoint_detection, constant_hazard, MultivariateT
    data = data.copy()
    
    # select latency and error metrics from microservices
    selected_cols = []
    for c in data.columns:
        if 'queue-master' in c or 'rabbitmq_' in c: continue
        if "latency" in c or "latency-50" in c or "_error" in c:
            selected_cols.append(c)
    
    if selected_cols:
        data = data[selected_cols]
    else:
        # Warning: no latency or error columns found, use all time series
        warnings.warn("No latency or error columns found. Performing anomaly detection on all time series.")
        # Remove time column if it exists, use all other columns
        non_time_cols = [c for c in data.columns if c != 'time']
        if non_time_cols:
            data = data[non_time_cols]
        else:
            # If no non-time columns, we can't perform anomaly detection
            warnings.warn("No non-time columns found. Cannot perform anomaly detection.")
            return []

    # handle na
    data = drop_constant(data)
    data = data.ffill()
    data = data.fillna(0)
    
    # normalize data, handling edge cases
    for c in data.columns:
        col_min = np.min(data[c])
        col_max = np.max(data[c])
        col_range = col_max - col_min
        
        # Handle constant data (no variance)
        if col_range == 0 or np.isnan(col_range):
            # If all values are the same, set to 0
            data[c] = 0
        else:
            # Normal normalization
            data[c] = (data[c] - col_min) / col_range
    
    data = data.ffill()
    data = data.fillna(0)
    
    # Check if we have any data left after preprocessing
    if data.empty or data.shape[0] == 0:
        warnings.warn("No data available after preprocessing. Cannot perform anomaly detection.")
        return []
        
    data = data.to_numpy()

    R, maxes = online_changepoint_detection(
        data,
        partial(constant_hazard, 50),
        MultivariateT(dims=data.shape[1])
    )
    cps = find_cps(maxes)
    anomalies = [p[0] for p in cps]
    # anomalies, merged_anomalies = find_anomalies(data=R[Nw,Nw:-1].tolist(), time_col=time_col)
    
    return anomalies
    
