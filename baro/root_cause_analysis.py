import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde


from .utility import drop_time, drop_constant, drop_near_constant


def select_useful_cols(data):
    """Select useful columns from the dataset based on certain criteria.

    Parameters:
    - data : pandas.DataFrame
        The dataset to select columns from.

    Returns:
    - selected_cols : list
        A list of selected column names.
    """
    selected_cols = []
    for c in data.columns:
        # keep time
        if "time" in c:
            selected_cols.append(c)

        # cpu
        if c.endswith("_cpu") and data[c].std() > 1:
            selected_cols.append(c)

        # mem
        if c.endswith("_mem") and data[c].std() > 1:
            selected_cols.append(c)

        # latency
        # if ("lat50" in c or "latency" in c) and (data[c] * 1000).std() > 10:
        if "lat50" in c and (data[c] * 1000).std() > 10:
            selected_cols.append(c)
    return selected_cols


def drop_extra(df: pd.DataFrame):
    """Drop extra columns from the DataFrame.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame to remove extra columns from.

    Returns:
    - df : pandas.DataFrame
        The DataFrame after removing extra columns.
    """
    if "time.1" in df:
        df = df.drop(columns=["time.1"])

    # remove cols has "frontend-external" in name
    # remove cols start with "main_" or "PassthroughCluster_", etc.
    for col in df.columns:
        if (
            "frontend-external" in col
            or col.startswith("main_")
            or col.startswith("PassthroughCluster_")
            or col.startswith("redis_")
            or col.startswith("rabbitmq")
            or col.startswith("queue")
            or col.startswith("session")
            or col.startswith("istio-proxy")
        ):
            df = df.drop(columns=[col])

    return df


def convert_mem_mb(df: pd.DataFrame):
    """Convert memory values in the DataFrame to MBs.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame containing memory values.

    Returns:
    - df : pandas.DataFrame
        The DataFrame with memory values converted to MBs.
    """

    # Convert memory to MBs
    def update_mem(x):
        if not x.name.endswith("_mem"):
            return x
        x /= 1e6
        # x = x.astype(int)
        return x

    return df.apply(update_mem)


def preprocess(data, dataset=None, dk_select_useful=False):
    """Preprocess the dataset.

    Parameters:
    - data : pandas.DataFrame
        The dataset to preprocess.
    - dataset : str, optional
        The dataset name. Default is None.
    - dk_select_useful : bool, optional
        Whether to select useful columns. Default is False.

    Returns:
    - data : pandas.DataFrame
        The preprocessed dataset.
    """
    data = drop_constant(drop_time(data))
    data = convert_mem_mb(data)

    if dk_select_useful is True:
        data = drop_extra(data)
        data = drop_near_constant(data)
        data = data[select_useful_cols(data)]
    return data


def nsigma(
    data,
    inject_time=None,
    dataset=None,
    num_loop=None,
    sli=None,
    anomalies=None,
    **kwargs,
):
    """Perform nsigma analysis on the dataset.

    Parameters:
    - data : pandas.DataFrame
        The dataset to perform nsigma analysis on.
    - inject_time : int, optional
        The time of injection of anomalies. Default is None.
    - dataset : str, optional
        The dataset name. Default is None.
    - num_loop : int, optional
        Number of loops. Default is None.
    - sli : int, optional
        SLI (Service Level Indicator). Default is None.
    - anomalies : list, optional
        List of anomalies. Default is None.
    - kwargs : dict
        Additional keyword arguments.

    Returns:
    - dict
        A dictionary containing node names and ranks.
    """
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )

    anomal_df = preprocess(
        data=anomal_df,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = StandardScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }


def kde_cdf_fast(a: np.ndarray, b: np.ndarray, num_points=1000) -> np.ndarray:
    """
    Estimate the cumulative distribution function (CDF) of array `a` using kernel density estimation (KDE),
    and evaluate the CDF at the values in array `b`.

    Parameters
    ----------
    a : np.ndarray
        1D array of samples used to fit the KDE and estimate the CDF.
    b : np.ndarray
        1D array of values at which to evaluate the estimated CDF.
    num_points : int, optional
        Number of points in the grid used for KDE and CDF estimation (default is 1000).

    Returns
    -------
    cdf_values : np.ndarray
        Array of CDF values for each element in `b`, based on the KDE of `a`.
    """

    # Fit KDE
    kde = gaussian_kde(a, bw_method=0.5)

    # Create a grid of x-values over a reasonable range
    grid_x = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), num_points)
    pdf = kde(grid_x)

    # Approximate the CDF with cumulative sum
    dx = grid_x[1] - grid_x[0]
    cdf = np.cumsum(pdf) * dx

    # Normalize to ensure max CDF is 1
    cdf /= cdf[-1]

    # Interpolate CDF values for b
    cdf_values = np.interp(b, grid_x, cdf)

    return cdf_values


def robust_scorer(
    data,
    inject_time=None,
    dataset=None,
    num_loop=None,
    sli=None,
    anomalies=None,
    **kwargs,
):
    """Perform root cause analysis using RobustScorer.

    Parameters:
    - data : pandas.DataFrame
        The datas to perform RobustScorer.
    - inject_time : int, optional
        The time of fault injection time. Default is None.
    - dataset : str, optional
        The dataset name. Default is None.
    - num_loop : int, optional
        Number of loops. Default is None. Just for future API compatible
    - sli : int, optional
        SLI (Service Level Indicator). Default is None. Just for future API compatible
    - anomalies : list, optional
        List of anomalies. Default is None.
    - kwargs : dict
        Additional keyword arguments.

    Returns:
    - dict
        A dictionary containing node names and ranks. `ranks` is a ranked list of root causes.
    """

    data.fillna(0, inplace=True)

    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )

    anomal_df = preprocess(
        data=anomal_df,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        cdf_scores = kde_cdf_fast(a, b)
        score = np.mean(cdf_scores)

        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }
