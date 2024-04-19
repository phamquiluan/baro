import requests

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def drop_constant(df: pd.DataFrame):
    return df.loc[:, (df != df.iloc[0]).any()]


def drop_near_constant(df: pd.DataFrame, threshold: float = 0.1):
    return df.loc[:, (df != df.iloc[0]).mean() > threshold]


def drop_time(df: pd.DataFrame):
    if "time" in df:
        df = df.drop(columns=["time"])
    if "Time" in df:
        df = df.drop(columns=["Time"])
    if "timestamp" in df:
        df = df.drop(columns=["timestamp"])
    return df


def visualize_metrics(data: pd.DataFrame, filename=None, figsize=None):
    """Visualize the metrics."""
    if figsize is None:
        figsize = (25, 25)

    data = drop_time(data)
    services = []
    metrics = []
    for c in data.columns:
        try:
            service, metric_name = c.split("_", 1)
        except Exception as e:
            print(f"Can not parse {c}")
            continue  # ignore
            # raise e
        if service not in services:
            services.append(service)
        if metric_name not in metrics:
            metrics.append(metric_name)

    n_services = len(services)
    n_metrics = len(metrics)

    fig, axs = plt.subplots(n_services, n_metrics, figsize=figsize)
    fig.tight_layout(pad=3.0)
    for i, service in enumerate(services):
        for j, metric in enumerate(metrics):
            # print(f"{service}_{metric}")
            try:
                axs[i, j].plot(data[f"{service}_{metric}"])
            except Exception:
                pass
            axs[i, j].set_title(f"{service}_{metric}")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    # close the figure
    plt.close(fig)


def download_data(remote_url=None, local_path=None):
    """Download sample metrics data""" 
    if remote_url is None:
        remote_url = "https://github.com/phamquiluan/baro/releases/download/0.0.4/simple_data.csv"
    if local_path is None:
        local_path = "data.csv"

    response = requests.get(remote_url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(
        desc=f"Downloading {local_path}..",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(local_path, "wb") as ref:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            ref.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
