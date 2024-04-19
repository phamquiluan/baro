import pandas as pd
import matplotlib.pyplot as plt


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
