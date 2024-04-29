import os
import shutil
import requests
import json
import zipfile
from os.path import join 

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def download_online_boutique_dataset(local_path=None):
    """
    Download the Online Boutique dataset from Zenodo.
    """
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "fse-ob")):
        return
    download_data("https://zenodo.org/records/11046533/files/fse-ob.zip?download=1", "fse-ob.zip")
    with zipfile.ZipFile("fse-ob.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("fse-ob.zip")


def download_sock_shop_dataset(local_path=None):
    """
    Download the Sock Shop dataset from Zenodo. 
    """
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "fse-ss")):
        return   
    download_data("https://zenodo.org/records/11046533/files/fse-ss.zip?download=1", "fse-ss.zip")
    with zipfile.ZipFile("fse-ss.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("fse-ss.zip")


def download_train_ticket_dataset(local_path=None):
    """
    Download the Train Ticket dataset from Zenodo. 
    """
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "fse-tt")):
        return
    download_data("https://zenodo.org/records/11046533/files/fse-tt.zip?download=1", "fse-tt.zip")
    with zipfile.ZipFile("fse-tt.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("fse-tt.zip")
    

def load_json(filename: str):
    with open(filename) as f:
        data = json.load(f)
    return data
              
              
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


def read_data(data_path):
    """Read csv data for root cause analysis."""    
    data = pd.read_csv(data_path)
    data_dir = os.path.dirname(data_path)

    ############# PREPROCESSING ###############
    if "time.1" in data:
        data = data.drop(columns=["time.1"])
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill()
    data = data.fillna(0)

    # remove latency-50 columns
    data = data.loc[:, ~data.columns.str.endswith("latency-50")]
    # rename latency-90 columns to latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )

    # cut the data into 10 mins
    data_length = 300
    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip())
    normal_df = data[data["time"] < inject_time].tail(data_length)
    anomal_df = data[data["time"] >= inject_time].head(data_length)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)    
    return data

def to_service_ranks(ranks):
    """Convert fine-grained ranking to service ranks"""
    _service_ranks = [r.split("_")[0] for r in ranks]
    service_ranks = []
    # remove duplicates
    for s in _service_ranks:
        if s not in service_ranks:
            service_ranks.append(s)
    return service_ranks


def reproduce_baro(dataset=None, fault=None):
    assert dataset in ["fse-ob", "fse-ss", "fse-tt"], f"{dataset} is not supported!"
    assert fault in [None, "all", "cpu", "mem", "delay", "loss"], f"{fault} is not supported!"
    if fault is None:
        fault = "all"
    
    data_paths = list(glob.glob(f"./data/{dataset}/**/simple_data.csv", recursive=True))
    if fault != "all":
        data_paths = [p for p in data_paths if fault in p]
    
    top1_cnt, top2_cnt, top3_cnt, top4_cnt, top5_cnt, total_cnt = 0, 0, 0, 0, 0, 0

    for data_path in data_paths:
        # read data
        data = read_data(data_path)
        data_dir = os.path.dirname(data_path)
        service, metric = basename(dirname(dirname(data_path))).split("_")

        ############# READ ANOMALY DETECTION OUTPUT ###############
        # To reproduce the anomaly detection output, please check
        # the notebook ./tutorials/reproduce_multivariate_bocpd.ipynb
        anomalies = load_json(join(data_dir, "naive_bocpd.json"))
        anomalies = [i[0] for i in anomalies]

        ############# ROOT CAUSE ANALYSIS ###############
        ranks = robust_scorer(data, anomalies=anomalies)["ranks"]
        service_ranks = to_service_ranks(ranks)

        ############## EVALUATION ###############
        if service in service_ranks[:1]:
            top1_cnt += 1
        if service in service_ranks[:2]:
            top2_cnt += 1
        if service in service_ranks[:3]:
            top3_cnt += 1
        if service in service_ranks[:4]:
            top4_cnt += 1
        if service in service_ranks[:5]:
            top5_cnt += 1
        total_cnt += 1

    ############## EVALUATION ###############
    top1_accuracy = top1_cnt / total_cnt
    top2_accuracy = top2_cnt / total_cnt
    top3_accuracy = top3_cnt / total_cnt
    top4_accuracy = top4_cnt / total_cnt
    top5_accuracy = top5_cnt / total_cnt
    avg5_accuracy = (top1_accuracy + top2_accuracy + top3_accuracy + top4_accuracy + top5_accuracy) / 5

    print("====== Reproduce BARO =====")
    print(f"Dataset:   {dataset}")
    print(f"Fault:     {fault}")
    print(f"Avg@5 Acc: {avg5_accuracy}")