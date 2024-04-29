
import os
import glob 
import shutil
import requests
import json
import zipfile
from os.path import join, basename, dirname

import numpy as np
from tqdm import tqdm
import pandas as pd
from baro.utility import (
    read_data,
    load_json,
    to_service_ranks,
    download_online_boutique_dataset,
    download_sock_shop_dataset,
    download_train_ticket_dataset,
)
from baro.root_cause_analysis import robust_scorer


def reproduce_baro(dataset=None, fault=None):
    assert dataset in ["fse-ob", "fse-ss", "fse-tt"], f"{dataset} is not supported!"
    assert fault in [None, "all", "cpu", "mem", "delay", "loss"], f"{fault} is not supported!"
    if fault is None:
        fault = "all"
    
    if not os.path.exists(f"data/{dataset}"):
        if dataset == "fse-ob":
            download_online_boutique_dataset()
        elif dataset == "fse-ss":
            download_sock_shop_dataset()
        elif dataset == "fse-tt":
            download_train_ticket_dataset()
    
    data_paths = list(glob.glob(f"./data/{dataset}/**/simple_data.csv", recursive=True))
    if fault != "all":
        data_paths = [p for p in data_paths if fault in p]
    
    top1_cnt, top2_cnt, top3_cnt, top4_cnt, top5_cnt, total_cnt = 0, 0, 0, 0, 0, 0

    for data_path in tqdm(data_paths, desc=f"Running"):
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
    print(f"Dataset   : {dataset}")
    print(f"Fault type: {fault}")
    print(f"Avg@5 Acc : {avg5_accuracy:.2f}")