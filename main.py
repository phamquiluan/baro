# usage: python eval.py -i data/sock-shop -m rcd --iter-num 5 --length 300 -w 5
import argparse
import glob
import json
import os
import shutil
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from os.path import abspath, basename, dirname, exists, join

# turn off all warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from cfm.benchmark.evaluation import Evaluator
from cfm.classes.graph import Node

from cfm.io.time_series import drop_constant, drop_time, preprocess
from cfm.utility import dump_json, is_py310, load_json
from cfm.utility.visualization import draw_adj

from baro.root_cause_analysis import robust_scorer


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


data_paths = ["data/fse-ob/cartservice_cpu/1/simple_data.csv"]


def process(data_path):
    print("Process", data_path)

    data = pd.read_csv(data_path)

    # remove lat-50, only selecte lat-90 
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]

    # handle inf
    data = data.replace([np.inf, -np.inf], np.nan)

    # read inject time
    with open("data/fse-ob/cartservice_cpu/1/inject_time.txt") as f:
        inject_time = int(f.readlines()[0].strip())

    length = 10
    normal_df = data[data["time"] < inject_time].tail(length * 60 // 2)
    anomal_df = data[data["time"] >= inject_time].head(length * 60 // 2)

    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # num column, exclude time
    num_node = len(data.columns) - 1

    # rename latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )

    anomalies = load_json("data/fse-ob/cartservice_cpu/1/naive_bocpd.json")
    anomalies = [i[0] for i in anomalies]

    out = func(
        data,
        inject_time,
        dataset=dataset,
        anomalies=anomalies,
        # dk_select_useful=args.useful,
        sli=sli,
        verbose=args.verbose,
        n_iter=num_node,
        gamma=args.gamma,
        addup=args.addup,
        normalize=args.normalize,
        service=service,
        fault_type=fault_type,
        case=case,
        rank=args.rank,
    )
    root_causes = out.get("ranks")


if args.worker_num > 1:
    with Pool(min(args.worker_num, os.cpu_count() - 2)) as p:
        list(tqdm(p.imap(process, data_paths), total=len(data_paths)))
else:  # single worker
    # for data_path in tqdm(sorted(data_paths)):
    for data_path in sorted(data_paths):
        process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
# remove set milliseconds to 0
time_taken = time_taken - timedelta(microseconds=time_taken.microseconds)
with open(join(output_path, "time_taken.txt"), "w") as f:
    s = f"Time taken: {time_taken}"
    f.write(s)


