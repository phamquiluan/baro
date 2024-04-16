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

from baro.root_cause_analysis import robust_scorer


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


data = pd.read_csv("data/fse-ob/cartservice_cpu/1/simple_data.csv")

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

out = robust_scorer(data, inject_time, dataset="fse-ob", anomalies=anomalies)
root_causes = out.get("ranks")


print(root_causes)


