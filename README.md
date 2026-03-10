# BARO+: Enhancing Root Cause Analysis via Multivariate Bayesian Online Change Point Detection and LLM-based Reranking

[![DOI](https://zenodo.org/badge/787200147.svg)](https://zenodo.org/doi/10.5281/zenodo.11063695)
[![pypi package](https://img.shields.io/pypi/v/fse-baro.svg)](https://pypi.org/project/fse-baro)
[![Downloads](https://static.pepy.tech/badge/fse-baro)](https://pepy.tech/project/fse-baro)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/phamquiluan/baro/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/phamquiluan/baro/tree/main)
[![Build and test](https://github.com/phamquiluan/baro/actions/workflows/build-and-test.yml/badge.svg?branch=main)](https://github.com/phamquiluan/baro/actions/workflows/build-and-test.yml)
[![Upload Python Package](https://github.com/phamquiluan/baro/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phamquiluan/baro/actions/workflows/python-publish.yml)

**BARO+** is an end-to-end root cause analysis approach for microservices failures that combines statistical anomaly scoring with LLM-based re-ranking. It extends [BARO](https://dl.acm.org/doi/10.1145/3660805) (FSE'24) by adding a multi-modal LLM re-ranking stage that analyzes metrics, logs, and traces to improve Top-1 accuracy and provide human-readable explanations for root cause decisions.

**Table of Contents**
  * [Overview](#overview)
  * [Installation](#installation)
  * [How-to-use](#how-to-use)
    + [Data format](#data-format)
    + [BARO+ usage example](#baro-usage-example)
    + [BARO baseline usage](#baro-baseline-usage)
  * [Reproducibility](#reproducibility)
    + [BARO+ benchmarking (RCAEval)](#baro-benchmarking-rcaeval)
    + [BARO+ evaluation](#baro-evaluation)
    + [BARO baseline (FSE'24)](#baro-baseline-fse24)
  * [Download Datasets](#download-datasets)
  * [Citation](#citation)
  * [Contact](#contact)

## Overview

BARO+ operates in two stages:

1. **BARO (RobustScorer):** IQR-based anomaly scoring that ranks metrics by deviation from pre-fault baselines. This produces a ranked list of candidate root causes.

2. **LLM Re-ranking:** The top-k candidates from BARO are re-ranked by an LLM that analyzes multi-modal observability data:
   - **Metrics context:** Pre-computed statistical summaries (pre/post means, change percentages, anomaly severity)
   - **Logs context:** Priority-sampled log messages (error logs first, then random samples)
   - **Traces context:** Aggregated trace statistics (latency changes, error rates, slowest operations)

   The LLM returns a re-ranked list along with a natural language explanation of its reasoning.

BARO+ supports 11 LLMs across three providers (Anthropic, OpenAI, Google) and gracefully degrades to BARO when the LLM is unavailable.

## Installation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phamquiluan/baro/blob/main/tutorials/how-to-use-baro.ipynb)

Clone BARO+ from GitHub

```bash
git clone https://github.com/phamquiluan/baro.git && cd baro
```

Install BARO+ from PyPI

```bash
# install BARO+ from PyPI
pip install fse-baro
```

OR, build BARO+ from source

```
# build BARO+ from source
pip install -e .
```

BARO+ has been tested on Linux and Windows, with different Python versions. More details are in [INSTALL.md](./INSTALL.md).

## How-to-use

### Data format

BARO+ supports two data formats:

- **RCAEval format (recommended):** JSON metrics (`metrics.json`) with an injection timestamp (`inject_time.txt`), plus optional `logs.csv` and `traces.csv`. Each dataset directory contains these files. See `rcaeval-data/` for examples.

- **FSE format (legacy):** A `pandas.DataFrame` with a `time` column and metric columns named `<service>_<metric>` (e.g., `cart_cpu`).


### BARO+ usage example

BARO+ uses BARO's RobustScorer for initial scoring, then re-ranks the top-k candidates with an LLM.

```python
from data_loader import DataLoader
from baro.root_cause_analysis import robust_scorer
from baro.context_builder import ContextBuilder
from baro.llm_reranker import LLMReranker

# Load a dataset
loader = DataLoader("rcaeval-data", filter_patterns=["re3ob_adservice_f3_1"])
dataset = next(iter(loader))

# Stage 1: BARO scoring
result = robust_scorer(dataset["metrics"], inject_time=dataset["inject_time"])
ranked = result["scores"]  # [(metric_name, score), ...]

# Aggregate to service level and take top-5
services = ContextBuilder.aggregate_by_service(ranked)
top_5 = services[:5]
top_5_names = [s for s, _ in top_5]

# Build multi-modal context
metrics_ctx = ContextBuilder.build_metrics_context(
    dataset["metrics"], top_5_names, dataset["inject_time"]
)
logs_ctx = ContextBuilder.build_logs_context(
    dataset["logs"], top_5_names, dataset["inject_time"]
)
traces_ctx = ContextBuilder.build_traces_context(
    dataset["traces"], top_5_names, dataset["inject_time"]
)

# Stage 2: LLM re-ranking
reranker = LLMReranker(model="claude-sonnet")
result = reranker.rerank(top_5, metrics_ctx, logs_ctx, traces_ctx, dataset["inject_time"])

print("Ranking:", result["ranking"])
print("Reasoning:", result["reasoning"])
```

### BARO baseline usage

The original BARO pipeline (without LLM re-ranking) is still available:

```python
from baro.anomaly_detection import bocpd
from baro.root_cause_analysis import robust_scorer
from baro.utility import download_data, read_data

# download a sample data to data.csv
download_data()

# read data from data.csv
data = read_data("data.csv")

# perform anomaly detection
anomalies = bocpd(data)
print("Anomalies are detected at timestep:", anomalies[0])

# perform root cause analysis
root_causes = robust_scorer(data, anomalies=anomalies)["ranks"]

# print the top 5 root causes
print("Top 5 root causes:", root_causes[:5])
```


## Reproducibility

### BARO+ benchmarking (RCAEval)

BARO+ is evaluated on the [RCAEval](https://github.com/phamquiluan/RCAEval) benchmark with 90 code-level fault injection cases (RE3) across three microservice systems (Online Boutique, Sock Shop, Train Ticket).

Run BARO+ with a specific LLM:

```bash
python run_bench.py --filter re3 --methods baro+sonnet
```

Run the BARO baseline on all 735 cases:

```bash
python run_bench.py --methods baro
```

### BARO+ evaluation

Aggregate results and generate tables:

```bash
python run_eval.py --by-system
python run_eval.py --by-benchmark
python run_eval.py --csv output/results.csv
```

### Hallucination analysis

Analyze the faithfulness of LLM-generated explanations:

```bash
python analyze_hallucinations.py --data-path rcaeval-data --output output/hallucination_analysis.csv
```

### BARO baseline (FSE'24)

To reproduce the original BARO results from the FSE'24 paper:

```bash
python main.py --dataset OnlineBoutique --fault-type all
python main.py --dataset SockShop --fault-type all
python main.py --dataset TrainTicket --fault-type all
```

## Download Datasets

Our datasets and their description are publicly available in Zenodo repository with the following information:

- Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11046533.svg)](https://doi.org/10.5281/zenodo.11046533)
- Dataset URL: https://zenodo.org/records/11046533

We also provide utility functions to download our datasets using Python. The downloaded datasets will be available at directory `data`.

```python
from baro.utility import (
    download_online_boutique_dataset,
    download_sock_shop_dataset,
    download_train_ticket_dataset,
)
download_online_boutique_dataset()
download_sock_shop_dataset()
download_train_ticket_dataset()
```

## Citation

```bibtex
@inproceedings{pham2024baro,
  title={BARO: Robust root cause analysis for microservices via multivariate bayesian online change point detection},
  author={Pham, Luan and Ha, Huong and Zhang, Hongyu},
  journal={Proceedings of the ACM on Software Engineering},
  volume={1},
  number={FSE},
  pages={2214--2237},
  year={2024},
}
```

## Contact

[phamquiluan\@gmail.com](mailto:phamquiluan@gmail.com?subject=BARO+)
