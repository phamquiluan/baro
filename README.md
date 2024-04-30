# 🕵️ BARO: Root Cause Analysis for Microservices 

[![DOI](https://zenodo.org/badge/787200147.svg)](https://zenodo.org/doi/10.5281/zenodo.11063695)
[![pypi package](https://img.shields.io/pypi/v/fse-baro.svg)](https://pypi.org/project/fse-baro)
[![Downloads](https://static.pepy.tech/badge/fse-baro)](https://pepy.tech/project/fse-baro)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/phamquiluan/baro/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/phamquiluan/baro/tree/main)
[![Build and test](https://github.com/phamquiluan/baro/actions/workflows/build-and-test.yml/badge.svg?branch=main)](https://github.com/phamquiluan/baro/actions/workflows/build-and-test.yml)
[![Upload Python Package](https://github.com/phamquiluan/baro/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phamquiluan/baro/actions/workflows/python-publish.yml)

**BARO** is an end-to-end anomaly detection and root cause analysis approach for microservices's failures. This repository includes artifacts for reuse and reproduction of experimental results presented in our FSE'24 paper titled _"BARO: Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection"_.


## Installation

Install from PyPI

```bash
pip install fse-baro
```

Or, build from source

```bash
git clone https://github.com/phamquiluan/baro.git && cd baro
pip install -e .
```

BARO has been tested on Linux and Windows, with different Python versions. More details are in [INSTALL.md](./INSTALL.md).

## How-to-use

### Data format

The data must be a `pandas.DataFrame` that consists of multivariate time series metrics data. We require the data to have a column named `time` that stores the timestep. Each other column stores a time series for metrics data with the name format of `<service>_<metric>`. For example, the column `cart_cpu` stores the CPU utilization of service `cart`. A sample of valid data could be downloaded using the `download_data()` method that we will demonstrated shortly below.


### Sample Python commands to use BARO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1znckFNPny9zU0Rlc9_Q99E6h3hsJq764?usp=sharing)

BARO consists of two modules, namely MultivariateBOCPD (implemented in `baro.anomaly_detection.bocpd`) and RobustScorer (implemented in `baro.root_cause_analysis.robust_scorer`). We expose these two functions for users/researchers to reuse them more conveniently. The sample commands to run BARO are presented as follows,

```python
import pandas as pd 
from baro.anomaly_detection import bocpd
from baro.root_cause_analysis import robust_scorer
from baro.utility import download_data, read_data

# download a sample data to data.csv
download_data()

# read data, perform anomaly detection and rca using bocpd and robust_scorer
data = read_data("data.csv")
anomalies = bocpd(data)
root_causes = robust_scorer(data, anomalies=anomalies)
print(root_causes)
```

## Reproducibility

As presented in Table 3, BARO achieves Avg@5 of 0.91, 0.96, 0.95, 0.62, and 0.86 for CPU, MEM, DELAY, LOSS, and ALL fault types on the Online Boutique dataset. To reproduce the RCA performance of our BARO as presented in the Table 3. You can run the following commands:

**Reproduce RCA performance on the Online Boutique dataset, fault type CPU**

```bash
$ python main.py --dataset OnlineBoutique --fault-type cpu
```

Expected output

```
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: cpu
Avg@5 Acc : 0.91
```

**Reproduce RCA performance on the Online Boutique dataset, fault type MEM**

```bash
$ python main.py --dataset OnlineBoutique --fault-type mem
```

Expected output

```
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: mem
Avg@5 Acc : 0.96
```

**Reproduce RCA performance on the Online Boutique dataset, fault type DELAY**

```bash
$ python main.py --dataset OnlineBoutique --fault-type delay
```

Expected output

```
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: delay
Avg@5 Acc : 0.95
```

**Reproduce RCA performance on the Online Boutique dataset, fault type LOSS**

```bash
$ python main.py --dataset OnlineBoutique --fault-type loss
```

Expected output

```
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: loss
Avg@5 Acc : 0.62
```

**Reproduce RCA performance on the Online Boutique dataset, fault type ALL**

```bash
$ python main.py --dataset OnlineBoutique --fault-type all
```

Expected output

```
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: all
Avg@5 Acc : 0.86
```



----
We have prepared two Google Colab Notebooks as follows,
1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/120svKTl53cK8KId1rFSrw0BOqnReMB0j?usp=sharing): This notebook reproduces the RCA performance of BARO (also at [tutorials/reproducibility.ipynb](https://github.com/phamquiluan/baro/blob/main/tutorials/reproducibility.ipynb)).
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TObicUGcP9Z9xqML-iJxDo_Vlttp1Lpm?usp=sharing): This nodebook reproduces the output of the Multivariate BOCPD module. 


## Download Paper

TBD

## Download Datasets

Our datasets are publicly available in Zenodo repository with the following information:

- Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11046533.svg)](https://doi.org/10.5281/zenodo.11046533)
- Dataset URL: https://zenodo.org/records/11046533

## Running Time \& Instrumentation Cost

Please refer to our [docs/running_time_and_instrumentation_cost.md](docs/running_time_and_instrumentation_cost.md) document.

## Citation

```
@inproceedings{pham2024baro,
  title={BARO: Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection},
  author={Luan Pham, Huong Ha, and Hongyu Zhang},
  booktitle={Proceedings of the ACM on Software Engineering, Vol 1},
  year={2024},
  organization={ACM}
}
```

## Contact

[luan.pham\@rmit.edu.au](mailto:luan.pham@rmit.edu.au?subject=BARO)
