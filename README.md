# 🕵️ BARO: Root Cause Analysis for Microservices 

[![pypi package](https://img.shields.io/pypi/v/fse-baro.svg)](https://pypi.org/project/fse-baro)
[![Downloads](https://static.pepy.tech/badge/fse-baro)](https://pepy.tech/project/fse-baro)
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

More details are in [INSTALL.md](./INSTALL.md).

## How-to-use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1znckFNPny9zU0Rlc9_Q99E6h3hsJq764?usp=sharing)


```python
from baro import BARO

m = BARO()

anomalies = m.detect_anomalies(data)
root_causes = m.rca(data, anomalies=anomalies)
print(root_causes)
```

## Download Paper

TBD

## Download Datasets

Our datasets are publicly available in Zenodo repository with the following information:

- Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11046533.svg)](https://doi.org/10.5281/zenodo.11046533)
- Dataset URL: https://zenodo.org/records/11046533

## Reproducibility

Check the Jupyter Notebook at [tutorials/reproducibility.ipynb](https://github.com/phamquiluan/baro/blob/main/tutorials/reproducibility.ipynb) to reproduce the performance of BARO.

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
