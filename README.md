[![pypi package](https://img.shields.io/pypi/v/fse-baro.svg)](https://pypi.org/project/fse-baro)
[![Build and test](https://github.com/phamquiluan/baro/actions/workflows/python-package.yml/badge.svg)](https://github.com/phamquiluan/baro/actions/workflows/python-package.yml)

# WORK IN PROGRESSS

https://2024.esec-fse.org/track/fse-2024-artifacts#submission-for-replicated-and-reproduced-badges

TODO:

- [ ] reproduce RobustScorer
- [ ] reproduce BOCPD
- [ ] make the API
- [ ] restructure + lint code
- [ ] better readme
- [ ] make a Helm chart

## Installation

Build from source

```bash
git clone https://github.com/phamquiluan/baro.git && cd baro
pip install -e .
```

Install from [PyPI](https://pypi.org/project/fse-baro)

```bash
pip install fse-baro
```

## How-to-use

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

TBD 

## Performance comparison

TBD 

## Citation

```
@inproceedings{pham2024baro,
  title={BARO: Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection},
  author={Luan Pham, Huong Ha, and Hongyu Zhang},
  booktitle={Proceedings of the 32nd ACM Symposium on the Foundations of Software Engineering (FSE'24)},
  year={2024},
  organization={ACM}
}
```
