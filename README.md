[![pypi package](https://img.shields.io/pypi/v/fse-baro.svg)](https://pypi.org/project/fse-baro)
[![Build and test](https://github.com/phamquiluan/baro/actions/workflows/python-package.yml/badge.svg)](https://github.com/phamquiluan/baro/actions/workflows/python-package.yml)

# WORK IN PROGRESSS

TBD

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

TBD
