# BARO: Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection

[![pypi package](https://img.shields.io/pypi/v/fse-baro.svg)](https://pypi.org/project/fse-baro)
[![Build and test](https://github.com/phamquiluan/baro/actions/workflows/build-and-test.yml/badge.svg?branch=main)](https://github.com/phamquiluan/baro/actions/workflows/build-and-test.yml)


In the progress of preparing the Artifact submission:
- Deadline 30/4
- https://2024.esec-fse.org/track/fse-2024-artifacts#submission-for-replicated-and-reproduced-badges
- Can apply for all three badges: Available, Function, Reusable

**Functional: The artifacts associated with the research are found to be documented, consistent, complete, exercisable, and include appropriate evidence of verification and validation.	
**

**Reusable: Functional + the artifacts associated with the paper are of a quality that significantly exceeds minimal functionality. They are very carefully documented and well-structured to the extent that reuse and repurposing is facilitated. In particular, norms and standards of the research community for artifacts of this type are strictly adhered to.	
**

TODO:

- [ ] make the end-to-end API
- [ ] restructure + lint code
- [ ] better readme
- [ ] reproduce BOCPD

## Installation

Install from [PyPI](https://pypi.org/project/fse-baro)

```bash
pip install fse-baro
```

Or, build from source

```bash
git clone https://github.com/phamquiluan/baro.git && cd baro
pip install -e .
```

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

Our datasets are publicly available in Zenodo repository with the following information

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
