# Installation

We assume the users already satisfy the [REQUIREMENTS.md](REQUIREMENTS.md) and have Python 3.10 installed ([Section 1](https://github.com/phamquiluan/baro/blob/main/INSTALL.md#1-install-python310)). Then, users can install BARO from PyPI ([Section 2](https://github.com/phamquiluan/baro/blob/main/INSTALL.md#2-install-baro-from-pypi)) or build BARO from source ([Section 3](https://github.com/phamquiluan/baro/blob/main/INSTALL.md#3-install-baro-from-source)). In addition, users who familiar with Continuous Integration (CI) can take a look at our [build-and-test.yml](https://github.com/phamquiluan/baro/blob/main/.github/workflows/build-and-test.yml) configuration to see how we test our BARO on Linux and Windows machine from Python 3.7 to 3.12.


## Installation Instruction

### Install Python 3.10

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
sudo apt-get install -y python3.10 python3.10-dev python3.10-venv
```

### Clone BARO from GitHub


```bash
git clone https://github.com/phamquiluan/baro.git && cd baro
```


### Create and activate a virtual environment

```
# create a virtual environment
python3.10 -m venv env

# activate the environment
. env/bin/activate
```

### Install BARO from [PyPI](https://pypi.org/project/fse-baro) or Build BARO from source

```bash
# install BARO from PyPI
pip install fse-baro

# build BARO from source
pip install -e .
```

## Test the installation

Users can perform testing using the following commands:

```bash
pytest tests/test.py
```

<details>
<summary>The expected output would look like this</summary>

```bash

(ins)(env) luan@machine:~/ws/baro$ pytest tests/test.py 
============================================ test session starts =============================================
platform linux -- Python 3.10.13, pytest-7.4.0, pluggy-1.3.0
rootdir: /home/luan/ws/baro
collected 4 items                                                                                            

tests/test.py ....                                                                                     [100%]

======================================= 4 passed in 501.44s (0:08:21) ========================================
(ins)(env) luan@machine:~/ws/baro$ 

```
</details>
