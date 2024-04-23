# Installation

We assume the users already satisfy the [REQUIREMENTS.md](REQUIREMENTS.md). Then, users can install BARO from PyPI ([Section 2](https://github.com/phamquiluan/baro/blob/main/INSTALL.md#2-install-baro-from-pypi)) or build BARO from source ([Section 3](https://github.com/phamquiluan/baro/blob/main/INSTALL.md#3-install-baro-from-source)).


## 1. Install Python3.10

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
sudo apt-get install -y python3.10 python3.10-dev python3.10-venv
```

## 2. Install BARO from [PyPI](https://pypi.org/project/fse-baro)
### 2.1. Create and activate a virtual environment

```
# create a virtual environment
python3.10 -m venv env

# activate the environment
. env/bin/activate
```

### 2.2. Install BARO from [PyPI](https://pypi.org/project/fse-baro)

```bash
pip install fse-baro
```

## 2.3. Check the installed BARO

Users can perform sanity check using 
```bash
pip show fse-baro
# or 
python -c 'from baro.root_cause_analysis import robust_scorer'
```

<details>
<summary>The expected output would look like this</summary>

```bash

(ins)(env) luan@machine:~/tmp$ pip show fse-baro
Name: fse-baro
Version: 0.0.7
Summary: BARO: Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection
Home-page: 
Author: 
Author-email: Luan Pham <phamquiluan@gmail.com>
License: MIT License
        
        Copyright (c) 2024 Luan Pham
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        
Location: /home/luan/tmp/env/lib/python3.10/site-packages
Requires: matplotlib, numpy, pandas, pytest, requests, scikit-learn, tqdm
Required-by: 
(ins)(env) luan@machine:~/tmp$ python -c 'from baro.root_cause_analysis import robust_scorer'
(ins)(env) luan@machine:~/tmp$ 
```
</details>






## 3. Install BARO from source
### 3.1. Clone BARO from GitHub


```bash
git clone https://github.com/phamquiluan/baro.git
cd baro
```

### 3.2. Create and activate a virtual environment

```bash
# create a virtual environment
python3.10 -m venv env

# activate the environment
. env/bin/activate
```

### 3.3. Install BARO with editable mode

```bash
pip install -e .
```

## 2.3. Check the installed BARO

Users can test the installed BARO using 

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
collected 2 items                                                                                            

tests/test.py ..                                                                                       [100%]

============================================= 2 passed in 4.78s ==============================================
(ins)(env) luan@machine:~/ws/baro$ 

```
</details>
