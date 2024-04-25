# Supplementary Material: Running Time \& Instrumentation Cost


## I. Running Time

On three datasets, BARO takes an average of 30 to 173 seconds to perform anomaly detection and root cause analysis. The highest recorded running time of BARO is 7 minutes in a case of the Train Ticket dataset. RobustScorer is very fast when it takes around 0.01 seconds to analyze the root cause, i.e., at least 3 to 1000 times faster than other methods like $\epsilon$-Diagnosis, RCD, CIRCA, and CausalRCA. The detailed running time of our proposed method and different anomaly detection and root cause analysis modules are shown in Tables S1 and S2, respectively.


Table S1. Average execution time (in seconds) of five anomaly detection methods across three different datasets.

| Method      	| Online Boutique 	| Sock Shop 	| Train Ticket 	|
|-------------	|:---------------:	|:---------:	|:------------:	|
| N-Sigma     	|            0.16 	|      0.11 	|         0.53 	|
| BIRCH       	|            0.05 	|      0.04 	|         0.11 	|
| SPOT        	|            3.17 	|       1.9 	|         11.4 	|
| UniBCD      	|           819.7 	|    638.19 	|      3292.48 	|
| **BARO (Ours)** 	|           44.83 	|        30 	|       173.37 	|





Table S2. Average execution time (in seconds) of six root cause analysis methods across three different datasets.

| Method      	| Online Boutique 	| Sock Shop 	| Train Ticket 	|
|-------------	|:---------------:	|:---------:	|:------------:	|
| CausalRCA   	|          299.18 	|    287.18 	|      2638.51 	|
| Ɛ-Diagnosis 	|            3.94 	|      3.97 	|        14.83 	|
| RCD         	|           10.74 	|      5.62 	|        24.21 	|
| CIRCA       	|           13.52 	|     13.47 	|      7564.88 	|
| Nsigma      	|            0.01 	|      0.01 	|         0.01 	|
| **BARO (Ours)** 	|            0.01 	|      0.01 	|         0.01 	|



## II. Instrumentation Cost

BARO requires monitoring agents (e.g., Istio, cAdvisor, Prometheus) installed alongside application services to collect system-level (e.g., CPU/Mem/Disk usage) and application-level metrics (e.g., response time, request count per minute). These metrics are used for BARO to perform anomaly detection and root cause analysis. BARO requires Python and some standard packages (e.g. pandas, numpy). To install and use BARO, users can follow our [README.md](../README.md) document.
