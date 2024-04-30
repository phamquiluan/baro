from argparse import ArgumentParser, RawTextHelpFormatter
from baro.reproducibility import reproduce_baro, reproduce_bocpd

DATASET_MAPS = {
    "OnlineBoutique": "fse-ob",
    "SockShop": "fse-ss",
    "TrainTicket": "fse-tt"
}

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("--anomaly-detection", action="store_true", help="This flag is used to reproduce the anomaly detection results. Using\nthis flag omits the `--fault-type` argument")
parser.add_argument("--saved", action="store_true", help="This flag is used to use saved anomaly detection results to reproduce\nthe presented results without running anomaly detection again")
parser.add_argument("--dataset", type=str, default=None, help="Valid options are: ['OnlineBoutique', 'SockShop', and 'TrainTicket']")
parser.add_argument("--fault-type", type=str, default=None, help="Valid options are: ['cpu', 'mem', 'delay', 'loss', and 'all']. If 'all' is\nselected, the program will run the root cause analysis for all fault types")

args = parser.parse_args()
if args.dataset not in DATASET_MAPS:
    print(f"Dataset {args.dataset} is not supported! Valid datasets are {list(DATASET_MAPS.keys())}")
    exit(0)
if args.fault_type not in [None, "all", "cpu", "mem", "delay", "loss"]:
    print(f"Fault type {args.fault_type} is not supported! Valid fault types are [None, 'all', 'cpu', 'mem', 'delay', 'loss']")
    exit(0)

if not args.anomaly_detection:
    reproduce_baro(dataset=DATASET_MAPS[args.dataset], fault=args.fault_type)
else:
    reproduce_bocpd(dataset=DATASET_MAPS[args.dataset], saved=args.saved)