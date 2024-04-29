from argparse import ArgumentParser
from baro.reproducibility import reproduce_baro, reproduce_bocpd

DATASET_MAPS = {
    "OnlineBoutique": "fse-ob",
    "SockShop": "fse-ss",
    "TrainTicket": "fse-tt"
}

parser = ArgumentParser()
parser.add_argument("--anomaly-detection", action="store_true")
parser.add_argument("--saved", action="store_true")
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--fault-type", type=str, default=None)

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