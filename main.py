from argparse import ArgumentParser
from baro.reproducibility import reproduce_baro

DATASET_MAPS = {
    "OnlineBoutique": "fse-ob",
    "SockShop": "fse-ss",
    "TrainTicket": "fse-tt"
}

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--fault-type", type=str, default=None)

args = parser.parse_args()
if args.dataset not in DATASET_MAPS:
    print(f"{args.dataset} is not supported! Valid datasets are {list(DATASET_MAPS.keys())}")
if args.fault_type not in [None, "all", "cpu", "mem", "delay", "loss"]:
    print(f"{args.fault_type} is not supported! Valid fault types are [None, 'all', 'cpu', 'mem', 'delay', 'loss']")

reproduce_baro(dataset=DATASET_MAPS[args.dataset], fault=args.fault_type)