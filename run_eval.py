#!/usr/bin/env python3
"""
Evaluation script for analyzing benchmark results.

Usage:
    python run_eval.py [OPTIONS]

Examples:
    # Evaluate all methods
    python run_eval.py

    # Evaluate specific methods
    python run_eval.py --methods baro,nsigma

    # Group by system
    python run_eval.py --by-system

    # Export to CSV
    python run_eval.py --csv results.csv
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from data_loader import DataLoader


def load_result(result_file: Path) -> Optional[Dict[str, Any]]:
    """Load a single result file.

    Args:
        result_file: Path to result JSON file.

    Returns:
        Result dictionary or None if file is invalid.
    """
    try:
        with open(result_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def calculate_top_k_accuracy(
    results: List[Dict],
    ground_truth: str,
    k_values: List[int] = [1, 2, 3, 4, 5],
) -> Dict[str, float]:
    """Calculate top-k accuracy for ranked results.

    Uses substring matching: ground_truth "carts" matches "carts_latency".

    Args:
        results: List of {"candidate": name, "score": value} dicts.
        ground_truth: Ground truth service name.
        k_values: List of k values to compute.

    Returns:
        Dictionary with top-k accuracies and avg5.
    """
    accuracies = {}

    for k in k_values:
        top_k = [r["candidate"] for r in results[:k]]
        hit = any(ground_truth in c for c in top_k)
        accuracies[f"top{k}"] = 1.0 if hit else 0.0

    accuracies["avg5"] = sum(accuracies[f"top{k}"] for k in k_values[:5]) / 5.0

    return accuracies


def parse_dataset_id(dataset_id: str) -> Dict[str, str]:
    """Parse metadata from dataset ID.

    Args:
        dataset_id: Dataset identifier (e.g., "re1ss_carts_mem_4")

    Returns:
        Dictionary with parsed metadata.
    """
    import re

    match = re.match(
        r"^re(\d)(ob|ss|tt)_(.+?)_(cpu|mem|delay|loss|disk|socket|f\d)_(\d+)$",
        dataset_id,
    )

    if match:
        level, system, service, fault, instance = match.groups()
        system_names = {"ob": "OnlineBoutique", "ss": "SockShop", "tt": "TrainTicket"}
        return {
            "benchmark": f"RE{level}",
            "system": system_names.get(system, system.upper()),
            "service": service,
            "fault": fault,
            "instance": instance,
        }
    else:
        parts = dataset_id.split("_")
        return {
            "benchmark": "unknown",
            "system": "unknown",
            "service": parts[1] if len(parts) > 1 else "unknown",
            "fault": parts[2] if len(parts) > 2 else "unknown",
            "instance": parts[3] if len(parts) > 3 else "0",
        }


def aggregate_results(
    results: List[Dict[str, Any]],
    group_by: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate accuracy results.

    Args:
        results: List of result dictionaries with accuracies.
        group_by: Optional grouping key (benchmark, system, fault).

    Returns:
        Dictionary mapping group names to average accuracies.
    """
    if group_by is None:
        # Overall average
        if not results:
            return {"all": {"top1": 0, "top3": 0, "top5": 0, "avg5": 0, "n": 0}}

        avg = {
            "top1": sum(r["top1"] for r in results) / len(results),
            "top3": sum(r["top3"] for r in results) / len(results),
            "top5": sum(r["top5"] for r in results) / len(results),
            "avg5": sum(r["avg5"] for r in results) / len(results),
            "n": len(results),
        }
        return {"all": avg}

    # Group by specified key
    groups = defaultdict(list)
    for r in results:
        key = r.get(group_by, "unknown")
        groups[key].append(r)

    aggregated = {}
    for key, items in sorted(groups.items()):
        aggregated[key] = {
            "top1": sum(r["top1"] for r in items) / len(items),
            "top3": sum(r["top3"] for r in items) / len(items),
            "top5": sum(r["top5"] for r in items) / len(items),
            "avg5": sum(r["avg5"] for r in items) / len(items),
            "n": len(items),
        }

    return aggregated


def print_table(
    aggregated: Dict[str, Dict[str, float]],
    method: str,
    group_name: str = "Group",
):
    """Print results as a formatted table.

    Args:
        aggregated: Aggregated results dictionary.
        method: Method name.
        group_name: Name of the grouping dimension.
    """
    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}")
    print(f"{'='*60}")

    # Header
    print(f"{'':15} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Avg@5':>8} {'N':>6}")
    print("-" * 60)

    # Rows
    for key, metrics in aggregated.items():
        print(
            f"{key:15} {metrics['top1']:>8.3f} {metrics['top3']:>8.3f} "
            f"{metrics['top5']:>8.3f} {metrics['avg5']:>8.3f} {metrics['n']:>6}"
        )

    print("-" * 60)


def export_csv(
    all_results: Dict[str, List[Dict]],
    output_file: Path,
):
    """Export results to CSV file.

    Args:
        all_results: Dictionary mapping method names to result lists.
        output_file: Output CSV file path.
    """
    import csv

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "dataset_id", "benchmark", "system", "service", "fault",
            "top1", "top3", "top5", "avg5"
        ])

        for method, results in all_results.items():
            for r in results:
                writer.writerow([
                    method,
                    r["dataset_id"],
                    r["benchmark"],
                    r["system"],
                    r["service"],
                    r["fault"],
                    r["top1"],
                    r["top3"],
                    r["top5"],
                    r["avg5"],
                ])

    print(f"Results exported to {output_file}")


def export_latex(
    all_results: Dict[str, List[Dict]],
    output_file: Path,
    group_by: str = "system",
):
    """Export results to LaTeX table.

    Args:
        all_results: Dictionary mapping method names to result lists.
        output_file: Output LaTeX file path.
        group_by: Grouping dimension for table rows.
    """
    methods = list(all_results.keys())

    with open(output_file, "w") as f:
        # Header
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Root Cause Analysis Results on RCAEval}\n")
        f.write("\\label{tab:rcaeval}\n")

        cols = "l" + "c" * len(methods)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write("\\toprule\n")

        # Method headers
        f.write(f"{group_by.capitalize()}")
        for method in methods:
            f.write(f" & {method.upper()}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        # Get all groups
        all_groups = set()
        for results in all_results.values():
            for r in results:
                all_groups.add(r.get(group_by, "unknown"))

        # Rows
        for group in sorted(all_groups):
            f.write(f"{group}")
            for method in methods:
                results = [r for r in all_results[method] if r.get(group_by) == group]
                if results:
                    avg5 = sum(r["avg5"] for r in results) / len(results)
                    f.write(f" & {avg5:.2f}")
                else:
                    f.write(" & -")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory with results (default: output)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="rcaeval-data",
        help="Path to RCAEval data for ground truth (default: rcaeval-data)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Methods to evaluate, comma-separated (default: all found)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter datasets by pattern",
    )
    parser.add_argument(
        "--by-benchmark",
        action="store_true",
        help="Group results by benchmark (RE1/RE2/RE3)",
    )
    parser.add_argument(
        "--by-system",
        action="store_true",
        help="Group results by system (OnlineBoutique/SockShop/TrainTicket)",
    )
    parser.add_argument(
        "--by-fault",
        action="store_true",
        help="Group results by fault type",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--latex",
        type=str,
        default=None,
        help="Export results to LaTeX table",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)

    # Discover methods from output directory
    available_methods = [
        d.name for d in output_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    if not available_methods:
        print("No results found in output directory")
        sys.exit(1)

    # Filter methods if specified
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        methods = [m for m in methods if m in available_methods]
    else:
        methods = available_methods

    print(f"Evaluating methods: {methods}")

    # Load ground truth from data loader
    filter_patterns = None
    if args.filter:
        filter_patterns = [p.strip() for p in args.filter.split(",")]

    loader = DataLoader(data_path=args.data_path, filter_patterns=filter_patterns)
    ground_truth_map = {}
    for item in loader:
        ground_truth_map[item["dataset_id"]] = item["root_cause_service"]

    print(f"Loaded {len(ground_truth_map)} datasets for evaluation")

    # Load and evaluate results
    all_results = {}

    for method in methods:
        method_dir = output_dir / method
        results = []

        for result_file in sorted(method_dir.glob("*.json")):
            result = load_result(result_file)
            if result is None or "error_type" in result:
                continue

            dataset_id = result["dataset"]
            if dataset_id not in ground_truth_map:
                continue

            if args.filter and not any(p in dataset_id for p in filter_patterns):
                continue

            ground_truth = ground_truth_map[dataset_id]
            metadata = parse_dataset_id(dataset_id)

            # Calculate accuracies
            accuracies = calculate_top_k_accuracy(result["results"], ground_truth)

            results.append({
                "dataset_id": dataset_id,
                **metadata,
                **accuracies,
            })

        all_results[method] = results

    # Determine grouping
    group_by = None
    if args.by_benchmark:
        group_by = "benchmark"
    elif args.by_system:
        group_by = "system"
    elif args.by_fault:
        group_by = "fault"

    # Print results
    for method in methods:
        aggregated = aggregate_results(all_results[method], group_by)
        print_table(aggregated, method, group_by or "Overall")

    # Export if requested
    if args.csv:
        export_csv(all_results, Path(args.csv))

    if args.latex:
        export_latex(all_results, Path(args.latex), group_by or "system")


if __name__ == "__main__":
    main()
