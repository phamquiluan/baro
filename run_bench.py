#!/usr/bin/env python3
"""
Benchmarking script for running BARO on RCAEval datasets.

Usage:
    python run_bench.py [OPTIONS]

Examples:
    # Run BARO on all datasets
    python run_bench.py

    # Run on specific datasets
    python run_bench.py --filter re1ob

    # Force re-run even if results exist
    python run_bench.py --force
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from tqdm import tqdm

from data_loader import DataLoader
from baro.root_cause_analysis import robust_scorer, robust_scorer_dict


def save_results(
    output_dir: Path,
    dataset_id: str,
    method: str,
    results: List[tuple],
    elapsed_time: float,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
):
    """Save benchmark results to JSON file.

    Args:
        output_dir: Output directory path.
        dataset_id: Dataset identifier.
        method: Method name (e.g., "baro", "nsigma").
        results: List of (candidate, score) tuples.
        elapsed_time: Execution time in seconds.
        error_type: Optional error type if method failed.
        error_message: Optional error message.
    """
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    result_file = method_dir / f"{dataset_id}_{method}.json"

    result_data = {
        "dataset": dataset_id,
        "method": method,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "time": elapsed_time,
        "results": [{"candidate": c, "score": s} for c, s in results],
    }

    if error_type:
        result_data["error_type"] = error_type
        result_data["error_message"] = error_message
        result_data["results"] = []

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)


def result_exists(output_dir: Path, dataset_id: str, method: str) -> bool:
    """Check if results already exist for a dataset/method combination."""
    result_file = output_dir / method / f"{dataset_id}_{method}.json"
    return result_file.exists()


def run_baro(metrics: Dict, inject_time: int) -> List[tuple]:
    """Run BARO RobustScorer on metrics dict.

    Args:
        metrics: Dictionary of metric time-series data.
        inject_time: Fault injection timestamp.

    Returns:
        List of (metric_name, score) tuples, sorted descending by score.
    """
    result = robust_scorer(metrics, inject_time=inject_time)
    return result.get("scores", [])


def run_nsigma(metrics: Dict, inject_time: int) -> List[tuple]:
    """Run N-Sigma baseline on metrics dict.

    Similar to robust_scorer_dict but uses mean/std instead of median/IQR.

    Args:
        metrics: Dictionary of metric time-series data.
        inject_time: Fault injection timestamp.

    Returns:
        List of (metric_name, score) tuples, sorted descending by score.
    """
    ranked_list = []

    for metric_name, metric_data in metrics.items():
        # Split by inject_time
        pre_data = [float(v) for ts, v in metric_data if ts < inject_time]
        post_data = [float(v) for ts, v in metric_data if ts >= inject_time]

        if len(pre_data) < 4 or not post_data:
            continue

        # Calculate mean and std from pre-fault data
        mean = sum(pre_data) / len(pre_data)
        variance = sum((x - mean) ** 2 for x in pre_data) / len(pre_data)
        std = variance ** 0.5

        if std == 0:
            std = 1

        # Score = max z-score in post-fault period
        max_val = max(post_data)
        score = abs(max_val - mean) / std
        ranked_list.append((metric_name, score))

    ranked_list.sort(key=lambda x: x[1], reverse=True)
    return ranked_list


AVAILABLE_METHODS = {
    "baro": run_baro,
    "nsigma": run_nsigma,
}


def calculate_top_k_accuracy(
    ranked_list: List[tuple],
    ground_truth: str,
    k_values: List[int] = [1, 2, 3, 4, 5],
) -> Dict[str, float]:
    """Calculate top-k accuracy for a single dataset.

    Uses substring matching: ground_truth "carts" matches "carts_latency".

    Args:
        ranked_list: List of (candidate, score) tuples.
        ground_truth: Ground truth service name.
        k_values: List of k values to compute.

    Returns:
        Dictionary with top-k accuracies and avg5.
    """
    results = {}

    for k in k_values:
        top_k = [c for c, _ in ranked_list[:k]]
        hit = any(ground_truth in c for c in top_k)
        results[f"top{k}"] = 1.0 if hit else 0.0

    results["avg5"] = sum(results[f"top{k}"] for k in k_values[:5]) / 5.0

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BARO on RCAEval datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_bench.py --filter re1ob
    python run_bench.py --methods baro,nsigma
    python run_bench.py --force
        """,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="rcaeval-data",
        help="Path to RCAEval data directory (default: rcaeval-data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter datasets by pattern (e.g., 're1ob', 're2ss')",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="baro",
        help="Methods to run, comma-separated (default: baro)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results already exist",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per method in seconds (default: 120)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Print evaluation summary after running",
    )

    args = parser.parse_args()

    # Parse methods
    methods = [m.strip() for m in args.methods.split(",")]
    for method in methods:
        if method not in AVAILABLE_METHODS:
            print(f"Error: Unknown method '{method}'")
            print(f"Available methods: {', '.join(AVAILABLE_METHODS.keys())}")
            sys.exit(1)

    # Parse filter patterns
    filter_patterns = None
    if args.filter:
        filter_patterns = [p.strip() for p in args.filter.split(",")]

    # Initialize data loader
    loader = DataLoader(data_path=args.data_path, filter_patterns=filter_patterns)
    output_dir = Path(args.output)

    print(f"Data path: {args.data_path}")
    print(f"Output dir: {output_dir}")
    print(f"Methods: {methods}")
    print(f"Filter: {filter_patterns}")
    print(f"Datasets found: {len(loader)}")
    print()

    # Track statistics
    stats = {method: {"total": 0, "skipped": 0, "success": 0, "error": 0} for method in methods}
    all_results = {method: [] for method in methods}

    # Run benchmarks
    for item in tqdm(loader, desc="Running"):
        dataset_id = item["dataset_id"]
        metrics = item["metrics"]
        inject_time = item["inject_time"]
        ground_truth = item["root_cause_service"]

        for method in methods:
            stats[method]["total"] += 1

            # Check if result exists
            if not args.force and result_exists(output_dir, dataset_id, method):
                stats[method]["skipped"] += 1
                continue

            # Run method
            try:
                start_time = time.time()
                method_func = AVAILABLE_METHODS[method]
                results = method_func(metrics, inject_time)
                elapsed_time = time.time() - start_time

                save_results(output_dir, dataset_id, method, results, elapsed_time)
                stats[method]["success"] += 1

                # Track for evaluation
                all_results[method].append({
                    "dataset_id": dataset_id,
                    "ground_truth": ground_truth,
                    "results": results,
                })

            except Exception as e:
                elapsed_time = time.time() - start_time
                save_results(
                    output_dir,
                    dataset_id,
                    method,
                    [],
                    elapsed_time,
                    error_type="unknown",
                    error_message=str(e),
                )
                stats[method]["error"] += 1

    # Print statistics
    print("\n=== Benchmark Statistics ===")
    for method in methods:
        s = stats[method]
        print(f"\n{method.upper()}:")
        print(f"  Total:   {s['total']}")
        print(f"  Skipped: {s['skipped']}")
        print(f"  Success: {s['success']}")
        print(f"  Errors:  {s['error']}")

    # Print evaluation if requested
    if args.eval:
        print("\n=== Evaluation Summary ===")
        for method in methods:
            if not all_results[method]:
                print(f"\n{method.upper()}: No results to evaluate")
                continue

            accuracies = []
            for item in all_results[method]:
                acc = calculate_top_k_accuracy(item["results"], item["ground_truth"])
                accuracies.append(acc)

            # Calculate averages
            avg_top1 = sum(a["top1"] for a in accuracies) / len(accuracies)
            avg_top3 = sum(a["top3"] for a in accuracies) / len(accuracies)
            avg_top5 = sum(a["top5"] for a in accuracies) / len(accuracies)
            avg5 = sum(a["avg5"] for a in accuracies) / len(accuracies)

            print(f"\n{method.upper()} (n={len(accuracies)}):")
            print(f"  Top-1: {avg_top1:.3f}")
            print(f"  Top-3: {avg_top3:.3f}")
            print(f"  Top-5: {avg_top5:.3f}")
            print(f"  Avg@5: {avg5:.3f}")


if __name__ == "__main__":
    main()
