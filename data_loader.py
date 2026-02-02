"""
Data loader for RCAEval benchmark datasets.

Supports loading metrics from the RCAEval format (metrics.json + inject_time.txt).
"""

import json
import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any


class DataLoader:
    """Load RCAEval datasets from a directory.

    Each dataset directory should contain:
    - metrics.json: Dictionary mapping metric names to time-series data
    - inject_time.txt: Single integer (Unix timestamp of fault injection)

    Dataset naming convention:
        re{level}{system}_{service}_{fault}_{instance}
        Example: re1ss_carts_mem_4

    Where:
    - level: 1, 2, or 3 (RE1/RE2/RE3 benchmarks)
    - system: ob (Online Boutique), ss (Sock Shop), tt (Train Ticket)
    - service: target service name (ground truth root cause)
    - fault: cpu, mem, delay, loss, disk, socket, f1-f5
    - instance: 1-6 (repetition number)
    """

    # Regex pattern for parsing dataset IDs
    DATASET_PATTERN = re.compile(
        r"^re(\d)(ob|ss|tt)_(.+?)_(cpu|mem|delay|loss|disk|socket|f\d)_(\d+)$"
    )

    # System name mappings
    SYSTEM_NAMES = {
        "ob": "OnlineBoutique",
        "ss": "SockShop",
        "tt": "TrainTicket",
    }

    def __init__(
        self,
        data_path: str = "rcaeval-data",
        filter_patterns: Optional[List[str]] = None,
    ):
        """Initialize the data loader.

        Args:
            data_path: Path to the RCAEval data directory.
            filter_patterns: Optional list of patterns to filter datasets.
                Examples: ["re1ob", "re2ss"], ["re1"], ["ss_carts"]
        """
        self.data_path = Path(data_path)
        self.filter_patterns = filter_patterns
        self._datasets = None

    def _discover_datasets(self) -> List[str]:
        """Discover all valid dataset directories.

        Returns:
            List of dataset IDs (directory names).
        """
        datasets = []

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        for item in sorted(self.data_path.iterdir()):
            if not item.is_dir():
                continue

            # Check if directory has required files
            metrics_file = item / "metrics.json"
            inject_time_file = item / "inject_time.txt"

            if not metrics_file.exists() or not inject_time_file.exists():
                continue

            dataset_id = item.name

            # Apply filter patterns if specified
            if self.filter_patterns:
                if not any(p in dataset_id for p in self.filter_patterns):
                    continue

            datasets.append(dataset_id)

        return datasets

    def _parse_dataset_id(self, dataset_id: str) -> Dict[str, Any]:
        """Parse metadata from dataset ID.

        Args:
            dataset_id: Dataset directory name (e.g., "re1ss_carts_mem_4")

        Returns:
            Dictionary with parsed metadata.
        """
        match = self.DATASET_PATTERN.match(dataset_id)

        if match:
            level, system, service, fault, instance = match.groups()
            return {
                "dataset_name": f"RE{level}",
                "system_code": system,
                "system_name": self.SYSTEM_NAMES.get(system, system.upper()),
                "root_cause_service": service,
                "fault_type": fault,
                "case_idx": int(instance),
            }
        else:
            # Fallback parsing for non-standard names
            parts = dataset_id.split("_")
            return {
                "dataset_name": "unknown",
                "system_code": "unknown",
                "system_name": "Unknown",
                "root_cause_service": parts[1] if len(parts) > 1 else "unknown",
                "fault_type": parts[2] if len(parts) > 2 else "unknown",
                "case_idx": int(parts[3]) if len(parts) > 3 else 0,
            }

    def _load_metrics(self, dataset_path: Path) -> Dict[str, List[List]]:
        """Load metrics from metrics.json.

        Args:
            dataset_path: Path to dataset directory.

        Returns:
            Dictionary mapping metric names to time-series data.
            Format: {"metric_name": [[timestamp, value], ...], ...}
        """
        metrics_file = dataset_path / "metrics.json"
        with open(metrics_file, "r") as f:
            return json.load(f)

    def _load_inject_time(self, dataset_path: Path) -> int:
        """Load fault injection timestamp.

        Args:
            dataset_path: Path to dataset directory.

        Returns:
            Unix timestamp of fault injection.
        """
        inject_time_file = dataset_path / "inject_time.txt"
        with open(inject_time_file, "r") as f:
            return int(f.read().strip())

    @property
    def datasets(self) -> List[str]:
        """Get list of discovered datasets."""
        if self._datasets is None:
            self._datasets = self._discover_datasets()
        return self._datasets

    def __len__(self) -> int:
        """Return number of datasets."""
        return len(self.datasets)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over datasets.

        Yields:
            Dictionary for each dataset containing:
            - metrics: Dict[str, List[List[timestamp, value]]]
            - inject_time: int (Unix timestamp)
            - root_cause_service: str (ground truth)
            - dataset_id: str (e.g., "re1ss_carts_mem_4")
            - dataset_name: str (RE1/RE2/RE3)
            - system_name: str (OnlineBoutique/SockShop/TrainTicket)
            - fault_type: str (cpu/mem/delay/loss/disk/socket/f1-f5)
            - case_idx: int (repetition number)
        """
        for dataset_id in self.datasets:
            dataset_path = self.data_path / dataset_id

            # Load data
            metrics = self._load_metrics(dataset_path)
            inject_time = self._load_inject_time(dataset_path)

            # Parse metadata
            metadata = self._parse_dataset_id(dataset_id)

            yield {
                "dataset_id": dataset_id,
                "metrics": metrics,
                "inject_time": inject_time,
                **metadata,
            }

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get a specific dataset by ID.

        Args:
            dataset_id: Dataset directory name.

        Returns:
            Dictionary with dataset data and metadata.
        """
        dataset_path = self.data_path / dataset_id

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")

        metrics = self._load_metrics(dataset_path)
        inject_time = self._load_inject_time(dataset_path)
        metadata = self._parse_dataset_id(dataset_id)

        return {
            "dataset_id": dataset_id,
            "metrics": metrics,
            "inject_time": inject_time,
            **metadata,
        }


def load_rcaeval_data(
    data_path: str = "rcaeval-data",
    filter_patterns: Optional[List[str]] = None,
) -> DataLoader:
    """Convenience function to create a DataLoader.

    Args:
        data_path: Path to RCAEval data directory.
        filter_patterns: Optional patterns to filter datasets.

    Returns:
        DataLoader instance.
    """
    return DataLoader(data_path=data_path, filter_patterns=filter_patterns)
