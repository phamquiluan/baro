"""
Context Builder for BARO+ LLM Re-ranking

This module prepares LLM-friendly context from raw observability data through
a 3-tier token reduction strategy:
1. Temporal filtering (90% reduction)
2. Service filtering (70% reduction)
3. Statistical summarization (90% reduction)

Target: Reduce 8M-40M tokens per case to <20K tokens for LLM processing.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict


class ContextBuilder:
    """Builder for multi-modal context for LLM re-ranking."""

    @staticmethod
    def aggregate_by_service(ranked_metrics: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Aggregate metric-level BARO scores to service-level.

        Algorithm:
        - Extract service prefix: "frontend_cpu" -> "frontend"
        - Aggregate scores: max score among service's metrics
        - Sort by aggregated score descending

        Args:
            ranked_metrics: List of (metric_name, score) tuples from BARO

        Returns:
            List of (service_name, aggregated_score) tuples sorted by score descending

        Example:
            Input: [("frontend_cpu", 8.2), ("frontend_mem", 6.1), ("cartservice_cpu", 7.4)]
            Output: [("frontend", 8.2), ("cartservice", 7.4)]
        """
        service_scores = {}

        for metric_name, score in ranked_metrics:
            # Extract service name (everything before the last underscore)
            # Handle special cases like "loadgenerator" (no underscore)
            parts = metric_name.rsplit('_', 1)
            service_name = parts[0] if len(parts) > 1 else metric_name

            # Use max score among service's metrics
            if service_name not in service_scores:
                service_scores[service_name] = score
            else:
                service_scores[service_name] = max(service_scores[service_name], score)

        # Sort by score descending
        ranked_services = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_services

    @staticmethod
    def _calculate_statistics(values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for a list of values."""
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    @staticmethod
    def _classify_anomaly_severity(change_pct: float) -> str:
        """Classify anomaly severity based on change percentage."""
        abs_change = abs(change_pct)
        if abs_change > 300:
            return "high"
        elif abs_change > 100:
            return "medium"
        elif abs_change > 30:
            return "low"
        else:
            return "minimal"

    @staticmethod
    def build_metrics_context(
        metrics: Dict[str, List[List]],
        top_k_services: List[str],
        inject_time: int,
        pre_window: int = 120,
        post_window: int = 120
    ) -> Dict[str, Dict]:
        """
        Build statistical summary of metric anomalies for top-k services.

        Args:
            metrics: Dict mapping metric names to [[timestamp, value], ...] lists
            top_k_services: List of top-k service names to include
            inject_time: Fault injection timestamp
            pre_window: Seconds before inject_time for baseline (default: 120s)
            post_window: Seconds after inject_time for anomaly detection (default: 120s)

        Returns:
            Dict mapping service names to their metric summaries:
            {
                "frontend": {
                    "cpu": {
                        "pre_mean": 15.2, "post_mean": 78.3, "change_pct": 415.1,
                        "anomaly_severity": "high", "pre_std": 2.1, "post_max": 95.2
                    },
                    "mem": {...}, "latency": {...}
                },
                "cartservice": {...}
            }
        """
        context = defaultdict(dict)

        for metric_name, time_series in metrics.items():
            # Extract service and metric type
            parts = metric_name.rsplit('_', 1)
            if len(parts) < 2:
                continue  # Skip metrics without clear service_metrictype format

            service_name, metric_type = parts[0], parts[1]

            # Only include top-k services
            if service_name not in top_k_services:
                continue

            # Convert to numpy arrays for efficient filtering
            if not time_series:
                continue

            timestamps = np.array([point[0] for point in time_series])
            values = np.array([point[1] for point in time_series])

            # Tier 1: Temporal filtering
            pre_mask = (timestamps >= inject_time - pre_window) & (timestamps < inject_time)
            post_mask = (timestamps >= inject_time) & (timestamps <= inject_time + post_window)

            pre_values = values[pre_mask]
            post_values = values[post_mask]

            if len(pre_values) == 0 or len(post_values) == 0:
                continue

            # Calculate statistics
            pre_stats = ContextBuilder._calculate_statistics(pre_values.tolist())
            post_stats = ContextBuilder._calculate_statistics(post_values.tolist())

            # Calculate change percentage
            pre_mean = pre_stats["mean"]
            post_mean = post_stats["mean"]
            if pre_mean != 0:
                change_pct = ((post_mean - pre_mean) / abs(pre_mean)) * 100
            else:
                change_pct = 0.0 if post_mean == 0 else float('inf')

            # Build summary
            metric_summary = {
                "pre_mean": round(pre_mean, 2),
                "pre_std": round(pre_stats["std"], 2),
                "post_mean": round(post_mean, 2),
                "post_std": round(post_stats["std"], 2),
                "post_max": round(post_stats["max"], 2),
                "change_pct": round(change_pct, 1),
                "anomaly_severity": ContextBuilder._classify_anomaly_severity(change_pct)
            }

            context[service_name][metric_type] = metric_summary

        return dict(context)

    @staticmethod
    def build_logs_context(
        logs_df: Optional[pd.DataFrame],
        top_k_services: List[str],
        inject_time: int,
        time_window: int = 180,
        samples_per_service: int = 15
    ) -> Optional[Dict[str, List[str]]]:
        """
        Extract and sample relevant log entries for top-k services.

        Sampling strategy:
        1. Temporal filter: ±time_window seconds around inject_time
        2. Service filter: container_name matches top-k services
        3. Priority sampling:
           - Error keywords ("error", "timeout", "failed", "exception") -> high priority
           - Random sample from remaining logs

        Args:
            logs_df: DataFrame with columns: timestamp, container_name, message
            top_k_services: List of top-k service names to include
            inject_time: Fault injection timestamp
            time_window: Seconds around inject_time to include (default: ±180s)
            samples_per_service: Max log messages per service (default: 15)

        Returns:
            Dict mapping service names to sampled log messages:
            {
                "frontend": [
                    "[1731903240] request started",
                    "[1731903241] error connecting to cartservice",
                    ...
                ],
                "cartservice": [...]
            }
            None if logs_df is None or empty
        """
        if logs_df is None or logs_df.empty:
            return None

        # Check required columns
        required_cols = ['timestamp', 'container_name', 'message']
        if not all(col in logs_df.columns for col in required_cols):
            return None

        context = {}

        # Error keywords for priority sampling
        error_keywords = ['error', 'timeout', 'failed', 'exception', 'panic',
                          'fatal', 'critical', 'warn', 'denied', 'refused']

        for service_name in top_k_services:
            # Tier 1: Temporal filtering
            time_mask = (logs_df['timestamp'] >= inject_time - time_window) & \
                       (logs_df['timestamp'] <= inject_time + time_window)

            # Tier 2: Service filtering
            service_mask = logs_df['container_name'].str.contains(service_name, case=False, na=False)

            filtered_logs = logs_df[time_mask & service_mask].copy()

            if filtered_logs.empty:
                continue

            # Tier 3: Priority sampling
            sampled_messages = []

            # Priority 1: Error messages (up to 10)
            error_logs = filtered_logs[
                filtered_logs['message'].str.lower().str.contains('|'.join(error_keywords), na=False)
            ]
            if not error_logs.empty:
                error_sample = error_logs.head(10)
                sampled_messages.extend([
                    f"[{int(row['timestamp'])}] {row['message'][:200]}"  # Truncate long messages
                    for _, row in error_sample.iterrows()
                ])

            # Priority 2: Random sample from remaining logs (fill to samples_per_service)
            remaining_count = samples_per_service - len(sampled_messages)
            if remaining_count > 0:
                non_error_logs = filtered_logs[
                    ~filtered_logs['message'].str.lower().str.contains('|'.join(error_keywords), na=False)
                ]
                if not non_error_logs.empty:
                    sample_size = min(remaining_count, len(non_error_logs))
                    random_sample = non_error_logs.sample(n=sample_size, random_state=42)
                    sampled_messages.extend([
                        f"[{int(row['timestamp'])}] {row['message'][:200]}"
                        for _, row in random_sample.iterrows()
                    ])

            if sampled_messages:
                context[service_name] = sampled_messages

        return context if context else None

    @staticmethod
    def build_traces_context(
        traces_df: Optional[pd.DataFrame],
        top_k_services: List[str],
        inject_time: int,
        time_window: int = 180
    ) -> Optional[Dict[str, Dict]]:
        """
        Extract and aggregate trace spans for top-k services.

        Args:
            traces_df: DataFrame with trace span data (columns: serviceName, duration,
                      startTimeMillis, operationName, tags.error, etc.)
            top_k_services: List of top-k service names to include
            inject_time: Fault injection timestamp (in seconds)
            time_window: Seconds around inject_time to include (default: ±180s)

        Returns:
            Dict mapping service names to aggregated trace statistics:
            {
                "frontend": {
                    "avg_duration_pre": 125.3,
                    "avg_duration_post": 3421.7,
                    "p95_duration_post": 8234.1,
                    "error_rate_pre": 0.0,
                    "error_rate_post": 0.15,
                    "slowest_operations": [("GetCart", 4521.3), ("Checkout", 3234.1)]
                },
                "cartservice": {...}
            }
            None if traces_df is None or empty
        """
        if traces_df is None or traces_df.empty:
            return None

        # Check required columns
        required_cols = ['serviceName', 'duration', 'startTimeMillis']
        if not all(col in traces_df.columns for col in required_cols):
            return None

        context = {}

        # Convert inject_time to milliseconds
        inject_time_ms = inject_time * 1000
        time_window_ms = time_window * 1000

        for service_name in top_k_services:
            # Tier 1: Temporal filtering
            time_mask = (traces_df['startTimeMillis'] >= inject_time_ms - time_window_ms) & \
                       (traces_df['startTimeMillis'] <= inject_time_ms + time_window_ms)

            # Tier 2: Service filtering
            service_mask = traces_df['serviceName'].str.contains(service_name, case=False, na=False)

            filtered_traces = traces_df[time_mask & service_mask].copy()

            if filtered_traces.empty:
                continue

            # Split into pre and post fault
            pre_traces = filtered_traces[filtered_traces['startTimeMillis'] < inject_time_ms]
            post_traces = filtered_traces[filtered_traces['startTimeMillis'] >= inject_time_ms]

            # Calculate duration statistics
            service_stats = {}

            if not pre_traces.empty:
                service_stats['avg_duration_pre'] = round(pre_traces['duration'].mean() / 1000, 2)  # Convert to ms
                service_stats['p95_duration_pre'] = round(pre_traces['duration'].quantile(0.95) / 1000, 2)
            else:
                service_stats['avg_duration_pre'] = 0.0
                service_stats['p95_duration_pre'] = 0.0

            if not post_traces.empty:
                service_stats['avg_duration_post'] = round(post_traces['duration'].mean() / 1000, 2)
                service_stats['p95_duration_post'] = round(post_traces['duration'].quantile(0.95) / 1000, 2)
            else:
                service_stats['avg_duration_post'] = 0.0
                service_stats['p95_duration_post'] = 0.0

            # Calculate error rates (if error information is available)
            if 'tags.error' in traces_df.columns or 'error' in traces_df.columns:
                error_col = 'tags.error' if 'tags.error' in traces_df.columns else 'error'

                if not pre_traces.empty:
                    pre_errors = pre_traces[error_col].fillna(False).astype(bool).sum()
                    service_stats['error_rate_pre'] = round(pre_errors / len(pre_traces), 3)
                else:
                    service_stats['error_rate_pre'] = 0.0

                if not post_traces.empty:
                    post_errors = post_traces[error_col].fillna(False).astype(bool).sum()
                    service_stats['error_rate_post'] = round(post_errors / len(post_traces), 3)
                else:
                    service_stats['error_rate_post'] = 0.0

            # Find slowest operations (post-fault)
            if not post_traces.empty and 'operationName' in traces_df.columns:
                op_durations = post_traces.groupby('operationName')['duration'].mean().sort_values(ascending=False)
                slowest_ops = [(op, round(dur / 1000, 2)) for op, dur in op_durations.head(3).items()]
                service_stats['slowest_operations'] = slowest_ops

            if service_stats:
                context[service_name] = service_stats

        return context if context else None
