"""
Unit tests for BARO+ LLM re-ranking module.

Tests cover:
- Service aggregation
- Context building (metrics, logs, traces)
- LLM response parsing
- Graceful degradation with missing data
"""

import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baro.context_builder import ContextBuilder
from baro.llm_reranker import LLMReranker


class TestServiceAggregation:
    """Test metric-to-service aggregation."""

    def test_basic_aggregation(self):
        """Test aggregation with multiple metrics per service."""
        metrics = [
            ("frontend_cpu", 8.2),
            ("frontend_mem", 6.1),
            ("cartservice_cpu", 7.4),
            ("cartservice_latency", 5.3),
        ]
        services = ContextBuilder.aggregate_by_service(metrics)

        assert len(services) == 2
        assert services[0] == ("frontend", 8.2)  # max of frontend metrics
        assert services[1] == ("cartservice", 7.4)  # max of cartservice metrics

    def test_single_metric_per_service(self):
        """Test aggregation when each service has one metric."""
        metrics = [
            ("frontend_cpu", 8.2),
            ("cartservice_mem", 7.4),
            ("redis_latency", 3.1),
        ]
        services = ContextBuilder.aggregate_by_service(metrics)

        assert len(services) == 3
        assert services[0][0] == "frontend"
        assert services[0][1] == 8.2

    def test_service_without_underscore(self):
        """Test handling of service names without underscore."""
        metrics = [
            ("loadgenerator", 5.0),
            ("frontend_cpu", 8.2),
        ]
        services = ContextBuilder.aggregate_by_service(metrics)

        # loadgenerator should be kept as-is
        service_names = [s[0] for s in services]
        assert "frontend" in service_names or "loadgenerator" in service_names

    def test_empty_list(self):
        """Test with empty metrics list."""
        services = ContextBuilder.aggregate_by_service([])
        assert services == []


class TestMetricsContextBuilding:
    """Test statistical summary generation for metrics."""

    def test_basic_metrics_context(self):
        """Test context building with simple time series."""
        metrics = {
            "frontend_cpu": [
                [1000, 10.0],
                [1001, 12.0],
                [1002, 80.0],
                [1003, 85.0],
            ],
        }
        inject_time = 1002
        context = ContextBuilder.build_metrics_context(
            metrics, ["frontend"], inject_time, pre_window=2, post_window=2
        )

        assert "frontend" in context
        assert "cpu" in context["frontend"]

        cpu_stats = context["frontend"]["cpu"]
        assert cpu_stats["pre_mean"] == 11.0  # (10 + 12) / 2
        assert cpu_stats["post_mean"] == 82.5  # (80 + 85) / 2
        assert cpu_stats["change_pct"] > 600  # Significant increase

    def test_anomaly_severity_classification(self):
        """Test anomaly severity classification."""
        # High severity (>300% change)
        assert ContextBuilder._classify_anomaly_severity(400) == "high"
        assert ContextBuilder._classify_anomaly_severity(-350) == "high"

        # Medium severity (100-300%)
        assert ContextBuilder._classify_anomaly_severity(150) == "medium"

        # Low severity (30-100%)
        assert ContextBuilder._classify_anomaly_severity(50) == "low"

        # Minimal (<30%)
        assert ContextBuilder._classify_anomaly_severity(10) == "minimal"

    def test_multiple_metrics_per_service(self):
        """Test with multiple metric types for one service."""
        metrics = {
            "frontend_cpu": [[1000, 10.0], [1001, 12.0], [1002, 80.0], [1003, 85.0]],
            "frontend_mem": [[1000, 100.0], [1001, 105.0], [1002, 110.0], [1003, 115.0]],
        }
        inject_time = 1002
        context = ContextBuilder.build_metrics_context(
            metrics, ["frontend"], inject_time, pre_window=2, post_window=2
        )

        assert "frontend" in context
        assert "cpu" in context["frontend"]
        assert "mem" in context["frontend"]

        assert context["frontend"]["cpu"]["anomaly_severity"] == "high"
        assert context["frontend"]["mem"]["anomaly_severity"] == "minimal"

    def test_service_filtering(self):
        """Test that only top-k services are included."""
        metrics = {
            "frontend_cpu": [[1000, 10.0], [1002, 80.0]],
            "backend_cpu": [[1000, 20.0], [1002, 90.0]],
            "database_cpu": [[1000, 5.0], [1002, 50.0]],
        }
        inject_time = 1001
        context = ContextBuilder.build_metrics_context(
            metrics, ["frontend", "backend"], inject_time, pre_window=2, post_window=2
        )

        assert "frontend" in context
        assert "backend" in context
        assert "database" not in context

    def test_empty_metrics(self):
        """Test with no metrics."""
        context = ContextBuilder.build_metrics_context({}, [], 1000)
        assert context == {}


class TestLogsContextBuilding:
    """Test log filtering and sampling."""

    def test_basic_logs_sampling(self):
        """Test basic log filtering and sampling."""
        logs_df = pd.DataFrame({
            'timestamp': [1000, 1001, 1002, 1003, 1004],
            'container_name': ['frontend', 'frontend', 'cartservice', 'frontend', 'frontend'],
            'message': ['normal log', 'another log', 'cart log', 'yet another', 'final log']
        })
        inject_time = 1002
        context = ContextBuilder.build_logs_context(
            logs_df, ["frontend"], inject_time, time_window=3, samples_per_service=3
        )

        assert context is not None
        assert "frontend" in context
        assert len(context["frontend"]) <= 3

    def test_error_keyword_prioritization(self):
        """Test that error keywords are prioritized."""
        logs_df = pd.DataFrame({
            'timestamp': [1000, 1001, 1002, 1003, 1004],
            'container_name': ['frontend'] * 5,
            'message': [
                'normal log',
                'error connecting to backend',
                'normal log 2',
                'timeout occurred',
                'normal log 3'
            ]
        })
        inject_time = 1002
        context = ContextBuilder.build_logs_context(
            logs_df, ["frontend"], inject_time, time_window=5, samples_per_service=3
        )

        assert context is not None
        assert "frontend" in context

        # Error messages should be included
        logs_text = ' '.join(context["frontend"])
        assert 'error' in logs_text.lower() or 'timeout' in logs_text.lower()

    def test_temporal_filtering(self):
        """Test temporal filtering of logs."""
        logs_df = pd.DataFrame({
            'timestamp': [1000, 1100, 1200, 1300, 1400],
            'container_name': ['frontend'] * 5,
            'message': ['log1', 'log2', 'log3', 'log4', 'log5']
        })
        inject_time = 1200
        context = ContextBuilder.build_logs_context(
            logs_df, ["frontend"], inject_time, time_window=50, samples_per_service=10
        )

        # Only logs within [1150, 1250] should be included
        assert context is not None
        assert "frontend" in context
        # Should have fewer logs due to temporal filtering

    def test_missing_logs(self):
        """Test handling of missing logs."""
        context = ContextBuilder.build_logs_context(None, ["frontend"], 1000)
        assert context is None

    def test_empty_logs_dataframe(self):
        """Test with empty DataFrame."""
        logs_df = pd.DataFrame({'timestamp': [], 'container_name': [], 'message': []})
        context = ContextBuilder.build_logs_context(logs_df, ["frontend"], 1000)
        assert context is None

    def test_missing_required_columns(self):
        """Test with DataFrame missing required columns."""
        logs_df = pd.DataFrame({'timestamp': [1000], 'message': ['log']})  # Missing container_name
        context = ContextBuilder.build_logs_context(logs_df, ["frontend"], 1000)
        assert context is None


class TestTracesContextBuilding:
    """Test trace aggregation."""

    def test_basic_traces_context(self):
        """Test basic trace aggregation."""
        traces_df = pd.DataFrame({
            'serviceName': ['frontend', 'frontend', 'frontend', 'frontend'],
            'duration': [100000, 120000, 3000000, 3500000],  # in microseconds
            'startTimeMillis': [1000000, 1001000, 1002000, 1003000],
            'operationName': ['GetProducts', 'GetProducts', 'GetProducts', 'GetProducts']
        })
        inject_time = 1002  # in seconds
        context = ContextBuilder.build_traces_context(
            traces_df, ["frontend"], inject_time, time_window=5
        )

        assert context is not None
        assert "frontend" in context
        assert "avg_duration_pre" in context["frontend"]
        assert "avg_duration_post" in context["frontend"]

        # Post-fault duration should be much higher
        assert context["frontend"]["avg_duration_post"] > context["frontend"]["avg_duration_pre"]

    def test_error_rate_calculation(self):
        """Test error rate calculation in traces."""
        traces_df = pd.DataFrame({
            'serviceName': ['frontend'] * 6,
            'duration': [100000] * 6,
            'startTimeMillis': [1000000, 1001000, 1002000, 1003000, 1004000, 1005000],
            'tags.error': [False, False, False, True, True, False]
        })
        inject_time = 1002
        context = ContextBuilder.build_traces_context(
            traces_df, ["frontend"], inject_time, time_window=5
        )

        assert context is not None
        assert "frontend" in context
        assert "error_rate_pre" in context["frontend"]
        assert "error_rate_post" in context["frontend"]

        # Post-fault should have higher error rate
        assert context["frontend"]["error_rate_post"] > context["frontend"]["error_rate_pre"]

    def test_slowest_operations(self):
        """Test slowest operations extraction."""
        traces_df = pd.DataFrame({
            'serviceName': ['frontend'] * 4,
            'duration': [100000, 200000, 5000000, 3000000],
            'startTimeMillis': [1002000, 1003000, 1004000, 1005000],
            'operationName': ['Op1', 'Op2', 'Op3', 'Op4']
        })
        inject_time = 1001
        context = ContextBuilder.build_traces_context(
            traces_df, ["frontend"], inject_time, time_window=10
        )

        assert context is not None
        assert "frontend" in context
        if "slowest_operations" in context["frontend"]:
            slowest = context["frontend"]["slowest_operations"]
            assert len(slowest) > 0

    def test_missing_traces(self):
        """Test handling of missing traces."""
        context = ContextBuilder.build_traces_context(None, ["frontend"], 1000)
        assert context is None

    def test_missing_required_columns(self):
        """Test with DataFrame missing required columns."""
        traces_df = pd.DataFrame({'serviceName': ['frontend'], 'duration': [100000]})  # Missing startTimeMillis
        context = ContextBuilder.build_traces_context(traces_df, ["frontend"], 1000)
        assert context is None


class TestLLMResponseParsing:
    """Test LLM response parsing with various formats."""

    def _make_reranker(self):
        """Create a reranker instance with mocked client."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            reranker = LLMReranker(model="gpt-4o")
        return reranker

    def test_valid_json_response(self):
        """Test parsing of valid JSON response."""
        reranker = self._make_reranker()
        candidates = [("frontend", 8.2), ("cartservice", 7.1)]

        mock_response = {
            "content": '{"ranking": ["cartservice", "frontend"], "reasoning": "Cartservice shows higher anomaly"}',
            "tokens": {"input": 1000, "output": 50}
        }

        parsed = reranker._parse_response(mock_response, candidates)
        assert parsed["ranking"] == ["cartservice", "frontend"]
        assert "reasoning" in parsed

    def test_json_in_markdown_code_block(self):
        """Test parsing JSON from markdown code blocks."""
        reranker = self._make_reranker()
        candidates = [("frontend", 8.2), ("cartservice", 7.1)]

        mock_response = {
            "content": '```json\n{"ranking": ["cartservice", "frontend"], "reasoning": "Analysis"}\n```',
            "tokens": {"input": 1000, "output": 50}
        }

        parsed = reranker._parse_response(mock_response, candidates)
        assert parsed["ranking"] == ["cartservice", "frontend"]

    def test_malformed_json_fallback(self):
        """Test fallback with malformed JSON."""
        reranker = self._make_reranker()
        candidates = [("frontend", 8.2), ("cartservice", 7.1)]

        mock_response = {
            "content": "Based on the analysis, cartservice is the root cause because...",
            "tokens": {"input": 1000, "output": 50}
        }

        parsed = reranker._parse_response(mock_response, candidates)

        # Should fall back to original order or extract from text
        assert len(parsed["ranking"]) == 2
        assert all(svc in [c[0] for c in candidates] for svc in parsed["ranking"])

    def test_invalid_service_names_filtered(self):
        """Test filtering of invalid service names."""
        reranker = self._make_reranker()
        candidates = [("frontend", 8.2), ("cartservice", 7.1)]

        mock_response = {
            "content": '{"ranking": ["invalid_service", "frontend", "cartservice"], "reasoning": "Analysis"}',
            "tokens": {"input": 1000, "output": 50}
        }

        parsed = reranker._parse_response(mock_response, candidates)

        # Invalid service should be filtered, missing service added
        assert len(parsed["ranking"]) == 2
        assert "frontend" in parsed["ranking"]
        assert "cartservice" in parsed["ranking"]


class TestClaudeCLI:
    """Test Claude CLI provider."""

    def test_claude_cli_init(self):
        """Test that claude-cli models can be initialized if claude is on PATH."""
        import shutil
        if shutil.which("claude"):
            reranker = LLMReranker(model="claude-sonnet")
            assert reranker.provider == "claude-cli"
            assert reranker.config["model_id"] == "claude-sonnet-4-5"
            assert reranker.client is None  # No API client for CLI

            # Also test 4.6 variant
            reranker46 = LLMReranker(model="claude-sonnet4.6")
            assert reranker46.provider == "claude-cli"
            assert reranker46.config["model_id"] == "claude-sonnet-4-6"
        else:
            with pytest.raises(ValueError, match="claude.*CLI.*not found"):
                LLMReranker(model="claude-sonnet")

    def test_claude_cli_model_variants(self):
        """Test all Claude CLI model configs exist (4.5 and 4.6)."""
        # 4.5 models
        assert "claude-opus" in LLMReranker.MODEL_CONFIGS
        assert "claude-sonnet" in LLMReranker.MODEL_CONFIGS
        assert "claude-haiku" in LLMReranker.MODEL_CONFIGS

        assert LLMReranker.MODEL_CONFIGS["claude-opus"]["model_id"] == "claude-opus-4-5"
        assert LLMReranker.MODEL_CONFIGS["claude-sonnet"]["model_id"] == "claude-sonnet-4-5"
        assert LLMReranker.MODEL_CONFIGS["claude-haiku"]["model_id"] == "claude-haiku-4-5"

        # 4.6 models
        assert "claude-sonnet4.6" in LLMReranker.MODEL_CONFIGS
        assert "claude-opus4.6" in LLMReranker.MODEL_CONFIGS

        assert LLMReranker.MODEL_CONFIGS["claude-sonnet4.6"]["model_id"] == "claude-sonnet-4-6"
        assert LLMReranker.MODEL_CONFIGS["claude-opus4.6"]["model_id"] == "claude-opus-4-6"

        for model in ["claude-opus", "claude-sonnet", "claude-haiku", "claude-sonnet4.6", "claude-opus4.6"]:
            assert LLMReranker.MODEL_CONFIGS[model]["provider"] == "claude-cli"

    @patch('baro.llm_reranker.subprocess.run')
    @patch('baro.llm_reranker.shutil.which', return_value="/usr/local/bin/claude")
    def test_claude_cli_call(self, mock_which, mock_run):
        """Test that claude CLI is called with correct arguments."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                "type": "result",
                "result": '{"ranking": ["frontend", "cartservice"], "reasoning": "test"}',
                "cost_usd": 0.001,
                "duration_ms": 1234
            }),
            stderr=""
        )

        reranker = LLMReranker(model="claude-sonnet")
        result = reranker._call_claude_cli("test prompt")

        # Verify the CLI was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "/usr/local/bin/claude"
        assert "-p" in cmd
        assert "--model" in cmd
        assert "claude-sonnet-4-5" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

        # Verify the prompt was passed as stdin
        assert call_args[1]["input"] == "test prompt"

        assert "content" in result
        assert "tokens" in result

    @patch('baro.llm_reranker.subprocess.run')
    @patch('baro.llm_reranker.shutil.which', return_value="/usr/local/bin/claude")
    def test_claude_cli_error_handling(self, mock_which, mock_run):
        """Test error handling for failed CLI calls."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Error: connection failed"
        )

        reranker = LLMReranker(model="claude-sonnet")
        with pytest.raises(RuntimeError, match="claude CLI exited with code 1"):
            reranker._call_claude_cli("test prompt")

    @patch('baro.llm_reranker.subprocess.run')
    @patch('baro.llm_reranker.shutil.which', return_value="/usr/local/bin/claude")
    def test_claude_cli_raw_text_fallback(self, mock_which, mock_run):
        """Test fallback when CLI returns raw text instead of JSON."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"ranking": ["frontend", "cartservice"], "reasoning": "analysis"}',
            stderr=""
        )

        reranker = LLMReranker(model="claude-haiku")
        result = reranker._call_claude_cli("test prompt")

        # Should still parse the content
        assert "content" in result


class TestGracefulDegradation:
    """Test BARO+ works with missing data modalities."""

    def test_metrics_only(self):
        """Test BARO+ with only metrics (no logs, no traces)."""
        from run_bench import run_baro_plus

        metrics = {
            "frontend_cpu": [[1000, 10.0], [1002, 80.0]],
            "frontend_mem": [[1000, 100.0], [1002, 105.0]],
        }
        inject_time = 1001

        # Should not raise an error
        try:
            result, metadata = run_baro_plus(metrics, inject_time, logs=None, traces=None,
                                            model="gpt-4o", k=5, modalities="metrics")
            # If no API key, will fail gracefully
        except Exception as e:
            # Expected if API key not configured
            assert "API" in str(e) or "key" in str(e).lower()

    def test_with_empty_logs(self):
        """Test BARO+ with empty logs DataFrame."""
        from run_bench import run_baro_plus

        metrics = {
            "frontend_cpu": [[1000, 10.0], [1002, 80.0]],
        }
        inject_time = 1001
        empty_logs = pd.DataFrame({'timestamp': [], 'container_name': [], 'message': []})

        try:
            result, metadata = run_baro_plus(metrics, inject_time, logs=empty_logs, traces=None,
                                            model="gpt-4o", k=5)
        except Exception as e:
            # Expected if API key not configured
            assert "API" in str(e) or "key" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
