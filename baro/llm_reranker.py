"""
LLM Re-ranker for BARO+

This module implements the LLM-based re-ranking on top of BARO's statistical scoring.
Supports multiple LLM providers (OpenAI, Anthropic, Claude CLI) with graceful error handling.
"""

import os
import time
import json
import re
import shutil
import subprocess
from typing import Dict, List, Tuple, Optional
import logging

# LLM API clients (imported lazily to handle missing packages)
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass


logger = logging.getLogger(__name__)


class LLMReranker:
    """
    LLM-based re-ranker for root cause candidates.

    Supports multiple models:
    - OpenAI: gpt-4o, gpt-4o-mini
    - Anthropic API: claude-3-5-sonnet-20241022
    - Claude CLI: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6
    """

    # Model configurations
    MODEL_CONFIGS = {
        "gpt-4o": {
            "provider": "openai",
            "model_id": "gpt-4o-2024-08-06",
            "max_tokens": 1000,
            "temperature": 0.0
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "max_tokens": 1000,
            "temperature": 0.0
        },
        "claude-3-5-sonnet": {
            "provider": "anthropic",
            "model_id": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "temperature": 0.0
        },
        # Claude CLI models (uses `claude` command, no API key needed)
        "claude-opus": {
            "provider": "claude-cli",
            "model_id": "claude-opus-4-5",
            "max_tokens": 1000,
            "temperature": 0.0
        },
        "claude-sonnet": {
            "provider": "claude-cli",
            "model_id": "claude-sonnet-4-5",
            "max_tokens": 1000,
            "temperature": 0.0
        },
        "claude-haiku": {
            "provider": "claude-cli",
            "model_id": "claude-haiku-4-5",
            "max_tokens": 1000,
            "temperature": 0.0
        },
        # Claude 4.6 models
        "claude-sonnet4.6": {
            "provider": "claude-cli",
            "model_id": "claude-sonnet-4-6",
            "max_tokens": 1000,
            "temperature": 0.0
        },
        "claude-opus4.6": {
            "provider": "claude-cli",
            "model_id": "claude-opus-4-6",
            "max_tokens": 1000,
            "temperature": 0.0
        },
    }

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize LLM re-ranker.

        Args:
            model: Model identifier ("gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet")
            api_key: API key (reads from environment if not provided)

        Raises:
            ValueError: If model is not supported or API client not available
        """
        if model not in self.MODEL_CONFIGS:
            raise ValueError(f"Model '{model}' not supported. Available: {list(self.MODEL_CONFIGS.keys())}")

        self.model = model
        self.config = self.MODEL_CONFIGS[model]
        self.provider = self.config["provider"]

        # Initialize API client
        if self.provider == "claude-cli":
            self.claude_path = shutil.which("claude")
            if not self.claude_path:
                raise ValueError("'claude' CLI not found on PATH. Install Claude Code first.")
            self.client = None

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI package not installed. Run: pip install openai")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = openai.OpenAI(api_key=self.api_key)

        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Anthropic package not installed. Run: pip install anthropic")
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def rerank(
        self,
        candidates: List[Tuple[str, float]],
        metrics_context: Dict[str, Dict],
        logs_context: Optional[Dict[str, List[str]]] = None,
        traces_context: Optional[Dict[str, Dict]] = None,
        inject_time: Optional[int] = None
    ) -> Dict:
        """
        Re-rank top-k candidates using LLM with multi-modal context.

        Args:
            candidates: List of (service_name, score) tuples from BARO
            metrics_context: Statistical summary of metrics for each service
            logs_context: Sampled log messages for each service (optional)
            traces_context: Aggregated trace statistics for each service (optional)
            inject_time: Fault injection timestamp (optional, for context)

        Returns:
            {
                "ranking": ["service1", "service2", ...],
                "scores": [(service1, rank_score1), ...],
                "reasoning": "LLM's explanation",
                "model": "gpt-4o",
                "tokens": {"input": 1234, "output": 567},
                "latency_ms": 1234,
                "error": None
            }
        """
        start_time = time.time()

        try:
            # Build prompt from available context
            prompt = self._build_prompt(candidates, metrics_context, logs_context, traces_context, inject_time)

            # Call LLM with retry logic
            llm_response = self._call_llm_with_retry(prompt, max_retries=3)

            # Parse response and extract ranking
            parsed = self._parse_response(llm_response, candidates)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "ranking": parsed["ranking"],
                "scores": [(svc, len(candidates) - i) for i, svc in enumerate(parsed["ranking"])],
                "reasoning": parsed["reasoning"],
                "model": self.model,
                "tokens": parsed.get("tokens", {}),
                "latency_ms": latency_ms,
                "error": None
            }

        except Exception as e:
            # Fallback to original BARO ranking on any error
            logger.error(f"LLM re-ranking failed: {e}. Falling back to original BARO ranking.")
            original_ranking = [svc for svc, _ in candidates]
            return {
                "ranking": original_ranking,
                "scores": candidates,
                "reasoning": f"Error: {str(e)}. Returned original BARO ranking.",
                "model": self.model,
                "tokens": {},
                "latency_ms": int((time.time() - start_time) * 1000),
                "error": str(e)
            }

    def _build_prompt(
        self,
        candidates: List[Tuple[str, float]],
        metrics_ctx: Dict[str, Dict],
        logs_ctx: Optional[Dict[str, List[str]]],
        traces_ctx: Optional[Dict[str, Dict]],
        inject_time: Optional[int]
    ) -> str:
        """
        Construct structured prompt from available data.

        Args:
            candidates: Top-k candidates with BARO scores
            metrics_ctx: Metrics statistical summary
            logs_ctx: Sampled log messages (optional)
            traces_ctx: Trace aggregations (optional)
            inject_time: Fault injection time (optional)

        Returns:
            Formatted prompt string
        """
        # System instructions
        system_prompt = """You are an expert SRE analyzing microservice failures. Your task is to identify the root cause service by analyzing observability data.

Key principles:
1. Look for anomalies that appeared AFTER the fault injection
2. High severity metric changes (>300%) are strong indicators
3. Error messages in logs provide direct evidence
4. Trace latency increases and errors indicate service degradation
5. Consider cascading failures - the root cause may affect downstream services"""

        # Task definition
        task = "Re-rank the following candidate services from most likely to least likely to be the root cause of the failure."

        # Format candidates
        candidates_text = "\n".join([
            f"  {i+1}. {svc} (BARO anomaly score: {score:.2f})"
            for i, (svc, score) in enumerate(candidates)
        ])

        # Format metrics context
        metrics_text = self._format_metrics_context(metrics_ctx)

        # Format logs context (if available)
        logs_text = ""
        if logs_ctx:
            logs_text = self._format_logs_context(logs_ctx)

        # Format traces context (if available)
        traces_text = ""
        if traces_ctx:
            traces_text = self._format_traces_context(traces_ctx)

        # Construct full prompt
        prompt_parts = [
            system_prompt,
            f"\n## Task\n{task}",
            f"\n## BARO Top-{len(candidates)} Candidates\n{candidates_text}",
            f"\n## Metrics Analysis\n{metrics_text}"
        ]

        if logs_text:
            prompt_parts.append(f"\n## Log Messages\n{logs_text}")

        if traces_text:
            prompt_parts.append(f"\n## Distributed Traces\n{traces_text}")

        if inject_time:
            prompt_parts.append(f"\n## Additional Context\nFault injection time: {inject_time}")

        # Output format instructions
        output_format = """
## Output Format
Respond with a JSON object containing:
{
  "ranking": ["service1", "service2", ...],
  "reasoning": "Brief explanation (2-3 sentences) of why this ranking is chosen"
}

Ensure all service names in the ranking match exactly the candidate names provided above."""

        prompt_parts.append(output_format)

        return "\n".join(prompt_parts)

    def _format_metrics_context(self, metrics_ctx: Dict[str, Dict]) -> str:
        """Format metrics context for prompt."""
        lines = []
        for service, metrics in metrics_ctx.items():
            lines.append(f"\n### {service}")
            for metric_type, stats in metrics.items():
                severity = stats.get('anomaly_severity', 'unknown')
                change = stats.get('change_pct', 0)
                pre_mean = stats.get('pre_mean', 0)
                post_mean = stats.get('post_mean', 0)

                lines.append(
                    f"  - {metric_type}: {pre_mean:.2f} → {post_mean:.2f} "
                    f"({change:+.1f}% change, {severity} severity)"
                )

        return "\n".join(lines) if lines else "No metrics data available"

    def _format_logs_context(self, logs_ctx: Dict[str, List[str]]) -> str:
        """Format logs context for prompt."""
        lines = []
        for service, messages in logs_ctx.items():
            lines.append(f"\n### {service}")
            for msg in messages[:10]:  # Limit to 10 messages per service
                lines.append(f"  {msg}")

        return "\n".join(lines) if lines else "No logs data available"

    def _format_traces_context(self, traces_ctx: Dict[str, Dict]) -> str:
        """Format traces context for prompt."""
        lines = []
        for service, stats in traces_ctx.items():
            lines.append(f"\n### {service}")

            # Duration changes
            pre_dur = stats.get('avg_duration_pre', 0)
            post_dur = stats.get('avg_duration_post', 0)
            if pre_dur > 0 and post_dur > 0:
                change_pct = ((post_dur - pre_dur) / pre_dur) * 100
                lines.append(f"  - Latency: {pre_dur:.1f}ms → {post_dur:.1f}ms ({change_pct:+.1f}%)")

            # Error rates
            pre_err = stats.get('error_rate_pre', 0)
            post_err = stats.get('error_rate_post', 0)
            if post_err > 0:
                lines.append(f"  - Error rate: {pre_err:.1%} → {post_err:.1%}")

            # Slowest operations
            slowest = stats.get('slowest_operations', [])
            if slowest:
                ops_text = ", ".join([f"{op}({dur:.1f}ms)" for op, dur in slowest[:3]])
                lines.append(f"  - Slowest operations: {ops_text}")

        return "\n".join(lines) if lines else "No traces data available"

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Dict:
        """
        Call LLM API with exponential backoff retry.

        Args:
            prompt: The prompt to send
            max_retries: Maximum number of retry attempts

        Returns:
            API response dict

        Raises:
            Exception: If all retries fail
        """
        retry_delays = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s

        for attempt in range(max_retries):
            try:
                if self.provider == "claude-cli":
                    return self._call_claude_cli(prompt)

                elif self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.config["model_id"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config["temperature"],
                        max_tokens=self.config["max_tokens"],
                        timeout=30.0
                    )

                    return {
                        "content": response.choices[0].message.content,
                        "tokens": {
                            "input": response.usage.prompt_tokens,
                            "output": response.usage.completion_tokens
                        }
                    }

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.config["model_id"],
                        max_tokens=self.config["max_tokens"],
                        temperature=self.config["temperature"],
                        messages=[{"role": "user", "content": prompt}],
                        timeout=30.0
                    )

                    return {
                        "content": response.content[0].text,
                        "tokens": {
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens
                        }
                    }

            except Exception as e:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delays[attempt])
                else:
                    raise

    def _call_claude_cli(self, prompt: str) -> Dict:
        """
        Call Claude via the `claude` CLI tool (pipe mode).

        Uses: claude -p --model <model_id> --output-format json

        Args:
            prompt: The prompt to send

        Returns:
            Dict with "content" and "tokens" keys

        Raises:
            RuntimeError: If the CLI call fails
        """
        cmd = [
            self.claude_path,
            "-p",
            "--model", self.config["model_id"],
            "--output-format", "json",
            "--max-turns", "1",
        ]

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited with code {result.returncode}: {result.stderr[:500]}"
            )

        # Parse JSON output from claude CLI
        try:
            cli_output = json.loads(result.stdout)
        except json.JSONDecodeError:
            # If not valid JSON, treat raw stdout as the response text
            return {
                "content": result.stdout.strip(),
                "tokens": {}
            }

        # Extract content from claude CLI JSON output
        # claude --output-format json returns: {"type":"result","result":"...","cost_usd":...,"duration_ms":...,"...}
        content = cli_output.get("result", result.stdout.strip())
        tokens = {}

        # Extract token usage if available
        if "usage" in cli_output:
            tokens = {
                "input": cli_output["usage"].get("input_tokens", 0),
                "output": cli_output["usage"].get("output_tokens", 0),
            }

        # Extract cost info if available
        cost_usd = cli_output.get("cost_usd")
        if cost_usd is not None:
            tokens["cost_usd"] = cost_usd

        return {
            "content": content,
            "tokens": tokens
        }

    def _parse_response(self, response: Dict, candidates: List[Tuple[str, float]]) -> Dict:
        """
        Extract ranking and reasoning from LLM response.

        Expects JSON format:
        {"ranking": ["service1", "service2", ...], "reasoning": "..."}

        Falls back to regex extraction if JSON parsing fails.

        Args:
            response: LLM API response
            candidates: Original candidates (for fallback)

        Returns:
            {"ranking": [...], "reasoning": "...", "tokens": {...}}
        """
        content = response.get("content", "")

        # Try JSON parsing first
        try:
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content

            parsed = json.loads(json_str)

            ranking = parsed.get("ranking", [])
            reasoning = parsed.get("reasoning", "No reasoning provided")

            # Validate ranking contains valid service names
            valid_services = {svc for svc, _ in candidates}
            validated_ranking = [svc for svc in ranking if svc in valid_services]

            # Fill missing services from original order
            for svc, _ in candidates:
                if svc not in validated_ranking:
                    validated_ranking.append(svc)

            return {
                "ranking": validated_ranking[:len(candidates)],
                "reasoning": reasoning,
                "tokens": response.get("tokens", {})
            }

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"JSON parsing failed: {e}. Attempting regex fallback.")

            # Fallback: try to extract service names with regex
            service_names = [svc for svc, _ in candidates]
            found_services = []

            for service in service_names:
                if service in content:
                    found_services.append(service)

            # If we found some services in the response, use that order
            if found_services:
                # Fill in missing services
                for service in service_names:
                    if service not in found_services:
                        found_services.append(service)

                return {
                    "ranking": found_services,
                    "reasoning": "Extracted from unstructured response",
                    "tokens": response.get("tokens", {})
                }

            # Last resort: return original BARO order
            logger.warning("Could not extract ranking from response. Using original BARO order.")
            return {
                "ranking": service_names,
                "reasoning": "Failed to parse LLM response. Using original BARO ranking.",
                "tokens": response.get("tokens", {})
            }
