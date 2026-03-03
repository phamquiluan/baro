#!/usr/bin/env python3
"""Count tokens per RE3 failure case to show raw observability data exceeds LLM context windows.

Reads metrics.json, logs.csv, traces.csv for each RE3 case, counts tokens using
tiktoken (cl100k_base / GPT-4 tokenizer), and produces:
  - output/re3_token_counts.csv  — per-case token counts
  - output/re3_token_counts.pdf  — stacked bar plot with context window reference lines
"""

import csv
import json
import os
from pathlib import Path

import tiktoken
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "rcaeval-data"
OUTPUT_DIR = Path(__file__).parent / "output"
RE3_PREFIX = "re3"

SYSTEM_LABELS = {"ob": "Online Boutique", "ss": "Sock Shop", "tt": "Train Ticket"}
SYSTEM_ORDER = ["ob", "ss", "tt"]

# Context window reference lines (tokens)
CONTEXT_LINES = {
    "128K": 128_000,
    "1M": 1_000_000,
}

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (GPT-4 tokenizer)."""
    return len(enc.encode(text))


def count_file_tokens(path: Path) -> int:
    """Read a file and return its token count. Returns 0 if file doesn't exist."""
    if not path.exists():
        return 0
    text = path.read_text(errors="replace")
    return count_tokens(text)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def parse_dataset_id(dataset_id: str):
    """Parse re3{system}_{service}_{fault}_{instance} into components."""
    # e.g. re3ob_adservice_f3_1
    rest = dataset_id[len("re3"):]          # ob_adservice_f3_1
    system_code = rest[:2]                   # ob
    remainder = rest[3:]                     # adservice_f3_1
    # Split from the right: last part is instance, second-to-last is fault
    parts = remainder.rsplit("_", 2)
    if len(parts) == 3:
        service, fault, instance = parts
    else:
        service = remainder
        fault = "?"
        instance = "?"
    return system_code, service, fault, instance


def discover_re3_cases() -> list[dict]:
    """Find all RE3 case directories and return sorted metadata."""
    cases = []
    for item in sorted(DATA_DIR.iterdir()):
        if not item.is_dir() or not item.name.startswith(RE3_PREFIX):
            continue
        dataset_id = item.name
        system_code, service, fault, instance = parse_dataset_id(dataset_id)
        cases.append({
            "dataset_id": dataset_id,
            "system": system_code,
            "service": service,
            "fault": fault,
            "instance": instance,
            "path": item,
        })
    # Sort by system order, then dataset_id
    order_map = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    cases.sort(key=lambda c: (order_map.get(c["system"], 99), c["dataset_id"]))
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_existing_csv() -> list[dict] | None:
    """Load token counts from existing CSV if available."""
    csv_path = OUTPUT_DIR / "re3_token_counts.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ("metrics_tokens", "logs_tokens", "traces_tokens", "total_tokens"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def main():
    import sys
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # If --plot-only, just regenerate the plot from existing CSV
    if "--plot-only" in sys.argv:
        rows = load_existing_csv()
        if rows is None:
            print("No existing CSV found. Run without --plot-only first.")
            sys.exit(1)
        print(f"Loaded {len(rows)} rows from existing CSV")
        make_plot(rows)
        return

    cases = discover_re3_cases()
    print(f"Found {len(cases)} RE3 cases")

    # Count tokens
    rows = []
    for i, case in enumerate(cases):
        p = case["path"]
        m_tok = count_file_tokens(p / "metrics.json")
        l_tok = count_file_tokens(p / "logs.csv")
        t_tok = count_file_tokens(p / "traces.csv")
        total = m_tok + l_tok + t_tok
        row = {
            "dataset_id": case["dataset_id"],
            "system": SYSTEM_LABELS[case["system"]],
            "fault": case["fault"],
            "metrics_tokens": m_tok,
            "logs_tokens": l_tok,
            "traces_tokens": t_tok,
            "total_tokens": total,
        }
        rows.append(row)
        print(f"  [{i+1}/{len(cases)}] {case['dataset_id']}: "
              f"metrics={m_tok:,}  logs={l_tok:,}  traces={t_tok:,}  total={total:,}")

    # Write CSV
    csv_path = OUTPUT_DIR / "re3_token_counts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset_id", "system", "fault",
            "metrics_tokens", "logs_tokens", "traces_tokens", "total_tokens",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to {csv_path}")

    # --- Plot ---
    make_plot(rows)


def make_plot(rows: list[dict]):
    """Create stacked bar chart of token counts per case."""
    pdf_path = OUTPUT_DIR / "re3_token_counts.pdf"

    n = len(rows)
    indices = np.arange(n)

    metrics = np.array([r["metrics_tokens"] for r in rows])
    logs = np.array([r["logs_tokens"] for r in rows])
    traces = np.array([r["traces_tokens"] for r in rows])
    # Remap short labels from old CSV to full names
    _remap = {"OB": "Online Boutique", "SS": "Sock Shop", "TT": "Train Ticket"}
    systems = [_remap.get(r["system"], r["system"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.subplots_adjust(bottom=0.15)

    bar_w = 0.8
    ax.bar(indices, metrics, bar_w, label="Metrics", color="#7EB0E0")
    ax.bar(indices, logs, bar_w, bottom=metrics, label="Logs", color="#EE854A")
    ax.bar(indices, traces, bar_w, bottom=metrics + logs, label="Traces", color="#6ACC64")

    # Context window reference lines
    line_styles = {"128K": ":", "1M": "--"}
    for label, val in CONTEXT_LINES.items():
        ax.axhline(y=val, color="red", linestyle=line_styles.get(label, "--"),
                   linewidth=1.2, alpha=0.8)
        ax.text(n + 0.5, val, f"{label} context", va="center", fontsize=13, color="red")

    # Log scale
    ax.set_yscale("log")
    ax.set_ylabel("Token count (log scale)", fontsize=14)

    # System group labels on x-axis
    prev_sys = None
    group_starts = []
    for i, s in enumerate(systems):
        if s != prev_sys:
            group_starts.append((i, s))
            prev_sys = s

    # Place system labels below bars using figure-level annotation
    ax.set_xticks([])
    ax.set_xlabel("")
    for g_idx, (start, sys_label) in enumerate(group_starts):
        end = group_starts[g_idx + 1][0] if g_idx + 1 < len(group_starts) else n
        mid = (start + end - 1) / 2
        if start > 0:
            ax.axvline(x=start - 0.5, color="black", linewidth=1, linestyle="-")
        ax.annotate(sys_label, xy=(mid, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=13, fontweight="bold")
        if sys_label == "Sock Shop":
            ss_totals = metrics[start:end] + logs[start:end] + traces[start:end]
            ax.text(mid, ss_totals.max() * 1.3,
                    "(*) The Sock Shop system\ndoes not have trace data.",
                    ha="center", va="bottom", fontsize=11, fontstyle="italic",
                    color="black")

    ax.legend(loc="upper left", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlim(-0.5, n - 0.5)

    # Y-axis formatting
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"
    ))

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Plot written to {pdf_path}")

    # Summary stats
    totals = [r["total_tokens"] for r in rows]
    print(f"\nSummary:")
    print(f"  Min total tokens:    {min(totals):>12,}")
    print(f"  Max total tokens:    {max(totals):>12,}")
    print(f"  Mean total tokens:   {int(np.mean(totals)):>12,}")
    print(f"  Median total tokens: {int(np.median(totals)):>12,}")
    cases_over_128k = sum(1 for t in totals if t > 128_000)
    cases_over_1m = sum(1 for t in totals if t > 1_000_000)
    print(f"  Cases > 128K tokens: {cases_over_128k}/{len(totals)}")
    print(f"  Cases > 1M tokens:   {cases_over_1m}/{len(totals)}")


if __name__ == "__main__":
    main()
