#!/usr/bin/env python3
"""
Analyze `tracing-subscriber` JSON output from `chess-cli` and produce
an aggregate performance summary plus a simple plot.

Usage:
    python tools/analyze_trace.py trace.json

The script expects one JSON object per line (as produced by
`--json-trace` in the CLI) and looks for span "close" events with
`time.busy`/`time.idle` fields.

Console output:
    - per-span totals: count, total busy time, mean busy time
    - grouped by `span.name` (e.g. chess_response_u8, coarse, refine)

Plot:
    - bar chart of total busy milliseconds per span name, saved next
      to the input file as `<name>.perf.png`.
"""

from dataclasses import dataclass
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from trace.traceplot import plot_totals

def parse_time(s: str) -> float:
    """
    Parse a `time.busy` / `time.idle` string emitted by tracing-subscriber
    into seconds (float).

    Examples: "156µs", "5.88ms", "416ns", "1.23s"
    """
    s = s.strip()
    if not s:
        return 0.0

    units = [
        ("ns", 1e-9),
        ("µs", 1e-6),
        ("us", 1e-6),
        ("ms", 1e-3),
        ("s", 1.0),
    ]
    for suffix, scale in units:
        if s.endswith(suffix):
            val = float(s[: -len(suffix)])
            return val * scale
    # Fallback: assume seconds if no suffix
    try:
        return float(s)
    except ValueError:
        return 0.0

@dataclass
class TraceRecord:
    busy: float
    idle: float
    target: str
    name: str
    parent: str

    @staticmethod
    def from_json(data:dict):
        pass


def load_spans(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Load span close events from a tracing JSON log and aggregate
    per (target, span_name).
    """
    stats: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "busy_s": 0.0, "idle_s": 0.0}
    )

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue

            # We configured FmtSpan::CLOSE, so spans emit "close" events.
            if evt.get("message") != "close":
                continue

            target = evt.get("target", "")
            span = evt.get("span") or {}
            name = span.get("name")
            if not name:
                continue

            # Build a simple path from ancestor spans + current span
            stack = [s.get("name") for s in evt.get("spans", []) if s.get("name")]
            path = " > ".join(stack + [name])

            busy = parse_time(evt.get("time.busy", "0"))
            idle = parse_time(evt.get("time.idle", "0"))

            key = (target, path)
            stats[key]["count"] += 1
            stats[key]["busy_s"] += busy
            stats[key]["idle_s"] += idle

    return stats


def print_summary(stats: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    print("=== Tracing summary (per span) ===")
    rows = []
    for (target, name), s in stats.items():
        count = s["count"]
        busy_ms = s["busy_s"] * 1e3
        idle_ms = s["idle_s"] * 1e3
        mean_ms = busy_ms / count if count else 0.0
        rows.append((busy_ms, target, name, count, mean_ms, idle_ms))

    # Sort by total busy time descending
    rows.sort(reverse=True)

    # Highlight coarse-to-fine total vs stages if present.
    total_key = None
    for (target, name), _ in stats.items():
        if name.endswith("find_corners_coarse_to_fine_image_trace"):
            total_key = (target, name)
            break

    if total_key is not None:
        s = stats[total_key]
        total_busy_ms = s["busy_s"] * 1e3
        print("Coarse-to-fine total (find_corners_coarse_to_fine_image_trace):")
        print(f"  target:   {total_key[0]}")
        print(f"  total_ms: {total_busy_ms:.3f}")
        print()
        print("Stages inside coarse-to-fine (e.g., coarse / refine / merge):")

    header = f"{'target':30} {'span':35} {'count':>8} {'total_ms':>10} {'mean_ms':>9} {'idle_ms':>10}"
    print(header)
    print("-" * len(header))
    for busy_ms, target, name, count, mean_ms, idle_ms in rows:
        print(
            f"{target[:30]:30} {name[:35]:35} {count:8d} {busy_ms:10.3f} {mean_ms:9.3f} {idle_ms:10.3f}"
        )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze tracing JSON output from chess-cli."
    )
    parser.add_argument(
        "trace",
        type=Path,
        help="Path to JSON log produced by `chess-cli --json-trace`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output PNG for the performance plot (defaults to <trace>.perf.png).",
    )
    args = parser.parse_args()

    stats = load_spans(args.trace)
    if not stats:
        print("No span close events found in trace file.")
        return

    print_summary(stats)

    out = args.out or args.trace.with_suffix(".perf.png")
    plot_totals(stats, out)
    plt.show()


if __name__ == "__main__":
    main()
