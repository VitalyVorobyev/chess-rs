import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

def plot_totals(stats: Dict[Tuple[str, str], Dict[str, float]], out_path: Path) -> None:
    # Aggregate by span path (across targets)
    by_span: Dict[str, float] = defaultdict(float)
    for (_target, path), s in stats.items():
        by_span[path] += s["busy_s"] * 1e3  # ms

    if not by_span:
        print("No span data to plot.")
        return

    names = list(by_span.keys())
    totals = [by_span[n] for n in names]

    # Sort by total descending
    names, totals = zip(*sorted(zip(names, totals), key=lambda p: p[1], reverse=True))

    plt.figure(figsize=(10, 4 + 0.25 * len(names)))
    y_pos = range(len(names))
    plt.barh(y_pos, totals)
    plt.yticks(y_pos, names)
    plt.xlabel("Total busy time (ms)")
    plt.title("Total span time by name")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    print(f"Saved performance plot to {out_path}")
