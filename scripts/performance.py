#!/usr/bin/env python3
"""
Plot render time (nanoseconds) vs number of threads from a TSV and save as PNG.
"""

from pathlib import Path
from typing import Optional
import argparse
import csv
import matplotlib.pyplot as plt


def plot_performance(tsv_path: Path, out_path: Optional[Path] = None) -> Path:
    """
    Plot time vs threads from a TSV and save as PNG.

    Parameters
    ----------
    tsv_path : Path
        Path to the input TSV with two columns:
        - 'num_threads' (int): number of threads
        - 'duration_ns' (int): time taken in nanoseconds
    out_path : Optional[Path]
        Where to save the PNG. If None, saves next to the TSV with the same
        basename and '.png' extension.

    Returns
    -------
    Path
        The path to the saved PNG.
    """
    tsv_path = tsv_path.resolve()
    if out_path is None:
        script_dir = Path(__file__).resolve().parent
        out_path = script_dir / (tsv_path.stem + ".png") if out_path is None else Path(out_path).resolve()
    else:
        out_path = Path(out_path).resolve()

    threads = []
    duration_ns = []
    with tsv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("num_threads") or not row.get("duration_ns"):
                continue
            threads.append(int(row["num_threads"]))
            duration_ns.append(int(row["duration_ns"]))

    if not threads:
        raise ValueError("No rows read from TSV. Check delimiter and header names.")

    # Sort by thread count
    pairs = sorted(zip(threads, duration_ns), key=lambda x: x[0])
    threads, duration_ns = map(list, zip(*pairs))

    # Plot
    plt.figure(figsize=(7, 4.2))
    plt.scatter(threads, duration_ns)
    plt.xlabel("Number of threads")
    plt.ylabel("Time (ns)")
    plt.title("Render time vs threads")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot -> {out_path}")
    return out_path

def main():
    default_tsv = (Path(__file__).resolve().parent / ".." / "timings_threads.tsv").resolve()

    ap = argparse.ArgumentParser(description="Plot time (ns) vs threads from a TSV and save PNG.")
    ap.add_argument("--tsv", type=Path, default=default_tsv,
                    help=f"Path to TSV (default: {default_tsv})")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (default: input .tsv replaced with .png)")
    args = ap.parse_args()

    plot_performance(args.tsv, args.out)


if __name__ == "__main__":
    main()
