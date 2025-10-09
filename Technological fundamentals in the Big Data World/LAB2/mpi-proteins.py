#!/usr/bin/env python3
"""
mpi-proteins.py

Parallel (MPI) solution for Lab2 using mpi4py.

High-level flow
---------------
Rank 0:
  - Loads the CSV (columns: protid, enzyme, hydrofob, sequence)
  - Asks user for the search pattern (UPPERCASE)
  - Splits the data into N chunks (N = number of MPI processes)
  - Broadcasts the pattern to all ranks
  - Scatters the chunks to each rank

All ranks:
  - Receive their chunk (list of rows)
  - Count overlapping occurrences of the pattern for each sequence
  - Keep only rows with matches > 0
  - Send local results back to Rank 0

Rank 0:
  - Gathers partial results from all ranks
  - Picks the winner (max matches; tie-breaker: max hydrofob)
  - (Optional) Plots Top-10 bar chart (single process only)
  - Prints parallel runtime and (optional) speedup if --serial-time given

Run
---
mpiexec -n 4 python mpi-proteins.py
mpiexec -n 8 python mpi-proteins.py --serial-time 1.234

Dependencies
------------
pip install mpi4py pandas matplotlib
(Windows: install MS-MPI or IntelMPI; Linux/macOS: OpenMPI or MPICH)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from mpi4py import MPI

# Plotting only on rank 0
import matplotlib
matplotlib.use("Agg")  # never open GUI; always safe in MPI
import matplotlib.pyplot as plt


# ---------- Shared utilities (same logic as in serial version) ----------

def count_overlapping(haystack: str, needle: str) -> int:
    """
    Count overlapping occurrences of `needle` in `haystack`.

    Example:
      haystack="AAAA", needle="AA"  -> 3
    """
    if not needle:
        return 0
    n, m = len(haystack), len(needle)
    if m > n:
        return 0
    c = 0
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            c += 1
    return c


def pick_winner(rows: pd.DataFrame) -> Tuple[str, int, float]:
    """
    Given DataFrame with columns: protid, matches, hydrofob
    Returns (winner_protid, matches, hydrofob) with
      - max(matches) and tie-breaker by max(hydrofob).
    """
    if rows.empty:
        return ("", 0, 0.0)
    best = rows.sort_values(["matches", "hydrofob"], ascending=[False, False]).iloc[0]
    return (str(best["protid"]), int(best["matches"]), float(best["hydrofob"]))


def plot_top10(df_hits: pd.DataFrame, out_path: Path) -> None:
    """
    Plot Top-10 by matches (tie: hydrofob). Annotate the bar that has the
    maximum hydrofob among those ten. Save figure to `out_path`.
    """
    if df_hits.empty:
        print("[INFO] Nothing to plot (no matches).")
        return

    top = df_hits.sort_values(["matches", "hydrofob"], ascending=[False, False]).head(10)
    idx_max_h = top["hydrofob"].idxmax()
    max_h_row = top.loc[idx_max_h]

    plt.figure(figsize=(12, 6))
    x_labels = top["protid"].astype(str).tolist()
    heights = top["matches"].tolist()
    plt.bar(range(len(top)), heights)

    max_pos = top.index.get_loc(idx_max_h)
    plt.text(
        x=max_pos,
        y=heights[max_pos] + max(1, 0.03 * max(heights)),
        s=f"max hydrofob: {max_h_row['hydrofob']:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.xticks(range(len(top)), x_labels, rotation=45, ha="right")
    plt.xlabel("Protein ID (protid)")
    plt.ylabel("Matches (count of pattern occurrences)")
    plt.title("Top-10 proteins by matches (MPI)")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Chart saved to: {out_path}")


# ---------- MPI helpers ----------

def split_into_chunks(df: pd.DataFrame, n_parts: int) -> List[pd.DataFrame]:
    """
    Split DataFrame into `n_parts` (as even as possible).
    We keep only the columns we need to reduce pickled payload.
    """
    cols = ["protid", "hydrofob", "sequence"]  # enzyme not needed for logic
    df = df[cols].copy()
    sizes = [(len(df) * (i + 1)) // n_parts - (len(df) * i) // n_parts for i in range(n_parts)]
    chunks = []
    start = 0
    for sz in sizes:
        chunks.append(df.iloc[start:start + sz])
        start += sz
    return chunks


def rows_to_python_objects(df: pd.DataFrame) -> List[Tuple[str, float, str]]:
    """
    Convert a small DataFrame chunk to a pure-Python list of tuples to simplify
    MPI scatter/gather (avoid issues with numpy dtypes).
    Each row is (protid, hydrofob, sequence).
    """
    # Ensure proper types
    return list(zip(df["protid"].astype(str),
                    pd.to_numeric(df["hydrofob"], errors="coerce").fillna(0).astype(float),
                    df["sequence"].astype(str)))


def local_count(chunk: List[Tuple[str, float, str]], pattern: str) -> List[Tuple[str, float, int]]:
    """
    Compute matches for a chunk: [(protid, hydrofob, sequence), ...]
    Returns list of rows with positive matches:
      [(protid, hydrofob, matches), ...]
    """
    out = []
    for protid, hydrofob, seq in chunk:
        m = count_overlapping(seq, pattern)
        if m > 0:
            out.append((protid, hydrofob, m))
    return out


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="MPI version of protein pattern search.")
    parser.add_argument("--csv", type=str, default="proteins.csv",
                        help="Path to CSV (default: proteins.csv in current working directory).")
    parser.add_argument("--serial-time", type=float, default=None,
                        help="Optional: serial runtime in seconds, to print speedup.")
    parser.add_argument("--no-plot", action="store_true",
                        help="If set, do not generate the Top-10 chart.")
    args = parser.parse_args()

    base_dir = Path(".").resolve()
    csv_path = base_dir / args.csv
    chart_path = base_dir / "top10_matches_mpi.png"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Rank 0 prepares data and pattern ---
    if rank == 0:
        # Load CSV on rank 0 only (avoid N-fold I/O)
        if not csv_path.exists():
            sys.exit(f"[ERROR] CSV not found: {csv_path}. "
                     f"Generate it first with: python proteins-generator.py 50000")

        df = pd.read_csv(csv_path, dtype={"protid": str})
        # Normalize columns we rely on
        required_cols = {"protid", "enzyme", "hydrofob", "sequence"}
        missing = required_cols - set(df.columns)
        if missing:
            sys.exit(f"[ERROR] CSV missing columns: {missing}. Got: {list(df.columns)}")

        df["sequence"] = df["sequence"].astype(str).str.upper()
        df["enzyme"]   = df["enzyme"].astype(str).str.upper()
        df["hydrofob"] = pd.to_numeric(df["hydrofob"], errors="coerce").fillna(0)

        # Ask for pattern (UPPERCASE)
        pattern = input("Enter pattern to search (will be converted to UPPERCASE): ").strip().upper()
        if not pattern:
            print("[ERROR] Empty pattern is not allowed.", flush=True)
            pattern = ""  # still broadcast something; others will skip work

        # Split into chunks and convert to plain Python objects
        chunks_df = split_into_chunks(df, size)
        payload = [rows_to_python_objects(ch) for ch in chunks_df]
    else:
        pattern = None
        payload = None

    # Broadcast pattern to all ranks
    pattern = comm.bcast(pattern, root=0)

    # Scatter data chunks to all ranks (each rank gets a Python list of tuples)
    my_chunk: List[Tuple[str, float, str]] = comm.scatter(payload, root=0)

    # --- Start timing just before parallel computation ---
    t0 = MPI.Wtime()

    # Compute local results (keep only hits)
    local_hits: List[Tuple[str, float, int]] = []
    if pattern:
        local_hits = local_count(my_chunk, pattern)

    # Gather results from all ranks at root
    gathered: List[List[Tuple[str, float, int]]] = comm.gather(local_hits, root=0)

    elapsed = MPI.Wtime() - t0

    # --- Rank 0 merges and reports ---
    if rank == 0:
        # Flatten lists and convert to DataFrame for convenience
        all_hits = [row for part in gathered for row in part]  # type: ignore
        df_hits = pd.DataFrame(all_hits, columns=["protid", "hydrofob", "matches"])

        print(f"[TIME] MPI search finished in {elapsed:.3f} seconds with {size} ranks.")

        if args.serial_time is not None and args.serial_time > 0:
            speedup = args.serial_time / elapsed
            print(f"[SPEEDUP] {args.serial_time:.3f} / {elapsed:.3f} = {speedup:.2f}x")

        # Winner (if any)
        winner_protid, winner_matches, winner_h = pick_winner(df_hits) if not df_hits.empty else ("", 0, 0.0)
        if winner_matches == 0:
            print("[RESULT] No matches found for the given pattern.")
        else:
            print(f"[RESULT] Winner protid: {winner_protid} "
                  f"(matches={winner_matches}, hydrofob={winner_h:.2f})")

        # Optional plotting
        if not args.no_plot and not df_hits.empty:
            plot_top10(df_hits, chart_path)


if __name__ == "__main__":
    main()
