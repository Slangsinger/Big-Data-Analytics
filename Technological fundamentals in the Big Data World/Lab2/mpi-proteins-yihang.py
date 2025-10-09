from mpi4py import MPI
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # ---------------- MPI init ----------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---------------- Pattern input (rank 0) ----------------
    if rank == 0:
        pattern = input("Enter pattern to search: ").strip().upper()
    else:
        pattern = None
    pattern = comm.bcast(pattern, root=0)

    # ---------------- Timer start ----------------
    t0 = MPI.Wtime()

    # Each rank accumulates local results
    local_records = pd.DataFrame()

    # ---------------- Stream CSV and scatter work ----------------
    data_path="E:\\UC3M\\Technological_fundamentals_in_the_Big_Data_World\\proteins.csv"
    work=None
    if rank == 0:
        cols_used = ["protid", "hydrofob", "sequence"]
        data= pd.read_csv(data_path, usecols=cols_used)
        parts = np.array_split(data, size)

        # Scatter to all ranks
        work = comm.scatter(parts, root=0)

    else:
        work = comm.scatter(None, root=0)

    # First work batch is in `work` now for every rank that reached here.
    # We’ll loop until rank 0 sends an empty list to indicate end.

    if work is not None:

        # Build a small DataFrame for vectorized counting (fast)
        dfw = work
        counts = dfw["sequence"].str.count(pattern)
        has = counts > 0
        if has.any():
            local_records = dfw.loc[has, ["protid", "hydrofob"]].copy()
            local_records["occurrences"] = counts[has].to_numpy()

    # Ask rank 0 for the next batch
    comm.barrier()


    # ---------------- Gather & reduce ----------------
    all_records = comm.gather(local_records, root=0)

    if rank == 0:
        # print(f"The length of all_records is {len(all_records)}")
        res = pd.concat(all_records, ignore_index=True)
        if not res.empty:
            # Top-10 by occurrences, break ties by hydrofob
            top10 = res.sort_values(by=["occurrences", "hydrofob"], ascending=[False, False]).head(10)

            # Best protein (max occurrences, tie -> max hydrofob)
            best = top10.iloc[0]

            # Timing
            t_par = MPI.Wtime() - t0
            print(f"Parallel elapsed time: {t_par:.3f} s")


            # Plot (only rank 0 draws)
            plt.figure(figsize=(10, 6))
            plt.bar(top10["protid"].astype(str), top10["occurrences"].astype(int))
            plt.xlabel("Protein ID")
            plt.ylabel("Occurrences")
            plt.title("Top 10 proteins with most pattern matches")
            plt.tight_layout()
            plt.show()

            print(f"Protein with max occurrences: ID={best['protid']}, "
                  f"Occurrences={int(best['occurrences'])}, "
                  f"Hydrofob={best['hydrofob']}")
        else:
            print("No matches found.")
            print(f"Parallel elapsed time: {MPI.Wtime() - t0:.3f} s")
    else:

        # Non-root ranks just exit
        pass