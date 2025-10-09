import pandas as pd
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    # read pattern to search for from keyboard
    pattern = input("Enter pattern to search: ").strip().upper()

    # start timer
    start_time = time.time()

    # read CSV
    data_path="E:\\UC3M\\Technological_fundamentals_in_the_Big_Data_World\\proteins.csv"

    start_reading_csv_time = time.time()
    df = pd.read_csv(data_path)
    end_reading_csv_time = time.time()

    # search occurrences
    results = []
    counts = df["sequence"].str.count(pattern)
    has = counts > 0
    res_df = None
    if has.any():
        res_df = df.loc[has, ["protid", "hydrofob"]].copy()
        res_df["occurrences"] = counts[has].to_numpy()

    # calculate execution time
    exec_time = time.time() - start_time
    io_time = end_reading_csv_time - start_reading_csv_time
    print(f"Total Execution time: {exec_time:.2f} seconds")
    print(f"IO time: {io_time:.2f} seconds")

    if res_df is not None:

        # get top 10 proteins with more matches
        top10 = res_df.sort_values(by=["occurrences", "hydrofob"], ascending=[False, False]).head(10)

        # plot bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(top10["protid"].astype(str), top10["occurrences"])
        plt.xlabel("Protein ID")
        plt.ylabel("Occurrences")
        plt.title("Top 10 Proteins with Most Pattern Matches")
        plt.show()

        # find protein with max occurrences (and max hydrofob in tie)
        best = top10.iloc[0]
        print(
            f"Protein with max occurrences: ID={best['protid']}, Occurrences={best['occurrences']}, Hydrofob={best['hydrofob']}")
    else:
        print("No matches found.")

