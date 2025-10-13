import pandas as pd
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    # Read pattern to search for from keyboard and change it to UPPERCASE
    pattern = input("Enter pattern to search: ").strip().upper()

    # start timer
    start_time = time.time()

    # read CSV
    data_path = "/Users/miguelrodriguezlosada/Documents/UC3M/MSc in Big Data Analytics/1st semiquarter/Technological fundamentals in the Big Data World/Labs/Lab2/proteins.csv"

    start_reading_csv_time = time.time()
    df = pd.read_csv(data_path)
    end_reading_csv_time = time.time()

    print(f"CSV reading time: {end_reading_csv_time - start_reading_csv_time:.2f} seconds")

    # search occurrences
    counts = df["sequence"].str.count(pattern)
    has = counts > 0
    res_df = None
    if has.any():
        res_df = df.loc[has, ["protid", "hydrofob"]].copy()
        res_df["occurrences"] = counts[has]

        # --- Find protein with max occurrences (break ties with hydrofob) ---
        max_occ = res_df["occurrences"].max()
        candidates = res_df[res_df["occurrences"] == max_occ]
        best_protein = candidates.loc[candidates["hydrofob"].idxmax()]

        print(f"Protein with max occurrences: ID={best_protein['protid']}, "
              f"Occurrences={best_protein['occurrences']}, "
              f"Hydrofob={best_protein['hydrofob']}")

        # --- Plot top 10 proteins by occurrences ---
        top10 = res_df.nlargest(10, "occurrences")
        colors = ["red" if pid == best_protein["protid"] else "blue" for pid in top10["protid"]]

        plt.bar(top10["protid"].astype(str), top10["occurrences"], color=colors)
        plt.xlabel("Protein ID")
        plt.ylabel("Occurrences")
        plt.title("Top 10 Proteins with Most Pattern Matches")
        plt.show()

    # total execution time
    end_time = time.time()
    print(f"Total Execution time: {end_time - start_time:.2f} seconds")
    print(f"IO time: {end_reading_csv_time - start_reading_csv_time:.2f} seconds")
