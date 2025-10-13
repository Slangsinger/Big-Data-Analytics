import pandas as pd
import matplotlib.pyplot as plt
import time

if __name__ == "__main__": # code is executed only when the script is run directly, not when it’s imported.

    # reading pattern to search for from keyboard and change it to UPPERCASE
    pattern = input("Enter pattern to search: ").strip().upper() # strip() removes any leading and trailing spaces from the string.

    if not pattern:
        print("No pattern entered. Exiting.")
        exit()

    # starting timer
    start_time = time.time()

    # reading CSV
    data_path = "/Users/miguelrodriguezlosada/Documents/UC3M/MSc in Big Data Analytics/1st semiquarter/Technological fundamentals in the Big Data World/Labs/Lab2/proteins.csv"

    start_reading_csv_time = time.time()
    df = pd.read_csv(data_path)
    end_reading_csv_time = time.time()

    # search occurrences
    counts = df["sequence"].str.count(pattern)
    has = counts > 0
    res_df = None # initializing

    if has.any(): # checking if at least one protein contains the pattern
        res_df = df.loc[has, ["protid", "hydrofob"]].copy() # Filters the DataFrame df to only proteins that matched (has)
        res_df["occurrences"] = counts[has] # Adds a new column "occurrences" with the number of matches for each
        
        # total execution time
        end_time = time.time() # ending timer

        print(f"Total Execution time: {end_time - start_time:.2f} seconds")
        print(f"IO time: {end_reading_csv_time - start_reading_csv_time:.2f} seconds")

        # finding protein with max occurrences (break ties with hydrofob)
        max_occ = res_df["occurrences"].max()
        candidates = res_df[res_df["occurrences"] == max_occ]
        best_protein = candidates.loc[candidates["hydrofob"].idxmax()]

        # plotting top 10 proteins by occurrences
        top10 = res_df.nlargest(10, "occurrences")
        colors = ["red" if pid == best_protein["protid"] else "blue" for pid in top10["protid"]]

        plt.bar(top10["protid"].astype(str), top10["occurrences"], color=colors)
        plt.xlabel("Protein ID")
        plt.ylabel("Occurrences")
        plt.title("Top 10 Proteins with Most Pattern Matches")
        plt.show()

        print(f"Protein with max occurrences: ID={best_protein['protid']}, "
              f"Occurrences={best_protein['occurrences']}, "
              f"Hydrofob={best_protein['hydrofob']}")            
    
    else:
        print("No matches found for the given pattern.")