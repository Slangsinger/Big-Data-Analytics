import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the dataset
proteins = pd.read_csv("proteins.csv")

# Ask the user for the pattern to search
pattern = input("Enter the amino acid sequence to search: ").upper()

t0 = time.time()
# Initialize results dictionary
results = {"protein_id": [], "occurrences": []}

# Iterate through the dataset
for _, row in proteins.iterrows():
    count = str(row['sequence']).upper().count(pattern)
    if count > 0:
        results["protein_id"].append(row['protid'])
        results["occurrences"].append(count)
tf = time.time()

# Display the result
print(f"\nPattern to look for: {pattern}")

if len(results['occurrences']) > 0:
    print(pd.DataFrame(results))
else:
    print('There are no occurences of that sequence in the database')

print(f'Exectuting time = {tf - t0}')

