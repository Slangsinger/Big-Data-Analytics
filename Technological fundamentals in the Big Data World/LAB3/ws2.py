import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Request data from Idescat API
url = "https://api.idescat.cat/onomastica/v1/nadons/dades.json"
params = {
    "id": "40683",   # Maria
    "lang": "en",
    "class": "t"     # female
}

response = requests.get(url, params=params)
data = response.json()

# Convert the JSON into a dataframe
records = []
for entry in data['onomastica_nadons']['ff']['f']:
    records.append({
        'year': int(entry['c']),
        'rank_total': int(entry['rank']['total']),
        'rank_female': int(entry['rank']['sex']),
        'count': int(entry['pos1']['v']),
        'per_thousand_total': float(entry['pos1']['w']['total']),
        'per_thousand_female': float(entry['pos1']['w']['sex'])
    })

df = pd.DataFrame(records)

# Select last 5 years
df_last5 = df[df['year'] >= df['year'].max() - 4]

# Compute averages
avg_rank_total = df_last5['rank_total'].mean()
avg_rank_female = df_last5['rank_female'].mean()
avg_per_thousand_total = df_last5['per_thousand_total'].mean()
avg_per_thousand_female = df_last5['per_thousand_female'].mean()

# Print results
print("Average overall rank (last 5 years):", round(avg_rank_total, 1))
print("Average female rank (last 5 years):", round(avg_rank_female, 1))
print("\nFrequency of the name 'Maria' per year:")
print(df_last5[['year', 'count']])
print("\nNewborns named Maria per thousand (total):", round(avg_per_thousand_total, 2))
print("Newborns named Maria per thousand (female):", round(avg_per_thousand_female, 2))

# Plot evolution of the name over time
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['count'], marker='o')
plt.title("Frequency of the Name 'Maria' in Catalonia (Newborn Girls)")
plt.xlabel("Year")
plt.ylabel("Number of Girls Named Maria")
plt.grid(True)
plt.tight_layout()
plt.savefig("maria_evolution.png", dpi=300)
plt.show()
