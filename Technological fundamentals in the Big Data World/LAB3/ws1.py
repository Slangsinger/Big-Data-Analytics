import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from pyjstat import pyjstat
import os


# get the data from eurostat
url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ei_bsco_m?format=JSON&unit=BAL&indic=BS-CSMCI&s_adj=SA&lang=EN"
response = requests.get(url)

data = response.json()

dataset = pyjstat.Dataset.read(json.dumps(data))
df = dataset.write('dataframe')
df = df[['Geopolitical entity (reporting)', 'Time', 'value']]
df['year'] = df['Time'].str[:4].astype(int)


last5 = df[df['year'] >= 2020]


avg_eu = last5[last5['Geopolitical entity (reporting)'] == 'European Union - 27 countries (from 2020)']['value'].mean()
print("Average EU Consumer Confidence (last 5 years):", round(avg_eu, 2))

# --- 2. EU and Spain data for 2024 ---
eu_2024 = df[(df['Geopolitical entity (reporting)'] == 'European Union - 27 countries (from 2020)') & (df['year'] == 2024)]
es_2024 = df[(df['Geopolitical entity (reporting)'] == 'Spain') & (df['year'] == 2024)]

print("\nEU 2024 data:")
print(eu_2024[['Time', 'value']])

print("\nSpain 2024 data:")
print(es_2024[['Time', 'value']])

# --- Plot (last 5 years) ---
eu_last5 = df[(df['Geopolitical entity (reporting)'] == 'European Union - 27 countries (from 2020)') & (df['year'] >= 2020)]
es_last5 = df[(df['Geopolitical entity (reporting)'] == 'Spain') & (df['year'] >= 2020)]

plt.figure(figsize=(10, 6))
plt.plot(eu_last5['Time'], eu_last5['value'], label='EU')
plt.plot(es_last5['Time'], es_last5['value'], label='Spain')
plt.xticks(eu_last5['Time'][::3], rotation=45)  # fewer date labels
plt.title("Consumer Confidence Indicator (Last 5 Years)")
plt.xlabel("Month")
plt.ylabel("Balance (Seasonally Adjusted)")
plt.legend()
plt.grid(True)

# --- Save plot in same folder as script ---
filename = "consumer_confidence_last5years.png"
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, filename)

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"\nPlot saved as: {save_path}")
