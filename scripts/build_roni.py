from pathlib import Path
import pandas as pd

# Output file
ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "raw_data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "RONI.csv"

# RONI website
url = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/"

# Read all HTML tables on the page
tables = pd.read_html(url)

# NOAA RONI table is table 8 on the page
roni = tables[8].copy()

# Copy the year and month timeframe
roni = roni[["Year The year", "DJF Dec Jan Feb"]].copy()
roni = roni.rename(columns={
    "Year The year": "Year",
    "DJF Dec Jan Feb": "RONI"
})

# Convert to numeric
roni["Year"] = pd.to_numeric(roni["Year"], errors="coerce")
roni["DJF"] = pd.to_numeric(roni["RONI"], errors="coerce")

# Remove repeated header rows/invalid rows
roni = roni.dropna(subset=["Year", "RONI"])

roni["Year"] = roni["Year"].astype(int)

# NOAA year 2016 DJF = Dec 2015 + Jan 2016 + Feb 2016
roni["winter"] = ((roni["Year"] - 1).astype(str) + "-" + roni["Year"].astype(str))

# Restrict to observation period of CoastSat data
roni = roni[(roni["Year"] >= 2000) & (roni["Year"] <= 2022)].copy()
roni = roni[["winter", "RONI"]]

print(roni)
roni.to_csv(OUTPUT_FILE, index=False)
print("\nSaved RONI.csv")