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

# Copy the year and seasonal RONI timeframes
roni = roni[
    [
        "Year The year",
        "OND Oct Nov Dec",
        "NDJ Nov Dec Jan",
        "DJF Dec Jan Feb",
        "JFM Jan Feb Mar",
        "FMA Feb Mar Apr"
    ]
].copy()

roni = roni.rename(columns={
    "Year The year": "Year",
    "OND Oct Nov Dec": "OND",
    "NDJ Nov Dec Jan": "NDJ",
    "DJF Dec Jan Feb": "DJF",
    "JFM Jan Feb Mar": "JFM",
    "FMA Feb Mar Apr": "FMA"
})

# Convert to numeric
roni["Year"] = pd.to_numeric(roni["Year"], errors="coerce")

for col in ["OND", "NDJ", "DJF", "JFM", "FMA"]:
    roni[col] = pd.to_numeric(roni[col], errors="coerce")

# Remove repeated header rows/invalid rows
roni = roni.dropna(subset=["Year"])
roni["Year"] = roni["Year"].astype(int)

# Sort by year so the seasonal alignment is correct
roni = roni.sort_values("Year").reset_index(drop=True)

# For winter 2015-2016: OND and NDJ come from 2015, while DJF, JFM, and FMA come from 2016.
# Shift OND and NDJ down one row so they line up with the ending year of the winter season.
roni["OND_winter"] = roni["OND"].shift(1)
roni["NDJ_winter"] = roni["NDJ"].shift(1)

# Calculate one average RONI value for the broader winter period
roni["RONI"] = roni[
    ["OND_winter", "NDJ_winter", "DJF", "JFM", "FMA"]
].mean(axis=1)

# Remove rows that do not have a complete winter average
roni = roni.dropna(
    subset=["OND_winter", "NDJ_winter", "DJF", "JFM", "FMA", "RONI"]
)

# NOAA year 2016 = winter 2015-2016
roni["winter"] = (
    (roni["Year"] - 1).astype(str)
    + "-"
    + roni["Year"].astype(str)
)

# Restrict to observation period of CoastSat data
roni = roni[
    (roni["Year"] >= 2000)
    & (roni["Year"] <= 2022)
].copy()

roni = roni[["winter", "RONI"]]

print(roni)
roni.to_csv(OUTPUT_FILE, index=False)
print("\nSaved RONI.csv")
