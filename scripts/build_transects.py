from pathlib import Path
import pandas as pd

# Set paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (ROOT_DIR / "raw_data" / "shoreline_data" / "shoreline_data_run6" / "USA_CA")

OUTPUT_DIR = ROOT_DIR / "raw_data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "california_transect_lookup.csv"

# Get all beach segment folders with transects
site_folders = []

for folder in DATA_DIR.iterdir():
    if folder.is_dir():
        site_folders.append(folder)

print("Found", len(site_folders), "site folders.")

# Get transects from each file
all_transects = []
missing_files = []

for site_folder in site_folders:

    site_id = site_folder.name

    transect_file = site_folder / "Transect_coordinates_and_beach_slopes.csv"
    if not transect_file.exists():
        transect_file = site_folder / "transect_coordinates_and_beach_slopes.csv"
    if not transect_file.exists():
        missing_files.append(site_id)
        continue

    try:
        df = pd.read_csv(transect_file)

        df["site_id"] = site_id

        all_transects.append(df)

    except Exception as e:
        print(f"Could not read {site_id}: {e}")

# Combine all transects into one table
if all_transects:

    transects = pd.concat(all_transects, ignore_index=True)

    transects = transects.rename(columns={
        "Transect id": "transect_id",
        "Longitude - Origin": "origin_lat",
        "Latitude - Origin": "origin_lon",
        "Longitude - Seawards point": "seaward_lat",
        "Latitude - Seawards point": "seaward_lon",
        "Beach face slope": "beach_slope"
    })
    
    transects = transects.drop(columns=["Unnamed: 0"], errors="ignore")

    print("\nCombined dataset created successfully.")
    print(f"Total transects: {len(transects):,}")

else:
    raise ValueError("No transect files were successfully loaded.")

# Save transects to CSV
transects.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved master transect table to:", OUTPUT_FILE.resolve())

