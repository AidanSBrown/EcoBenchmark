from src.data_loader import TreeDataset
import pandas as pd
from pathlib import Path

# Initialize dataset
ds = TreeDataset(processed_dir="data/processed")

print(f"Dataset found {len(ds)} images.")
print(f"Looking for annotations in: {ds.annotation_dir.absolute()}")

# Check the first 3 items
for i, (img_path, ground_truth) in enumerate(ds.load_data()):
    if i >= 3: break
    
    img_name = Path(img_path).name
    print(f"\n--- Image: {img_name} ---")
    print(f"Ground Truth Trees Found: {len(ground_truth)}")
    
    # Check if the CSV file actually exists physically
    expected_csv = ds.annotation_dir / f"{Path(img_path).stem}.csv"
    if expected_csv.exists():
        print(f"CSV File: FOUND at {expected_csv}")
        # Peek at the raw CSV to see if it's empty
        df = pd.read_csv(expected_csv)
        print(f"CSV Row Count: {len(df)}")
        print(f"CSV Columns: {df.columns.tolist()}")
    else:
        print(f"CSV File: MISSING at {expected_csv}")