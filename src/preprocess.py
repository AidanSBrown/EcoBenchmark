import os
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil

# --- CONFIGURATION ---
RAW_IMG_DIR = Path("data/raw/images")
# UPDATE THIS LINE to point to your actual single CSV file
MASTER_CSV_PATH = Path("data/raw/csv/testing_live.csv") 

PROCESSED_IMG_DIR = Path("data/processed/images")
PROCESSED_CSV_DIR = Path("data/processed/annotations")
TARGET_SIZE = (400, 400) 

def resize_and_scale():
    # 1. Setup Directories
    if PROCESSED_IMG_DIR.exists(): shutil.rmtree(PROCESSED_IMG_DIR)
    if PROCESSED_CSV_DIR.exists(): shutil.rmtree(PROCESSED_CSV_DIR)
    PROCESSED_IMG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_CSV_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Master CSV
    if not MASTER_CSV_PATH.exists():
        print(f"CRITICAL ERROR: Master CSV not found at {MASTER_CSV_PATH}")
        return
    
    print(f"Loading master annotations from {MASTER_CSV_PATH}...")
    master_df = pd.read_csv(MASTER_CSV_PATH)
    
    # Ensure strict column names (DeepForest standard)
    # If your column is 'image_name' or 'image_path', rename it to 'image_path' for consistency
    if 'image_name' in master_df.columns:
        master_df = master_df.rename(columns={'image_name': 'image_path'})
    
    # Clean up image_path to match filenames (remove folders if present)
    # e.g., "training/plot_01.png" -> "plot_01.png"
    master_df['filename_only'] = master_df['image_path'].apply(lambda x: Path(x).name)

    # 3. Process Images
    image_files = sorted(list(RAW_IMG_DIR.glob("*.png")) + list(RAW_IMG_DIR.glob("*.jpg")) + list(RAW_IMG_DIR.glob("*.tif")))
    print(f"Found {len(image_files)} raw images.")

    matches_found = 0
    
    for img_path in image_files:
        # A. Resize Image
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            resized_img.save(PROCESSED_IMG_DIR / img_path.name)
            
        # B. Filter Master CSV for this image
        # We match based on the filename (e.g. "plot_01.png")
        img_annotations = master_df[master_df['filename_only'] == img_path.name].copy()
        
        if not img_annotations.empty:
            # Calculate scale factors
            scale_x = TARGET_SIZE[0] / orig_w
            scale_y = TARGET_SIZE[1] / orig_h
            
            # Scale coordinates
            img_annotations['xmin'] = (img_annotations['xmin'] * scale_x).astype(int)
            img_annotations['ymin'] = (img_annotations['ymin'] * scale_y).astype(int)
            img_annotations['xmax'] = (img_annotations['xmax'] * scale_x).astype(int)
            img_annotations['ymax'] = (img_annotations['ymax'] * scale_y).astype(int)
            
            # Clip
            img_annotations['xmin'] = img_annotations['xmin'].clip(0, TARGET_SIZE[0])
            img_annotations['ymin'] = img_annotations['ymin'].clip(0, TARGET_SIZE[1])
            img_annotations['xmax'] = img_annotations['xmax'].clip(0, TARGET_SIZE[0])
            img_annotations['ymax'] = img_annotations['ymax'].clip(0, TARGET_SIZE[1])
            
            # Save individual CSV for this image (Benchmark expects 1 CSV per image)
            save_path = PROCESSED_CSV_DIR / f"{img_path.stem}.csv"
            
            # Keep only necessary columns
            cols_to_save = ['xmin', 'ymin', 'xmax', 'ymax', 'label']
            img_annotations[cols_to_save].to_csv(save_path, index=False)
            
            matches_found += 1
        else:
            print(f"WARNING: No annotations found in master CSV for image: {img_path.name}")

    print(f"\nProcessing Complete.")
    print(f"Images: {len(image_files)}")
    print(f"Annotated: {matches_found}")

if __name__ == "__main__":
    resize_and_scale()