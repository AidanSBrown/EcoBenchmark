import os
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil

# Config
RAW_IMG_DIR = Path("data/images") # Original high-res images
RAW_CSV_DIR = Path("data/csv") # csv of annotations in original image coordinates (e.g. 2000x2000)
PROCESSED_IMG_DIR = Path("data/processed/images")
PROCESSED_CSV_DIR = Path("data/processed/annotations")
TARGET_SIZE = (400, 400) # (Width, Height) - DeepForest standard

def setup_dirs():
    if PROCESSED_IMG_DIR.exists(): shutil.rmtree(PROCESSED_IMG_DIR)
    if PROCESSED_CSV_DIR.exists(): shutil.rmtree(PROCESSED_CSV_DIR)
    PROCESSED_IMG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_CSV_DIR.mkdir(parents=True, exist_ok=True)

def resize_and_scale():
    setup_dirs()
    image_files = sorted(list(RAW_IMG_DIR.glob("*.png")) + list(RAW_IMG_DIR.glob("*.jpg")) + list(RAW_IMG_DIR.glob("*.tif")))
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        # 1. Load Original
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            
            # 2. Resize Image
            # Use LANCZOS for high-quality downsampling to preserve tree details
            resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Save processed image
            save_path = PROCESSED_IMG_DIR / img_path.name
            resized_img.save(save_path)
            
            # 3. Scale Annotations
            csv_path = RAW_CSV_DIR / f"{img_path.stem}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # Calculate scale factors
                scale_x = TARGET_SIZE[0] / orig_w
                scale_y = TARGET_SIZE[1] / orig_h
                
                # Apply scaling to coordinates
                df['xmin'] = (df['xmin'] * scale_x).astype(int)
                df['ymin'] = (df['ymin'] * scale_y).astype(int)
                df['xmax'] = (df['xmax'] * scale_x).astype(int)
                df['ymax'] = (df['ymax'] * scale_y).astype(int)
                
                # Clip to new boundaries (0-400)
                df['xmin'] = df['xmin'].clip(0, TARGET_SIZE[0])
                df['ymin'] = df['ymin'].clip(0, TARGET_SIZE[1])
                df['xmax'] = df['xmax'].clip(0, TARGET_SIZE[0])
                df['ymax'] = df['ymax'].clip(0, TARGET_SIZE[1])
                
                # Save processed CSV
                df.to_csv(PROCESSED_CSV_DIR / csv_path.name, index=False)

    print("Preprocessing complete. Data is now standardized to 400x400.")

if __name__ == "__main__":
    resize_and_scale()