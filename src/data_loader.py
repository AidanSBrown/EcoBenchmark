import pandas as pd
from typing import List, Dict, Generator, Tuple
from pathlib import Path

class TreeDataset:
    """
    Loads PREPROCESSED (400x400) data.
    """
    def __init__(self, processed_dir: str = "data/processed"):
        self.image_dir = Path(processed_dir) / "images"
        self.annotation_dir = Path(processed_dir) / "annotations"
        self.image_files = sorted(list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.tif")))

    def __len__(self):
        return len(self.image_files)

    def load_data(self) -> Generator[Tuple[str, List[Dict]], None, None]:
        for img_path in self.image_files:
            csv_path = self.annotation_dir / f"{img_path.stem}.csv"
            ground_truth = []
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    ground_truth.append({
                        "box": [row['xmin'], row['ymin'], row['xmax'], row['ymax']], # Now in 400px coords
                        "label": row['label'],
                        "score": 1.0
                    })

            yield str(img_path), ground_truth