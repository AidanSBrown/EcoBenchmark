from .base_model import TreeDetector
from typing import List, Dict, Any

# Safe import for DeepForest
try:
    from deepforest import main
except ImportError:
    main = None

class DeepForestWrapper(TreeDetector):
    def __init__(self, model_path: str = None):
        if main is None:
            raise ImportError("DeepForest is not installed.")
        
        self.model = main.deepforest()
        if model_path:
            self.model.load_model(model_path)
        else:
            self.model.use_release()

    def detect(self, image_path: str, config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # Predict on the image
        # return_plot=False ensures we just get the dataframe
        boxes_df = self.model.predict_image(path=image_path, return_plot=False)
        
        if boxes_df is None or boxes_df.empty:
            return []

        predictions = []
        for _, row in boxes_df.iterrows():
            predictions.append({
                "box": [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                "label": row['label'],
                "score": float(row['score'])
            })
            
        return predictions