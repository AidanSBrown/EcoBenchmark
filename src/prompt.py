from typing import Any, Dict, List


def _get_system_prompt(self, detailed_reasoning: bool = False) -> str:
    base_prompt = """
    You are an expert ecologist. Detect ALL trees in this image.
    The image is a 50m x 50m aerial patch.
    
    OUTPUT FORMAT:
    Return valid JSON.
    Bounding boxes must be in [ymin, xmin, ymax, xmax] format.
    Use NORMALIZED COORDINATES (0 to 1000). 
    Example: Top-left corner is [0, 0], center is [500, 500], bottom-right is [1000, 1000].
    
    {
        "trees": [
            {"box_2d": [ymin, xmin, ymax, xmax], "label": "Alive"}, ...
        ]
    }
    """
    # ... (rest of reasoning logic) ...
    return base_prompt

def _parse_response(self, text: str) -> List[Dict[str, Any]]:
    # ... (json loading) ...
    # logic inside the loop:
        box = item["box_2d"]
        # Convert 0-1000 scale to 0-400 scale
        # Also swap from [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
        
        ymin, xmin, ymax, xmax = box
        
        standard_box = [
            (xmin / 1000) * 400,
            (ymin / 1000) * 400,
            (xmax / 1000) * 400,
            (ymax / 1000) * 400
        ]
        # ...