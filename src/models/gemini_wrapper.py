import os
import json
import time
import google.generativeai as genai
from typing import List, Dict, Any
from pathlib import Path
from .base_model import TreeDetector

class GeminiDetector(TreeDetector):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def _get_system_prompt(self, detailed_reasoning: bool = False) -> str:
        prompt = """
        You are an expert forest ecologist. Detect ALL trees in this image.
        The image is a 50m x 50m aerial patch (400x400 pixels).
        
        CRITICAL INSTRUCTION:
        Return ONLY valid JSON. Do not write an introduction or conclusion.
        
        OUTPUT FORMAT:
        {
            "trees": [
                {"box_2d": [ymin, xmin, ymax, xmax], "label": "Alive"},
                {"box_2d": [ymin, xmin, ymax, xmax], "label": "Dead"}
            ]
        }
        
        COORDINATE SYSTEM:
        - Use NORMALIZED coordinates (0 to 1000).
        - [0, 0] is Top-Left. [1000, 1000] is Bottom-Right.
        """
        
        if detailed_reasoning:
            prompt += """
            VISUAL GUIDES:
            - Alive: Continuous green texture, rounded canopy.
            - Dead: Gray/white skeletal branches, no leaves, jagged structure.
            """
        return prompt

    def detect(self, image_path: str, config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        config = config or {}
        use_reasoning = config.get("use_reasoning", False)
        
        # 1. Upload the file
        # Gemini requires uploading the file for processing
        sample_file = genai.upload_file(path=image_path, display_name="Tree Patch")
        
        # 2. Generate Content
        prompt = self._get_system_prompt(detailed_reasoning=use_reasoning)
        
        try:
            response = self.model.generate_content([prompt, sample_file])
            
            # Clean up the text response (remove markdown ```json ... ```)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3]
            elif text.startswith("```"):
                text = text[3:-3]
                
            data = json.loads(text)
            
            # 3. Parse and Convert Coordinates
            predictions = []
            for item in data.get("trees", []):
                # Gemini output: [ymin, xmin, ymax, xmax] in 0-1000 scale
                box = item["box_2d"]
                ymin, xmin, ymax, xmax = box
                
                # Convert to [xmin, ymin, xmax, ymax] in 0-400 pixel scale
                # (Since our preprocessing standardized images to 400x400)
                standard_box = [
                    (xmin / 1000) * 400,
                    (ymin / 1000) * 400,
                    (xmax / 1000) * 400,
                    (ymax / 1000) * 400
                ]
                
                predictions.append({
                    "box": standard_box,
                    "label": item["label"],
                    "score": 1.0 
                })
                
            return predictions

        except Exception as e:
            print(f"Gemini Error on {image_path}: {e}")
            return []
        finally:
            # Cleanup: Delete the file from Google's server to save space/privacy
            try:
                sample_file.delete()
            except:
                pass