import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict
from models.gemini_wrapper import GeminiDetector
import time

# Import your modules
from data_loader import TreeDataset
from models.deepforest_wrapper import DeepForestWrapper
from metrics import match_predictions

# Load API keys from .env file
load_dotenv()

def save_predictions(results: List[Dict], filename: str):
    os.makedirs("data/predictions", exist_ok=True)
    with open(f"data/predictions/{filename}", "w") as f:
        json.dump(results, f, indent=2)

def run_benchmark(limit=None):
    # 1. Setup Data
    dataset = TreeDataset(processed_dir="data/processed")
    print(f"Loaded {len(dataset)} images for benchmarking.")

    # 2. Initialize Models
    models = {}
    
    # DeepForest (Always runs locally)
    try:
        print("Initializing DeepForest...")
        models["DeepForest"] = DeepForestWrapper() 
    except Exception as e:
        print(f"Skipping DeepForest: {e}")

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("Initializing Gemini 1.5 Pro...")
        # REMOVE 'pass' and UNCOMMENT this line:
        models["Gemini-ZeroShot"] = GeminiDetector(api_key=gemini_key)
    else:
        print("Skipping Gemini (No API Key)")

    # 3. Run Inference Loop
    results_log = []
    
    # Iterate through images
    count = 0
    for img_path, ground_truth in tqdm(dataset.load_data(), total=len(dataset)):
        if limit and count >= limit: break


        if "Gemini-ZeroShot" in models:
            time.sleep(4) # to not overload free tier API with too many requests in a short time
        
        image_id = os.path.basename(img_path)
        
        image_id = os.path.basename(img_path)
        
        for model_name, model_instance in models.items():
            # A. Detect
            try:
                predictions = model_instance.detect(img_path)
            except Exception as e:
                print(f"Error {model_name} on {image_id}: {e}")
                predictions = []

            # B. Evaluate (Match predictions to Ground Truth)
            metrics = match_predictions(ground_truth, predictions, iou_threshold=0.4)
            
            # C. Log Result
            results_log.append({
                "image": image_id,
                "model": model_name,
                "ground_truth_count": len(ground_truth),
                "prediction_count": len(predictions),
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "precision": metrics["TP"] / (metrics["TP"] + metrics["FP"] + 1e-6),
                "recall": metrics["TP"] / (metrics["TP"] + metrics["FN"] + 1e-6)
            })
        
        count += 1

    # 4. Save and Summarize
    save_predictions(results_log, "benchmark_results.json")
    
    # Simple CLI Summary
    import pandas as pd
    df = pd.DataFrame(results_log)
    if not df.empty:
        summary = df.groupby("model")[["precision", "recall", "TP", "FP", "FN"]].mean()
        print("\n=== BENCHMARK SUMMARY (Average per Image) ===")
        print(summary)
    else:
        print("No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Run only on N images for testing")
    args = parser.parse_args()
    
    run_benchmark(limit=args.limit)