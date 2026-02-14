from typing import List, Dict, Any

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Boxes format: [xmin, ymin, xmax, ymax]
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def match_predictions(ground_truth: List[Dict], predictions: List[Dict], iou_threshold=0.4):
    """
    Matches predictions to ground truth using IoU.
    """
    # Sort predictions by confidence score (descending)
    predictions = sorted(predictions, key=lambda x: x.get('score', 0.0), reverse=True)
    
    gt_matched = [False] * len(ground_truth)
    pred_matched = [False] * len(predictions)
    
    matches = [] 

    for i, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truth):
            if gt_matched[j]: continue 
            
            iou = calculate_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            gt_matched[best_gt_idx] = True
            pred_matched[i] = True
            matches.append({
                "gt": ground_truth[best_gt_idx],
                "pred": pred,
                "iou": best_iou
            })
            
    tp = sum(gt_matched)
    fn = len(ground_truth) - tp
    fp = len(predictions) - tp
    
    return {"TP": tp, "FP": fp, "FN": fn, "matches": matches}