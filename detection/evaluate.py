import os
import json
import yaml
from ultralytics import YOLO

def evaluate_models():
    print("Evaluating models on Target SEA Test Set...")
    
    yaml_path = "detection/test.yaml"
    data = {
        "path": os.path.abspath('data/test_sea'),
        "train": "images/val",
        "val": "images/val",
        "test": "images/val",
        "nc": 1,
        "names": {0: "helmet"}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
        
    results_dict = {}
    
    baseline_weights = 'outputs/yolo_baseline/weights/best.pt'
    if os.path.exists(baseline_weights):
        print("\n--- Evaluating Baseline ---")
        model_b = YOLO(baseline_weights)
        metrics_b = model_b.val(data=yaml_path, imgsz=256)
        results_dict['baseline'] = {
            'map50': metrics_b.box.map50,
            'map': metrics_b.box.map,
            'precision': metrics_b.box.mp,
            'recall': metrics_b.box.mr
        }
    else:
        results_dict['baseline'] = None
        print("Baseline weights not found.")
        
    adapted_weights = 'outputs/yolo_adapted/weights/best.pt'
    if os.path.exists(adapted_weights):
        print("\n--- Evaluating Adapted ---")
        model_a = YOLO(adapted_weights)
        metrics_a = model_a.val(data=yaml_path, imgsz=256)
        results_dict['adapted'] = {
            'map50': metrics_a.box.map50,
            'map': metrics_a.box.map,
            'precision': metrics_a.box.mp,
            'recall': metrics_a.box.mr
        }
    else:
        results_dict['adapted'] = None
        print("Adapted weights not found.")
        
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print("\n================ Results ================")
    print(json.dumps(results_dict, indent=4))
    if results_dict.get('baseline') and results_dict.get('adapted'):
        imp = results_dict['adapted']['map50'] - results_dict['baseline']['map50']
        print(f"Improvement in mAP@50: {imp:.4f}")
    print("==========================================\n")

if __name__ == "__main__":
    evaluate_models()
