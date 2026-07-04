import os
import yaml
from ultralytics import YOLO

def create_yaml(data_type, base_dir):
    yaml_path = f"detection/{data_type}.yaml"
    data = {
        "path": os.path.abspath(base_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "helmet"}
    }
    os.makedirs("detection", exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path

def train_baseline(epochs=5):
    print("Training YOLO Baseline on Source (India)...")
    yaml_path = create_yaml('baseline', 'data/source_india')
    model = YOLO('yolov8n.pt')
    
    results = model.train(data=yaml_path, epochs=epochs, imgsz=256, batch=4, project='outputs', name='yolo_baseline')
    print("Baseline training complete.")
    
def train_adapted(epochs=5):
    print("Training YOLO Adapted on Adapted domain (India->SEA)...")
    yaml_path = create_yaml('adapted', 'data/adapted')
    model = YOLO('yolov8n.pt')
    
    results = model.train(data=yaml_path, epochs=epochs, imgsz=256, batch=4, project='outputs', name='yolo_adapted')
    print("Adapted training complete.")
