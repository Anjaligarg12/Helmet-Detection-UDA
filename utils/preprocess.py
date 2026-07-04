import os
import cv2
import numpy as np

def create_directories():
    dirs = [
        "data/source_india/images/train", "data/source_india/labels/train",
        "data/source_india/images/val", "data/source_india/labels/val",
        "data/target_sea/images/train", "data/target_sea/labels/train",
        "data/adapted/images/train", "data/adapted/labels/train",
        "data/adapted/images/val", "data/adapted/labels/val",
        "data/test_sea/images/val", "data/test_sea/labels/val",
        "outputs/samples", "outputs/metrics"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_synthetic_data(num_samples=50):
    print("Generating synthetic dataset...")
    create_directories()
    
    # Check if already generated
    if os.path.exists("data/source_india/images/train/india_0.jpg"):
        print("Data already exists. Skipping generation.")
        return

    for split in ['train', 'val']:
        # We need validation data for test_sea too. We'll put target_sea val there.
        samples = num_samples if split == 'train' else int(num_samples * 0.2)
        if samples == 0: samples = 10
        
        for i in range(samples):
            # --- Source India ---
            # 150 = Grayish dusty
            img_src = np.ones((256, 256, 3), dtype=np.uint8) * 150
            noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
            img_src = cv2.add(img_src, noise)
            
            w, h = np.random.randint(30, 60), np.random.randint(30, 60)
            cx, cy = np.random.randint(40, 216), np.random.randint(40, 216)
            
            # Draw helmet (Red shape)
            cv2.ellipse(img_src, (cx, cy), (w//2, h//2), 0, 0, 360, (0, 0, 255), -1)
            
            src_img_path = f"data/source_india/images/{split}/india_{i}.jpg"
            cv2.imwrite(src_img_path, img_src)
            
            src_lbl_path = f"data/source_india/labels/{split}/india_{i}.txt"
            with open(src_lbl_path, "w") as f:
                f.write(f"0 {cx/256} {cy/256} {w/256} {h/256}\n")

            # --- Target SEA ---
            # Greenish lush
            img_tgt = np.ones((256, 256, 3), dtype=np.uint8) * np.array([50, 150, 50], dtype=np.uint8) # BGR
            noise = np.random.randint(0, 30, (256, 256, 3), dtype=np.uint8)
            img_tgt = cv2.add(img_tgt, noise)
            
            w, h = np.random.randint(30, 60), np.random.randint(30, 60)
            cx, cy = np.random.randint(40, 216), np.random.randint(40, 216)
            
            # Draw helmet (Blue shape)
            cv2.ellipse(img_tgt, (cx, cy), (w//2, h//2), 0, 0, 360, (255, 0, 0), -1)
            
            tgt_base_dir = "data/target_sea" if split == 'train' else "data/test_sea"
            tgt_img_path = f"{tgt_base_dir}/images/{split}/sea_{i}.jpg"
            cv2.imwrite(tgt_img_path, img_tgt)
            
            tgt_lbl_path = f"{tgt_base_dir}/labels/{split}/sea_{i}.txt"
            with open(tgt_lbl_path, "w") as f:
                f.write(f"0 {cx/256} {cy/256} {w/256} {h/256}\n")
                
    print("Dataset generation complete.")

if __name__ == "__main__":
    generate_synthetic_data()
