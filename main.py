import os
import sys

from utils.preprocess import generate_synthetic_data
from cyclegan.train import train_cyclegan
from cyclegan.inference import adapt_dataset
from detection.train_yolo import train_baseline, train_adapted
from detection.evaluate import evaluate_models

def main():
    print("Starting UDA pipeline for Helmet Detection...")
    
    # 1. Data Generation (Synthetic dataset to enable easy runnable testing)
    generate_synthetic_data(num_samples=80)
    
    # 2. CycleGAN Training
    train_cyclegan(epochs=2)
    
    # 3. Domain Adaptation
    adapt_dataset('data/source_india', 'data/adapted', 'outputs/G_A2B.pth')
    
    # 4. Train YOLO Baseline
    train_baseline(epochs=5)
    
    # 5. Train YOLO Adapted
    train_adapted(epochs=5)
    
    # 6. Evaluate
    evaluate_models()
    
    print("\nPipeline completed successfully.")
    print("Check 'outputs/samples' for visual comparisons and 'outputs/metrics/results.json' for metrics.")

if __name__ == '__main__':
    main()
