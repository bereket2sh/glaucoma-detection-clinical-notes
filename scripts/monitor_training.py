#!/usr/bin/env python3
"""
Monitor training progress
"""

import os
import time
import glob

print("="*80)
print("TRAINING PROGRESS MONITOR")
print("="*80)

while True:
    os.system('clear')
    print("="*80)
    print("TRAINING PROGRESS MONITOR")
    print("="*80)
    
    # Check for trained models
    models = glob.glob("best_*_model.pt")
    print(f"\n✓ Trained models: {len(models)}/3")
    for model in models:
        size_mb = os.path.getsize(model) / (1024*1024)
        print(f"  - {model} ({size_mb:.2f} MB)")
    
    # Check for results
    results = glob.glob("*_results.json")
    print(f"\n✓ Results files: {len(results)}/3")
    for result in results:
        print(f"  - {result}")
    
    # Check log file
    if os.path.exists("training_output.log"):
        print("\n📝 Last 10 lines of training log:")
        print("-" * 80)
        os.system("tail -10 training_output.log")
    
    print("\n" + "="*80)
    print("Press Ctrl+C to exit | Refreshing every 30 seconds...")
    print("="*80)
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        break
