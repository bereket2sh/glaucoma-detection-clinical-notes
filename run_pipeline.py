#!/usr/bin/env python3
"""
Master Pipeline Script - Run Complete Analysis
CSCE566 - Data Mining - Glaucoma Detection
"""

import os
import sys
import time
import subprocess

print("="*80)
print("GLAUCOMA DETECTION PROJECT - COMPLETE PIPELINE")
print("CSCE566 - Data Mining Final Project")
print("="*80)

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Get Python path from virtual environment
        python_path = sys.executable
        result = subprocess.run(
            [python_path, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Completed in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_name}")
        return False

def check_file_exists(filename):
    """Check if a file exists"""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"  ✓ {filename} ({size:,} bytes)")
        return True
    else:
        print(f"  ❌ {filename} not found")
        return False

# Pipeline steps
steps = [
    {
        'script': '1_data_exploration.py',
        'description': 'Data Exploration and Visualization',
        'outputs': ['eda_visualizations.png', 'glaucoma_rate_by_race.png']
    },
    {
        'script': '2_data_preprocessing.py',
        'description': 'Data Preprocessing and Feature Engineering',
        'outputs': ['vocab.pkl', 'train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt']
    },
    {
        'script': 'train_all_models.py',
        'description': 'Train All Models (LSTM, GRU, Transformer)',
        'outputs': ['best_lstm_model.pt', 'best_gru_model.pt', 'best_transformer_model.pt',
                   'training_summary.csv']
    },
    {
        'script': '5_fairness_evaluation.py',
        'description': 'Fairness Evaluation Across Racial Groups',
        'outputs': ['model_comparison_table.csv', 'fairness_comparison.png', 'roc_curves_by_race.png']
    }
]

# Run pipeline
print("\nStarting complete pipeline...")
print(f"Working directory: {os.getcwd()}")
print(f"Python: {sys.executable}")

total_start = time.time()
success_count = 0

for i, step in enumerate(steps, 1):
    print(f"\n{'#'*80}")
    print(f"PIPELINE STEP {i}/{len(steps)}")
    print(f"{'#'*80}")
    
    # Run the script
    success = run_script(step['script'], step['description'])
    
    if success:
        success_count += 1
        # Check outputs
        print(f"\nChecking outputs...")
        for output_file in step['outputs']:
            check_file_exists(output_file)
    else:
        print(f"\n⚠️  Step {i} failed. Continuing to next step...")
        # Ask user if they want to continue
        user_input = input("\nContinue to next step? (y/n): ")
        if user_input.lower() != 'y':
            print("Pipeline stopped by user.")
            break

# Summary
total_time = time.time() - total_start

print("\n" + "="*80)
print("PIPELINE SUMMARY")
print("="*80)
print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Successful steps: {success_count}/{len(steps)}")

if success_count == len(steps):
    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("\nGenerated files:")
    
    all_outputs = [
        "eda_visualizations.png",
        "glaucoma_rate_by_race.png",
        "vocab.pkl",
        "train_dataset.pt",
        "val_dataset.pt", 
        "test_dataset.pt",
        "best_lstm_model.pt",
        "best_gru_model.pt",
        "best_transformer_model.pt",
        "training_summary.csv",
        "model_comparison_table.csv",
        "fairness_comparison.png",
        "roc_curves_by_race.png",
        "lstm_results.json",
        "gru_results.json",
        "transformer_results.json"
    ]
    
    for f in all_outputs:
        check_file_exists(f)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review results in training_summary.csv")
    print("2. Check fairness metrics in model_comparison_table.csv")
    print("3. View visualizations: *.png files")
    print("4. Write the 4-page report using these results")
    print("5. Create GitHub repository and upload code")
    print("6. Write reflection document")
    print("="*80)
else:
    print(f"\n⚠️  Pipeline completed with {len(steps) - success_count} failed step(s)")
    print("Please check error messages above.")

print("\n" + "="*80)
print("END OF PIPELINE")
print("="*80)
