#!/usr/bin/env python3
"""
Project Status Checker
Shows what's completed and what needs to be done
"""

import os
import glob

print("="*80)
print("CSCE566 - GLAUCOMA DETECTION PROJECT STATUS")
print("="*80)

# Check files
print("\n📁 PROJECT FILES STATUS:\n")

files_to_check = {
    "Data & Preprocessing": [
        ("clinical_notes.csv", "Raw dataset"),
        ("1_data_exploration.py", "EDA script"),
        ("2_data_preprocessing.py", "Preprocessing script"),
        ("eda_visualizations.png", "EDA visualizations"),
        ("glaucoma_rate_by_race.png", "Fairness visualization"),
        ("vocab.pkl", "Vocabulary"),
        ("train_dataset.pt", "Training dataset"),
        ("val_dataset.pt", "Validation dataset"),
        ("test_dataset.pt", "Test dataset"),
    ],
    "Models": [
        ("models.py", "Model architectures"),
        ("train_all_models.py", "Training pipeline"),
        ("test_train.py", "Quick test script"),
    ],
    "Evaluation": [
        ("5_fairness_evaluation.py", "Fairness evaluation"),
    ],
    "Documentation": [
        ("README.md", "Project documentation"),
        ("PROJECT_SUMMARY.md", "Detailed summary"),
        ("GETTING_STARTED.md", "Quick start guide"),
        ("REFLECTION_TEMPLATE.md", "Reflection template"),
        ("report_template.tex", "LaTeX report template"),
        ("requirements.txt", "Dependencies"),
    ],
    "Execution": [
        ("run_pipeline.py", "Master pipeline script"),
    ],
}

for category, files in files_to_check.items():
    print(f"\n{category}:")
    for filename, description in files:
        exists = os.path.exists(filename)
        status = "✓" if exists else "✗"
        color = "\033[92m" if exists else "\033[91m"
        reset = "\033[0m"
        size = f"({os.path.getsize(filename):,} bytes)" if exists else ""
        print(f"  {color}{status}{reset} {filename:35} - {description} {size}")

# Check for trained models
print("\n🤖 TRAINED MODELS:")
trained_models = glob.glob("best_*_model.pt")
if trained_models:
    for model in trained_models:
        size = os.path.getsize(model)
        print(f"  ✓ {model:35} ({size:,} bytes)")
else:
    print("  ⚠ No trained models found yet")
    print("  → Run: python train_all_models.py")

# Check for results
print("\n📊 RESULTS FILES:")
result_files = glob.glob("*_results.json")
if result_files:
    for result in result_files:
        print(f"  ✓ {result}")
else:
    print("  ⚠ No results files found yet")
    print("  → Run: python train_all_models.py")

# Check for visualizations
print("\n📈 VISUALIZATIONS:")
viz_files = [
    "eda_visualizations.png",
    "glaucoma_rate_by_race.png",
    "fairness_comparison.png",
    "roc_curves_by_race.png",
]
for viz in viz_files:
    exists = os.path.exists(viz)
    status = "✓" if exists else "⚠"
    print(f"  {status} {viz}")

if not all(os.path.exists(v) for v in viz_files[2:]):
    print("  → Run: python 5_fairness_evaluation.py (after training)")

# Summary
print("\n" + "="*80)
print("COMPLETION STATUS")
print("="*80)

completed_tasks = [
    ("✓", "Data Exploration", True),
    ("✓", "Data Preprocessing", True),
    ("✓", "Model Architectures", True),
    ("✓", "Training Pipeline", True),
    ("✓", "Fairness Evaluation Framework", True),
    ("✓", "Documentation & Templates", True),
    ("⚠", "Train Models", bool(trained_models)),
    ("⚠", "Fairness Analysis Results", os.path.exists("model_comparison_table.csv")),
    ("⚠", "Write Final Report", False),
    ("⚠", "GitHub Repository", False),
    ("⚠", "Reflection Document", False),
]

for status, task, done in completed_tasks:
    if done and status == "✓":
        print(f"  \033[92m✓\033[0m {task}")
    elif done:
        print(f"  \033[92m✓\033[0m {task}")
    else:
        print(f"  \033[93m⚠\033[0m {task} - TODO")

# Next steps
print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

if not trained_models:
    print("\n🚀 IMMEDIATE ACTION REQUIRED:")
    print("  1. Run training: python train_all_models.py")
    print("     (This will take 15-30 minutes on CPU)")
    print("\n  2. After training completes, run:")
    print("     python 5_fairness_evaluation.py")
else:
    print("\n✓ Models are trained!")
    if os.path.exists("model_comparison_table.csv"):
        print("✓ Fairness analysis complete!")
        print("\n📝 YOU CAN NOW:")
        print("  1. Write the 4-page report using:")
        print("     - report_template.tex")
        print("     - Results from model_comparison_table.csv")
        print("     - Figures: *.png files")
        print("\n  2. Write reflection using:")
        print("     - REFLECTION_TEMPLATE.md")
        print("\n  3. Setup GitHub:")
        print("     - Create repository")
        print("     - Upload code files")
        print("     - Add README.md")
        print("\n  4. Submit:")
        print("     - Zip: report.pdf + reflection.pdf")
        print("     - Due: 11/25/2025")
    else:
        print("\n⚠ Run fairness evaluation:")
        print("  python 5_fairness_evaluation.py")

print("\n" + "="*80)
print("For detailed instructions, read: GETTING_STARTED.md")
print("="*80 + "\n")
