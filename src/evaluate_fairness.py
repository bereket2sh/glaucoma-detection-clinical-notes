"""
Fairness Evaluation and Analysis
CSCE566 - Data Mining - Glaucoma Detection
Evaluate models across different racial groups
"""

import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from models import get_model

# Define GlaucomaDataset class (needed for loading saved datasets)
class GlaucomaDataset(Dataset):
    """Custom Dataset for Glaucoma Detection"""
    
    def __init__(self, dataframe):
        self.texts = np.array(dataframe['sequence'].tolist())
        self.labels = dataframe['label'].values
        self.races = dataframe['race'].values
        self.ages = dataframe['age'].values
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'text': torch.LongTensor(self.texts[idx]),
            'label': torch.LongTensor([self.labels[idx]]),
            'race': self.races[idx],
            'age': self.ages[idx]
        }

print("="*80)
print("FAIRNESS EVALUATION - RACE-BASED PERFORMANCE ANALYSIS")
print("="*80)

# Load test data
print("\n[1] Loading test data...")
test_df = pd.read_csv('../data/processed/test_processed.csv')
test_dataset = torch.load('../data/processed/test_dataset.pt', weights_only=False)

print(f"  Test samples: {len(test_df)}")

# Load vocabulary
with open('../data/processed/vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
vocab_size = len(vocab_data['vocab'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # Same as recall
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'total': int(tp + fp + tn + fn)
    }
    
    return metrics

def evaluate_model_by_race(model_name):
    """Evaluate a model across different racial groups"""
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name.upper()} MODEL")
    print(f"{'='*80}")
    
    # Load model
    print(f"\n[1] Loading {model_name} model...")
    model = get_model(model_name, vocab_size, device)
    checkpoint = torch.load(f'../outputs/models/best_{model_name}_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load predictions
    predictions = np.load(f'../outputs/{model_name}_predictions.npy')
    labels = np.load(f'../outputs/{model_name}_labels.npy')
    
    # Binary predictions
    pred_labels = (predictions >= 0.5).astype(int)
    
    print(f"  Total test samples: {len(labels)}")
    
    # Overall performance
    print(f"\n[2] Overall Performance:")
    overall_metrics = calculate_metrics(labels, pred_labels, predictions)
    
    print(f"  AUC: {overall_metrics['auc']:.4f}")
    print(f"  Sensitivity: {overall_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {overall_metrics['specificity']:.4f}")
    print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    
    # Performance by race
    print(f"\n[3] Performance by Race:")
    
    race_results = {'overall': overall_metrics}
    
    for race in ['white', 'black', 'asian']:
        race_mask = test_df['race'] == race
        race_count = race_mask.sum()
        
        if race_count > 0:
            race_labels = labels[race_mask]
            race_predictions = predictions[race_mask]
            race_pred_labels = pred_labels[race_mask]
            
            race_metrics = calculate_metrics(race_labels, race_pred_labels, race_predictions)
            race_results[race] = race_metrics
            
            print(f"\n  {race.upper()} (n={race_count}):")
            print(f"    AUC: {race_metrics['auc']:.4f}")
            print(f"    Sensitivity: {race_metrics['sensitivity']:.4f}")
            print(f"    Specificity: {race_metrics['specificity']:.4f}")
            print(f"    Accuracy: {race_metrics['accuracy']:.4f}")
    
    # Save results
    results = {
        'model_name': model_name,
        'overall': overall_metrics,
        'by_race': race_results
    }
    
    with open(f'../outputs/{model_name}_fairness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[4] Fairness results saved to outputs/{model_name}_fairness_results.json")
    
    return results

def create_comparison_table(all_results):
    """Create comparison table across models and racial groups"""
    print(f"\n{'='*80}")
    print("CREATING COMPARISON TABLE")
    print(f"{'='*80}")
    
    # Prepare data for table
    rows = []
    
    for model_name, results in all_results.items():
        # Overall
        overall = results['by_race']['overall']
        rows.append({
            'Model': model_name.upper(),
            'Group': 'Overall',
            'N': overall['total'],
            'AUC': f"{overall['auc']:.4f}",
            'Sensitivity': f"{overall['sensitivity']:.4f}",
            'Specificity': f"{overall['specificity']:.4f}",
            'Accuracy': f"{overall['accuracy']:.4f}"
        })
        
        # By race
        for race in ['white', 'black', 'asian']:
            if race in results['by_race']:
                metrics = results['by_race'][race]
                rows.append({
                    'Model': model_name.upper(),
                    'Group': race.capitalize(),
                    'N': metrics['total'],
                    'AUC': f"{metrics['auc']:.4f}",
                    'Sensitivity': f"{metrics['sensitivity']:.4f}",
                    'Specificity': f"{metrics['specificity']:.4f}",
                    'Accuracy': f"{metrics['accuracy']:.4f}"
                })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv('../outputs/model_comparison_table.csv', index=False)
    print("\n  ✓ Comparison table saved to 'outputs/model_comparison_table.csv'")
    
    # Print table
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    return df

def create_fairness_visualizations(all_results):
    """Create visualizations for fairness analysis"""
    print(f"\n{'='*80}")
    print("CREATING FAIRNESS VISUALIZATIONS")
    print(f"{'='*80}")
    
    models = list(all_results.keys())
    races = ['white', 'black', 'asian']
    metrics_to_plot = ['auc', 'sensitivity', 'specificity']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Performance by Racial Group', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Prepare data
        data = []
        for model_name in models:
            for race in races:
                if race in all_results[model_name]['by_race']:
                    value = all_results[model_name]['by_race'][race][metric]
                    data.append({
                        'Model': model_name.upper(),
                        'Race': race.capitalize(),
                        'Value': value
                    })
        
        df_plot = pd.DataFrame(data)
        
        # Create grouped bar chart
        x = np.arange(len(races))
        width = 0.25
        
        for i, model in enumerate(models):
            model_data = df_plot[df_plot['Model'] == model.upper()]
            values = [model_data[model_data['Race'] == race.capitalize()]['Value'].values[0] 
                     if len(model_data[model_data['Race'] == race.capitalize()]) > 0 else 0
                     for race in races]
            ax.bar(x + i * width, values, width, label=model.upper())
        
        ax.set_xlabel('Racial Group', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} by Race', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([r.capitalize() for r in races])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/fairness_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Fairness visualization saved to 'outputs/figures/fairness_comparison.png'")
    
    # Create individual ROC curves
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('ROC Curves by Model', fontsize=16, fontweight='bold')
    
    from sklearn.metrics import roc_curve
    
    for idx, model_name in enumerate(models):
        ax = axes2[idx]
        
        predictions = np.load(f'{model_name}_predictions.npy')
        labels = np.load(f'{model_name}_labels.npy')
        
        # Overall ROC
        fpr, tpr, _ = roc_curve(labels, predictions)
        overall_auc = all_results[model_name]['by_race']['overall']['auc']
        ax.plot(fpr, tpr, linewidth=2, label=f'Overall (AUC={overall_auc:.3f})', color='black')
        
        # By race
        colors = {'white': 'blue', 'black': 'red', 'asian': 'green'}
        for race in races:
            race_mask = test_df['race'] == race
            if race_mask.sum() > 0:
                race_labels = labels[race_mask]
                race_predictions = predictions[race_mask]
                fpr_race, tpr_race, _ = roc_curve(race_labels, race_predictions)
                race_auc = all_results[model_name]['by_race'][race]['auc']
                ax.plot(fpr_race, tpr_race, linewidth=2, 
                       label=f'{race.capitalize()} (AUC={race_auc:.3f})', 
                       color=colors[race], linestyle='--')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
        ax.set_title(f'{model_name.upper()} Model', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/roc_curves_by_race.png', dpi=300, bbox_inches='tight')
    print("  ✓ ROC curves saved to 'outputs/figures/roc_curves_by_race.png'")

# Main execution
if __name__ == "__main__":
    # Models to evaluate
    models = ['lstm', 'gru', 'transformer']
    
    # Evaluate each model
    all_results = {}
    for model_name in models:
        try:
            results = evaluate_model_by_race(model_name)
            all_results[model_name] = results
        except FileNotFoundError:
            print(f"\n  ⚠ Model {model_name} not found, skipping...")
    
    if all_results:
        # Create comparison table
        create_comparison_table(all_results)
        
        # Create visualizations
        create_fairness_visualizations(all_results)
        
        print("\n" + "="*80)
        print("FAIRNESS EVALUATION COMPLETE!")
        print("="*80)
    else:
        print("\n  ⚠ No trained models found. Please train models first.")
