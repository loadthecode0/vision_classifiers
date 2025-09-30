
import os
import torch
from finetune.trainer import finetune
from finetune.eval import evaluate_model
from utils.models import ViTClassifier
from finetune.results import save_metrics_json, save_metrics_csv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


def finetune_classifier(model_type, dataset_dir, epochs, lr, train_loader, val_loader, num_classes, classes, device, output_dir, topk=1):
    
    device = torch.device(device)
    print(device)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f'\n{"="*50}')
    print(f'Training {model_type.upper()} model...')
    print(f'{"="*50}')
    
    model = ViTClassifier(model_type, num_classes).to(device)
    
    train_losses, train_accuracies, val_losses, val_accuracies = \
        finetune(
            model, 
            train_loader, 
            val_loader, 
            device, epochs, 
            lr, 
            model_type, 
            f"{output_dir}/training",
            topk=topk
            )
    
    return model, train_losses, train_accuracies, val_losses, val_accuracies
    

def test_classifier(model, model_type, test_loader, classes, device, output_dir, topk=1):
    device = torch.device(device)
    accuracy, f1, auc_roc, predictions, labels, topk_acc = evaluate_model(model, test_loader, device, topk=topk)
    
    print(f'\n{model_type.upper()} Results:')
    print(f'  Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'  F1-Score: {f1:.4f}')
    print(f'  AUC-ROC: {auc_roc:.4f}')
    if topk_acc is not None:
        print(f'  Top-{topk} Accuracy: {topk_acc*100:.2f}%')
    print('\nDetailed Classification Report:')
    report_text = classification_report(labels, predictions, target_names=classes)
    print(report_text)

    testing_dir = f"{output_dir}/testing"
    os.makedirs(testing_dir, exist_ok=True)
    metrics = {
        'model_type': model_type,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'num_classes': len(classes)
    }
    if topk_acc is not None:
        metrics[f'top{topk}_accuracy'] = topk_acc
    save_metrics_json(metrics, f"{testing_dir}/metrics.json")
    # rows = [{'label': int(l), 'prediction': int(p)} for l, p in zip(labels, predictions)]
    # save_metrics_csv(rows, f"{testing_dir}/predictions.csv", fieldnames=['label', 'prediction'])
    with open(f"{testing_dir}/classification_report.txt", 'w') as f:
        f.write(report_text)
