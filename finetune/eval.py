
import torch

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def evaluate_model(model, test_loader, device, topk: int = 1):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    total = 0
    topk_correct = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            if topk and topk > 1:
                _, topk_pred = outputs.topk(topk, dim=1)
                topk_correct += (topk_pred.eq(labels.view(-1, 1)).any(dim=1)).sum().item()
                total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    try:
        auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc_roc = 0.0 

    topk_acc = None
    if topk and topk > 1:
        if total == 0:
            total = len(test_loader.dataset)
        topk_acc = topk_correct / float(total)
    
    return accuracy, f1, auc_roc, all_predictions, all_labels, topk_acc