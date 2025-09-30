import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import json

def calculate_metrics(y_true, y_pred, y_scores, num_classes, topk: int = 1):
    """Calculate evaluation metrics"""
    
    top1_acc = accuracy_score(y_true, y_pred)
    topk_acc = None
    if topk and topk > 1:
        # y_scores: (N, C)
        topk_idx = np.argpartition(-y_scores, kth=min(topk-1, y_scores.shape[1]-1), axis=1)[:, :topk]
        
        matches = [y_true[i] in topk_idx[i] for i in range(len(y_true))]
        topk_acc = float(np.mean(matches))
    
    f1 = f1_score(y_true, y_pred, average='macro')
    
    try:
        if num_classes == 2:
            auc_roc = roc_auc_score(y_true, y_scores[:, 1])
        else:
            y_true_binarized = label_binarize(y_true, classes=range(num_classes))
            auc_roc = roc_auc_score(y_true_binarized, y_scores, average='macro', multi_class='ovr')
    except Exception as e:
        print(f"Warning: Could not calculate AUC-ROC: {e}")
        auc_roc = 0.0
    
    metrics = {
        'top1_accuracy': top1_acc,
        'f1_score': f1,
        'auc_roc': auc_roc
    }
    if topk_acc is not None:
        metrics[f'top{topk}_accuracy'] = topk_acc
    return metrics