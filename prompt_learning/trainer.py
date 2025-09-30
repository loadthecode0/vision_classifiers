import os
from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from finetune.results import save_metrics_json, save_metrics_csv, plot_training_curves


def compute_topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    if k <= 1:
        _, preds = torch.max(logits, 1)
        return (preds == labels).sum().item()
    _, topk_pred = logits.topk(k, dim=1)
    return (topk_pred.eq(labels.view(-1, 1)).any(dim=1)).sum().item()


def train_prompt_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    output_dir: Optional[str] = None,
    save_best: bool = True,
    save_curves: bool = True,
    save_metrics: bool = True,
    topk: int = 1,
    method: str = "coop"
) -> Tuple[List[float], List[float], List[float], List[float]]:
    
    # Setup optimizer - only optimize prompt parameters
    if method.lower() == "coop" or method.lower() == "maple":
        trainable_params = model.prompt_learner.parameters()
    elif method.lower() == "cocoop":
        trainable_params = list(model.prompt_learner.meta_net.parameters()) + [model.prompt_learner.ctx]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    train_topk_accuracies: List[float] = []
    val_topk_accuracies: List[float] = []

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        correct = 0
        topk_correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += compute_topk_correct(logits, labels, 1)
            if topk and topk > 1:
                topk_correct += compute_topk_correct(logits, labels, topk)

            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / max(1, total)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        if topk and topk > 1:
            train_topk_accuracies.append(100.0 * topk_correct / max(1, total))

        # val
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_topk_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_running_loss += loss.item()
                val_total += labels.size(0)
                val_correct += compute_topk_correct(logits, labels, 1)
                if topk and topk > 1:
                    val_topk_correct += compute_topk_correct(logits, labels, topk)

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100.0 * val_correct / max(1, val_total)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if topk and topk > 1:
            val_topk_accuracies.append(100.0 * val_topk_correct / max(1, val_total))

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                torch.save(best_state, os.path.join(output_dir, "best_model.pt"))
                print(f"✅ Saved new best model (Val Acc = {best_val_acc:.2f}%) at epoch {epoch+1}")


        # step scheduler per epoch
        scheduler.step()

        # log
        if topk and topk > 1 and train_topk_accuracies and val_topk_accuracies:
            print(f"Epoch {epoch+1}: Train {train_loss:.4f}/{train_acc:.2f}% Top{topk} {train_topk_accuracies[-1]:.2f}%, "
                  f"Val {val_loss:.4f}/{val_acc:.2f}% Top{topk} {val_topk_accuracies[-1]:.2f}%")
        else:
            print(f"Epoch {epoch+1}: Train {train_loss:.4f}/{train_acc:.2f}%, Val {val_loss:.4f}/{val_acc:.2f}%")

    # save artifacts
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if save_best and best_state is not None:
            torch.save(best_state, os.path.join(output_dir, "best_model.pt"))
        if save_metrics:
            metrics = {
                "epochs": epochs,
                "lr": lr,
                "best_val_acc": best_val_acc,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies,
                "topk": topk if topk and topk > 1 else 1,
            }
            if topk and topk > 1:
                metrics["train_topk_accuracies"] = train_topk_accuracies
                metrics["val_topk_accuracies"] = val_topk_accuracies
            save_metrics_json(metrics, os.path.join(output_dir, "metrics.json"))
            rows = [
                {
                    "epoch": i + 1,
                    "train_loss": train_losses[i],
                    "val_loss": val_losses[i],
                    "train_acc": train_accuracies[i],
                    "val_acc": val_accuracies[i],
                    **({f"train_top{topk}_acc": train_topk_accuracies[i], f"val_top{topk}_acc": val_topk_accuracies[i]} if (topk and topk > 1) else {}),
                }
                for i in range(len(train_losses))
            ]
            save_metrics_csv(rows, os.path.join(output_dir, "metrics.csv"))
        if save_curves:
            plot_training_curves(
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                os.path.join(output_dir, "curves.png"),
                title="Prompt Learning Training Curves",
                topk=topk if (topk and topk > 1) else None,
                train_topk_accuracies=train_topk_accuracies if (topk and topk > 1) else None,
                val_topk_accuracies=val_topk_accuracies if (topk and topk > 1) else None,
            )

    return train_losses, train_accuracies, val_losses, val_accuracies


def test_prompt_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    output_dir: str,
    classes: List[str],
    topk: int = 1,
) -> None:

    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    topk_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            if topk and topk > 1:
                topk_correct += compute_topk_correct(logits, labels, topk)
                total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0
    topk_acc = None
    if topk and topk > 1:
        topk_acc = topk_correct / float(total)

    report = classification_report(all_labels, all_preds, target_names=classes)

    testing_dir = os.path.join(output_dir, "testing")
    os.makedirs(testing_dir, exist_ok=True)
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "auc_roc": auc,
        "num_classes": len(classes),
    }
    if topk_acc is not None:
        metrics[f"top{topk}_accuracy"] = topk_acc
    save_metrics_json(metrics, os.path.join(testing_dir, "metrics.json"))
    with open(os.path.join(testing_dir, "classification_report.txt"), "w") as f:
        f.write(report)


