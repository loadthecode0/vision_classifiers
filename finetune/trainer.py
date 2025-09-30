import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from finetune.results import save_metrics_json, save_metrics_csv, plot_training_curves

def finetune(model, train_loader, val_loader, device, epochs=5, lr=1e-4, model_type="imagenet", output_dir=None, save_best=True, save_curves=True, save_metrics=True, topk=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    train_topk_accuracies, val_topk_accuracies = [], []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
  
        # train

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        topk_correct = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if topk and topk > 1:
                _, topk_pred = outputs.topk(topk, dim=1)
                topk_correct += (topk_pred.eq(labels.view(-1, 1)).any(dim=1)).sum().item()
            
            postfix = {'Loss': f'{loss.item():.4f}', 'Acc': f'{100 * correct / total:.2f}%'}
            if topk and topk > 1:
                postfix[f'Top{topk}'] = f'{100 * topk_correct / total:.2f}%'
            progress_bar.set_postfix(postfix)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        if topk and topk > 1:
            train_topk_acc = 100 * topk_correct / total
            train_topk_accuracies.append(train_topk_acc)
        
        # val

        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        val_topk_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                if topk and topk > 1:
                    _, topk_pred = outputs.topk(topk, dim=1)
                    val_topk_correct += (topk_pred.eq(labels.view(-1, 1)).any(dim=1)).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        if topk and topk > 1:
            val_topk_acc = 100 * val_topk_correct / val_total
            val_topk_accuracies.append(val_topk_acc)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # log
        print(f'Epoch [{epoch+1}/{epochs}]:')
        if topk and topk > 1:
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Top{topk}: {train_topk_accuracies[-1]:.2f}%')
            print(f'  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%, Top{topk}: {val_topk_accuracies[-1]:.2f}%')
        else:
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%')
        print(f'  Best Val Acc: {best_val_acc:.2f}%')
        print('-' * 50)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nLoaded best model with validation accuracy: {best_val_acc:.2f}%')
    
    print(f"Output dir: {output_dir}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if save_best and best_model_state is not None:
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pt'))
        if save_metrics:
            metrics = {
                'model_type': model_type,
                'epochs': epochs,
                'lr': lr,
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'topk': topk if topk and topk > 1 else 1,
            }
            if topk and topk > 1:
                metrics['train_topk_accuracies'] = train_topk_accuracies
                metrics['val_topk_accuracies'] = val_topk_accuracies
            save_metrics_json(metrics, os.path.join(output_dir, 'metrics.json'))
            rows = [
                {
                    'epoch': i + 1,
                    'train_loss': train_losses[i],
                    'val_loss': val_losses[i],
                    'train_acc': train_accuracies[i],
                    'val_acc': val_accuracies[i],
                    **({f'train_top{topk}_acc': train_topk_accuracies[i], f'val_top{topk}_acc': val_topk_accuracies[i]} if (topk and topk > 1) else {})
                }
                for i in range(len(train_losses))
            ]
            save_metrics_csv(rows, os.path.join(output_dir, 'metrics.csv'))
        if save_curves:
            plot_training_curves(
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                os.path.join(output_dir, 'curves.png'),
                title=f'{model_type} Training Curves',
                topk=topk if (topk and topk > 1) else None,
                train_topk_accuracies=train_topk_accuracies if (topk and topk > 1) else None,
                val_topk_accuracies=val_topk_accuracies if (topk and topk > 1) else None,
            )
    
    return train_losses, train_accuracies, val_losses, val_accuracies
