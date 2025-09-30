import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, epochs, lr, model_name, device, output_dir):
    """Train a prompt learning model"""
    
    if model_name == "CoOp" or model_name == "MaPLe":
        trainable_params = model.prompt_learner.parameters()
    elif model_name == "CoCoOp":
        trainable_params = list(model.prompt_learner.meta_net.parameters()) + [model.prompt_learner.ctx]
    
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [{model_name}]')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Validation [{model_name}]'):
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{epochs} - {model_name}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(output_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            print(f'  ✓ Saved best model with val acc: {val_acc:.2f}%')
        
        scheduler.step()
    
    print(f'\n{model_name} - Best validation accuracy: {best_val_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }