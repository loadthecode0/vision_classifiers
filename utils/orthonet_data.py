import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, CLIPVisionModel
import timm
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from clip_zero_shot.image_transforms import *

class OrthonetKaggleDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, use_masks=False, fine_transforms=False, limit_rows=None):
        """
        Dataset class for Orthonet data on Kaggle
        csv_file: path to CSV file (train.csv or test.csv)
        image_dir: path to directory containing images
        use_masks: whether to use mask information for preprocessing
        """
        self.df = pd.read_csv(csv_file)
        if limit_rows is not None:
            self.df = self.df.head(limit_rows)
        self.image_dir = image_dir
        self.transform = transform
        self.use_masks = use_masks
        self.fine_transforms = fine_transforms
        
        # Get unique labels and create mapping
        self.classes = sorted(self.df['labels'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Print dataset statistics
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Dataset size: {len(self.df)}")
        
        # Check mask availability if using masks
        if 'valid_mask' in self.df.columns and self.use_masks:
            mask_count = self.df['valid_mask'].sum()
            print(f"Images with valid masks: {mask_count}/{len(self.df)} ({mask_count/len(self.df)*100:.1f}%)")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filenames']
        label = row['labels']
        
        # Load image
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')

        if self.fine_transforms:
            # Preprocessing steps
            # image = crop_black_border(image)
            # image = center_and_pad_square(image)
            image = apply_clahe(image)
            image = image.convert('RGB')
            
        
        # Optionally apply mask-based preprocessing
        if self.use_masks and 'masks' in row and 'valid_mask' in row and row['valid_mask']:
            try:
                mask_filename = row['masks']
                mask_path = os.path.join(self.image_dir, mask_filename)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')  # Load as grayscale
                    # Apply mask to focus on implant region (optional enhancement)
                    image = self._apply_mask_enhancement(image, mask)
            except:
                # If mask loading fails, continue with original image
                pass
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.class_to_idx[label]
        return image, label_idx
    
    def _apply_mask_enhancement(self, image, mask):
        """
        Optional: Apply mask-based enhancement to focus on implant region
        This is a simple implementation - you can make it more sophisticated
        """
        import numpy as np
        
        # Convert to numpy for processing
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Normalize mask to 0-1
        mask_array = mask_array / 255.0
        
        # Create a soft mask to avoid hard edges
        # This gives more weight to masked regions while keeping some background
        enhanced_img = img_array * (0.3 + 0.7 * mask_array[:, :, np.newaxis])
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced_img)

class OrthonetKaggleDatasetForDINOv2(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, use_masks=False, limit_rows=None):
        """
        Dataset class for Orthonet data on Kaggle
        csv_file: path to CSV file (train.csv or test.csv)
        image_dir: path to directory containing images
        use_masks: whether to use mask information for preprocessing
        """
        self.df = pd.read_csv(csv_file)
        if limit_rows is not None:
            self.df = self.df.head(limit_rows)
        self.image_dir = image_dir
        self.transform = transform
        self.use_masks = use_masks
        
        # Get unique labels and create mapping
        self.classes = sorted(self.df['labels'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Print dataset statistics
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Dataset size: {len(self.df)}")
        
        # Check mask availability if using masks
        if 'valid_mask' in self.df.columns and self.use_masks:
            mask_count = self.df['valid_mask'].sum()
            print(f"Images with valid masks: {mask_count}/{len(self.df)} ({mask_count/len(self.df)*100:.1f}%)")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filenames']
        label = row['labels']
        
        # Load image
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        # Optionally apply mask-based preprocessing
        if self.use_masks and 'masks' in row and 'valid_mask' in row and row['valid_mask']:
            try:
                mask_filename = row['masks']
                mask_path = os.path.join(self.image_dir, mask_filename)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')  # Load as grayscale
                    # Apply mask to focus on implant region (optional enhancement)
                    image = self._apply_mask_enhancement(image, mask)
            except:
                # If mask loading fails, continue with original image
                pass
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.class_to_idx[label]
        return image, label_idx
    
    def _apply_mask_enhancement(self, image, mask):
        """
        Optional: Apply mask-based enhancement to focus on implant region
        This is a simple implementation - you can make it more sophisticated
        """
        import numpy as np
        
        # Convert to numpy for processing
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Normalize mask to 0-1
        mask_array = mask_array / 255.0
        
        # Create a soft mask to avoid hard edges
        # This gives more weight to masked regions while keeping some background
        enhanced_img = img_array * (0.3 + 0.7 * mask_array[:, :, np.newaxis])
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced_img)

def get_transforms(resize_dim=224, model_type="clip"):
    # Data transforms
    if model_type.lower() in ["imagenet", "dinov2"]:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    if model_type.lower()=="clip":
        mean=[0.48145466, 0.4578275, 0.40821073]
        std=[0.26862954, 0.26130258, 0.27577711]

    transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform
# Optional: More advanced transforms for medical images
def get_adv_transforms(resize_dim=224,model_type="clip"):

    if model_type.lower() in ["imagenet", "dinov2"]:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    if model_type.lower()=="clip":
        mean=[0.48145466, 0.4578275, 0.40821073]
        std=[0.26862954, 0.26130258, 0.27577711]

    medical_transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.RandomRotation(degrees=10),  # Small rotations for medical images
        transforms.RandomHorizontalFlip(p=0.5),  # Careful with medical images
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust for X-ray variations
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return medical_transform

def create_dataloaders(dataset_dir, model_type, batch_size=16, test_split=0.2, random_state=42, limit_rows=None):  
    
    # set this to override:
    # dataset_dir = 

    train_df = pd.read_csv(dataset_dir + '/train.csv')

    # Split training data into train/validation sets (80/20 split)
    train_split, val_split = train_test_split(
        train_df, 
        test_size=test_split, 
        random_state=random_state, 
        stratify=train_df['labels']  # Ensure balanced split across classes
        )

    print(f"Original training data: {len(train_df)} samples")
    print(f"Split into: {len(train_split)} train + {len(val_split)} validation")

    # Save temporary CSV files for the splits
    train_split.to_csv(dataset_dir+'/train_split.csv', index=False)
    val_split.to_csv(dataset_dir+'/val_split.csv', index=False)

    if model_type.lower() in ["imagenet", "clip"]:
        transform = get_transforms(224, model_type)
        dataset_class = OrthonetKaggleDataset
    elif model_type.lower() == "dinov2":
        transform = get_transforms(518, model_type)
        dataset_class = OrthonetKaggleDatasetForDINOv2

    # Create datasets
    train_dataset = dataset_class( dataset_dir+'/train_split.csv',
                                        dataset_dir + '/orthonet data/orthonet data', 
                                        transform=transform, 
                                        use_masks=False,
                                        limit_rows=limit_rows)

    val_dataset = dataset_class( dataset_dir+'/val_split.csv',
                                    dataset_dir + '/orthonet data/orthonet data/',
                                    transform=transform, 
                                    use_masks=False,
                                    limit_rows=limit_rows)

    test_dataset = dataset_class( dataset_dir + '/test.csv',
                                        dataset_dir+'/orthonet data/orthonet data/',
                                        transform=transform, 
                                        use_masks=False,
                                        limit_rows=limit_rows)  # Test data has no masks

    print("\n" + "="*50)
    print("DATASET CONFIGURATION")
    print("="*50)
    print("✓ Training data: Standard images (80% of original train)")
    print("✓ Validation data: Standard images (20% of original train)")  
    print("✓ Test data: Standard images (separate test set)")
    print("✓ Stratified split maintains class balance across train/val")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nSanity check: class_to_idx mappings")
    print("Train mapping:", train_dataset.class_to_idx)
    print("Val mapping:", val_dataset.class_to_idx)
    print("Test mapping:", test_dataset.class_to_idx)

    # Check for mismatches
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print("⚠️ Train/Val label mapping mismatch!")
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        print("⚠️ Train/Test label mapping mismatch!")


    return train_loader, val_loader, test_loader, train_dataset.classes, train_dataset.num_classes