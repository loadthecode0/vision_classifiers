import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
from clip_zero_shot.image_transforms import apply_clahe, center_and_pad_square, crop_black_border

class PacemakerDataset(Dataset):
    def __init__(self, root_dir, split='train', fine_transforms=False, transform=None, collapse_to_manufacturer=False):
        self.transform = transform
        self.fine_transforms = fine_transforms
        self.samples = []
        self.collapse_to_manufacturer = collapse_to_manufacturer

        data_dir = Path(root_dir) / ('Train' if split != 'test' else 'Test')

        for model_dir in data_dir.iterdir():
            if model_dir.is_dir():
                for img_file in list(model_dir.glob('*.JPG')) + list(model_dir.glob('*.jpg')) + list(model_dir.glob('*.png')):
                    label = model_dir.name
                    if self.collapse_to_manufacturer:
                        # Extract manufacturer prefix ("BIO", "BOS", "MDT", "SOR", "STJ")
                        if " " in label:
                            label = label.split()[0]
                        elif "-" in label:
                            label = label.split("-")[0]
                        label = label.strip().upper()
                    self.samples.append({'path': str(img_file), 'label': label})

        # Create label mapping
        self.classes = sorted(list(set(s['label'] for s in self.samples)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Train/val split
        if split in ['train', 'val']:
            labels = [s['label'] for s in self.samples]
            train_samples, val_samples = train_test_split(
                self.samples, test_size=0.2, random_state=42, stratify=labels
            )
            self.samples = train_samples if split == 'train' else val_samples

        print(f"{split.upper()}: {len(self.samples)} images, {self.num_classes} classes (collapse={self.collapse_to_manufacturer})")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')

        if self.fine_transforms:
            # image = crop_black_border(image)
            # image = center_and_pad_square(image)
            image = apply_clahe(image)
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.class_to_idx[sample['label']]
        return image, label_idx


# Standard transforms
def get_transforms(augment=False, resize_dim=224):
    if augment:
        return transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# Create dataloaders
def create_dataloaders(dataset_dir, model_type, batch_size=16, collapse_to_manufacturer=False):
    if model_type.lower() in ["imagenet", "clip"]:
        train_dataset = PacemakerDataset(dataset_dir, 'train', True, get_transforms(augment=True, resize_dim=224), collapse_to_manufacturer)
        val_dataset = PacemakerDataset(dataset_dir, 'val', True, get_transforms(augment=False, resize_dim=224), collapse_to_manufacturer)
        test_dataset = PacemakerDataset(dataset_dir, 'test', True, get_transforms(augment=False, resize_dim=224), collapse_to_manufacturer)
    elif model_type.lower() == "dinov2":
        train_dataset = PacemakerDataset(dataset_dir, 'train', True, get_transforms(augment=True, resize_dim=518), collapse_to_manufacturer)
        val_dataset = PacemakerDataset(dataset_dir, 'val', True, get_transforms(augment=False, resize_dim=518), collapse_to_manufacturer)
        test_dataset = PacemakerDataset(dataset_dir, 'test', True, get_transforms(augment=False, resize_dim=518), collapse_to_manufacturer)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    print(f"\nReady: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    print(f"Classes: {train_dataset.num_classes} -> {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes, train_dataset.num_classes


