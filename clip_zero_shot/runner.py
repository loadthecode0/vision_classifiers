import torch
import json
import os
from utils.orthonet_data import OrthonetKaggleDataset
from utils.pacemakers_data import PacemakerDataset, get_transforms
from clip_zero_shot.zero_shot import CLIPZeroShot
from clip_zero_shot.utils import calculate_metrics

from torch.utils.data import DataLoader, Dataset


class ZeroShotRunner:

    def __init__(self, dataset = "orthonet", model_name = "ViT-L/14@336px", batch_size=16):

        self.dataset = dataset
        self.dataset_dir = f"./{self.dataset}"
        self.model_name = model_name
        self.output_file = f"./results/{dataset}/zero_shot/testing/metrics.json"
        self.method = "zero_shot_clip"
        self.batch_size = batch_size
        self.topk = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_classifier = CLIPZeroShot(model_name=model_name, device=self.device)

        if dataset == "orthonet":
            
            self.HIP_TEMPLATES = [
                "an X-ray of a {} hip implant",
                # "a radiograph of a {} hip prosthesis",
                # "a medical X-ray showing a {} hip replacement",
                # "an orthopedic radiograph of {} in the hip",
                # "a {} hip implant seen in an X-ray",
            ]

            self.KNEE_TEMPLATES = [
                "an X-ray of a {} knee implant",
                # "a radiograph of a {} knee prosthesis",
                # "a medical X-ray showing a {} knee replacement",
                # "an orthopedic radiograph of {} in the knee joint",
                # "a {} knee implant seen in an X-ray",
            ]

            self.test_dataset = OrthonetKaggleDataset(self.dataset_dir+'/test.csv',
                                        self.dataset_dir + '/orthonet data/orthonet data', 
                                        transform=self.clip_classifier.preprocess,
                                        use_masks=False,
                                        fine_transforms = True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


            
        
        elif self.dataset == "pacemakers":
            self.TEMPLATES = [
                "an image of a {} implant"
            ]
            self.test_dataset = PacemakerDataset(
                self.dataset_dir,
                split='test',
                transform=self.clip_classifier.preprocess,
                fine_transforms=True,
            )
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
            self.topk = 3

        print(f"Number of test samples: {len(self.test_dataset)}")
        print(f"Number of classes: {len(self.test_dataset.classes)}")
        print(f"Classes: {self.test_dataset.classes}")

    def create_multi_prompt_features(self, clip_classifier, classes):
        """Generate averaged text features per class with hip/knee specific templates."""
        all_text_features = []
        
        for class_name in classes:
            clean_name = class_name.replace("_", " ")
            
            if self.dataset == "orthonet":
                if "Hip" in class_name:
                    templates = self.HIP_TEMPLATES
                elif "Knee" in class_name:
                    templates = self.KNEE_TEMPLATES
                else:
                    templates = ["an X-ray of {} implant"]  # fallback
            elif self.dataset == "pacemakers":
                templates = self.TEMPLATES
                
            # Create prompts
            class_prompts = [t.format(clean_name) for t in templates]

            print(class_prompts[:3])
            
            features = clip_classifier.encode_text_prompts(class_prompts)
            
            mean_feature = features.mean(dim=0)
            mean_feature = mean_feature / mean_feature.norm()  # normalize
            all_text_features.append(mean_feature)
        
        return torch.stack(all_text_features)

    def run(self):
        
        
        print("\nEncoding multi-template text prompts...")
        text_features = self.create_multi_prompt_features(
            self.clip_classifier, 
            self.test_dataset.classes
        )
        

        predictions, labels, similarities = self.clip_classifier.predict(self.test_loader, text_features)
        metrics = calculate_metrics(labels, predictions, similarities, len(self.test_dataset.classes), topk=self.topk)
        
        print("\n" + "="*50)
        print("CLIP Zero-Shot Classification Results")
        print("="*50)
        print(f"Model: {self.model_name}")
        # print(f"Template: '{self.template}'")
        print(f"Dataset: Orthonet ({len(self.test_dataset.classes)} classes)")
        print("-"*50)
        print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")

        if self.topk and self.topk > 1:
            print(f"Top-{self.topk} Accuracy: {metrics[f'top{self.topk}_accuracy']:.4f}")
        print(f"F1-Score (macro): {metrics['f1_score']:.4f}")
        print(f"AUC-ROC (macro): {metrics['auc_roc']:.4f}")
        print("="*50)
        
        
        # Save results
        results = {
            'model_name': self.model_name,
            # 'template': template,
            'dataset': 'Orthonet',
            'num_classes': len(self.test_dataset.classes),
            'classes': self.test_dataset.classes,
            'num_test_samples': len(self.test_dataset),
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'true_labels': labels.tolist()
        }
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {self.output_file}")
        
        # Per-class performance analysis
        print("\nPer-class accuracy:")
        from collections import Counter
        for class_idx, class_name in enumerate(self.test_dataset.classes):
            class_mask = labels == class_idx
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == labels[class_mask]).mean()
                print(f"  {class_name}: {class_acc:.4f} ({class_mask.sum()} samples)")
        
        # Confusion analysis
        print("\nTop confusion pairs:")
        confusion_pairs = Counter()
        for true_label, pred_label in zip(labels, predictions):
            if true_label != pred_label:
                true_class = self.test_dataset.classes[true_label]
                pred_class = self.test_dataset.classes[pred_label]
                confusion_pairs[(true_class, pred_class)] += 1
        
        for (true_class, pred_class), count in confusion_pairs.most_common(5):
            print(f"  {true_class} -> {pred_class}: {count} times")
