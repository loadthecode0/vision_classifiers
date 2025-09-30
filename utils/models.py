import torch
import torch.nn as nn
from transformers import ViTForImageClassification, CLIPVisionModel
import timm

class ViTClassifier(nn.Module):
    def __init__(self, model_type, num_classes):
        super().__init__()
        self.model_type = model_type

        if model_type == 'imagenet':
            self.backbone = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k'
            )
            self.backbone.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

        elif model_type == 'clip':
            self.backbone = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16')
            self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

        elif model_type == 'dinov2':
            self.backbone = timm.create_model(
                'vit_small_patch14_dinov2.lvd142m',
                pretrained=True,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x): # want to return logits
        if self.model_type == 'imagenet':
            return self.backbone(x).logits

        elif self.model_type == 'clip':
            features = self.backbone(x).pooler_output
            return self.classifier(features)

        elif self.model_type == 'dinov2':
            return self.backbone(x)
