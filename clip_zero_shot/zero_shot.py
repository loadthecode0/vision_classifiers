import clip
import torch
import numpy as np
import torch.nn.functional as F

class CLIPZeroShot:
    """CLIP Zero-Shot Classification Pipeline"""
    
    def __init__(self, model_name='ViT-B/32', device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
    def create_text_prompts(self, class_names, template="an X-ray of a {} implant"):
        """Create text prompts for each class"""
        prompts = []
        for class_name in class_names:
            # Clean class name and create prompt
            clean_name = class_name.replace('_', ' ').lower()
            prompt = template.format(clean_name)
            prompts.append(prompt)
        
        return prompts
    
    def encode_text_prompts(self, prompts):
        """Encode text prompts into embeddings"""
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
            
        return text_features
    
    def predict(self, dataloader, text_features):
        """Perform zero-shot prediction on test data"""
        all_predictions = []
        all_labels = []
        all_similarities = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                
                # Encode images
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, p=2, dim=1)
                
                # Compute similarities
                similarities = torch.matmul(image_features, text_features.T)
                
                # Get predictions
                predictions = similarities.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_similarities.append(similarities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.vstack(all_similarities)
