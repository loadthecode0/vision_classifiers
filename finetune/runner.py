from utils.orthonet_data import create_dataloaders as orthonet_dataloaders
from utils.pacemakers_data import create_dataloaders as pacemakers_dataloaders
from finetune.finetune_utils import finetune_classifier, test_classifier
from utils.models import ViTClassifier
import torch
from config import config

class FinetuneRunner:

    def __init__(self, dataset = "orthonet", model_type="imagenet", batch_size=16):
        self.dataset = dataset
        self.dataset_dir = f"./{dataset}"
        self.model_type = model_type
        self.method = "finetune"
        self.output_dir = f"./results/{self.dataset}/{self.method}/{self.model_type}"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = config["finetune"][self.dataset][self.model_type]["epochs"]
        self.lr = config["finetune"][self.dataset][self.model_type]["lr"]
        self.batch_size = config["finetune"][self.dataset][self.model_type]["batch_size"]
        self.topk=1

        if dataset == "orthonet":
            create_dataloaders = orthonet_dataloaders
        elif dataset == "pacemakers":
            create_dataloaders = pacemakers_dataloaders
            self.topk=3

        self.train_loader, self.val_loader, self.test_loader, self.classes, self.num_classes = \
            create_dataloaders(self.dataset_dir, self.model_type, self.batch_size)
        
        self.train_fn = finetune_classifier
        self.test_fn = test_classifier

    def run_training(self):
        
        print(self.epochs, self.lr, self.batch_size)
        _, self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies = \
            self.train_fn(
                self.model_type, 
                self.dataset_dir, 
                self.epochs, 
                self.lr, 
                self.train_loader, 
                self.val_loader, 
                self.num_classes, 
                self.classes, 
                self.device, 
                self.output_dir,
                self.topk
            )

    def run_testing(self):
        
        model = ViTClassifier(self.model_type, self.num_classes).to(self.device)
        state_dict = torch.load(f"{self.output_dir}/training/best_model.pt", map_location=self.device)
        model.load_state_dict(state_dict)
        self.test_fn(model, self.model_type, self.test_loader, self.classes, self.device, self.output_dir, self.topk)
