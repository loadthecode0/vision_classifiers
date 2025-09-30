import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from prompt_learning.utils import Config
from prompt_learning.trainer import train_prompt_model, test_prompt_model
from utils.models import ViTClassifier
from utils.orthonet_data import create_dataloaders as orthonet_dataloaders
from utils.pacemakers_data import create_dataloaders as pacemakers_dataloaders
import torch
import os
from prompt_learning.coop import CoOp
from prompt_learning.cocoop import CoCoOp
from prompt_learning.maple import MaPLe
from prompt_learning.maple import load_maple_clip
from config import config


class PromptLearningRunner:

    def __init__(self, method = "coop", dataset = "orthonet", model_type: str = "clip"):
        self.method = method
        self.dataset = dataset
        self.config = Config(method, dataset)
        self.model_type = model_type
        self.backbone = "ViT-B/32"
        self.output_dir = f"./results/{self.dataset}/{self.method}/{self.model_type}"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = config[self.method][self.dataset]["epochs"]
        self.lr = config[self.method][self.dataset]["lr"]
        self.batch_size = config[self.method][self.dataset]["batch_size"]
        self.topk = 1
        self._tokenizer = _Tokenizer()

        # load and freeze base clip model

        if self.method.lower() == "maple":
            self.clip_model, self.preprocess = load_maple_clip(self.backbone, self.device, n_ctx=self.config.n_ctx)
        else:
            self.clip_model, self.preprocess = clip.load(self.backbone, device=self.device)
        
        # self.clip_model, self.preprocess = clip.load(self.backbone, device=self.device)
        self.clip_model = self.clip_model.to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.float()
        print("CLIP model loaded and frozen")

        # get datasets and dataloaders
        if self.dataset == "orthonet":
            create_dataloaders = orthonet_dataloaders
        elif self.dataset == "pacemakers":
            create_dataloaders = pacemakers_dataloaders
            self.topk = 3

        self.train_loader, self.val_loader, self.test_loader, self.classes, self.num_classes = create_dataloaders("./" + self.dataset, self.model_type, self.batch_size)

        # get prompt learner model
        if self.method.lower() == "coop":
            self.model_class = CoOp
        elif self.method.lower() == "cocoop":
            self.model_class = CoCoOp
        elif self.method.lower() == "maple":
            self.model_class = MaPLe

        self.model = self.model_class(self.config, self.classes, self.clip_model, self._tokenizer).to(self.device)
        

    def run_training(self):
    
        os.makedirs(os.path.join(self.output_dir, 'training'), exist_ok=True)
        return train_prompt_model(
            self.model,
            self.train_loader,
            self.val_loader,
            self.device,
            epochs=self.epochs,
            lr=self.lr,
            output_dir=os.path.join(self.output_dir, 'training'),
            topk=self.topk,
            method=self.method
        )

    def run_testing(self):
        model = self.model_class(self.config, self.classes, self.clip_model, self._tokenizer).to(self.device)
        state_dict = torch.load(os.path.join(self.output_dir, 'training', 'best_model.pt'), map_location=self.device)
        model.load_state_dict(state_dict)
        test_prompt_model(
            model,
            test_loader=self.test_loader,
            device=self.device,
            output_dir=self.output_dir,
            classes=self.classes,
            topk=self.topk,
        )
