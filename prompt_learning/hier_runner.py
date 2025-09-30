import torch
import os
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from prompt_learning.utils import Config
from prompt_learning.coop import CoOp
from prompt_learning.cocoop import CoCoOp
from prompt_learning.maple import MaPLe
from prompt_learning.trainer import train_prompt_model, test_prompt_model
from utils.pacemakers_data import create_dataloaders as pacemakers_dataloaders
from prompt_learning.maple import load_maple_clip
from config import config

class HierarchicalPromptLearningRunner:
    def __init__(self, method="coop", model_type="clip"):
        self.method = method.lower()
        self.dataset = "pacemakers"
        self.config = Config(self.method, self.dataset)
        self.model_type = model_type
        self.backbone = "ViT-B/32"
        self.output_dir = f"./results/{self.dataset}/{self.method}/{self.model_type}/hierarchical"
        self.stage1_dir = os.path.join(self.output_dir, "stage1")
        self.stage2_dir = os.path.join(self.output_dir, "stage2")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = config[self.method][self.dataset]["epochs"]
        self.lr = config[self.method][self.dataset]["lr"]
        self.batch_size = config[self.method][self.dataset]["batch_size"]
        self.topk = 3  # for pacemakers

        # tokenizer + clip backbone
        self._tokenizer = _Tokenizer()

        if self.method.lower() == "maple":
            self.clip_model, self.preprocess = load_maple_clip(self.backbone, self.device, n_ctx=self.config.n_ctx)
        else:
            self.clip_model, self.preprocess = clip.load(self.backbone, device=self.device)
        
        # self.clip_model, self.preprocess = clip.load(self.backbone, device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.float()

        self.train_loader_stage1, self.val_loader_stage1, self.test_loader_stage1, \
            self.classes_stage1, self.num_classes_stage1 = pacemakers_dataloaders(
                "./pacemakers", self.model_type, self.batch_size, collapse_to_manufacturer=True
            )

        self.train_loader_stage2, self.val_loader_stage2, self.test_loader_stage2, \
            self.classes_stage2, self.num_classes_stage2 = pacemakers_dataloaders(
                "./pacemakers", self.model_type, self.batch_size, collapse_to_manufacturer=False
            )

        # pick model class
        if self.method == "coop":
            self.model_class = CoOp
        elif self.method == "cocoop":
            self.model_class = CoCoOp
        elif self.method == "maple":
            self.model_class = MaPLe
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def run_stage1(self):
        """Train model on manufacturer-level classification (5 classes)."""
        print("Train model on manufacturer-level classification (5 classes).")
        model = self.model_class(self.config, self.classes_stage1, self.clip_model, self._tokenizer).to(self.device)
        # stage1_dir = os.path.join(self.output_dir, "stage1")
        os.makedirs(self.stage1_dir, exist_ok=True)

        train_prompt_model(
            model,
            self.train_loader_stage1,
            self.val_loader_stage1,
            self.device,
            epochs=self.epochs,
            lr=self.lr,
            output_dir=self.stage1_dir,
            topk=self.topk,
            method=self.method
        )
        return model, self.stage1_dir

    def run_stage2(self):
        """Fine-tune on fine-grained classification (45 classes) with Stage-1 init."""
        model = self.model_class(self.config, self.classes_stage2, self.clip_model, self._tokenizer).to(self.device)
        state_dict = torch.load(os.path.join(self.stage1_dir, "best_model.pt"), map_location=self.device)

        pretrained_prompts = {
            k: v for k, v in state_dict.items()
            if "prompt_learner" in k and not any(x in k for x in ["token_prefix", "token_suffix"])
        }

        model_dict = model.state_dict()
        model_dict.update(pretrained_prompts)
        model.load_state_dict(model_dict)

        # stage2_dir = os.path.join(self.output_dir, "stage2")
        os.makedirs(self.stage2_dir, exist_ok=True)

        train_prompt_model(
            model,
            self.train_loader_stage2,
            self.val_loader_stage2,
            self.device,
            epochs=self.epochs,
            lr=self.lr,
            output_dir=self.stage2_dir,
            topk=self.topk,
            method=self.method
        )

        return model, self.stage2_dir



    def run_testing(self):
        state_dict = torch.load(os.path.join(self.stage2_dir, "best_model.pt"), map_location=self.device)
        model = self.model_class(self.config, self.classes_stage2, self.clip_model, self._tokenizer).to(self.device)
        model.load_state_dict(state_dict)

        test_prompt_model(
            model,
            test_loader=self.test_loader_stage2,
            device=self.device,
            output_dir=self.stage2_dir,
            classes=self.classes_stage2,
            topk=self.topk
        )

    def run_testing_stage1(self):
        state_dict = torch.load(os.path.join(self.stage1_dir, "best_model.pt"), map_location=self.device)
        model = self.model_class(self.config, self.classes_stage1, self.clip_model, self._tokenizer).to(self.device)
        model.load_state_dict(state_dict)

        test_prompt_model(
            model,
            test_loader=self.test_loader_stage1,
            device=self.device,
            output_dir=self.stage1_dir,
            classes=self.classes_stage1,
            topk=self.topk
        )
