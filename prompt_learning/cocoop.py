import torch
import torch.nn as nn
import clip
from prompt_learning.utils import TextEncoder

class ConditionalPromptLearner(nn.Module):
    """Conditional prompt learner for CoCoOp"""
    def __init__(self, cfg, classnames, clip_model, _tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        
        self.meta_net = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 16),
            nn.ReLU(),
            nn.Linear(vis_dim // 16, ctx_dim * n_ctx)
        ).type(dtype)
        
        if cfg.ctx_init:
            ctx_init = cfg.ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()  # Move to CUDA
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.ctx = nn.Parameter(ctx_vectors)
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # Move to CUDA
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        
    def forward(self, im_features):
        batch_size = im_features.shape[0]
        ctx_shift = self.meta_net(im_features)
        ctx_shift = ctx_shift.view(batch_size, self.n_ctx, -1)
        
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)
        ctx = ctx + ctx_shift
        
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ctx_expanded = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        
        prompts = torch.cat([prefix, ctx_expanded, suffix], dim=2)
        
        return prompts

class CoCoOp(nn.Module):
    """CoCoOp model with conditional prompts"""
    def __init__(self, cfg, classnames, clip_model, _tokenizer):
        super().__init__()
        self.prompt_learner = ConditionalPromptLearner(cfg, classnames, clip_model, _tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        prompts = self.prompt_learner(image_features)
        
        batch_size = prompts.shape[0]
        n_cls = prompts.shape[1]
        
        text_features = []
        for i in range(batch_size):
            text_features_i = self.text_encoder(prompts[i], tokenized_prompts)
            text_features_i = text_features_i / text_features_i.norm(dim=-1, keepdim=True)
            text_features.append(text_features_i)
        
        text_features = torch.stack(text_features)
        
        logits = logit_scale * torch.bmm(image_features.unsqueeze(1), text_features.transpose(1, 2))
        logits = logits.squeeze(1)
        
        return logits