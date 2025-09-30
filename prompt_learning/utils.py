import torch
import torch.nn as nn
from config import config

class Config:
    def __init__(self, method, dataset):
        self.n_ctx = config[method][dataset]["n_ctx"]
        self.ctx_init = config[method][dataset]["ctx_init"]
        self.csc = config[method][dataset]["csc"]  # class-specific context
        self.class_token_position = config[method][dataset]["class_token_position"]
        self.input_size = (224, 224)

        if method.lower() == "maple":
            self.prompt_depth = config[method][dataset]["prompt_depth"]
            self.use_vision_residual = config[method][dataset]["use_vision_residual"]
            self.use_meta_net = config[method][dataset]["use_meta_net"]

    def get(self, key, default=None):
        """Allow dict-like access to config attributes."""
        return getattr(self, key, default)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Take features from the eot embedding (eot_token is the highest number in each sequence) 
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

# Configuration class for ease of use
class MaPLeConfig:
    def __init__(self, n_ctx=4, ctx_init="", compound_prompts_depth=9, use_meta_net=False, use_vision_residual=True):
        self.n_ctx = n_ctx  # number of learnable tokens
        self.ctx_init = ctx_init  # initialization string (if any)
        self.compound_prompts_depth = compound_prompts_depth  # number of layers to prompt
        self.use_meta_net = use_meta_net  # whether to use image conditioning
        self.use_vision_residual = use_vision_residual  # whether to add vision-specific residuals
        
    def get(self, key, default=None):
        """Allow dict-like access to config attributes."""
        return getattr(self, key, default)



class MaPLeTextEncoder(nn.Module):
    """Text encoder with deep prompt insertion"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def forward(self, prompts, tokenized_prompts, compound_prompts_depth):
        """
        prompts: list of text prompts for each layer
        tokenized_prompts: tokenized prompt indices
        """
        # Take the first layer's prompt as input
        # print(prompts)
        # print(prompts[0])
        print(prompts[0].shape)
        x = prompts[0] + self.positional_embedding.type(self.dtype)
        print(x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        print(x.shape)
        
        # Forward through transformer with deep prompting
        for i, block in enumerate(self.transformer.resblocks):
            if i < compound_prompts_depth:
                # Replace context tokens with learned prompts at this layer
                # This is the key to deep prompting
                x[:prompts[i].shape[1]] = prompts[i].permute(1, 0, 2)
            x = block(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Take features from [EOS] token
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

class MaPLeVisionEncoder(nn.Module):
    """Vision encoder with deep prompt insertion"""
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        
    def forward(self, x, vision_prompts, compound_prompts_depth):
        """
        x: image tensor
        vision_prompts: prompts derived from text prompts (the coupling!)
        """
        # Initial vision encoding
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # Add class embedding
        class_embedding = self.visual.class_embedding.to(x.dtype) + \
                         torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_embedding, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)

        # Determine the number of prompt tokens
        if vision_prompts.dim() == 4:  # batched (batch, n_layers, n_ctx, vis_dim)
            batch_size = x.shape[0]
            n_prompts = vision_prompts.shape[2]  # n_ctx
            # Expand first layer prompts to match batch size if needed
            prompt_tokens = vision_prompts[:, 0]  # (batch, n_ctx, vis_dim)
            if prompt_tokens.shape[0] == 1 and batch_size > 1:
                prompt_tokens = prompt_tokens.expand(batch_size, -1, -1)
        else:  # not batched (n_layers, n_ctx, vis_dim)
            batch_size = x.shape[0]
            n_prompts = vision_prompts.shape[1]  # n_ctx
            # Expand to match batch size
            prompt_tokens = vision_prompts[0].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add prompt tokens to the sequence
        x = torch.cat([prompt_tokens, x], dim=1)
        
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Forward through transformer with deep prompting
        for i, block in enumerate(self.visual.transformer.resblocks):
            if i < compound_prompts_depth:
                # Update prompt tokens at this layer with text-derived prompts
                if vision_prompts.dim() == 4:
                    # Get prompts for this layer
                    layer_prompts = vision_prompts[:, i]  # (batch, n_ctx, vis_dim)
                    if layer_prompts.shape[0] == 1 and batch_size > 1:
                        layer_prompts = layer_prompts.expand(batch_size, -1, -1)
                    x[:n_prompts] = layer_prompts.permute(1, 0, 2)  # (n_ctx, batch, vis_dim)
                else:
                    layer_prompts = vision_prompts[i].unsqueeze(0).expand(batch_size, -1, -1)
                    x[:n_prompts] = layer_prompts.permute(1, 0, 2)
            x = block(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Extract class token (skipping prompt tokens)
        x = self.visual.ln_post(x[:, n_prompts, :])
        
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        
        return x
