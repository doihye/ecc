from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .vision_transformer import VisionTransformer
from diffusers import DDPMScheduler

class ScaledDDPMScheduler(DDPMScheduler):
    def __init__(self, factor=1.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adjust_betas(factor)

    def _adjust_betas(self, factor):
        self.betas = self.betas ** factor

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def noise_sampling(self, x, timesteps=None):
        bs = x.shape[0]
        noise = torch.randn(x.shape, device=x.device)
        if timesteps == None:
            timesteps = torch.randint(0, self.config.num_train_timesteps, (bs,), device=x.device).long()
        samples = self.add_noise(x, noise, timesteps)
        return samples

class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)
        
        self.scheduler = ScaledDDPMScheduler(
            factor=1.2, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask, noise_block):
        noise_x = x.clone()
        x = self.patch_embed(x)
        
        assert mask is not None
        B, L, _ = x.shape
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=x.device).long()
        t = self.time_embed(timesteps, L+1)
        
        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        for idx, blk in enumerate(self.blocks, noise_block):
            if idx == noise_block:
                x = x + t
                x = self.scheduler.noise_sampling(x, timesteps)
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        noise_x = self.scheduler.noise_sampling(noise_x, timesteps)
        return x, noise_x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask, mae_loss_coef=1.0):
        z, noise_x = self.encoder(x, mask, noise_block=2)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(noise_x, x_rec, reduction='none')
        loss_recon = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        
        visible = 1 - mask
        loss_denoise = F.l1_loss(x, x_rec, reduction='none')
        loss_denoise = (loss_recon * visible).sum() / (visible.sum() + 1e-5) / self.in_chans
        loss = mae_loss_coef * loss_recon + loss_denoise
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

def build_simmim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    return model
