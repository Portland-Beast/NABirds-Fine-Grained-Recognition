from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torchvision.models as tvm

from birdwatching.paths import CHECKPOINTS_DIR


class BirdsDetector(nn.Module):

    def __init__(self, n_classes: int):   #MobileNet V2 model
        super().__init__()
        backbone = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = 1280
        self.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, n_classes))
        self.bbox_head  = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 4), nn.Sigmoid())

    def forward(self, x):
        feat = self.pool(self.features(x)).flatten(1)
        return self.classifier(feat), self.bbox_head(feat)


class EfficientNetDetector(nn.Module):   #EfficientNet B0 model

    def __init__(self, n_classes: int):
        super().__init__()
        backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = 1280
        self.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, n_classes))
        self.bbox_head  = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 4), nn.Sigmoid())

    def forward(self, x):
        feat = self.pool(self.features(x)).flatten(1)
        return self.classifier(feat), self.bbox_head(feat)


class _AttentionCapture(nn.Module):


    def __init__(self, base):
        super().__init__()
        self.base = base
        self.attn_weights = None

    def forward(self, x, attn_mask=None, is_causal=False):  
        a = self.base
        B, N, C = x.shape
        qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)
        q = q * a.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        self.attn_weights = attn
        attn = a.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, a.attn_dim)
        x = a.norm(x)
        return a.proj_drop(a.proj(x))  


class TinyViTPSMDetector(nn.Module):   #TinyViT PSM model


    def __init__(self, n_classes: int, img_size: int = 224, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required for tinyvit_psm") from e

        backbone = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained,
                                     num_classes=0, img_size=img_size)
        self.embed_dim = backbone.embed_dim
        self.num_heads = backbone.blocks[0].attn.num_heads

        for blk in backbone.blocks:
            blk.attn = _AttentionCapture(blk.attn)

        self.patch_embed = backbone.patch_embed
        self.cls_token   = backbone.cls_token
        self.pos_embed   = backbone.pos_embed
        self.pos_drop    = backbone.pos_drop
        self.blocks      = backbone.blocks

        part_block = copy.deepcopy(backbone.blocks[-1])
        part_block.attn = part_block.attn.base
        self.part_block = part_block
        self.part_norm  = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.classifier = nn.Linear(self.embed_dim, n_classes)
        self.bbox_head  = nn.Sequential(nn.Linear(self.embed_dim, 4), nn.Sigmoid())

        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def _embed(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1) 
        x = torch.cat([cls, x], dim=1) + self.pos_embed   #concatenate the class token and the position embedding
        return self.pos_drop(x)

    def forward(self, x):
        x = self._embed(x)
        attn_maps = []
        for blk in self.blocks:
            x = blk(x)
            attn_maps.append(blk.attn.attn_weights)

        rollout = attn_maps[0]
        for a in attn_maps[1:]:
            rollout = a @ rollout

        top_idx = rollout[:, :, 0, 1:].argmax(dim=2) + 1
        B = x.size(0)
        parts  = torch.stack([x[b, top_idx[b]] for b in range(B)], dim=0)   #stack the parts
        concat = torch.cat([x[:, :1], parts], dim=1)

        feat = self.part_norm(self.part_block(concat))[:, 0]
        return self.classifier(feat), self.bbox_head(feat)   


class TimmFeatureDetector(nn.Module):

    def __init__(self, backbone_name: str, n_classes: int,
                 pretrained: bool = True, img_size: int = 224):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required for TimmFeatureDetector") from e

        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                              num_classes=0, img_size=img_size)   #create the backbone model
        except TypeError:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                              num_classes=0)   #create the backbone model
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, n_classes))
        self.bbox_head  = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 4), nn.Sigmoid())
        nn.init.trunc_normal_(self.classifier[1].weight, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), self.bbox_head(feat)   #return the classifier and the bounding box head


MODEL_NAMES = ('mobilenet_v2', 'efficientnet_b0', 'tinyvit_psm',
               'tinyvit_micro', 'efficientformer_s1')


def build_model(name: str, n_classes: int, img_size: int = 224) -> nn.Module:
    name = name.lower()
    if name == 'mobilenet_v2':
        return BirdsDetector(n_classes)
    if name == 'efficientnet_b0':
        return EfficientNetDetector(n_classes)
    if name == 'tinyvit_psm':
        return TinyViTPSMDetector(n_classes, img_size=img_size)
    if name == 'tinyvit_micro':
        return TimmFeatureDetector('tiny_vit_5m_224.dist_in22k', n_classes, img_size=img_size)
    if name == 'efficientformer_s1':
        return TimmFeatureDetector('efficientformerv2_s1.snap_dist_in1k', n_classes, img_size=img_size)
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")


def default_checkpoint_path(name: str, suffix: str = 'best') -> str:
    return str(CHECKPOINTS_DIR / f"{suffix}_model_{name}.pth")
