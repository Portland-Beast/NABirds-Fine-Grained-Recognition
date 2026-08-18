from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader

from birdwatching.dataset import BirdsDataset
from birdwatching.losses import BBoxLoss
from birdwatching.model import MODEL_NAMES, build_model, default_checkpoint_path
from birdwatching.paths import resolve_data_root


class ModelEMA:

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict:
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()  
                p.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


N_CLASSES = 555
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=(num_workers > 0),
                      persistent_workers=False)


def run_epoch(model, loader, optimizer, ce_loss, bbox_loss, bbox_weight,
              training: bool, ema: ModelEMA = None,
              use_cutmix: bool = False, cutmix_bbox_loss: bool = True):
    model.train(training)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels, bboxes, _ in loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            bboxes = bboxes.to(DEVICE, non_blocking=True)

            cls_out, bbox_out = model(imgs)

            if training and use_cutmix and random.random() < 0.5:   #apply cutmix if the random number is less than 0.5
                lam = float(np.random.beta(1.0, 1.0))
                idx = torch.randperm(imgs.size(0), device=imgs.device)
                W, H = imgs.size(3), imgs.size(2)
                cw = int(W * (1.0 - lam) ** 0.5); ch = int(H * (1.0 - lam) ** 0.5)   #
                cx = random.randint(0, W); cy = random.randint(0, H)
                x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
                y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
                imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
                lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
                cls_out, bbox_out = model(imgs)
                cls_loss = lam * ce_loss(cls_out, labels) + (1 - lam) * ce_loss(cls_out, labels[idx])
                loss = cls_loss + bbox_weight * bbox_loss(bbox_out, bboxes) if cutmix_bbox_loss else cls_loss
            else:
                cls_loss = ce_loss(cls_out, labels)
                loss = cls_loss + bbox_weight * bbox_loss(bbox_out, bboxes)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)   #update the EMA model

            total_loss += loss.item()
            correct += (cls_out.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def make_optimizer(model, lr_head, lr_backbone, weight_decay):    
    head_ids, head_params = set(), []
    for attr in ('classifier', 'bbox_head', 'part_block', 'part_norm'):
        mod = getattr(model, attr, None)
        if mod is not None:
            for p in mod.parameters():
                if id(p) not in head_ids:
                    head_params.append(p); head_ids.add(id(p))
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]  #get the backbone parameters
    return torch.optim.AdamW(
        [{'params': backbone_params, 'lr': lr_backbone},
         {'params': head_params,     'lr': lr_head}],
        weight_decay=weight_decay,
    )


def make_scheduler(args, optimizer):
    warmup_cfg = min(2, args.epochs)
    done = max(args.start_epoch - 1, 0)
    warm_remain = max(0, warmup_cfg - min(done, warmup_cfg))
    cos_remain  = max(args.epochs - warmup_cfg, 1) - max(0, done - warmup_cfg)  #calculate the number of epochs remaining

    if warm_remain > 0:
        w = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_remain)
        sched = SequentialLR(optimizer, [w, CosineAnnealingLR(optimizer, T_max=max(cos_remain,1), eta_min=1e-6)],
                             milestones=[warm_remain]) if cos_remain > 0 else w
    elif cos_remain > 0:
        sched = CosineAnnealingLR(optimizer, T_max=max(cos_remain, 1), eta_min=1e-6)
    else:
        return None
    return sched


def train(args):
    data_root = resolve_data_root(args.data_root)
    print(f"Device       : {DEVICE}")
    print(f"Model        : {args.model}  |  img_size={args.img_size}  batch={args.batch_size}")
    print(f"Loss (bbox)  : {args.loss}  bbox_weight={args.bbox_weight}")
    print(f"Label smooth : {args.label_smoothing}  CutMix: {args.cutmix_mode}")
    print(f"Epochs       : {args.epochs}  finetune={args.finetune_epochs}")

    cutmix_on = args.cutmix_mode in ('current', 'cls_only')
    cutmix_bbox = args.cutmix_mode != 'cls_only'

    train_ds = BirdsDataset(data_root, 'train', args.img_size, augment=True,  aug_strength=args.aug_strength)
    val_ds   = BirdsDataset(data_root, 'val',   args.img_size, augment=False)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, N_CLASSES, img_size=args.img_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {n_params:,}  ({n_params/1e6:.2f} M)")
    if n_params > 10_000_000:
        print("  WARNING: exceeds 10 M parameter cap!")

    save_path  = args.save_path  or default_checkpoint_path(args.model, 'best')
    final_path = args.final_path or default_checkpoint_path(args.model, 'final')

    best_val_acc = 0.0
    if args.start_epoch > 1 and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        print(f"Resumed from {save_path}")
        best_val_acc = -1.0

    ce_loss   = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    bbox_loss = BBoxLoss(args.loss)
    optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
    scheduler = make_scheduler(args, optimizer)
    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    if ema:
        print(f"EMA          : decay={args.ema_decay}")

    no_improve = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        run_epoch(model, train_loader, optimizer, ce_loss, bbox_loss,
                  args.bbox_weight, training=True, ema=ema,
                  use_cutmix=cutmix_on, cutmix_bbox_loss=cutmix_bbox)

        _, val_acc = run_epoch(model, val_loader, optimizer, ce_loss, bbox_loss,
                               args.bbox_weight, training=False)

        improved = False
        marker = ''
        if ema:
            backup = ema.apply_to(model)
            _, ema_acc = run_epoch(model, val_loader, optimizer, ce_loss, bbox_loss,
                                   args.bbox_weight, training=False)
            if ema_acc > val_acc and ema_acc > best_val_acc:  #save the model if the EMA accuracy is greater than the validation 
                torch.save(model.state_dict(), save_path)
                best_val_acc = ema_acc; improved = True; marker = '  *** saved (EMA) ***'
            ema.restore(model, backup)
            if not improved and val_acc > best_val_acc:
                best_val_acc = val_acc; torch.save(model.state_dict(), save_path)
                improved = True; marker = '  *** saved ***'
            display_acc = ema_acc
        else:
            if val_acc > best_val_acc:
                best_val_acc = val_acc; torch.save(model.state_dict(), save_path)
                improved = True; marker = '  *** saved ***'
            display_acc = val_acc

        if scheduler:
            scheduler.step()

        no_improve = 0 if improved else no_improve + 1
        es_str = f' [no-improve {no_improve}/{args.patience}]' if args.patience > 0 else ''
        print(f"Epoch {epoch:3d}/{args.epochs} | val acc {display_acc:.4f}{marker}{es_str}")

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nBest val acc: {best_val_acc:.4f}")

    if args.finetune_epochs > 0:   #fine-tune the model on the train+val dataset
        print(f"\nFine-tuning {args.finetune_epochs} epochs on train+val …")
        combined = ConcatDataset([
            BirdsDataset(data_root, 'train', args.img_size, augment=True, aug_strength=args.aug_strength),
            BirdsDataset(data_root, 'val',   args.img_size, augment=True, aug_strength=args.aug_strength),
        ])
        comb_loader = make_loader(combined, args.batch_size, shuffle=True, num_workers=args.num_workers)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        opt2 = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
        ft_sched = CosineAnnealingLR(opt2, T_max=args.finetune_epochs, eta_min=1e-7)
        ft_ema = ModelEMA(model, args.ema_decay) if args.ema else None

        for epoch in range(1, args.finetune_epochs + 1):
            loss, acc = run_epoch(model, comb_loader, opt2, ce_loss, bbox_loss,
                                  args.bbox_weight, training=True, ema=ft_ema)
            ft_sched.step()
            print(f"  FT {epoch}/{args.finetune_epochs} | loss {loss:.4f} acc {acc:.4f}")

        if ft_ema:
            ft_ema.apply_to(model)
        torch.save(model.state_dict(), final_path)
        print(f"Final model -> {final_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',    choices=MODEL_NAMES, default='tinyvit_micro')
    p.add_argument('--loss',     choices=('smooth_l1', 'giou', 'l1+giou'), default='giou')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs',   type=int, default=12)
    p.add_argument('--start_epoch', type=int, default=1)
    p.add_argument('--lr_head',      type=float, default=1e-3)
    p.add_argument('--lr_backbone',  type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--bbox_weight',  type=float, default=2.0)
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--aug_strength', choices=('basic', 'strong'), default='strong')
    p.add_argument('--cutmix_mode',  choices=('none', 'current', 'cls_only'), default='none')
    p.add_argument('--cutmix',       action='store_true')
    p.add_argument('--ema',          action='store_true')
    p.add_argument('--ema_decay',    type=float, default=0.999)
    p.add_argument('--finetune_epochs', type=int, default=3)
    p.add_argument('--finetune_lr',     type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=0,
                   help='Early-stop after N epochs with no val improvement. 0=disabled.')
    p.add_argument('--save_path',  type=str, default=None)
    p.add_argument('--final_path', type=str, default=None)
    p.add_argument('--data_root',  type=str, default=None)
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
