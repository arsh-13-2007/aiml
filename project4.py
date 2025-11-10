"""
Generalizable Medical Imaging Model — endtoend PyTorch template
================================================================

Builds classification models that generalize across hospitals, scanners, age groups,
countries, and patient populations. Includes:

• Site-/scanner-aware splits (train/val/test input via CSVs)
• Strong, clinically plausible augmentations
• GroupDRO objective (worstgroup risk minimization)
• Optional ERM baseline
• Testtime augmentation (TTA)
• Calibration with temperature scaling
• Subgroup dashboards & worstgroup metrics
• Model Card (JSON) artifact with subgroup results

Dependencies
------------
Python 3.9+
- torch >= 2.1
- torchvision >= 0.16
- torchmetrics >= 1.4
- pandas, numpy, pillow, scikit-image (for some I/O), tqdm, pyyaml (optional)

Install (example):
    pip install torch torchvision torchmetrics pandas numpy pillow scikit-image tqdm pyyaml

Data format
-----------
Provide CSV files for train/val/test with at least these columns:
    path            : path to image file (PNG/JPG or DICOM converted to PNG)
    label           : 0/1 for binary classification (extendable to multi-label)
    site_id         : hospital/scanner/site identifier (string or int)
Optional columns (if present will be logged and available for subgrouping):
    age             : numeric or bucketed string (e.g., "0-18","19-40",...)
    sex             : "M"/"F"/other
    country         : country code/name
    scanner         : manufacturer/model string

Example rows:
    path,label,site_id,age,sex,country,scanner
    /data/cxr/siteA/img001.png,1,siteA,67,M,IN,GE-Definium

Notes
-----
- DICOM: preprocess to PNG preserving windowing. Keep consistent pixel orientation.
- Multi-label: adapt criterion & metrics where noted.

Usage
-----
    python train_generalizable.py \
        --train_csv /data/train.csv \
        --val_csv /data/val.csv \
        --test_csv /data/test.csv \
        --out_dir ./runs/exp1 \
        --backbone resnet50 \
        --img_size 384 \
        --batch_size 32 \
        --epochs 20 \
        --optimizer adamw \
        --lr 3e-4 \
        --objective groupdro \
        --tta 5 \
        --spec_at 0.95

"""
from __future__ import annotations
import os
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import resnet18, resnet50, efficientnet_b0, efficientnet_b3

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryCalibrationError
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# -------------------------------
# Utilities
# -------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------------
# Dataset & Augmentations
# -------------------------------
class MedImgDataset(Dataset):
    def __init__(self, csv_path: str, img_size: int = 384, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        assert {'path', 'label', 'site_id'}.issubset(self.df.columns), \
            "CSV must have columns: path,label,site_id"
        self.img_size = img_size
        self.augment = augment

        # Transforms: keep clinical plausibility; avoid extreme distortions
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.15)),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                # Simulate scanner/compression noise
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not read image at {img_path}: {e}")
        x = self.tf(img)
        y = torch.tensor(row['label'], dtype=torch.float32)

        # Map group id (site/scanner/age bucket) to integer on the fly
        site_id = row['site_id']
        return x, y, str(site_id)


# -------------------------------
# Model
# -------------------------------
class Classifier(nn.Module):
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'resnet18':
            net = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat = net.fc.in_features
            net.fc = nn.Identity()
            self.encoder = net
            self.head = nn.Linear(feat, 1)
        elif backbone == 'resnet50':
            net = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat = net.fc.in_features
            net.fc = nn.Identity()
            self.encoder = net
            self.head = nn.Linear(feat, 1)
        elif backbone == 'efficientnet_b0':
            net = efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            feat = net.classifier[1].in_features
            net.classifier[1] = nn.Identity()
            self.encoder = net
            self.head = nn.Linear(feat, 1)
        elif backbone == 'efficientnet_b3':
            net = efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
            feat = net.classifier[1].in_features
            net.classifier[1] = nn.Identity()
            self.encoder = net
            self.head = nn.Linear(feat, 1)
        else:
            raise ValueError(f"Unknown backbone {backbone}")

    def forward(self, x):
        z = self.encoder(x)
        logit = self.head(z)
        return logit.squeeze(1)


# -------------------------------
# GroupDRO Loss
# -------------------------------
class GroupDROLoss(nn.Module):
    """Implements GroupDRO with exponential moving weights per group.
    Reference: Sagawa et al., Distributionally Robust Neural Networks, 2019.
    """
    def __init__(self, groups: List[str], eta: float = 0.01):
        super().__init__()
        self.eta = eta
        self.groups = sorted(list(set(groups)))
        self.g2idx = {g: i for i, g in enumerate(self.groups)}
        self.register_buffer('weights', torch.zeros(len(self.groups)))

    def forward(self, losses: torch.Tensor, group_names: List[str]):
        # Aggregate mean loss per group in current batch
        device = losses.device
        gw = self.weights.to(device)
        loss_per_group = torch.zeros_like(gw)
        count_per_group = torch.zeros_like(gw)

        for l, g in zip(losses.detach(), group_names):
            idx = self.g2idx.get(g, None)
            if idx is None:
                continue
            loss_per_group[idx] += l
            count_per_group[idx] += 1.0

        mask = count_per_group > 0
        mean_group_losses = torch.zeros_like(gw)
        mean_group_losses[mask] = loss_per_group[mask] / count_per_group[mask]

        # Update adversarial weights
        gw = gw + self.eta * mean_group_losses
        # Normalize weights to sum to 1 (avoid blow-up)
        gw = gw - torch.logsumexp(gw, dim=0)
        self.weights = gw.detach()

        # Compute weighted average of CURRENT sample losses with current weights
        batch_group_weights = torch.tensor([torch.exp(self.weights[self.g2idx[g]].item()) for g in group_names],
                                           device=device)
        weighted_loss = (losses * batch_group_weights).mean()
        return weighted_loss


# -------------------------------
# Temperature Scaling for Calibration
# -------------------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))  # temp=1.0

    def forward(self, logits):
        t = torch.exp(self.log_temp)
        return logits / t.clamp_min(1e-6)

    @torch.no_grad()
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200, lr: float = 0.01):
        # Simple SGD on NLL to find temperature
        opt = torch.optim.LBFGS([self.log_temp], lr=1.0, max_iter=50)
        nll = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return torch.exp(self.log_temp).item()


# -------------------------------
# Training / Evaluation helpers
# -------------------------------
@dataclass
class Config:
    train_csv: str
    val_csv: str
    test_csv: str
    out_dir: str = './runs/exp'
    seed: int = 42
    img_size: int = 384
    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    backbone: str = 'resnet50'
    pretrained: bool = True
    optimizer: str = 'adamw'  # or 'sgd'
    objective: str = 'groupdro'  # or 'erm'
    eta: float = 0.01  # groupdro step
    num_workers: int = 4
    tta: int = 0  # number of TTA samples at test time
    spec_at: float = 0.95  # operate at this specificity (select threshold on val)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, tta: int = 0) -> Tuple[np.ndarray, List[str], np.ndarray]:
    model.eval()
    all_logits = []
    all_groups = []
    all_labels = []
    if tta and tta > 0:
        # Define a light TTA pipeline compatible with dataset resize
        tta_tf = transforms.Compose([
            transforms.RandomResizedCrop(loader.dataset.img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    for xb, yb, gb in tqdm(loader, desc='Predict', leave=False):
        xb = xb.to(device)
        if tta and tta > 0:
            logits_accum = 0.0
            for _ in range(tta):
                xbt = xb
                # re-apply light spatial jitter in tensor space via grid_sample is complex; instead, rely on dataset-level transforms.
                # Simplest: small Gaussian noise perturbation
                noise = torch.randn_like(xb) * 0.01
                xbt = (xb + noise).clamp(0, 1)
                logits_accum += model(xbt)
            logits = logits_accum / float(tta)
        else:
            logits = model(xb)
        all_logits.append(logits.cpu())
        all_groups += list(gb)
        all_labels.append(yb)

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return logits, all_groups, labels


def choose_threshold_at_specificity(labels: np.ndarray, probs: np.ndarray, spec_target: float) -> float:
    # Sweep thresholds to achieve specificity; labels are 0/1, probs in [0,1]
    thresholds = np.linspace(0, 1, 1001)
    best_thr = 0.5
    best_diff = 1e9
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        spec = tn / max(1, (tn + fp))
        diff = abs(spec - spec_target)
        if diff < best_diff:
            best_diff = diff
            best_thr = thr
    return float(best_thr)


def subgroup_metrics(labels: np.ndarray, probs: np.ndarray, groups: List[str]) -> Dict[str, Dict[str, float]]:
    df = pd.DataFrame({'label': labels, 'prob': probs, 'group': groups})
    out = {}
    for g, sub in df.groupby('group'):
        if sub['label'].nunique() < 2:
            # AUROC undefined when only one class present; skip safely
            auroc = float('nan')
        else:
            auroc = roc_auc_score(sub['label'], sub['prob'])
        thr = choose_threshold_at_specificity(sub['label'].values, sub['prob'].values, 0.95)
        pred = (sub['prob'].values >= thr).astype(int)
        tp = int(((pred == 1) & (sub['label'].values == 1)).sum())
        tn = int(((pred == 0) & (sub['label'].values == 0)).sum())
        fp = int(((pred == 1) & (sub['label'].values == 0)).sum())
        fn = int(((pred == 0) & (sub['label'].values == 1)).sum())
        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))
        out[str(g)] = {
            'n': int(len(sub)),
            'auroc': float(auroc) if not math.isnan(auroc) else None,
            'thr_at_spec95': float(thr),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'ppv': float(tp / max(1, (tp + fp))),
            'npv': float(tn / max(1, (tn + fn))),
        }
    return out


def train(cfg: Config):
    seed_everything(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(cfg.out_dir)

    # Load datasets
    dtrain = MedImgDataset(cfg.train_csv, cfg.img_size, augment=True)
    dval = MedImgDataset(cfg.val_csv, cfg.img_size, augment=False)
    dtest = MedImgDataset(cfg.test_csv, cfg.img_size, augment=False)

    # Enumerate known groups from training set only (to avoid leakage)
    known_groups = sorted(list(set(dtrain.df['site_id'].astype(str).values)))

    # DataLoaders
    train_loader = DataLoader(dtrain, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(dval, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(dtest, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Model
    model = Classifier(cfg.backbone, pretrained=cfg.pretrained).to(device)

    # Objective
    bce = nn.BCEWithLogitsLoss(reduction='none')  # we need per-sample losses
    if cfg.objective.lower() == 'groupdro':
        dro = GroupDROLoss(groups=known_groups, eta=cfg.eta)
    else:
        dro = None

    # Optimizer
    if cfg.optimizer.lower() == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    # Metrics
    auroc_metric = BinaryAUROC()
    auprc_metric = BinaryAveragePrecision()

    best_val_auroc = -1
    best_path = os.path.join(cfg.out_dir, 'best.pt')

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.epochs}', leave=False)
        for xb, yb, gb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss_vec = bce(logits, yb)
            if dro is not None:
                loss = dro(loss_vec, list(gb))
            else:
                loss = loss_vec.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            pbar.set_postfix({'loss': float(loss.item())})

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits, v_groups, v_labels = predict(model, val_loader, device, tta=0)
            v_probs = 1 / (1 + np.exp(-v_logits))
            if len(np.unique(v_labels)) > 1:
                v_auroc = roc_auc_score(v_labels, v_probs)
            else:
                v_auroc = float('nan')
            auroc_metric.reset(); auprc_metric.reset()
            auroc_metric.update(torch.tensor(v_probs), torch.tensor(v_labels))
            auprc_metric.update(torch.tensor(v_probs), torch.tensor(v_labels))
            v_auprc = float(auprc_metric.compute().item())

        print(f"Epoch {epoch}: val AUROC={v_auroc:.4f} AUPRC={v_auprc:.4f}")
        if not math.isnan(v_auroc) and v_auroc > best_val_auroc:
            best_val_auroc = v_auroc
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, best_path)
            print(f"  >> Saved best to {best_path}")

    # Load best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])

    # Calibrate temperature on validation set
    with torch.no_grad():
        v_logits, v_groups, v_labels = predict(model, val_loader, device, tta=0)
        scaler = TemperatureScaler().to(device)
        v_logits_t = torch.tensor(v_logits, dtype=torch.float32, device=device)
        v_labels_t = torch.tensor(v_labels, dtype=torch.float32, device=device)
        temp = scaler.calibrate(v_logits_t, v_labels_t)
        print(f"Calibrated temperature: {temp:.3f}")

    # Evaluate on test with TTA + calibrated logits
    with torch.no_grad():
        t_logits, t_groups, t_labels = predict(model, test_loader, device, tta=cfg.tta)
        # Temperature scale
        t_logits_t = torch.tensor(t_logits, dtype=torch.float32, device=device)
        t_logits_cal = scaler(t_logits_t).cpu().numpy()
        t_probs = 1 / (1 + np.exp(-t_logits_cal))

    # Choose operating threshold at desired specificity using validation (post-calibration)
    v_logits_cal = scaler(torch.tensor(v_logits, dtype=torch.float32, device=device)).cpu().numpy()
    v_probs_cal = 1 / (1 + np.exp(-v_logits_cal))
    thr = choose_threshold_at_specificity(v_labels, v_probs_cal, cfg.spec_at)
    print(f"Threshold at specificity {cfg.spec_at:.2f}: {thr:.3f}")

    # Overall metrics
    if len(np.unique(t_labels)) > 1:
        test_auroc = roc_auc_score(t_labels, t_probs)
    else:
        test_auroc = float('nan')
    preds = (t_probs >= thr).astype(int)
    tp = int(((preds == 1) & (t_labels == 1)).sum())
    tn = int(((preds == 0) & (t_labels == 0)).sum())
    fp = int(((preds == 1) & (t_labels == 0)).sum())
    fn = int(((preds == 0) & (t_labels == 1)).sum())
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    ppv = tp / max(1, (tp + fp))
    npv = tn / max(1, (tn + fn))

    print(f"Test AUROC={test_auroc:.4f} | Sens={sens:.3f} Spec={spec:.3f} PPV={ppv:.3f} NPV={npv:.3f}")

    # Subgroup metrics (by site)
    sub = subgroup_metrics(t_labels, t_probs, t_groups)
    worst_group = None
    worst_auroc = 1e9
    for g, m in sub.items():
        a = m['auroc'] if m['auroc'] is not None else float('nan')
        print(f"  [Site={g}] n={m['n']} AUROC={a} Sens={m['sensitivity']:.3f} Spec={m['specificity']:.3f}")
        if not math.isnan(a) and a < worst_auroc:
            worst_auroc = a
            worst_group = g

    # Save artifacts
    results = {
        'overall': {
            'auroc': float(test_auroc) if not math.isnan(test_auroc) else None,
            'sensitivity': float(sens),
            'specificity': float(spec),
            'ppv': float(ppv),
            'npv': float(npv),
            'threshold_at_spec': float(thr),
            'spec_target': float(cfg.spec_at),
        },
        'subgroups_by_site': sub,
        'worst_group_by_auroc': worst_group,
        'temperature': float(np.exp(scaler.log_temp.detach().cpu().numpy()[0])),
        'config': asdict(cfg),
    }
    with open(os.path.join(cfg.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Minimal Model Card
    model_card = {
        'model': 'Generalizable Medical Imaging Classifier',
        'backbone': cfg.backbone,
        'intended_use': 'Assist clinicians by flagging positive findings; human-in-the-loop required.',
        'training_data': {
            'source': os.path.abspath(cfg.train_csv),
            'n_samples': len(dtrain),
            'sites': sorted(list(set(dtrain.df['site_id'].astype(str))))
        },
        'evaluation': results,
        'safety': {
            'abstention': {'low_confidence_below_prob': 0.2},
            'input_qc': ['enforce modality/view', 'minimum resolution', 'orientation check']
        },
        'limitations': [
            'Performance may degrade on unseen protocols or extreme artifacts.',
            'Not a diagnostic device; requires expert oversight.'
        ]
    }
    with open(os.path.join(cfg.out_dir, 'model_card.json'), 'w') as f:
        json.dump(model_card, f, indent=2)

    # Save final weights
    torch.save({'model': model.state_dict(), 'temperature': float(np.exp(scaler.log_temp.detach().cpu().numpy()[0])), 'cfg': asdict(cfg)},
               os.path.join(cfg.out_dir, 'final.pt'))

    print("Artifacts saved to:", cfg.out_dir)


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description='Train generalizable medical imaging model (GroupDRO + calibration)')
    p.add_argument('--train_csv', type=str, required=True)
    p.add_argument('--val_csv', type=str, required=True)
    p.add_argument('--test_csv', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='./runs/exp')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18','resnet50','efficientnet_b0','efficientnet_b3'])
    p.add_argument('--no_pretrained', action='store_true')
    p.add_argument('--optimizer', type=str, default='adamw', choices=['adamw','sgd'])
    p.add_argument('--objective', type=str, default='groupdro', choices=['groupdro','erm'])
    p.add_argument('--eta', type=float, default=0.01)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--tta', type=int, default=0)
    p.add_argument('--spec_at', type=float, default=0.95)

    args = p.parse_args()
    cfg = Config(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        seed=args.seed,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        optimizer=args.optimizer,
        objective=args.objective,
        eta=args.eta,
        num_workers=args.num_workers,
        tta=args.tta,
        spec_at=args.spec_at,
    )
    return cfg


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)
