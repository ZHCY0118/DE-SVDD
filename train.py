#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ENC + SVDD — 统一训练脚本（MSL/SMAP/SMD/SWAT/WADI）
----------------------------------------------------------
整合目标：
- 训练主逻辑、模型结构、日志/ckpt 语义统一；
- 仅数据读取按数据集分流；
- 风格对齐统一测试脚本：整体逻辑一致，不同点仅在 data loader；
- 保持当前基准训练语义：
  * epoch=1..pretrain_epochs：w_svdd=0
  * epoch=pretrain_epochs+1：用训练集 zh 均值初始化 center
  * center 初始化后：w_svdd 立即 = w_svdd_base（不使用 ramp）
  * early-stop：仅在 center 初始化后启用，并在 center 初始化所在轮 reset
  * 每个 epoch 保存 last.pt；同时维护 best.pt（仅在 center 初始化后开始比较）
- 本版本仅保留 reconstruction loss 与 SVDD loss
- enc1 前加入轻量多尺度上下文学习组件
- enc2 为更轻的正常性提炼编码器 + CompactProjector

支持数据集：
- MSL / SMAP（复用 MSLLoader + discover_series）
- SMD
- SWAT / SWaT
- WADI

建议配置：
- data.dataset: MSL / SMAP / SMD / SWAT / WADI
- data.data_root: 对应数据根目录
- data.series: 可选；SMD/MSL/SMAP 可指定单序列，否则自动发现
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data_loader import (
    set_seed,
    ensure_dir,
    to_device,
    WindowedTSDataset,
    MSLLoader,
    discover_series,
)


# =========================
# Model
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # [B, L, D]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class AdaptiveMultiScaleContextBlock(nn.Module):
    """
    轻量多尺度上下文学习模块：
    - 输入: [B, C, L]
    - 输出: [B, C, L]
    - 三个不同时间尺度分支：k=3/5/7
    - 使用全局摘要生成自适应尺度权重，对不同尺度分支做动态融合
    """
    def __init__(self, n_features: int, hidden_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden = max(16, int(n_features * hidden_ratio))

        self.pre_norm = nn.BatchNorm1d(n_features)

        self.branch3 = nn.Sequential(
            nn.Conv1d(n_features, n_features, kernel_size=3, padding=1, groups=n_features, bias=False),
            nn.Conv1d(n_features, hidden, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(n_features, n_features, kernel_size=5, padding=2, groups=n_features, bias=False),
            nn.Conv1d(n_features, hidden, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(n_features, n_features, kernel_size=7, padding=3, groups=n_features, bias=False),
            nn.Conv1d(n_features, hidden, kernel_size=1, bias=False),
            nn.GELU(),
        )

        gate_hidden = max(8, hidden // 2)
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(n_features, gate_hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(gate_hidden, 3, kernel_size=1, bias=True),
        )

        self.post = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, n_features, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )

        self.out_norm = nn.BatchNorm1d(n_features)

    def forward(self, x):  # [B, C, L]
        residual = x
        x = self.pre_norm(x)

        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)

        logits = self.scale_gate(x).squeeze(-1)  # [B, 3]
        alpha = torch.softmax(logits, dim=1)

        a3 = alpha[:, 0].view(-1, 1, 1)
        a5 = alpha[:, 1].view(-1, 1, 1)
        a7 = alpha[:, 2].view(-1, 1, 1)

        fused = a3 * b3 + a5 * b5 + a7 * b7
        out = self.post(fused)
        out = self.out_norm(residual + out)
        return out


class AttentionPooling(nn.Module):
    """
    对时间维做可学习加权聚合，替代简单 mean pooling
    输入: [B, L, D]
    输出: [B, D]
    """
    def __init__(self, d_model: int, attn_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1, bias=False),
        )

    def forward(self, h):  # [B, L, D]
        a = self.score(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.sum(h * a, dim=1)
        return pooled


class TransformerEncoder1D(nn.Module):
    """
    通用 Transformer encoder。
    通过 use_context / pooling_type 控制是否启用多尺度上下文模块与 attention pooling。
    """
    def __init__(
        self,
        n_features,
        latent_dim=128,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_ff=256,
        dropout=0.1,
        use_context=False,
        context_hidden_ratio=2.0,
        pooling_type="mean",
        pool_attn_hidden=128,
    ):
        super().__init__()

        self.use_context = bool(use_context)
        self.pooling_type = str(pooling_type).lower()

        if self.use_context:
            self.context = AdaptiveMultiScaleContextBlock(
                n_features=n_features,
                hidden_ratio=context_hidden_ratio,
                dropout=dropout,
            )

        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        if self.pooling_type == "attn":
            self.pool = AttentionPooling(d_model=d_model, attn_hidden=pool_attn_hidden, dropout=dropout)
            head_in = d_model
        elif self.pooling_type == "meanmax":
            self.pool = None
            head_in = d_model * 2
        else:
            self.pool = None
            head_in = d_model

        self.head = nn.Linear(head_in, latent_dim)

    def forward(self, x):  # [B, C, L]
        if self.use_context:
            x = self.context(x)

        x = x.transpose(1, 2)  # [B, L, C]
        h = self.pos(self.in_proj(x))
        h = self.encoder(h)

        if self.pooling_type == "attn":
            pooled = self.pool(h)
        elif self.pooling_type == "meanmax":
            pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        else:
            pooled = h.mean(dim=1)

        return self.head(pooled)


class CompactProjector(nn.Module):
    """
    第二潜在空间压缩投影头：
    - 输入: enc2 backbone 输出的中间表示
    - 输出: 更紧致、归一化的 zh
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        z = self.net(x)
        z = self.out_norm(z)
        return z


class LSTMDecoder1D(nn.Module):
    def __init__(self, n_features, seq_len, latent_dim=128, hidden=256, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_layers = num_layers
        self.inp = nn.Linear(latent_dim, hidden)
        self.h0p = nn.Linear(latent_dim, hidden * num_layers)
        self.c0p = nn.Linear(latent_dim, hidden * num_layers)
        self.lstm = nn.LSTM(hidden, hidden, num_layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, z):
        B = z.size(0)
        token = torch.tanh(self.inp(z))
        inp = token.unsqueeze(1).repeat(1, self.seq_len, 1)
        h0 = torch.tanh(self.h0p(z)).view(self.num_layers, B, self.hidden)
        c0 = torch.tanh(self.c0p(z)).view(self.num_layers, B, self.hidden)
        h, _ = self.lstm(inp, (h0, c0))
        return self.out(h).transpose(1, 2)  # [B, C, L]


class EncSVDD_TS(nn.Module):
    def __init__(
        self,
        n_features,
        seq_len,
        latent_dim=128,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_ff=256,
        dropout=0.1,
        enc1_use_context=True,
        enc1_context_hidden_ratio=2.0,
        enc1_pooling="attn",
        enc1_pool_attn_hidden=128,
        enc2_d_model=None,
        enc2_nhead=None,
        enc2_num_layers=1,
        enc2_dim_ff=None,
        enc2_pooling="mean",
        compact_hidden_dim=None,
    ):
        super().__init__()

        self.enc1 = TransformerEncoder1D(
            n_features=n_features,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            use_context=enc1_use_context,
            context_hidden_ratio=enc1_context_hidden_ratio,
            pooling_type=enc1_pooling,
            pool_attn_hidden=enc1_pool_attn_hidden,
        )

        self.dec = LSTMDecoder1D(n_features, seq_len, latent_dim)

        enc2_d_model = int(enc2_d_model if enc2_d_model is not None else max(32, d_model // 2))
        enc2_nhead = int(enc2_nhead if enc2_nhead is not None else max(1, min(4, nhead // 2)))
        enc2_dim_ff = int(enc2_dim_ff if enc2_dim_ff is not None else max(64, dim_ff // 2))
        compact_hidden_dim = int(compact_hidden_dim if compact_hidden_dim is not None else max(64, latent_dim))

        self.enc2_backbone = TransformerEncoder1D(
            n_features=n_features,
            latent_dim=enc2_d_model,
            d_model=enc2_d_model,
            nhead=enc2_nhead,
            num_layers=enc2_num_layers,
            dim_ff=enc2_dim_ff,
            dropout=dropout,
            use_context=False,
            pooling_type=enc2_pooling,
        )

        self.compact_projector = CompactProjector(
            in_dim=enc2_d_model,
            hidden_dim=compact_hidden_dim,
            out_dim=latent_dim,
            dropout=dropout,
        )

        self.register_buffer("c", torch.zeros(latent_dim))
        self.center_initialized = False

    @torch.no_grad()
    def set_center(self, c_new: torch.Tensor):
        c_new = c_new.detach().to(self.c.device)
        self.c.copy_(c_new)
        self.center_initialized = True

    def forward(self, x):
        z = self.enc1(x)
        xh = self.dec(z)
        h2 = self.enc2_backbone(xh)
        zh = self.compact_projector(h2)
        return z, xh, zh


# =========================
# Utils
# =========================

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float, warmup: int, mode: str = "min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup = int(warmup)
        self.mode = mode
        self.best = None
        self.bad = 0
        self.epoch = 0

    def reset(self):
        self.best = None
        self.bad = 0
        self.epoch = 0

    def step(self, value: float) -> bool:
        self.epoch += 1
        if self.epoch <= self.warmup:
            return False
        if self.best is None:
            self.best, self.bad = float(value), 0
            return False

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best, self.bad = float(value), 0
            return False

        self.bad += 1
        return self.bad >= self.patience


@torch.no_grad()
def compute_center(model: EncSVDD_TS, loader, device, use_amp: bool, max_batches=None) -> torch.Tensor:
    model.eval()
    zs, n = [], 0
    for i, batch in enumerate(loader, 1):
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = to_device(xb, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, _, zh = model(xb)
        zs.append(zh.detach().float().cpu())
        n += zh.size(0)
        if max_batches is not None and i >= max_batches:
            break
    if n == 0:
        raise RuntimeError("compute_center: empty loader / no batches.")
    return torch.cat(zs, dim=0).mean(dim=0)


def make_optimizer(params, cfg: dict):
    lr = float(cfg["learning_rate"])
    wd = float(cfg.get("weight_decay", 0.0))
    betas = (float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.999)))
    name = str(cfg.get("optimizer", "adam")).lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)


# =========================
# Dataset loaders
# =========================

def canonical_dataset_name(name: str) -> str:
    n = str(name).strip().lower()
    mapping = {
        "msl": "MSL",
        "smap": "SMAP",
        "smd": "SMD",
        "swat": "SWAT",
        "swaT": "SWAT",
        "wadi": "WADI",
    }
    return mapping.get(n, str(name).upper())


def load_smd_train(data_root: str, series_name: str, normalize: str = "zscore"):
    root = Path(data_root)

    agg_tr = root / "train.npy"
    if agg_tr.exists() and series_name == "SMD":
        Xtr = np.load(agg_tr).astype(np.float32)
    else:
        xtr_path = root / "train" / f"{series_name}.txt"
        if not xtr_path.exists():
            raise FileNotFoundError(f"SMD 训练文件不存在: {xtr_path}")
        try:
            Xtr = np.loadtxt(xtr_path, delimiter=",", dtype=np.float32)
        except Exception:
            Xtr = np.loadtxt(xtr_path, dtype=np.float32)

    stats = {}
    if normalize == "zscore":
        mean = Xtr.mean(axis=0, keepdims=True)
        std = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mean) / std
        stats = {"mean": mean, "std": std}
    return Xtr, stats


def load_swat_series(data_root: str, normalize: str = "zscore"):
    root = Path(data_root)

    tr_candidates = [root / "train_data.npy", root / "train.npy"]
    te_candidates = [root / "test_data.npy", root / "test.npy"]
    lb_candidates = [root / "test_label.npy", root / "labels.npy", root / "label.npy"]

    tr_path = next((p for p in tr_candidates if p.exists()), None)
    te_path = next((p for p in te_candidates if p.exists()), None)
    lb_path = next((p for p in lb_candidates if p.exists()), None)

    if tr_path is None or te_path is None or lb_path is None:
        raise FileNotFoundError(
            f"SWAT 数据文件缺失，期望存在：\n"
            f"  train_data.npy 或 train.npy\n"
            f"  test_data.npy  或 test.npy\n"
            f"  test_label.npy 或 labels.npy 或 label.npy\n"
            f"实际: train={tr_path}, test={te_path}, label={lb_path}"
        )

    print(f"[INFO] SWAT 使用: {tr_path.name}, {te_path.name}, {lb_path.name}")

    Xtr = np.load(tr_path).astype(np.float32)
    Xte = np.load(te_path).astype(np.float32)
    y_raw = np.load(lb_path)

    if y_raw.ndim == 2:
        yte = (y_raw.max(axis=1) > 0).astype(np.int64)
    else:
        yte = y_raw.astype(np.int64)

    if len(yte) != Xte.shape[0]:
        m = min(len(yte), Xte.shape[0])
        print(f"[WARN] SWAT 标签长度({len(yte)}) != test 样本数({Xte.shape[0]})，截取前 {m} 条对齐。")
        Xte = Xte[:m]
        yte = yte[:m]

    stats = {}
    if normalize == "zscore":
        mean = Xtr.mean(axis=0, keepdims=True)
        std = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mean) / std
        Xte = (Xte - mean) / std
        stats = {"mean": mean, "std": std}

    return Xtr, Xte, yte, stats


def load_wadi_series(data_root: str, normalize: str = "zscore"):
    root = Path(data_root)

    tr_candidates = [root / "train_data.npy", root / "train.npy"]
    te_candidates = [root / "test_data.npy", root / "test.npy"]
    lb_candidates = [root / "test_label.npy", root / "labels.npy"]

    tr_path = next((p for p in tr_candidates if p.exists()), None)
    te_path = next((p for p in te_candidates if p.exists()), None)
    lb_path = next((p for p in lb_candidates if p.exists()), None)

    if tr_path is None or te_path is None or lb_path is None:
        raise FileNotFoundError(
            f"WADI 数据文件缺失，期望存在以下任一命名：\n"
            f"  train_data.npy 或 train.npy\n"
            f"  test_data.npy  或 test.npy\n"
            f"  test_label.npy 或 labels.npy\n"
            f"实际: train={tr_path}, test={te_path}, label={lb_path}"
        )

    print(f"[INFO] WADI 使用: {tr_path.name}, {te_path.name}, {lb_path.name}")

    Xtr = np.load(tr_path).astype(np.float32)
    Xte = np.load(te_path).astype(np.float32)
    y_raw = np.load(lb_path)

    if y_raw.ndim == 2:
        yte = (y_raw.max(axis=1) > 0).astype(np.int64)
    else:
        yte = y_raw.astype(np.int64)

    if len(yte) != Xte.shape[0]:
        m = min(len(yte), Xte.shape[0])
        print(f"[WARN] WADI 标签长度({len(yte)}) != test 样本数({Xte.shape[0]})，截取前 {m} 条对齐。")
        Xte = Xte[:m]
        yte = yte[:m]

    stats = {}
    if normalize == "zscore":
        mean = Xtr.mean(axis=0, keepdims=True)
        std = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mean) / std
        Xte = (Xte - mean) / std
        stats = {"mean": mean, "std": std}

    return Xtr, Xte, yte, stats


def discover_series_unified(data_cfg: dict) -> Tuple[str, List[str]]:
    dataset = canonical_dataset_name(data_cfg.get("dataset", data_cfg.get("dataset_only", "")))

    series = data_cfg.get("series", None)
    if series:
        return dataset, [str(series)]

    if dataset in ["MSL", "SMAP"]:
        series_list = discover_series(
            data_root=data_cfg["data_root"],
            dataset_only=dataset,
            meta_csv=data_cfg.get("meta_csv", None),
            include_expr=data_cfg.get("include", None),
        )
        return dataset, series_list

    if dataset == "SMD":
        train_dir = Path(data_cfg["data_root"]) / "train"
        if train_dir.exists():
            series_list = sorted(p.stem for p in train_dir.glob("*.txt"))
            if len(series_list) == 0:
                agg_tr = Path(data_cfg["data_root"]) / "train.npy"
                if agg_tr.exists():
                    series_list = ["SMD"]
        else:
            agg_tr = Path(data_cfg["data_root"]) / "train.npy"
            if agg_tr.exists():
                series_list = ["SMD"]
            else:
                series_list = []
        return dataset, series_list

    if dataset == "SWAT":
        return dataset, ["SWAT"]

    if dataset == "WADI":
        return dataset, [str(data_cfg.get("series_name", "WADI"))]

    raise ValueError(f"不支持的数据集: {dataset}")


def load_train_array_for_series(data_cfg: dict, dataset: str, series_name: str):
    normalize = data_cfg.get("normalize", "zscore")
    data_root = data_cfg["data_root"]

    if dataset in ["MSL", "SMAP"]:
        loader = MSLLoader(
            data_root=data_root,
            normalize=normalize,
            force_dim=data_cfg.get("force_dim", None),
            series=series_name,
            dataset_only=dataset,
            meta_csv=data_cfg.get("meta_csv", None),
        )
        Xtr, _, _, stats = loader.load_raw()
        return Xtr, stats

    if dataset == "SMD":
        return load_smd_train(data_root=data_root, series_name=series_name, normalize=normalize)

    if dataset == "SWAT":
        Xtr, _, _, stats = load_swat_series(data_root=data_root, normalize=normalize)
        return Xtr, stats

    if dataset == "WADI":
        Xtr, _, _, stats = load_wadi_series(data_root=data_root, normalize=normalize)
        return Xtr, stats

    raise ValueError(f"不支持的数据集: {dataset}")


# =========================
# Train core
# =========================

def build_model_from_cfg(cfg: dict, n_features: int, seq_len: int):
    tr_cfg = cfg["training"]
    model_cfg = cfg.get("model", {})

    latent_dim = int(tr_cfg.get("latent_dim", model_cfg.get("latent_dim", 128)))

    d_model = int(model_cfg.get("d_model", 128))
    nhead = int(model_cfg.get("nhead", 8))
    num_layers = int(model_cfg.get("num_layers", 3))
    dim_ff = int(model_cfg.get("dim_ff", 256))
    dropout = float(model_cfg.get("dropout", 0.1))

    enc1_use_context = bool(model_cfg.get("enc1_use_context", True))
    enc1_context_hidden_ratio = float(model_cfg.get("enc1_context_hidden_ratio", 2.0))
    enc1_pooling = str(model_cfg.get("enc1_pooling", "attn")).lower()
    enc1_pool_attn_hidden = int(model_cfg.get("enc1_pool_attn_hidden", d_model))

    enc2_d_model = int(model_cfg.get("enc2_d_model", max(32, d_model // 2)))
    enc2_nhead = int(model_cfg.get("enc2_nhead", max(1, min(4, nhead // 2))))
    enc2_num_layers = int(model_cfg.get("enc2_num_layers", 1))
    enc2_dim_ff = int(model_cfg.get("enc2_dim_ff", max(64, dim_ff // 2)))
    enc2_pooling = str(model_cfg.get("enc2_pooling", "mean")).lower()
    compact_hidden_dim = int(model_cfg.get("compact_hidden_dim", max(64, latent_dim)))

    model = EncSVDD_TS(
        n_features=n_features,
        seq_len=seq_len,
        latent_dim=latent_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        enc1_use_context=enc1_use_context,
        enc1_context_hidden_ratio=enc1_context_hidden_ratio,
        enc1_pooling=enc1_pooling,
        enc1_pool_attn_hidden=enc1_pool_attn_hidden,
        enc2_d_model=enc2_d_model,
        enc2_nhead=enc2_nhead,
        enc2_num_layers=enc2_num_layers,
        enc2_dim_ff=enc2_dim_ff,
        enc2_pooling=enc2_pooling,
        compact_hidden_dim=compact_hidden_dim,
    )

    model_meta = {
        "latent_dim": latent_dim,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_ff": dim_ff,
        "dropout": dropout,
        "enc1_use_context": enc1_use_context,
        "enc1_context_hidden_ratio": enc1_context_hidden_ratio,
        "enc1_pooling": enc1_pooling,
        "enc1_pool_attn_hidden": enc1_pool_attn_hidden,
        "enc2_d_model": enc2_d_model,
        "enc2_nhead": enc2_nhead,
        "enc2_num_layers": enc2_num_layers,
        "enc2_dim_ff": enc2_dim_ff,
        "enc2_pooling": enc2_pooling,
        "compact_hidden_dim": compact_hidden_dim,
    }
    return model, model_meta


def train_one_series(cfg: dict, dataset: str, series_name: str):
    data_cfg = cfg["data"]
    tr_cfg = cfg["training"]

    Xtr, stats = load_train_array_for_series(data_cfg, dataset, series_name)

    seq_len = int(data_cfg["window_size"])
    stride = int(data_cfg["step_size"])
    batch_size = int(tr_cfg["batch_size"])
    num_workers = int(tr_cfg.get("num_workers", 0))

    train_ds = WindowedTSDataset(Xtr, None, seq_len, stride, split="train", drop_anom_in_train=True)
    n_win = len(train_ds)
    print(f"[TRAIN] dataset={dataset} series={series_name} 训练窗口数={n_win} (batch_size={batch_size})")

    save_root = Path(tr_cfg["save_dir"])
    out_dir = save_root / series_name
    ensure_dir(str(out_dir))

    csv_name = str(tr_cfg.get("log_csv_name", "train_loss_history.csv"))
    csv_path = out_dir / csv_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch",
            "w_recon_eff",
            "w_svdd_eff",
            "G_rec",
            "G_svdd",
            "G_total",
            "epoch_time_sec",
            "peak_gpu_mem_mb_so_far",
        ])

    if n_win == 0:
        print(f"[WARN] dataset={dataset} series={series_name} 训练窗口数为 0，跳过训练。")
        return

    from torch.utils.data import DataLoader
    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dl_center = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    if len(dl_tr) == 0:
        print(f"[WARN] dataset={dataset} series={series_name} drop_last=True 导致无 batch（n_win={n_win} < batch_size={batch_size}），跳过训练。")
        return

    device = torch.device(tr_cfg["device"])
    if bool(tr_cfg.get("no_cudnn", False)):
        torch.backends.cudnn.enabled = False

    model, model_meta = build_model_from_cfg(cfg, n_features=Xtr.shape[1], seq_len=seq_len)
    model = model.to(device)

    optim = make_optimizer(model.parameters(), tr_cfg)
    use_amp = bool(tr_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    w_recon = float(tr_cfg.get("w_recon", 1.0))
    w_svdd_base = float(tr_cfg.get("w_svdd", 1.0))
    pretrain_epochs = int(tr_cfg.get("pretrain_epochs", 5))
    ramp_epochs = int(tr_cfg.get("svdd_ramp_epochs", 0))
    grad_clip = float(tr_cfg.get("grad_clip_norm", 0.0))
    log_interval = int(tr_cfg.get("log_interval", 100))
    num_epochs = int(tr_cfg["num_epochs"])

    stopper = EarlyStopper(
        patience=int(tr_cfg.get("early_stop_patience", 0)),
        min_delta=float(tr_cfg.get("early_stop_min_delta", 0.0)),
        warmup=int(tr_cfg.get("early_stop_warmup", 0)),
        mode=str(tr_cfg.get("early_stop_mode", "min")),
    )

    center_inited_epoch = None
    early_stop_active = False
    best_metric = None
    best_epoch = None
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    print(
        f"[MODEL] dataset={dataset} series={series_name} "
        f"enc1_use_context={model_meta['enc1_use_context']}, "
        f"enc1_context_hidden_ratio={model_meta['enc1_context_hidden_ratio']}, "
        f"enc1_pooling={model_meta['enc1_pooling']}, enc1_pool_attn_hidden={model_meta['enc1_pool_attn_hidden']} | "
        f"enc2_d_model={model_meta['enc2_d_model']}, enc2_nhead={model_meta['enc2_nhead']}, "
        f"enc2_num_layers={model_meta['enc2_num_layers']}, enc2_dim_ff={model_meta['enc2_dim_ff']}, "
        f"enc2_pooling={model_meta['enc2_pooling']}, compact_hidden_dim={model_meta['compact_hidden_dim']}"
    )

    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    total_train_start = time.perf_counter()
    epoch_times_sec = []
    last_epoch_ran = 0

    for epoch in range(1, num_epochs + 1):
        if is_cuda:
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        if (not model.center_initialized) and (epoch == pretrain_epochs + 1):
            c = compute_center(model, dl_center, device, use_amp=use_amp, max_batches=None)
            model.set_center(c)
            center_inited_epoch = epoch
            print(f"[CENTER INIT] dataset={dataset} series={series_name} epoch={epoch} center initialized from zh mean (train set).")

            stopper.reset()
            early_stop_active = True
            best_metric = None
            best_epoch = None
            print(f"[EARLY STOP] dataset={dataset} series={series_name} activated after center init (epoch={epoch}); early-stop counter reset.")

        w_svdd = w_svdd_base if model.center_initialized else 0.0

        model.train()
        sum_rec = sum_svdd = sum_tot = 0.0
        n_steps = 0

        for it, batch in enumerate(dl_tr, 1):
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = to_device(xb, device)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, xh, zh = model(xb)
                l_rec = F.mse_loss(xh, xb)
                l_svdd = torch.mean(torch.sum((zh - model.c) ** 2, dim=1)) if w_svdd > 0.0 else zh.new_tensor(0.0)
                loss = w_recon * l_rec + w_svdd * l_svdd

            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optim)
            scaler.update()

            sum_rec += float(l_rec.detach().cpu())
            sum_svdd += float(l_svdd.detach().cpu())
            sum_tot += float(loss.detach().cpu())
            n_steps += 1

            if log_interval > 0 and (it % log_interval == 0):
                print(
                    f"[E{epoch:03d}][{it:05d}/{len(dl_tr):05d}] dataset={dataset} series={series_name} "
                    f"w=(recon={w_recon:.4g}, svdd={w_svdd:.4g}) "
                    f"loss={sum_tot/n_steps:.6f} recon={sum_rec/n_steps:.6f} svdd={sum_svdd/n_steps:.6f}"
                )

        ep_rec = sum_rec / max(1, n_steps)
        ep_svdd = sum_svdd / max(1, n_steps)
        ep_tot = sum_tot / max(1, n_steps)

        if is_cuda:
            torch.cuda.synchronize(device)
        epoch_time_sec = time.perf_counter() - epoch_start
        epoch_times_sec.append(float(epoch_time_sec))
        last_epoch_ran = int(epoch)
        peak_gpu_mem_mb_so_far = (
            float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2) if is_cuda else 0.0
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                w_recon,
                w_svdd,
                ep_rec,
                ep_svdd,
                ep_tot,
                epoch_time_sec,
                peak_gpu_mem_mb_so_far,
            ])

        ckpt = {
            "model": model.state_dict(),
            "config": cfg,
            "series": series_name,
            "dataset": dataset,
            "epoch": epoch,
            "train_loss": ep_tot,
            "train_recon": ep_rec,
            "train_svdd": ep_svdd,
            "w_recon_eff": w_recon,
            "w_svdd_eff": w_svdd,
            "center_initialized": bool(model.center_initialized),
            "center_inited_epoch": center_inited_epoch,
            "center_c": model.c.detach().cpu(),
            "stats": stats,
        }

        torch.save(ckpt, last_path)

        best_updated = False
        if early_stop_active:
            mode = str(tr_cfg.get("early_stop_mode", "min")).lower()
            if best_metric is None:
                best_updated = True
            else:
                best_updated = (ep_tot < best_metric) if mode == "min" else (ep_tot > best_metric)

            if best_updated:
                best_metric = float(ep_tot)
                best_epoch = int(epoch)
                torch.save(ckpt, best_path)

        msg = (
            f"[EPOCH {epoch:03d}] dataset={dataset} series={series_name} "
            f"w=(recon={w_recon:.4g}, svdd={w_svdd:.4g}) "
            f"loss={ep_tot:.6f} recon={ep_rec:.6f} svdd={ep_svdd:.6f} "
            f"time={epoch_time_sec:.2f}s peak_gpu_mem={peak_gpu_mem_mb_so_far:.2f}MB -> saved {last_path}"
        )
        if best_updated:
            msg += f" | best updated (epoch={best_epoch}, metric={best_metric:.6f}) -> saved {best_path}"
        print(msg)

        if int(tr_cfg.get("early_stop_patience", 0)) > 0 and early_stop_active and stopper.step(ep_tot):
            print(f"[EARLY STOP] dataset={dataset} series={series_name} triggered at epoch={epoch}, best={stopper.best:.6f}")
            break

    if is_cuda:
        torch.cuda.synchronize(device)
    total_training_time_sec = time.perf_counter() - total_train_start
    peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2) if is_cuda else 0.0
    avg_epoch_time_sec = float(np.mean(epoch_times_sec)) if len(epoch_times_sec) > 0 else 0.0

    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "series": series_name,
                "pretrain_epochs": pretrain_epochs,
                "svdd_ramp_epochs": ramp_epochs,
                "w_recon_base": w_recon,
                "w_svdd_base": w_svdd_base,
                "center_initialized": bool(model.center_initialized),
                "center_inited_epoch": center_inited_epoch,
                "loss_csv": str(csv_path),
                "last_ckpt": str(last_path),
                "best_ckpt": str(best_path),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "trained_epochs": last_epoch_ran,
                "avg_epoch_time_sec": avg_epoch_time_sec,
                "total_training_time_sec": total_training_time_sec,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                **model_meta,
                "note": "Latent loss removed. Training uses only reconstruction loss and SVDD loss. enc1 is enhanced by an adaptive multi-scale context block and attention pooling. The second stage is redesigned as a lighter normality-refining encoder plus a compact projector, so that SVDD is imposed on a more compact second latent space. EarlyStop starts ONLY after center init; EarlyStopper is reset at the center-init epoch. SVDD warmup/ramp disabled: after center init, w_svdd is set to w_svdd_base immediately. This script also records epoch time, total training time, and peak GPU memory during training.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[TRAIN SUMMARY] dataset={dataset} series={series_name} "
        f"trained_epochs={last_epoch_ran} avg_epoch_time={avg_epoch_time_sec:.2f}s "
        f"total_training_time={total_training_time_sec:.2f}s peak_gpu_memory={peak_gpu_memory_mb:.2f}MB"
    )

    if device.type == "cuda" and torch.cuda.is_available():
        del model, optim, scaler, dl_tr, dl_center
        torch.cuda.empty_cache()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Unified ENC+SVDD train (clean, rec+svdd only, adaptive multi-scale context + compact enc2)"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML 配置路径")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    tr_cfg = cfg["training"]

    set_seed(int(tr_cfg["seed"]))

    dataset, series_list = discover_series_unified(data_cfg)
    if len(series_list) == 0:
        raise RuntimeError(f"dataset={dataset} 未发现可训练序列，请检查 data_root/series/meta_csv 配置。")

    print(f"[INFO] dataset={dataset}, num_series={len(series_list)}")
    for s in series_list:
        print(f"\n==================== TRAIN dataset={dataset} series={s} ====================")
        train_one_series(cfg, dataset, s)


if __name__ == "__main__":
    main()
