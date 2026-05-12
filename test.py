#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified ENC + SVDD Test Script
------------------------------
统一测试脚本：MSL / SMAP / SMD / SWAT / WADI

核心评分：
    svdd = ||zh - c||^2
    rec  = mean((x_hat - x)^2)
    score = s_svdd * svdd + s_rec * rec

输出约定：
- 每个序列输出一行：
    [series_name] F1=... P=... R=... AUC=... AUPR=...
- 最后输出平均值：
    [MEAN] F1=... P=... R=... AUC=... AUPR=...
- 同时写出 CSV 文件，包含每个序列和最终 MEAN

新增时间统计：
- 每个序列完整评测总时间：total_eval_time_sec
- 全数据集总评测时间：total_eval_time_sec_sum
- 每序列平均完整评测时间：mean_total_eval_time_sec_per_series
- 吞吐率：throughput_win_per_sec，单位为 win/s，即每秒处理的窗口数
"""

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from data_loader import (
    set_seed,
    ensure_dir,
    to_device,
    WindowedTSDataset,
    MSLLoader,
    discover_series,
)

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


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
        return x + self.pe[:x.size(1)].unsqueeze(0)


class AdaptiveMultiScaleContextBlock(nn.Module):
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

        logits = self.scale_gate(x).squeeze(-1)
        alpha = torch.softmax(logits, dim=1)

        a3 = alpha[:, 0].view(-1, 1, 1)
        a5 = alpha[:, 1].view(-1, 1, 1)
        a7 = alpha[:, 2].view(-1, 1, 1)

        fused = a3 * b3 + a5 * b5 + a7 * b7
        out = self.post(fused)
        out = self.out_norm(residual + out)
        return out


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, attn_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, attn_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1, bias=False),
        )

    def forward(self, h):
        a = self.score(h)
        a = torch.softmax(a, dim=1)
        return torch.sum(h * a, dim=1)


class TransformerEncoder1D(nn.Module):
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
        h = self.encoder(self.pos(self.in_proj(x)))

        if self.pooling_type == "attn":
            pooled = self.pool(h)
        elif self.pooling_type == "meanmax":
            pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        else:
            pooled = h.mean(dim=1)

        return self.head(pooled)


class CompactProjector(nn.Module):
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
        self.seq_len = int(seq_len)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
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

    def forward(self, x):
        z = self.enc1(x)
        xh = self.dec(z)
        h2 = self.enc2_backbone(xh)
        zh = self.compact_projector(h2)
        return z, xh, zh


# =========================
# Data helpers
# =========================

def infer_dataset_kind(data_cfg):
    ds = str(data_cfg.get("dataset_type", "") or "").strip().upper()
    if ds:
        return ds

    ds2 = str(data_cfg.get("dataset_only", "") or "").strip().upper()
    if ds2 in ("MSL", "SMAP"):
        return ds2

    root = str(data_cfg.get("data_root", "")).lower()
    if "smd" in root:
        return "SMD"
    if "msl" in root:
        return "MSL"
    if "smap" in root:
        return "SMAP"
    if "swat" in root:
        return "SWAT"
    if "wadi" in root:
        return "WADI"

    return "SINGLE_NPY"


def infer_output_csv_name(kind: str):
    return f"{kind}.csv"


def load_smd_series_raw(data_root: str, series_name: str):
    root = Path(data_root)

    def _load(p, dtype):
        try:
            return np.loadtxt(p, delimiter=",", dtype=dtype)
        except Exception:
            return np.loadtxt(p, dtype=dtype)

    xtr = root / "train" / f"{series_name}.txt"
    xte = root / "test" / f"{series_name}.txt"
    lbl1 = root / "test_label" / f"{series_name}.txt"
    lbl2 = root / "test_label" / f"{series_name}_label.txt"

    if not xtr.exists() or not xte.exists():
        raise FileNotFoundError(f"SMD 文件不存在: {xtr} 或 {xte}")

    Xtr = _load(xtr, np.float32)
    Xte = _load(xte, np.float32)

    if lbl1.exists():
        y_raw = _load(lbl1, np.int64)
    elif lbl2.exists():
        y_raw = _load(lbl2, np.int64)
    else:
        raise FileNotFoundError(f"SMD 标签文件不存在: {lbl1} 或 {lbl2}")

    if y_raw.ndim == 2:
        yte = (y_raw.max(axis=1) > 0).astype(np.int64)
    else:
        yte = y_raw.astype(np.int64)

    return Xtr, Xte, yte


def default_single_npy_candidates(kind: str):
    kind = kind.upper()
    if kind == "SWAT":
        return {
            "train": ["train_data.npy", "train.npy"],
            "test": ["test_data.npy", "test.npy"],
            "label": ["test_label.npy", "labels.npy", "label.npy"],
        }
    if kind == "WADI":
        return {
            "train": ["train_data.npy", "train.npy", "WADI_train.npy"],
            "test": ["test_data.npy", "test.npy", "WADI_test.npy"],
            "label": ["test_label.npy", "labels.npy", "label.npy", "WADI_test_label.npy"],
        }
    return {
        "train": ["train_data.npy", "train.npy"],
        "test": ["test_data.npy", "test.npy"],
        "label": ["test_label.npy", "labels.npy", "label.npy"],
    }


def load_single_npy_dataset(data_root: str, kind: str, data_cfg: dict):
    root = Path(data_root)
    defaults = default_single_npy_candidates(kind)

    tr_candidates = data_cfg.get("train_candidates", defaults["train"])
    te_candidates = data_cfg.get("test_candidates", defaults["test"])
    lb_candidates = data_cfg.get("label_candidates", defaults["label"])

    tr_path = next((root / x for x in tr_candidates if (root / x).exists()), None)
    te_path = next((root / x for x in te_candidates if (root / x).exists()), None)
    lb_path = next((root / x for x in lb_candidates if (root / x).exists()), None)

    if tr_path is None or te_path is None or lb_path is None:
        raise FileNotFoundError(
            f"{kind} 数据文件缺失。\n"
            f"train candidates={tr_candidates}\n"
            f"test candidates={te_candidates}\n"
            f"label candidates={lb_candidates}\n"
            f"resolved: train={tr_path}, test={te_path}, label={lb_path}"
        )

    Xtr = np.load(tr_path).astype(np.float32)
    Xte = np.load(te_path).astype(np.float32)
    y_raw = np.load(lb_path)

    if y_raw.ndim == 2:
        yte = (y_raw.max(axis=1) > 0).astype(np.int64)
    else:
        yte = y_raw.astype(np.int64)

    if len(yte) != Xte.shape[0]:
        m = min(len(yte), Xte.shape[0])
        Xte = Xte[:m]
        yte = yte[:m]

    return Xtr, Xte, yte


def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (X - mean) / (std + 1e-6)


def zscore_with_ckpt_stats_or_train(Xtr_raw, Xte_raw, ckpt_stats):
    if isinstance(ckpt_stats, dict):
        mean = ckpt_stats.get("mean", None)
        std = ckpt_stats.get("std", None)
        if mean is not None and std is not None:
            Xtr = apply_zscore(Xtr_raw, mean, std)
            Xte = apply_zscore(Xte_raw, mean, std)
            return Xtr, Xte

    mean = Xtr_raw.mean(axis=0, keepdims=True).astype(np.float32)
    std = (Xtr_raw.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    Xtr = (Xtr_raw - mean) / std
    Xte = (Xte_raw - mean) / std
    return Xtr, Xte


# =========================
# Utils
# =========================

def clean_scores(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        m = np.isfinite(x)
        med = np.median(x[m]) if m.any() else 0.0
        x[~m] = med
    return x


def metrics_at_thr(scores, labels, thr):
    pred = (scores >= thr).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    tn = ((pred == 0) & (labels == 0)).sum()
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(f1), float(p), float(r), int(tp), int(fp), int(fn), int(tn)


def best_f1_thr(scores, labels):
    scores = clean_scores(scores)
    thrs = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
    best_f1, best_thr = -1.0, float(thrs[0])
    for t in thrs:
        f1, *_ = metrics_at_thr(scores, labels, t)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return float(best_thr)


def simplex_grid_2d(n_div: int):
    for i in range(n_div + 1):
        j = n_div - i
        yield i / n_div, j / n_div


def search_best_weights(SVD, REC, labels, n_div=20):
    SVD, REC = clean_scores(SVD), clean_scores(REC)
    best = None
    for a, b in simplex_grid_2d(n_div):
        scores = a * SVD + b * REC
        thr = best_f1_thr(scores, labels)
        f1, p, r, tp, fp, fn, tn = metrics_at_thr(scores, labels, thr)
        if best is None or f1 > best["f1"]:
            best = dict(
                s_svdd=float(a),
                s_rec=float(b),
                thr=float(thr),
                f1=float(f1),
                p=float(p),
                r=float(r),
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
                tn=int(tn),
                scores=scores,
            )
    return best


def eval_fixed(SVD, REC, labels, s_svdd: float, s_rec: float):
    scores = clean_scores(s_svdd * clean_scores(SVD) + s_rec * clean_scores(REC))
    thr = best_f1_thr(scores, labels)
    f1, p, r, tp, fp, fn, tn = metrics_at_thr(scores, labels, thr)
    return {
        "s_svdd": float(s_svdd),
        "s_rec": float(s_rec),
        "thr": float(thr),
        "f1": float(f1),
        "p": float(p),
        "r": float(r),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "scores": scores,
    }


@torch.no_grad()
def infer_components(
    model,
    dl,
    device,
    use_amp: bool,
    measure_latency: bool = True,
    latency_warmup_batches: int = 10,
    latency_max_batches: int = 50,
):
    SVD, REC, Y = [], [], []
    model.eval()

    measured_time_sec = 0.0
    measured_batches = 0
    measured_samples = 0
    total_batches = 0
    total_samples = 0

    is_cuda = (device.type == "cuda") and torch.cuda.is_available()

    for batch_idx, batch in enumerate(dl, 1):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            xb, yb = batch[0], batch[1]
        else:
            raise RuntimeError("DataLoader batch format unexpected: expected (x, y).")

        xb = to_device(xb, device)
        bs = int(xb.size(0))
        total_batches += 1
        total_samples += bs

        do_measure = bool(measure_latency) and (batch_idx > latency_warmup_batches) and (measured_batches < latency_max_batches)

        amp_ctx = torch.cuda.amp.autocast(enabled=use_amp) if is_cuda else nullcontext()

        if do_measure and is_cuda:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with amp_ctx:
                _, xh, zh = model(xb)
                svdd = torch.sum((zh - model.c) ** 2, dim=1)
                rec = torch.mean((xh - xb) ** 2, dim=(1, 2))
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            measured_time_sec += (t1 - t0)
            measured_batches += 1
            measured_samples += bs
        elif do_measure:
            t0 = time.perf_counter()
            with amp_ctx:
                _, xh, zh = model(xb)
                svdd = torch.sum((zh - model.c) ** 2, dim=1)
                rec = torch.mean((xh - xb) ** 2, dim=(1, 2))
            t1 = time.perf_counter()
            measured_time_sec += (t1 - t0)
            measured_batches += 1
            measured_samples += bs
        else:
            with amp_ctx:
                _, xh, zh = model(xb)
                svdd = torch.sum((zh - model.c) ** 2, dim=1)
                rec = torch.mean((xh - xb) ** 2, dim=(1, 2))

        SVD.append(torch.nan_to_num(svdd, 0.0).float().cpu())
        REC.append(torch.nan_to_num(rec, 0.0).float().cpu())
        Y.append(torch.as_tensor(yb).cpu())

    latency_info = {
        "inference_time_sec_measured": float(measured_time_sec),
        "latency_measured_batches": int(measured_batches),
        "latency_measured_samples": int(measured_samples),
        "num_eval_batches": int(total_batches),
        "num_eval_samples": int(total_samples),
        "latency_warmup_batches": int(latency_warmup_batches),
        "inference_latency_ms_per_batch": float((measured_time_sec / measured_batches) * 1000.0) if measured_batches > 0 else float("nan"),
        "inference_latency_ms_per_sample": float((measured_time_sec / measured_samples) * 1000.0) if measured_samples > 0 else float("nan"),
        "inference_throughput_win_per_sec": float(measured_samples / measured_time_sec) if measured_time_sec > 0 else float("nan"),
    }

    return (
        torch.cat(SVD).numpy(),
        torch.cat(REC).numpy(),
        torch.cat(Y).numpy(),
        latency_info,
    )


def write_results_csv(csv_path: Path, rows: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series",
        "f1", "p", "r",
        "auc", "aupr",
        "s_svdd", "s_rec",
        "thr",
        "tp", "fp", "fn", "tn",
        "ckpt_name", "ckpt_epoch",
        "inference_latency_ms_per_batch",
        "inference_latency_ms_per_sample",
        "inference_throughput_win_per_sec",
        "inference_time_sec_measured",
        "total_eval_time_sec",
        "total_eval_throughput_win_per_sec",
        "test_peak_gpu_memory_mb",
        "num_eval_batches",
        "num_eval_samples",
        "latency_measured_batches",
        "latency_measured_samples",
        "latency_warmup_batches",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            w.writerow(out)


def load_ckpt_prefer_best(series_dir: Path):
    best_p = series_dir / "best.pt"
    last_p = series_dir / "last.pt"
    if best_p.exists():
        return best_p, torch.load(best_p, map_location="cpu")
    if last_p.exists():
        return last_p, torch.load(last_p, map_location="cpu")
    raise FileNotFoundError(f"Checkpoint not found: {best_p} or {last_p}")


def load_config(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================
# Per-series evaluation
# =========================

def build_model_from_cfg(cfg, ckpt, data_cfg, train_cfg, Xtr):
    ckpt_cfg = ckpt.get("config", None)

    if ckpt_cfg is not None and isinstance(ckpt_cfg, dict):
        ckpt_data_cfg = ckpt_cfg.get("data", {}) or {}
        ckpt_train_cfg = ckpt_cfg.get("training", {}) or {}
        ckpt_model_cfg = ckpt_cfg.get("model", {}) or {}

        win_size = int(ckpt_data_cfg.get("window_size", data_cfg["window_size"]))
        step_size = int(ckpt_data_cfg.get("step_size", data_cfg["step_size"]))
        latent_dim = int(ckpt_train_cfg.get("latent_dim", train_cfg["latent_dim"]))

        d_model = int(ckpt_model_cfg.get("d_model", 128))
        nhead = int(ckpt_model_cfg.get("nhead", 8))
        num_layers = int(ckpt_model_cfg.get("num_layers", 3))
        dim_ff = int(ckpt_model_cfg.get("dim_ff", 256))
        dropout = float(ckpt_model_cfg.get("dropout", 0.1))

        enc1_use_context = bool(ckpt_model_cfg.get("enc1_use_context", True))
        enc1_context_hidden_ratio = float(ckpt_model_cfg.get("enc1_context_hidden_ratio", 2.0))
        enc1_pooling = str(ckpt_model_cfg.get("enc1_pooling", "attn")).lower()
        enc1_pool_attn_hidden = int(ckpt_model_cfg.get("enc1_pool_attn_hidden", d_model))

        enc2_d_model = int(ckpt_model_cfg.get("enc2_d_model", max(32, d_model // 2)))
        enc2_nhead = int(ckpt_model_cfg.get("enc2_nhead", max(1, min(4, nhead // 2))))
        enc2_num_layers = int(ckpt_model_cfg.get("enc2_num_layers", 1))
        enc2_dim_ff = int(ckpt_model_cfg.get("enc2_dim_ff", max(64, dim_ff // 2)))
        enc2_pooling = str(ckpt_model_cfg.get("enc2_pooling", "mean")).lower()
        compact_hidden_dim = int(ckpt_model_cfg.get("compact_hidden_dim", max(64, latent_dim)))
    else:
        model_cfg = cfg.get("model", {})

        win_size = int(data_cfg["window_size"])
        step_size = int(data_cfg["step_size"])
        latent_dim = int(train_cfg["latent_dim"])

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
        n_features=int(Xtr.shape[1]),
        seq_len=win_size,
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

    model_meta = dict(
        win_size=win_size,
        step_size=step_size,
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
    return model, model_meta


def maybe_normalize_single_dataset(kind, data_cfg, ckpt, Xtr_raw, Xte_raw):
    normalize = str(data_cfg.get("normalize", "zscore")).lower()
    if normalize == "zscore":
        return zscore_with_ckpt_stats_or_train(Xtr_raw, Xte_raw, ckpt.get("stats", None))
    return Xtr_raw, Xte_raw


def evaluate_one_series(
    cfg, data_cfg, train_cfg, test_cfg, device, use_amp,
    kind, series_name, series_dir, data_bundle
):
    Xtr, Xte, yte = data_bundle

    ckpt_path, ckpt = load_ckpt_prefer_best(series_dir)
    model, meta = build_model_from_cfg(cfg, ckpt, data_cfg, train_cfg, Xtr)

    from torch.utils.data import DataLoader
    test_ds = WindowedTSDataset(
        Xte, yte,
        meta["win_size"],
        meta["step_size"],
        split="test",
        drop_anom_in_train=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=int(test_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(test_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    if "center_c" in ckpt and ckpt["center_c"] is not None:
        model.c.copy_(ckpt["center_c"].to(model.c.device))

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    eval_t0 = time.perf_counter()

    SVD, REC, labels, latency_info = infer_components(
        model,
        test_dl,
        device,
        use_amp=use_amp,
        measure_latency=bool(test_cfg.get("measure_latency", True)),
        latency_warmup_batches=int(test_cfg.get("latency_warmup_batches", 10)),
        latency_max_batches=int(test_cfg.get("latency_max_batches", 50)),
    )

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    eval_t1 = time.perf_counter()
    total_eval_time_sec = float(eval_t1 - eval_t0)

    if device.type == "cuda" and torch.cuda.is_available():
        test_peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        test_peak_gpu_memory_mb = float("nan")

    if bool(test_cfg.get("auto_search_weights", True)):
        best = search_best_weights(
            SVD, REC, labels,
            n_div=int(test_cfg.get("weight_search_n_div", 20)),
        )
    else:
        best = eval_fixed(
            SVD, REC, labels,
            s_svdd=float(test_cfg.get("s_svdd", 1.0)),
            s_rec=float(test_cfg.get("s_rec", 1.0)),
        )

    scores = clean_scores(best["scores"])

    auc, aupr = float("nan"), float("nan")
    if SKLEARN_AVAILABLE and len(np.unique(labels)) >= 2:
        try:
            auc = float(roc_auc_score(labels, scores))
        except Exception:
            auc = float("nan")
        try:
            aupr = float(average_precision_score(labels, scores))
        except Exception:
            aupr = float("nan")

    row = dict(
        series=series_name,
        f1=best["f1"],
        p=best["p"],
        r=best["r"],
        auc=auc,
        aupr=aupr,
        s_svdd=best["s_svdd"],
        s_rec=best["s_rec"],
        thr=best["thr"],
        tp=best["tp"],
        fp=best["fp"],
        fn=best["fn"],
        tn=best["tn"],
        ckpt_name=ckpt_path.name,
        ckpt_epoch=ckpt.get("epoch", ""),
        inference_latency_ms_per_batch=latency_info["inference_latency_ms_per_batch"],
        inference_latency_ms_per_sample=latency_info["inference_latency_ms_per_sample"],
        inference_throughput_win_per_sec=latency_info["inference_throughput_win_per_sec"],
        inference_time_sec_measured=latency_info["inference_time_sec_measured"],
        total_eval_time_sec=total_eval_time_sec,
        total_eval_throughput_win_per_sec=float(latency_info["num_eval_samples"] / total_eval_time_sec) if total_eval_time_sec > 0 else float("nan"),
        test_peak_gpu_memory_mb=test_peak_gpu_memory_mb,
        num_eval_batches=latency_info["num_eval_batches"],
        num_eval_samples=latency_info["num_eval_samples"],
        latency_measured_batches=latency_info["latency_measured_batches"],
        latency_measured_samples=latency_info["latency_measured_samples"],
        latency_warmup_batches=latency_info["latency_warmup_batches"],
    )

    print(
        f"[EFF] series={series_name} total_eval_time={row['total_eval_time_sec']:.4f} s "
        f"throughput={row['total_eval_throughput_win_per_sec']:.2f} win/s "
        f"infer_throughput={row['inference_throughput_win_per_sec']:.2f} win/s "
        f"latency={row['inference_latency_ms_per_sample']:.4f} ms/sample "
        f"({row['inference_latency_ms_per_batch']:.4f} ms/batch) "
        f"test_peak_mem={row['test_peak_gpu_memory_mb']:.2f} MB"
    )

    if device.type == "cuda":
        del model
        torch.cuda.empty_cache()

    return row


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser("Unified ENC+SVDD test")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg, train_cfg, test_cfg = cfg["data"], cfg["training"], cfg["testing"]

    kind = infer_dataset_kind(data_cfg)

    set_seed(int(train_cfg["seed"]))
    device = torch.device(train_cfg["device"])

    if bool(train_cfg.get("no_cudnn", False)):
        torch.backends.cudnn.enabled = False

    use_amp = bool(test_cfg.get("amp", False)) and (device.type == "cuda")

    save_root = Path(train_cfg["save_dir"])
    ensure_dir(str(save_root))
    out_csv = save_root / infer_output_csv_name(kind)

    rows = []

    if kind in ("MSL", "SMAP"):
        series_list = discover_series(
            data_root=data_cfg["data_root"],
            dataset_only=data_cfg.get("dataset_only", None),
            meta_csv=data_cfg.get("meta_csv", None),
            include_expr=data_cfg.get("include", None),
        )
        if len(series_list) == 0:
            raise RuntimeError("未发现任何可评估序列，请检查 data_root / dataset_only / meta_csv / include 配置。")

        for s in series_list:
            loader = MSLLoader(
                data_root=data_cfg["data_root"],
                normalize=data_cfg.get("normalize", "zscore"),
                force_dim=data_cfg.get("force_dim", None),
                series=s,
                dataset_only=data_cfg.get("dataset_only", None),
                meta_csv=data_cfg.get("meta_csv", None),
            )
            Xtr, Xte, yte, _ = loader.load_raw()
            series_dir = save_root / s
            try:
                row = evaluate_one_series(
                    cfg, data_cfg, train_cfg, test_cfg, device, use_amp,
                    kind, s, series_dir, (Xtr, Xte, yte)
                )
                rows.append(row)
                print(
                    f"[{s}] F1={row['f1']:.4f} P={row['p']:.4f} R={row['r']:.4f} "
                    f"AUC={row['auc']:.4f} AUPR={row['aupr']:.4f} "
                    f"TOTAL_TIME={row['total_eval_time_sec']:.4f}s "
                    f"THROUGHPUT={row['total_eval_throughput_win_per_sec']:.2f}win/s"
                )
            except FileNotFoundError:
                continue

    elif kind == "SMD":
        test_dir = Path(data_cfg["data_root"]) / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"SMD test 目录不存在: {test_dir}")

        series_list = sorted(p.stem for p in test_dir.glob("*.txt"))
        if len(series_list) == 0:
            raise RuntimeError("SMD test 目录下未发现任何 .txt 文件")

        for s in series_list:
            try:
                Xtr_raw, Xte_raw, yte = load_smd_series_raw(data_cfg["data_root"], s)
                normalize = str(data_cfg.get("normalize", "zscore")).lower()

                series_dir = save_root / s
                _, ckpt = load_ckpt_prefer_best(series_dir)

                if normalize == "zscore":
                    Xtr, Xte = zscore_with_ckpt_stats_or_train(Xtr_raw, Xte_raw, ckpt.get("stats", None))
                else:
                    Xtr, Xte = Xtr_raw, Xte_raw

                row = evaluate_one_series(
                    cfg, data_cfg, train_cfg, test_cfg, device, use_amp,
                    kind, s, series_dir, (Xtr, Xte, yte)
                )
                rows.append(row)
                print(
                    f"[{s}] F1={row['f1']:.4f} P={row['p']:.4f} R={row['r']:.4f} "
                    f"AUC={row['auc']:.4f} AUPR={row['aupr']:.4f} "
                    f"TOTAL_TIME={row['total_eval_time_sec']:.4f}s "
                    f"THROUGHPUT={row['total_eval_throughput_win_per_sec']:.2f}win/s"
                )
            except FileNotFoundError:
                continue

    elif kind in ("SWAT", "WADI", "SINGLE_NPY"):
        series_name = str(data_cfg.get("series", kind) or kind)
        series_dir = save_root / series_name

        _, ckpt = load_ckpt_prefer_best(series_dir)

        Xtr_raw, Xte_raw, yte = load_single_npy_dataset(data_cfg["data_root"], kind, data_cfg)
        Xtr, Xte = maybe_normalize_single_dataset(kind, data_cfg, ckpt, Xtr_raw, Xte_raw)

        row = evaluate_one_series(
            cfg, data_cfg, train_cfg, test_cfg, device, use_amp,
            kind, series_name, series_dir, (Xtr, Xte, yte)
        )
        rows.append(row)
        print(
            f"[{series_name}] F1={row['f1']:.4f} P={row['p']:.4f} R={row['r']:.4f} "
            f"AUC={row['auc']:.4f} AUPR={row['aupr']:.4f} "
            f"TOTAL_TIME={row['total_eval_time_sec']:.4f}s"
        )

    else:
        raise ValueError(f"Unsupported dataset_kind: {kind}")

    if not rows:
        return

    if len(rows) > 1:
        mean_row = dict(
            series="MEAN",
            f1=float(np.mean([r["f1"] for r in rows])),
            p=float(np.mean([r["p"] for r in rows])),
            r=float(np.mean([r["r"] for r in rows])),
            auc=float(np.nanmean([r["auc"] for r in rows])),
            aupr=float(np.nanmean([r["aupr"] for r in rows])),
            s_svdd=float("nan"),
            s_rec=float("nan"),
            thr=float("nan"),
            tp="",
            fp="",
            fn="",
            tn="",
            ckpt_name="",
            ckpt_epoch="",
            inference_latency_ms_per_batch=float("nan"),
            inference_latency_ms_per_sample=float("nan"),
            inference_throughput_win_per_sec=(
                float(np.sum([r["latency_measured_samples"] for r in rows]) / np.sum([r["inference_time_sec_measured"] for r in rows]))
                if np.sum([r["inference_time_sec_measured"] for r in rows]) > 0 else float("nan")
            ),
            inference_time_sec_measured=float(np.sum([r["inference_time_sec_measured"] for r in rows])),
            total_eval_time_sec=float(np.sum([r["total_eval_time_sec"] for r in rows])),
            total_eval_throughput_win_per_sec=(
                float(np.sum([r["num_eval_samples"] for r in rows]) / np.sum([r["total_eval_time_sec"] for r in rows]))
                if np.sum([r["total_eval_time_sec"] for r in rows]) > 0 else float("nan")
            ),
            test_peak_gpu_memory_mb=float("nan"),
            num_eval_batches=int(np.sum([r["num_eval_batches"] for r in rows])),
            num_eval_samples=int(np.sum([r["num_eval_samples"] for r in rows])),
            latency_measured_batches=int(np.sum([r["latency_measured_batches"] for r in rows])),
            latency_measured_samples=int(np.sum([r["latency_measured_samples"] for r in rows])),
            latency_warmup_batches="",
        )
        rows.append(mean_row)
        print(
            f"[MEAN] F1={mean_row['f1']:.4f} P={mean_row['p']:.4f} R={mean_row['r']:.4f} "
            f"AUC={mean_row['auc']:.4f} AUPR={mean_row['aupr']:.4f} "
            f"TOTAL_TIME_SUM={mean_row['total_eval_time_sec']:.4f}s "
            f"THROUGHPUT={mean_row['total_eval_throughput_win_per_sec']:.2f}win/s"
        )
    else:
        only = rows[0]
        mean_row = dict(
            series="MEAN",
            f1=only["f1"],
            p=only["p"],
            r=only["r"],
            auc=only["auc"],
            aupr=only["aupr"],
            s_svdd=float("nan"),
            s_rec=float("nan"),
            thr=float("nan"),
            tp="",
            fp="",
            fn="",
            tn="",
            ckpt_name="",
            ckpt_epoch="",
            inference_latency_ms_per_batch=only["inference_latency_ms_per_batch"],
            inference_latency_ms_per_sample=only["inference_latency_ms_per_sample"],
            inference_throughput_win_per_sec=only["inference_throughput_win_per_sec"],
            inference_time_sec_measured=only["inference_time_sec_measured"],
            total_eval_time_sec=only["total_eval_time_sec"],
            total_eval_throughput_win_per_sec=only["total_eval_throughput_win_per_sec"],
            test_peak_gpu_memory_mb=only["test_peak_gpu_memory_mb"],
            num_eval_batches=only["num_eval_batches"],
            num_eval_samples=only["num_eval_samples"],
            latency_measured_batches=only["latency_measured_batches"],
            latency_measured_samples=only["latency_measured_samples"],
            latency_warmup_batches=only["latency_warmup_batches"],
        )
        rows.append(mean_row)
        print(
            f"[MEAN] F1={mean_row['f1']:.4f} P={mean_row['p']:.4f} R={mean_row['r']:.4f} "
            f"AUC={mean_row['auc']:.4f} AUPR={mean_row['aupr']:.4f} "
            f"TOTAL_TIME_SUM={mean_row['total_eval_time_sec']:.4f}s "
            f"THROUGHPUT={mean_row['total_eval_throughput_win_per_sec']:.2f}win/s"
        )

    write_results_csv(out_csv, rows)

    detail_rows = [r for r in rows if r.get("series") != "MEAN"]
    if detail_rows:
        valid_latency_sample = [
            r["inference_latency_ms_per_sample"]
            for r in detail_rows
            if np.isfinite(r.get("inference_latency_ms_per_sample", np.nan))
        ]
        valid_latency_batch = [
            r["inference_latency_ms_per_batch"]
            for r in detail_rows
            if np.isfinite(r.get("inference_latency_ms_per_batch", np.nan))
        ]
        valid_peak_mem = [
            r["test_peak_gpu_memory_mb"]
            for r in detail_rows
            if np.isfinite(r.get("test_peak_gpu_memory_mb", np.nan))
        ]
        valid_total_eval_time = [
            r["total_eval_time_sec"]
            for r in detail_rows
            if np.isfinite(r.get("total_eval_time_sec", np.nan))
        ]
        valid_total_eval_windows = [
            r["num_eval_samples"]
            for r in detail_rows
            if np.isfinite(r.get("num_eval_samples", np.nan))
        ]
        valid_infer_time = [
            r["inference_time_sec_measured"]
            for r in detail_rows
            if np.isfinite(r.get("inference_time_sec_measured", np.nan))
        ]
        valid_infer_windows = [
            r["latency_measured_samples"]
            for r in detail_rows
            if np.isfinite(r.get("latency_measured_samples", np.nan))
        ]

        summary = {
            "dataset_kind": kind,
            "num_series": len(detail_rows),
            "mean_inference_latency_ms_per_sample": float(np.mean(valid_latency_sample)) if valid_latency_sample else float("nan"),
            "mean_inference_latency_ms_per_batch": float(np.mean(valid_latency_batch)) if valid_latency_batch else float("nan"),
            "mean_test_peak_gpu_memory_mb": float(np.mean(valid_peak_mem)) if valid_peak_mem else float("nan"),
            "total_eval_time_sec_sum": float(np.sum(valid_total_eval_time)) if valid_total_eval_time else float("nan"),
            "mean_total_eval_time_sec_per_series": float(np.mean(valid_total_eval_time)) if valid_total_eval_time else float("nan"),
            "total_eval_throughput_win_per_sec": (
                float(np.sum(valid_total_eval_windows) / np.sum(valid_total_eval_time))
                if valid_total_eval_windows and valid_total_eval_time and np.sum(valid_total_eval_time) > 0 else float("nan")
            ),
            "inference_throughput_win_per_sec": (
                float(np.sum(valid_infer_windows) / np.sum(valid_infer_time))
                if valid_infer_windows and valid_infer_time and np.sum(valid_infer_time) > 0 else float("nan")
            ),
            "rows_csv": str(out_csv),
            "measure_latency": bool(test_cfg.get("measure_latency", True)),
            "latency_warmup_batches": int(test_cfg.get("latency_warmup_batches", 10)),
            "latency_max_batches": int(test_cfg.get("latency_max_batches", 50)),
        }
        summary_path = save_root / f"{kind}_eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(
            f"[EVAL SUMMARY] kind={kind} total_eval_time={summary['total_eval_time_sec_sum']:.4f} s "
            f"mean_per_series_time={summary['mean_total_eval_time_sec_per_series']:.4f} s "
            f"throughput={summary['total_eval_throughput_win_per_sec']:.2f} win/s "
            f"infer_throughput={summary['inference_throughput_win_per_sec']:.2f} win/s "
            f"mean_latency={summary['mean_inference_latency_ms_per_sample']:.4f} ms/sample "
            f"mean_test_peak_mem={summary['mean_test_peak_gpu_memory_mb']:.2f} MB -> {summary_path}"
        )


if __name__ == "__main__":
    main()