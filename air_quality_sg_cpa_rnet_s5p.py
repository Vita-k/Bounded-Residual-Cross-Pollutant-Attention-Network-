# -*- coding: utf-8 -*-
"""
FULL SG-CPA-RNet + Sentinel-5P/TROPOMI EXPERIMENT
==================================================

This companion script extends `air_quality_final_compact_cpa.py` and directly
addresses the reviewer concern that earlier numerical results were obtained for
Compact CPA-BiLSTM without the satellite branch and without a trained selective
cross-pollutant mechanism.

What is tested here
-------------------
A0  SG-CPA-RNet ground + meteorology, selective cross-pollutant gate
A1  A0 + Sentinel-5P neighbourhood mean
A2  A0 + Sentinel-5P distance-weighted context
A3  A0 + Sentinel-5P wind-aware context
A4  Full SG-CPA-RNet + multi-feature S5P context + quality/age gate
A5  A4 without selective cross-pollutant gate (unconditional CPA)

The full model contains:
- pollutant-specific compact temporal encoders;
- Cross-Pollutant Multi-Head Attention;
- learned Selective Cross-Pollutant Gate;
- shared meteorological encoder;
- Sentinel-5P context encoder with learned quality/age-aware satellite gate;
- pollutant-specific temporal decoders;
- pollutant-specific residual heads;
- learned adaptive persistence-residual coefficient lambda_{p,t}.

Evaluation
----------
The default mode uses 3 expanding-window rolling-origin folds and 5 neural
seeds. This is intentionally separate from the older one-test-interval Compact
CPA-BiLSTM experiment.

Inputs
------
1) 2019-10-efir_1-3.xlsx
2) results_s5p/s5p_aligned_30min.csv
   Generate it first with:
      python prepare_sentinel5p_tropomi.py --ground-xlsx 2019-10-efir_1-3.xlsx

Outputs
-------
results_sg_cpa_s5p/
  rolling_fold_metrics.csv
  rolling_summary.csv
  gate_summary.csv
  fold_predictions_*.csv

Required packages are the same as the ground experiment plus those required by
`prepare_sentinel5p_tropomi.py` for the preprocessing step.
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import RobustScaler

# Import the validated ground+meteorology pipeline as the common data/metric base.
import air_quality_final_compact_cpa as base


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_SAT_CSV = Path("results_s5p/s5p_aligned_30min.csv")
DEFAULT_OUT = Path("results_sg_cpa_s5p")

SAT_PRODUCTS = ("CO", "NO2", "SO2")
SAT_MAX_AGE_H = 36.0
D_MODEL = 48
METEO_D = 48
SAT_D = 32
DECODER_D = 64


# -----------------------------------------------------------------------------
# Rolling-origin folds
# -----------------------------------------------------------------------------
def rolling_folds(n: int, n_folds: int = 3) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Expanding train, non-overlapping forward validation/test windows."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    # Deliberately conservative initial training interval for a one-month case study.
    if n_folds == 1:
        i1 = int(n * 0.70)
        i2 = int(n * 0.85)
        return [(np.arange(i1), np.arange(i1, i2), np.arange(i2, n))]

    # 3-fold default: train 55/65/75%, val next 10%, test next 10/10/15%.
    # For other fold counts use a similar expanding construction.
    initial = 0.55
    val_frac = 0.10
    remaining = 1.0 - initial - val_frac
    test_frac = remaining / n_folds

    folds = []
    for k in range(n_folds):
        train_end = initial + k * test_frac
        val_end = train_end + val_frac
        test_end = 1.0 if k == n_folds - 1 else min(1.0, val_end + test_frac)
        i1 = max(int(n * train_end), base.HISTORY_STEPS + base.HORIZON_STEPS + 10)
        i2 = max(int(n * val_end), i1 + 20)
        i3 = max(int(n * test_end), i2 + 20)
        i3 = min(i3, n)
        if i3 <= i2:
            continue
        folds.append((np.arange(i1), np.arange(i1, i2), np.arange(i2, i3)))
    return folds


# -----------------------------------------------------------------------------
# Satellite feature preparation: training-only filling/scaling
# -----------------------------------------------------------------------------
def load_satellite_frame(path: Path, grid_index: pd.DatetimeIndex) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Satellite feature file not found: {path}. Run prepare_sentinel5p_tropomi.py first."
        )
    sat = pd.read_csv(path, index_col=0, parse_dates=True)
    sat.index = pd.to_datetime(sat.index)
    sat = sat.sort_index().reindex(grid_index)
    return sat


def sat_columns(mode: str, sat: pd.DataFrame) -> List[str]:
    mode = mode.lower()
    if mode == "none":
        return []

    stat_map = {
        "mean": ["mean", "age_h", "available"],
        "distance": ["distance_weighted", "age_h", "available"],
        "wind": ["wind_weighted", "age_h", "available"],
        "full": [
            "mean", "distance_weighted", "wind_weighted", "std",
            "n_valid", "mean_qa", "mean_distance_km", "age_h", "available",
        ],
    }
    if mode not in stat_map:
        raise ValueError(f"Unknown satellite mode: {mode}")

    cols = []
    for pol in SAT_PRODUCTS:
        for stat in stat_map[mode]:
            c = f"S5P_{pol}_{stat}"
            if c in sat.columns:
                cols.append(c)
    if not cols:
        raise RuntimeError(f"No Sentinel-5P columns found for mode={mode}")
    return cols


@dataclass
class SatState:
    fill: Dict[str, float]
    scaler: RobustScaler
    scale_cols: List[str]


def fit_sat_state(sat: pd.DataFrame, cols: Sequence[str], tr_idx: np.ndarray) -> SatState:
    z = sat[list(cols)].copy()
    fill: Dict[str, float] = {}
    for c in cols:
        if c.endswith("_available"):
            fill[c] = 0.0
        elif c.endswith("_age_h"):
            fill[c] = SAT_MAX_AGE_H + 12.0
        elif c.endswith("_n_valid"):
            fill[c] = 0.0
        else:
            med = z.iloc[tr_idx][c].median(skipna=True)
            fill[c] = float(med) if pd.notna(med) and np.isfinite(med) else 0.0
        z[c] = z[c].fillna(fill[c])

    scale_cols = [c for c in cols if not c.endswith("_available")]
    sc = RobustScaler(quantile_range=(10, 90))
    if scale_cols:
        sc.fit(z.iloc[tr_idx][scale_cols])
    return SatState(fill=fill, scaler=sc, scale_cols=scale_cols)


def apply_sat_state(sat: pd.DataFrame, cols: Sequence[str], state: SatState) -> pd.DataFrame:
    z = sat[list(cols)].copy()
    for c, v in state.fill.items():
        z[c] = z[c].fillna(v)
    if state.scale_cols:
        z.loc[:, state.scale_cols] = state.scaler.transform(z[state.scale_cols])
    return z


# -----------------------------------------------------------------------------
# Window packs
# -----------------------------------------------------------------------------
@dataclass
class SGPack:
    Xg: np.ndarray
    Xm: np.ndarray
    Xs: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    last: np.ndarray
    delta: np.ndarray
    ts: np.ndarray


def make_sg_pack(
    fs: pd.DataFrame,
    sat_scaled: Optional[pd.DataFrame],
    grid: pd.DataFrame,
    idx: np.ndarray,
    branch: Dict[str, List[str]],
    meteo: Sequence[str],
    pcols: Dict[str, str],
    sat_cols: Sequence[str],
) -> SGPack:
    start, end = int(idx[0]), int(idx[-1])
    G, M, S, Y, Mask, Last, Delta, Ts = [], [], [], [], [], [], [], []

    first_target = start + base.HISTORY_STEPS - 1 + base.HORIZON_STEPS
    for ti in range(first_target, end + 1):
        hist_end = ti - base.HORIZON_STEPS
        hist_start = hist_end - base.HISTORY_STEPS + 1
        if hist_start < start:
            continue

        gb, yy, mm, ll, dd = [], [], [], [], []
        for p in base.TARGETS:
            gb.append(fs.iloc[hist_start:hist_end + 1][branch[p]].to_numpy(np.float32))
            target = grid.iloc[ti][pcols[p]]
            hist = grid.iloc[hist_start:hist_end + 1][pcols[p]].dropna()
            last = float(hist.iloc[-1]) if len(hist) else np.nan
            yy.append(float(target) if pd.notna(target) else np.nan)
            mm.append(float(pd.notna(target) and np.isfinite(last)))
            ll.append(last)
            dd.append(float(target - last) if pd.notna(target) and np.isfinite(last) else np.nan)

        G.append(np.stack(gb, axis=1))
        M.append(fs.iloc[hist_start:hist_end + 1][list(meteo)].to_numpy(np.float32))
        if sat_cols:
            S.append(sat_scaled.iloc[hist_start:hist_end + 1][list(sat_cols)].to_numpy(np.float32))
        else:
            S.append(np.zeros((base.HISTORY_STEPS, 1), dtype=np.float32))
        Y.append(yy); Mask.append(mm); Last.append(ll); Delta.append(dd); Ts.append(grid.index[ti])

    return SGPack(
        np.asarray(G, np.float32), np.asarray(M, np.float32), np.asarray(S, np.float32),
        np.asarray(Y, np.float32), np.asarray(Mask, np.float32),
        np.asarray(Last, np.float32), np.asarray(Delta, np.float32), np.asarray(Ts),
    )


@dataclass
class ScaleOnly:
    scale: np.ndarray


def fit_scale_only(pack: SGPack) -> ScaleOnly:
    scales = []
    for j in range(len(base.TARGETS)):
        x = pack.delta[:, j]
        x = x[np.isfinite(x)]
        if len(x):
            q10, q90 = np.quantile(x, [0.10, 0.90])
            sc = float(q90 - q10)
        else:
            sc = 1.0
        scales.append(sc if np.isfinite(sc) and sc > 1e-8 else 1.0)
    return ScaleOnly(np.asarray(scales, dtype=np.float32))


class SGDataset(Dataset):
    def __init__(self, pack: SGPack, scale: ScaleOnly):
        self.g = torch.tensor(pack.Xg, dtype=torch.float32)
        self.m = torch.tensor(pack.Xm, dtype=torch.float32)
        self.s = torch.tensor(pack.Xs, dtype=torch.float32)
        dnorm = pack.delta / scale.scale[None, :]
        self.d = torch.tensor(np.nan_to_num(dnorm, nan=0.0), dtype=torch.float32)
        self.mask = torch.tensor(pack.mask, dtype=torch.float32)

    def __len__(self):
        return len(self.g)

    def __getitem__(self, i):
        return self.g[i], self.m[i], self.s[i], self.d[i], self.mask[i]


# -----------------------------------------------------------------------------
# Full SG-CPA-RNet
# -----------------------------------------------------------------------------
class SGCPARNet(nn.Module):
    def __init__(
        self,
        ground_f: int,
        meteo_f: int,
        sat_f: int,
        use_satellite: bool = True,
        selective_gate: bool = True,
    ):
        super().__init__()
        self.use_satellite = use_satellite
        self.selective_gate = selective_gate
        P = len(base.TARGETS)

        self.encoders = nn.ModuleList([
            base.CompactPollutantEncoder(ground_f, D_MODEL) for _ in range(P)
        ])
        self.cpa = base.CPA(D_MODEL, heads=4, enabled=True)

        if selective_gate:
            self.cross_gate = nn.Sequential(
                nn.Linear(2 * D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1), nn.Sigmoid()
            )

        self.meteo_encoder = nn.Sequential(
            nn.Linear(meteo_f, METEO_D), nn.GELU(),
            nn.Linear(METEO_D, METEO_D), nn.GELU(), nn.LayerNorm(METEO_D)
        )

        if use_satellite:
            self.sat_encoder = nn.Sequential(
                nn.Linear(sat_f, SAT_D), nn.GELU(),
                nn.Linear(SAT_D, SAT_D), nn.GELU(), nn.LayerNorm(SAT_D)
            )
            # Learned regulation of sparse/asynchronous satellite context.
            self.sat_gate = nn.Sequential(
                nn.Linear(sat_f + SAT_D, SAT_D), nn.GELU(), nn.Linear(SAT_D, 1), nn.Sigmoid()
            )
        else:
            self.sat_encoder = None
            self.sat_gate = None

        context_d = D_MODEL + METEO_D + (SAT_D if use_satellite else 0)
        self.fusion = nn.ModuleList([
            nn.Sequential(nn.Linear(context_d, 96), nn.GELU(), nn.LayerNorm(96))
            for _ in range(P)
        ])
        self.decoders = nn.ModuleList([
            nn.LSTM(96, DECODER_D, batch_first=True, bidirectional=False)
            for _ in range(P)
        ])
        self.temporal_pool = nn.ModuleList([base.TemporalAttention(DECODER_D) for _ in range(P)])
        self.delta_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(DECODER_D, 48), nn.GELU(), nn.Linear(48, 1))
            for _ in range(P)
        ])
        self.lambda_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(DECODER_D, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(P)
        ])

    def forward(self, g, m, s, return_diag: bool = False):
        local = torch.stack([enc(g[:, :, j, :]) for j, enc in enumerate(self.encoders)], dim=2)
        cpa = self.cpa(local)

        if self.selective_gate:
            cross_g = self.cross_gate(torch.cat([local, cpa], dim=-1))  # B,T,P,1
            h = (1.0 - cross_g) * local + cross_g * cpa
        else:
            cross_g = torch.ones((*local.shape[:-1], 1), device=local.device, dtype=local.dtype)
            h = cpa

        met = self.meteo_encoder(m)

        sat_g = None
        if self.use_satellite:
            sat_raw = self.sat_encoder(s)
            sat_g = self.sat_gate(torch.cat([s, sat_raw], dim=-1))
            sat = sat_raw * sat_g
        else:
            sat = None

        outputs, lambdas = [], []
        for p in range(len(base.TARGETS)):
            parts = [h[:, :, p, :], met]
            if sat is not None:
                parts.append(sat)
            z = self.fusion[p](torch.cat(parts, dim=-1))
            o, _ = self.decoders[p](z)
            c, _ = self.temporal_pool[p](o)
            delta_raw = self.delta_heads[p](c)
            lam = self.lambda_heads[p](c)
            outputs.append(lam * delta_raw)
            lambdas.append(lam)

        effective_delta_norm = torch.cat(outputs, dim=1)
        if not return_diag:
            return effective_delta_norm
        diag = {
            "cross_gate": cross_g,
            "lambda": torch.cat(lambdas, dim=1),
            "sat_gate": sat_g,
        }
        return effective_delta_norm, diag


# -----------------------------------------------------------------------------
# Training/inference
# -----------------------------------------------------------------------------
def masked_huber(pred, target, mask, delta=1.0):
    e = pred - target
    ae = e.abs()
    loss = torch.where(ae <= delta, 0.5 * e ** 2, delta * (ae - 0.5 * delta))
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def train_model(model, trds, vads, seed: int):
    base.set_seed(seed)
    model = model.to(base.DEVICE)
    tr = DataLoader(trds, batch_size=base.BATCH_SIZE, shuffle=True)
    va = DataLoader(vads, batch_size=base.BATCH_SIZE, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    best, state, bad = np.inf, None, 0

    for _ in range(base.MAX_EPOCHS):
        model.train()
        for g, m, s, d, mask in tr:
            g, m, s, d, mask = [x.to(base.DEVICE) for x in (g, m, s, d, mask)]
            opt.zero_grad()
            loss = masked_huber(model(g, m, s), d, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval(); vals = []
        with torch.no_grad():
            for g, m, s, d, mask in va:
                g, m, s, d, mask = [x.to(base.DEVICE) for x in (g, m, s, d, mask)]
                vals.append(float(masked_huber(model(g, m, s), d, mask).item()))
        vl = float(np.mean(vals)) if vals else np.inf
        if vl < best - 1e-5:
            best, state, bad = vl, deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= base.PATIENCE:
                break
    if state is not None:
        model.load_state_dict(state)
    return model


@torch.no_grad()
def predict_model(model: SGCPARNet, pack: SGPack, scale: ScaleOnly):
    ds = SGDataset(pack, scale)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    effs, cross, lams, sats = [], [], [], []
    model.eval()
    for g, m, s, d, mask in dl:
        eff, diag = model(g.to(base.DEVICE), m.to(base.DEVICE), s.to(base.DEVICE), return_diag=True)
        effs.append(eff.cpu().numpy())
        cross.append(diag["cross_gate"].cpu().numpy())
        lams.append(diag["lambda"].cpu().numpy())
        if diag["sat_gate"] is not None:
            sats.append(diag["sat_gate"].cpu().numpy())

    eff = np.vstack(effs)
    delta_phys = eff * scale.scale[None, :]
    pred = np.clip(pack.last + delta_phys, 0.0, None)
    diagnostics = {
        "cross_gate": np.concatenate(cross, axis=0) if cross else None,
        "lambda": np.vstack(lams) if lams else None,
        "sat_gate": np.concatenate(sats, axis=0) if sats else None,
    }
    return pred, diagnostics


def train_ensemble(factory, tr: SGPack, va: SGPack, te: SGPack, scale: ScaleOnly, seeds: int):
    trds, vads = SGDataset(tr, scale), SGDataset(va, scale)
    preds, diags = [], []
    for k in range(seeds):
        seed = base.SEED + k
        model = train_model(factory(), trds, vads, seed)
        p, d = predict_model(model, te, scale)
        preds.append(p); diags.append(d)

    pred = np.mean(np.stack(preds), axis=0)
    # Gate summaries use the last trained seed only; forecasting metrics use ensemble mean.
    return pred, diags[-1]


# -----------------------------------------------------------------------------
# Baseline and metrics
# -----------------------------------------------------------------------------
def metrics_from_pack(pack: SGPack, pred: np.ndarray, label: str, fold: int) -> pd.DataFrame:
    df = base.regression_metrics(pack.y, pred, pack.mask, label)
    df.insert(0, "fold", fold)
    return df


def gate_summary(diag: dict, label: str, fold: int) -> pd.DataFrame:
    rows = []
    cg = diag.get("cross_gate")
    lam = diag.get("lambda")
    sg = diag.get("sat_gate")
    if cg is not None:
        # B,T,P,1
        for j, pol in enumerate(base.TARGETS):
            x = cg[:, :, j, 0].reshape(-1)
            rows.append({"fold": fold, "model": label, "gate": "cross_pollutant",
                         "pollutant": pol, "mean": float(np.mean(x)),
                         "median": float(np.median(x)), "std": float(np.std(x))})
    if lam is not None:
        for j, pol in enumerate(base.TARGETS):
            x = lam[:, j]
            rows.append({"fold": fold, "model": label, "gate": "adaptive_residual_lambda",
                         "pollutant": pol, "mean": float(np.mean(x)),
                         "median": float(np.median(x)), "std": float(np.std(x))})
    if sg is not None:
        x = sg.reshape(-1)
        rows.append({"fold": fold, "model": label, "gate": "satellite_context",
                     "pollutant": "all", "mean": float(np.mean(x)),
                     "median": float(np.median(x)), "std": float(np.std(x))})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base.DATA_PATH = args.ground_xlsx

    grid, pcols, mcols = base.load_data()
    sat = load_satellite_frame(args.sat_csv, grid.index)

    fold_defs = rolling_folds(len(grid), args.folds)
    all_metrics, all_gates = [], []

    variants = [
        ("SG-ground+meteo", "none", False, True),
        ("SG+S5P-mean", "mean", True, True),
        ("SG+S5P-distance", "distance", True, True),
        ("SG+S5P-wind", "wind", True, True),
        ("SG-CPA-RNet-full", "full", True, True),
        ("Full-without-selective-gate", "full", True, False),
    ]

    for fold_no, (tr_idx, va_idx, te_idx) in enumerate(fold_defs, start=1):
        print(f"\n=== Rolling fold {fold_no}/{len(fold_defs)} ===")
        f, branch, meteo = base.build_base_features(grid, pcols, mcols)
        fs_state = base.fit_feature_state(f, branch, meteo, tr_idx)
        fs, meteo_out = base.apply_feature_state(f, branch, meteo, fs_state, include_context=True)

        # Build a ground-only pack for comparable Persistence and HistGB baselines.
        tr0 = base.make_pack(fs, grid, tr_idx, branch, meteo_out, pcols)
        va0 = base.make_pack(fs, grid, va_idx, branch, meteo_out, pcols)
        te0 = base.make_pack(fs, grid, te_idx, branch, meteo_out, pcols)

        pers = te0.last.copy()
        all_metrics.append(metrics_from_pack(
            SGPack(te0.Xg, te0.Xm, np.zeros((len(te0.y), base.HISTORY_STEPS, 1), np.float32),
                   te0.y, te0.mask, te0.last, te0.delta, te0.ts),
            pers, "Persistence", fold_no
        ))
        hist_pred, _ = base.histgb_residual(tr0, va0, te0)
        all_metrics.append(metrics_from_pack(
            SGPack(te0.Xg, te0.Xm, np.zeros((len(te0.y), base.HISTORY_STEPS, 1), np.float32),
                   te0.y, te0.mask, te0.last, te0.delta, te0.ts),
            hist_pred, "HistGBResidual", fold_no
        ))

        for label, mode, use_sat, selective in variants:
            cols = sat_columns(mode, sat)
            if cols:
                sat_state = fit_sat_state(sat, cols, tr_idx)
                sat_scaled = apply_sat_state(sat, cols, sat_state)
            else:
                sat_scaled = None

            tr = make_sg_pack(fs, sat_scaled, grid, tr_idx, branch, meteo_out, pcols, cols)
            va = make_sg_pack(fs, sat_scaled, grid, va_idx, branch, meteo_out, pcols, cols)
            te = make_sg_pack(fs, sat_scaled, grid, te_idx, branch, meteo_out, pcols, cols)
            if min(len(tr.y), len(va.y), len(te.y)) < 10:
                raise RuntimeError(f"Too few windows in fold {fold_no} for {label}")

            scale = fit_scale_only(tr)
            ground_f, meteo_f, sat_f = tr.Xg.shape[-1], tr.Xm.shape[-1], tr.Xs.shape[-1]
            factory = lambda gf=ground_f, mf=meteo_f, sf=sat_f, us=use_sat, sel=selective: SGCPARNet(
                gf, mf, sf, use_satellite=us, selective_gate=sel
            )

            print(f"  {label} | sat={mode} | selective={selective}")
            pred, diag = train_ensemble(factory, tr, va, te, scale, args.seeds)
            all_metrics.append(metrics_from_pack(te, pred, label, fold_no))
            all_gates.append(gate_summary(diag, label, fold_no))

            pred_df = pd.DataFrame({"timestamp": pd.to_datetime(te.ts)})
            for j, pol in enumerate(base.TARGETS):
                pred_df[f"{pol}_observed"] = te.y[:, j]
                pred_df[f"{pol}_predicted"] = pred[:, j]
                pred_df[f"{pol}_persistence"] = te.last[:, j]
            pred_df.to_csv(args.out_dir / f"fold{fold_no}_{label.replace('+','plus').replace(' ','_')}.csv", index=False)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(args.out_dir / "rolling_fold_metrics.csv", index=False)

    summary = (metrics.groupby(["model", "pollutant"], as_index=False)
               .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                    R2_mean=("R2", "mean"), nMAE_mean=("nMAE", "mean")))

    # Improvement relative to Persistence within pollutant, based on fold-mean MAE.
    p_mae = summary.loc[summary.model == "Persistence", ["pollutant", "MAE_mean"]].rename(
        columns={"MAE_mean": "Persistence_MAE"}
    )
    summary = summary.merge(p_mae, on="pollutant", how="left")
    summary["MAE_improvement_vs_persistence_pct"] = (
        (summary["Persistence_MAE"] - summary["MAE_mean"]) / summary["Persistence_MAE"] * 100.0
    )
    summary.to_csv(args.out_dir / "rolling_summary.csv", index=False)

    gates = pd.concat(all_gates, ignore_index=True) if all_gates else pd.DataFrame()
    gates.to_csv(args.out_dir / "gate_summary.csv", index=False)

    print("\n=== Rolling-origin summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {args.out_dir.resolve()}")
    return metrics, summary, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-xlsx", type=Path, default=Path("2019-10-efir_1-3.xlsx"))
    ap.add_argument("--sat-csv", type=Path, default=DEFAULT_SAT_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="1 fold, 1 seed for a code/data check")
    args = ap.parse_args()
    if args.quick:
        args.folds = 1
        args.seeds = 1
    if not args.ground_xlsx.exists():
        raise FileNotFoundError(f"Ground workbook not found: {args.ground_xlsx}")
    run(args)


if __name__ == "__main__":
    main()
