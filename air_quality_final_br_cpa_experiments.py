# -*- coding: utf-8 -*-
"""
Final confirmatory experiments for BR-CPA-RNet.

Primary setting:
  BR-CPA-RNet = fixed bounded residual CPA (alpha=0.30)
  + compact S5P context, max age 48 h.

Predefined ablations/sensitivity:
  Persistence
  HistGBResidual
  Local-S5P-48h
  BR-CPA-no-S5P
  BR-CPA-RNet
  Adaptive-gated-S5P-48h
  BR-CPA-S5P-24h
  BR-CPA-S5P-12h

Statistics:
  paired moving-block bootstrap on absolute-error differences,
  stratified by rolling fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import air_quality_final_compact_cpa as base
import air_quality_sg_cpa_rnet_v2 as v2


PRIMARY_MODEL = "BR-CPA-RNet"
DEFAULT_SAT = Path("results_s5p_exact/s5p_aligned_30min.csv")
DEFAULT_OUT = Path("results_br_cpa_final")


def satellite_variant(sat, age_limit_h=None, no_satellite=False):
    z = sat.copy()

    for pol in v2.SAT_PRODUCTS:
        mean_col = f"S5P_{pol}_mean"
        age_col = f"S5P_{pol}_age_h"
        av_col = f"S5P_{pol}_available"

        if no_satellite:
            z[mean_col] = np.nan
            z[age_col] = np.nan
            z[av_col] = 0.0
            continue

        av = pd.to_numeric(z[av_col], errors="coerce").fillna(0.0) > 0.5
        age = pd.to_numeric(z[age_col], errors="coerce")

        if age_limit_h is not None:
            av = av & age.notna() & (age >= 0) & (age <= float(age_limit_h))

        z.loc[~av, mean_col] = np.nan
        z.loc[~av, age_col] = np.nan
        z[av_col] = av.astype(float)

    return z


def metrics_from_pack(pack, pred, label, fold):
    df = base.regression_metrics(pack.y, pred, pack.mask, label)
    df.insert(0, "fold", fold)
    return df


def prediction_long(fold, model, pack, pred):
    rows = []
    ts = pd.to_datetime(pack.ts)

    for j, pol in enumerate(base.TARGETS):
        valid = (
            (pack.mask[:, j] > 0.5)
            & np.isfinite(pack.y[:, j])
            & np.isfinite(pred[:, j])
        )

        for t, y, p, last in zip(
            ts[valid],
            pack.y[valid, j],
            pred[valid, j],
            pack.last[valid, j],
        ):
            rows.append(
                {
                    "fold": fold,
                    "timestamp": t,
                    "model": model,
                    "pollutant": pol,
                    "observed": float(y),
                    "predicted": float(p),
                    "persistence": float(last),
                    "abs_error": float(abs(y - p)),
                }
            )

    return pd.DataFrame(rows)


def circular_block_mean(x, block_length, rng):
    x = np.asarray(x, float)
    n = len(x)
    L = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / L))

    parts = []
    for _ in range(n_blocks):
        start = int(rng.integers(0, n))
        idx = (start + np.arange(L)) % n
        parts.append(x[idx])

    return float(np.mean(np.concatenate(parts)[:n]))


def paired_block_bootstrap(
    diff_by_fold: Dict[int, np.ndarray],
    comp_mae_by_fold: Dict[int, float],
    block_length=12,
    reps=5000,
    seed=20260827,
):
    clean = {}
    weights = {}

    for fold, x in diff_by_fold.items():
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x):
            clean[fold] = x
            weights[fold] = len(x)

    if not clean:
        return {
            "mean_abs_error_difference": np.nan,
            "improvement_pct": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "bootstrap_p_two_sided": np.nan,
            "n": 0,
        }

    total_n = int(sum(weights.values()))

    point = sum(weights[f] * np.mean(clean[f]) for f in clean) / total_n
    comp_mae = sum(
        weights[f] * comp_mae_by_fold[f] for f in clean
    ) / total_n

    improvement_pct = 100.0 * point / comp_mae if comp_mae > 0 else np.nan

    rng = np.random.default_rng(seed)
    boots = np.empty(reps, float)

    for b in range(reps):
        num = 0.0
        den = 0

        for f, x in clean.items():
            m = circular_block_mean(x, block_length, rng)
            num += weights[f] * m
            den += weights[f]

        boots[b] = num / den

    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = min(
        1.0,
        2.0 * min(np.mean(boots <= 0), np.mean(boots >= 0)),
    )

    return {
        "mean_abs_error_difference": float(point),
        "improvement_pct": float(improvement_pct),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_p_two_sided": float(p),
        "n": total_n,
    }


def significance_table(pred_long, block_length, reps):
    comparators = [
        "Persistence",
        "HistGBResidual",
        "Local-S5P-48h",
        "BR-CPA-no-S5P",
        "Adaptive-gated-S5P-48h",
    ]

    rows = []

    for pol in base.TARGETS:
        p = pred_long[
            (pred_long.model == PRIMARY_MODEL)
            & (pred_long.pollutant == pol)
        ][["fold", "timestamp", "abs_error"]].rename(
            columns={"abs_error": "primary_abs_error"}
        )

        for comp in comparators:
            c = pred_long[
                (pred_long.model == comp)
                & (pred_long.pollutant == pol)
            ][["fold", "timestamp", "abs_error"]].rename(
                columns={"abs_error": "comparator_abs_error"}
            )

            z = p.merge(c, on=["fold", "timestamp"], how="inner")

            diff_by_fold = {}
            comp_mae_by_fold = {}

            for fold, g in z.groupby("fold"):
                diff_by_fold[int(fold)] = (
                    g.comparator_abs_error.to_numpy(float)
                    - g.primary_abs_error.to_numpy(float)
                )
                comp_mae_by_fold[int(fold)] = float(
                    g.comparator_abs_error.mean()
                )

            res = paired_block_bootstrap(
                diff_by_fold,
                comp_mae_by_fold,
                block_length=block_length,
                reps=reps,
                seed=20260827 + len(rows) * 101,
            )

            rows.append(
                {
                    "primary_model": PRIMARY_MODEL,
                    "comparator": comp,
                    "pollutant": pol,
                    "block_length_steps": block_length,
                    "block_length_hours": block_length * 0.5,
                    "bootstrap_reps": reps,
                    **res,
                    "primary_better_ci95": (
                        np.isfinite(res["ci95_low"])
                        and res["ci95_low"] > 0
                    ),
                }
            )

    return pd.DataFrame(rows)


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base.DATA_PATH = args.ground_xlsx
    grid, pcols, mcols = base.load_data()

    sat_raw = v2.load_satellite_frame(args.sat_csv, grid.index)
    sat_cols = v2.compact_sat_columns(sat_raw)

    folds = v2.rolling_folds(len(grid), args.folds)

    variants = [
        ("Local-S5P-48h", "none", 48.0, False),
        ("BR-CPA-no-S5P", "fixed", None, True),
        (PRIMARY_MODEL, "fixed", 48.0, False),
        ("Adaptive-gated-S5P-48h", "gated", 48.0, False),
        ("BR-CPA-S5P-24h", "fixed", 24.0, False),
        ("BR-CPA-S5P-12h", "fixed", 12.0, False),
    ]

    all_metrics = []
    all_preds = []
    all_gates = []
    complexity_rows = []

    print("\nFINAL BR-CPA-RNet EXPERIMENT")
    print(f"Ground rows: {len(grid):,}")
    print(f"Rolling folds: {len(folds)}")
    print(f"Neural seeds: {args.seeds}")
    print(f"alpha_max: {args.alpha_max:.2f}")
    print(
        f"Bootstrap: {args.bootstrap_reps} reps, "
        f"block={args.block_length} steps "
        f"({args.block_length * 0.5:.1f} h)"
    )
    print("Primary S5P age limit: 48 h")
    print("12 h and 24 h variants are sensitivity analyses only.\n")

    for fold_no, (tr_idx, va_idx, te_idx) in enumerate(folds, start=1):
        print(f"=== Rolling fold {fold_no}/{len(folds)} ===")

        f, branch, meteo = base.build_base_features(grid, pcols, mcols)
        fs_state = base.fit_feature_state(f, branch, meteo, tr_idx)
        fs, meteo_out = base.apply_feature_state(
            f, branch, meteo, fs_state, include_context=True
        )

        tr0 = base.make_pack(fs, grid, tr_idx, branch, meteo_out, pcols)
        va0 = base.make_pack(fs, grid, va_idx, branch, meteo_out, pcols)
        te0 = base.make_pack(fs, grid, te_idx, branch, meteo_out, pcols)

        # Common 48 h reference pack.
        sat_ref = satellite_variant(sat_raw, 48.0, False)
        sat_state = v2.fit_sat_state(sat_ref, sat_cols, tr_idx)
        sat_scaled = v2.apply_sat_state(sat_ref, sat_cols, sat_state)

        te_ref = v2.make_sg_pack(
            fs, sat_scaled, grid, te_idx,
            branch, meteo_out, pcols, sat_cols
        )

        persistence = te0.last.copy()
        all_metrics.append(
            metrics_from_pack(te_ref, persistence, "Persistence", fold_no)
        )
        all_preds.append(
            prediction_long(fold_no, "Persistence", te_ref, persistence)
        )

        hist_pred, _ = base.histgb_residual(tr0, va0, te0)
        all_metrics.append(
            metrics_from_pack(te_ref, hist_pred, "HistGBResidual", fold_no)
        )
        all_preds.append(
            prediction_long(fold_no, "HistGBResidual", te_ref, hist_pred)
        )

        for label, cross_mode, age_limit, no_sat in variants:
            sat_v = satellite_variant(
                sat_raw,
                age_limit_h=age_limit,
                no_satellite=no_sat,
            )

            sat_state = v2.fit_sat_state(sat_v, sat_cols, tr_idx)
            sat_scaled = v2.apply_sat_state(sat_v, sat_cols, sat_state)

            tr = v2.make_sg_pack(
                fs, sat_scaled, grid, tr_idx,
                branch, meteo_out, pcols, sat_cols
            )
            va = v2.make_sg_pack(
                fs, sat_scaled, grid, va_idx,
                branch, meteo_out, pcols, sat_cols
            )
            te = v2.make_sg_pack(
                fs, sat_scaled, grid, te_idx,
                branch, meteo_out, pcols, sat_cols
            )

            scale = v2.fit_scale_only(tr)
            gf = tr.Xg.shape[-1]
            mf = tr.Xm.shape[-1]
            sf = tr.Xs.shape[-1]

            print(
                f"  {label} | cross={cross_mode} | "
                f"S5P={'OFF' if no_sat else str(int(age_limit)) + 'h'}"
            )

            def factory(cm=cross_mode, gf=gf, mf=mf, sf=sf):
                return v2.SGCPARNetV2(
                    ground_f=gf,
                    meteo_f=mf,
                    sat_f=sf,
                    cross_mode=cm,
                    alpha_max=args.alpha_max,
                )

            if fold_no == 1:
                tmp = factory()
                complexity_rows.append(
                    {
                        "model": label,
                        "trainable_parameters": v2.trainable_parameters(tmp),
                        "alpha_max": args.alpha_max,
                        "satellite_features": len(sat_cols),
                        "satellite_age_limit_h": (
                            np.nan if no_sat else age_limit
                        ),
                    }
                )

            pred, diag = v2.train_ensemble(
                factory, tr, va, te, scale, args.seeds
            )

            all_metrics.append(
                metrics_from_pack(te, pred, label, fold_no)
            )
            all_preds.append(
                prediction_long(fold_no, label, te, pred)
            )
            all_gates.append(
                v2.gate_summary(
                    diag, label, fold_no, args.alpha_max
                )
            )

    metrics = pd.concat(all_metrics, ignore_index=True)
    pred_long = pd.concat(all_preds, ignore_index=True)
    gates = pd.concat(all_gates, ignore_index=True)
    complexity = pd.DataFrame(complexity_rows)

    summary = (
        metrics.groupby(["model", "pollutant"], as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            nMAE_mean=("nMAE", "mean"),
        )
    )

    pmae = summary[
        summary.model == PRIMARY_MODEL
    ][["pollutant", "MAE_mean"]].rename(
        columns={"MAE_mean": "primary_MAE"}
    )

    summary = summary.merge(pmae, on="pollutant", how="left")
    summary["primary_improvement_over_model_pct"] = (
        (summary.MAE_mean - summary.primary_MAE)
        / summary.MAE_mean
        * 100.0
    )

    sig = significance_table(
        pred_long,
        block_length=args.block_length,
        reps=args.bootstrap_reps,
    )

    sensitivity = summary[
        summary.model.isin(
            [PRIMARY_MODEL, "BR-CPA-S5P-24h", "BR-CPA-S5P-12h"]
        )
    ].copy()
    sensitivity["setting_role"] = np.where(
        sensitivity.model == PRIMARY_MODEL,
        "PRIMARY",
        "SENSITIVITY_ONLY",
    )

    metrics.to_csv(args.out_dir / "final_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "final_summary.csv", index=False)
    pred_long.to_csv(
        args.out_dir / "final_predictions_long.csv", index=False
    )
    sig.to_csv(
        args.out_dir / "final_bootstrap_significance.csv", index=False
    )
    sensitivity.to_csv(
        args.out_dir / "final_satellite_age_sensitivity.csv", index=False
    )
    gates.to_csv(
        args.out_dir / "final_gate_summary.csv", index=False
    )
    complexity.to_csv(
        args.out_dir / "final_model_complexity.csv", index=False
    )

    print("\n=== FINAL ROLLING-ORIGIN SUMMARY ===")
    print(
        summary[
            [
                "model",
                "pollutant",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "R2_mean",
                "nMAE_mean",
                "primary_improvement_over_model_pct",
            ]
        ].to_string(index=False)
    )

    print("\n=== PAIRED MOVING-BLOCK BOOTSTRAP ===")
    print(
        sig[
            [
                "comparator",
                "pollutant",
                "mean_abs_error_difference",
                "improvement_pct",
                "ci95_low",
                "ci95_high",
                "bootstrap_p_two_sided",
                "primary_better_ci95",
            ]
        ].to_string(index=False)
    )

    print("\nInterpretation:")
    print("  difference > 0 => BR-CPA-RNet has lower MAE.")
    print("  primary_better_ci95=True => 95% CI stays above zero.")
    print("  12 h / 24 h are sensitivity analyses only.")
    print(f"\nSaved to: {args.out_dir.resolve()}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ground-xlsx",
        type=Path,
        default=Path("2019-10-efir_1-3.xlsx"),
    )
    ap.add_argument(
        "--sat-csv",
        type=Path,
        default=DEFAULT_SAT,
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alpha-max", type=float, default=0.30)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument(
        "--block-length",
        type=int,
        default=12,
        help="30-min steps; 12 = 6 h.",
    )
    ap.add_argument("--quick", action="store_true")

    args = ap.parse_args()

    if args.quick:
        args.folds = 1
        args.seeds = 1
        args.bootstrap_reps = 300

    if not args.ground_xlsx.exists():
        raise FileNotFoundError(
            f"Ground workbook not found: {args.ground_xlsx}"
        )
    if not args.sat_csv.exists():
        raise FileNotFoundError(
            f"Satellite CSV not found: {args.sat_csv}"
        )

    run(args)


if __name__ == "__main__":
    main()
