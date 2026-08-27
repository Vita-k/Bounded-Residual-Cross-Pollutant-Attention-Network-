# Sentinel-5P/TROPOMI extension for SG-CPA-RNet

## Goal
Address the reviewer comment that the reported numerical results were obtained for the Compact CPA-BiLSTM precursor without the Sentinel-5P branch and without a trained selective cross-pollutant mechanism.

## Step 1. Build causal Sentinel-5P features

```bash
python prepare_sentinel5p_tropomi.py --ground-xlsx 2019-10-efir_1-3.xlsx --out-dir results_s5p
```

The downloader uses the public `meeo-s5p` AWS Open Data archive, requests historical RPRO products first and uses OFFL as a fallback. It extracts a 50-km neighbourhood around the station (47.5839 N, 34.3585 E), applies product-specific `qa_value` filtering, and creates simple, distance-weighted, and wind-aware spatial summaries. The 30-min model grid receives only the latest satellite observation at or before the model time. Satellite age and availability are retained as explicit features.

Products:
- CO: `L2__CO____`, `carbonmonoxide_total_column`, `qa_value > 0.50`;
- NO2: `L2__NO2___`, `nitrogendioxide_tropospheric_column`, `qa_value > 0.75`;
- SO2: `L2__SO2___`, `sulfurdioxide_total_vertical_column`, `qa_value > 0.50`.

Main outputs:
- `s5p_pixel_samples.csv`;
- `s5p_overpass_features.csv`;
- `s5p_aligned_30min.csv`;
- `s5p_qc_summary.csv`.

## Step 2. Run the full selective model

Place `air_quality_final_compact_cpa.py`, `air_quality_sg_cpa_rnet_s5p.py`, the Excel workbook, and `results_s5p/` in the same working directory.

Quick data/code check:

```bash
python air_quality_sg_cpa_rnet_s5p.py --quick
```

Final experiment:

```bash
python air_quality_sg_cpa_rnet_s5p.py --folds 3 --seeds 5
```

The experiment compares:
1. Persistence;
2. HistGBResidual;
3. SG-CPA-RNet ground + meteorology with selective cross-pollutant gate;
4. + Sentinel-5P neighbourhood mean;
5. + distance-weighted Sentinel-5P context;
6. + wind-aware Sentinel-5P context;
7. full Sentinel-5P context with quality/age-aware gate;
8. full model without the selective cross-pollutant gate.

Main outputs:
- `rolling_fold_metrics.csv`;
- `rolling_summary.csv`;
- `gate_summary.csv`;
- per-fold forecast CSV files.

## What to report in the paper

The strongest reviewer-facing result is not merely a satellite map. Report a quantitative ablation table with MAE/RMSE/R² for CO and SO2 and the change in MAE relative to the ground+meteorology configuration. Separately report the learned mean selective cross-pollutant gate and adaptive residual coefficient by pollutant. This directly tests whether the selective mechanism is actually used and whether Sentinel-5P improves the one-hour forecast.

Do not interpret Sentinel-5P column values as surface concentrations. They are regional atmospheric context features. A satellite contribution should be claimed only if the full/ablation experiment shows a reproducible improvement across rolling-origin folds.
