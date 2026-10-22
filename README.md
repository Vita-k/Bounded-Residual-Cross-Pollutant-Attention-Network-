# BR-CPA-RNet: Bounded Residual Cross-Pollutant Attention Network


The repository corresponds to the manuscript version submitted to SCIA-2026. All numbers below are taken from Tables 2–4 of the paper and are reproduced by the files in `results_clean/`.

**Code release identifier:** `v1.1-scia2026` (Git tag; see *Version* below).

## What is in this repository

| Path | Content |
|---|---|
| `clean_efir.py` | Quality control of the one-minute ground record (rules R1–R4, Section 3.1 of the paper) |
| `BR_CPA_RNet_reproducible.py` | Single-file experiment: preprocessing, rolling-origin folds, reference models, BR-CPA-RNet, ablations, paired block bootstrap |
| `eco_monitor_v7.py` | Prototype dashboard (Fig. 4 of the paper) generated from the stored predictions |
| `results_clean/` | Stored outputs of the run reported in the paper |
| `requirements.txt` | Python dependencies |

The Sentinel-5P alignment script `<add file name>` produces `results_s5p_exact/*.csv` from the Copernicus Data Space Ecosystem catalogue; the aligned table is provided so that the experiment can be rerun without credentials.

## Data

* **Ground data.** Efir 1.3 automatic monitoring station, Elektrometalurhiv St., Nikopol, Ukraine; October 2019, 44,640 one-minute records of NO₂, SO₂, CO, O₃, H₂S and dust (µg/m³) together with relative humidity, wind, pressure, temperature, solar radiation and precipitation. Source: Ukrainian open-data portal, https://data.gov.ua/dataset/b319ac30-f404-4e5d-bd60-6880c046d730 (file `2019-10-efir_1-3.xlsx`).
* **Satellite data.** Sentinel-5P/TROPOMI Level-2 CO, NO₂ and SO₂, reprocessed products (RPRO; processor 02.04.00 for CO and NO₂, 02.07.00 for SO₂), Copernicus Data Space Ecosystem.
* **Channels used.** NO₂, O₃, H₂S and dust are almost entirely zero after quality control and are excluded as targets and inputs. The experiment uses P = 2 pollutant streams (CO, SO₂). Solar radiation and precipitation are not model inputs; solar radiation is used only by the dashboard to determine the atmospheric stability class.

### Quality control (`clean_efir.py`)

Flagged values are set to missing; no rows are removed, so the time grid and fold boundaries are unchanged.

| Rule | Condition | Action | Records |
|---|---|---|---|
| R1 | CO and SO₂ simultaneously zero (QC assumption: probable instrument downtime encoded as zero; no operation logs available) | CO, SO₂ → missing | 4,321 |
| R2 | malfunction episode 1 Oct 2019 19:40 – 2 Oct 2019 01:06 | all pollutant channels → missing | 327 |
| R3 | any pollutant value ≥ 30,000 µg/m³ (sensor limit code) | → missing | 0 after R2 |
| R4 | relative humidity > 100 % (error code 3,276.7 %) | RH → missing | 6,720 |

```bash
python clean_efir.py 2019-10-efir_1-3.xlsx data/2019-10-efir_1-3_clean.xlsx
```

## Temporal convention

The one-minute record is aggregated to a 30-min grid of 1,489 bins. Bins are left-closed and labelled by their start: the value labelled *t* is the mean of the records in [*t*, *t* + 30 min) and is complete at *t* + 30 min.

| Quantity | Definition | Example (Fig. 4a) |
|---|---|---|
| Forecast origin | label *t* of the latest input bin | 28 Oct 2019 18:00 |
| Actual issue time | *t* + 30 min | 18:30 |
| Target | mean over [*t* + 60 min, *t* + 90 min), bin label *t* + 1 h | 19:00–19:30 |
| Label horizon | target label − origin label | 1 h |
| Lead after issue | start / end of the target interval | 30 / 60 min |
| History | 4 bins (2 h) | 16:30–18:30 |

No record from the target interval enters the inputs. For Sentinel-5P, the latest valid observation with acquisition time ≤ *t* and age ≤ 48 h is used. This is a retrospective setting: the condition concerns acquisition time, not the time at which the reprocessed product became available to users, so the satellite contribution is a potentially optimistic estimate for operational use.

## Running the experiment

```bash
pip install -r requirements.txt
python BR_CPA_RNet_reproducible.py \
    --ground-xlsx data/2019-10-efir_1-3_clean.xlsx \
    --sat-csv results_s5p_exact/s5p_aligned_30min.csv \
    --out-dir results_clean
```

Defaults reproduce the paper: 3 expanding rolling-origin folds, 5 neural initialisations per fold (forecasts averaged), fixed α = 0.30, Sentinel-5P age limit 48 h, circular block bootstrap with 12-step (6 h) blocks and 5,000 resamples. `--quick` runs one fold and one seed for a smoke test. Run time of a full run depends on hardware (tens of minutes on a CPU).

| Output file | Paper |
|---|---|
| `final_summary.csv` | Table 2 |
| `final_bootstrap_significance.csv` | Table 3 |
| `final_satellite_age_sensitivity.csv` | Table 4 |
| `final_fold_metrics.csv` | per-fold metrics (Section 4.2) |
| `final_gate_summary.csv` | learned coefficients of the adaptive variant (Section 4.3) |
| `final_predictions_long.csv` | forecasts used for Fig. 3 and Fig. 4 |
| `final_model_complexity.csv` | trainable parameters |

## Results reported in the paper

Model names in the CSV files: `Local-S5P-48h` = Local-S5P-mean (α = 0), `BR-CPA-no-S5P` = Without S5P, `Adaptive-gated-S5P-48h` = Adaptive, `BR-CPA-S5P-12h` / `BR-CPA-S5P-24h` = A_max sensitivity runs. In `final_bootstrap_significance.csv`, `improvement_pct` is ΔMAE, % and `bootstrap_p_two_sided` is p (n = 507 paired test forecasts per pollutant).

### Table 2. One-hour forecasting accuracy (mean ± SD over three folds; MAE, RMSE in µg/m³)

| Model | CO MAE | CO RMSE | CO R² | SO₂ MAE | SO₂ RMSE | SO₂ R² |
|---|---|---|---|---|---|---|
| Persistence | 24.638 ± 9.821 | 34.930 ± 14.661 | 0.751 | 0.754 ± 0.207 | 1.034 ± 0.255 | 0.886 |
| HistGBResidual | 24.843 ± 12.555 | 35.046 ± 18.075 | 0.764 | 0.781 ± 0.295 | 1.046 ± 0.351 | 0.872 |
| Local-S5P-mean (α = 0) | 25.321 ± 12.150 | 36.000 ± 18.618 | 0.751 | 0.762 ± 0.219 | 1.001 ± 0.266 | 0.891 |
| Without S5P | 25.329 ± 11.538 | 35.898 ± 18.438 | 0.752 | 0.771 ± 0.251 | 1.019 ± 0.309 | 0.881 |
| **BR-CPA-RNet (α = 0.30)** | **24.343** ± 10.091 | 34.962 ± 16.687 | 0.760 | **0.749** ± 0.218 | 0.992 ± 0.263 | 0.892 |
| Adaptive (learned α ≤ 0.30) | 24.401 ± 11.110 | 34.604 ± 17.296 | 0.768 | 0.750 ± 0.201 | 0.997 ± 0.258 | 0.892 |

### Table 3. Paired MAE differences of BR-CPA-RNet (circular block bootstrap; positive values favour BR-CPA-RNet)

| Comparator / pollutant | ΔMAE, µg/m³ | ΔMAE, % | 95% CI, µg/m³ | p |
|---|---|---|---|---|
| Local-S5P-mean / CO | 0.978 | +3.86 | [0.379, 1.638] | < 0.001 |
| Local-S5P-mean / SO₂ | 0.0129 | +1.70 | [−0.0161, 0.0424] | 0.394 |
| Persistence / CO | 0.295 | +1.20 | [−1.595, 2.210] | 0.791 |
| Persistence / SO₂ | 0.0048 | +0.64 | [−0.0573, 0.0618] | 0.869 |
| HistGBResidual / CO | 0.500 | +2.01 | [−0.386, 1.524] | 0.298 |
| HistGBResidual / SO₂ | 0.0318 | +4.07 | [−0.0232, 0.0831] | 0.257 |
| Adaptive variant / CO | 0.057 | +0.24 | [−0.500, 0.626] | 0.846 |
| Adaptive variant / SO₂ | 0.0004 | +0.06 | [−0.0251, 0.0269] | 0.958 |
| Without Sentinel-5P / CO | 0.986 | +3.89 | [0.242, 1.838] | 0.008 |
| Without Sentinel-5P / SO₂ | 0.0212 | +2.75 | [−0.0291, 0.0693] | 0.395 |

### Table 4. Sensitivity to the Sentinel-5P age threshold A_max

| A_max | CO MAE | CO R² | SO₂ MAE | SO₂ R² |
|---|---|---|---|---|
| 12 h | 24.141 ± 10.391 | 0.765 | 0.762 ± 0.194 | 0.891 |
| 24 h | 25.673 ± 11.680 | 0.747 | 0.784 ± 0.234 | 0.881 |
| 48 h (primary) | 24.343 ± 10.091 | 0.760 | 0.749 ± 0.218 | 0.892 |

**Interpretation.** BR-CPA-RNet has the lowest mean MAE among the compared models, but only the differences from the matched channel-independent configuration (CO) and from the model without Sentinel-5P (CO) are significant; the differences from persistence and gradient boosting are not. The Sentinel-5P effect depends non-monotonically on the age threshold and was obtained with acquisition-time alignment of reprocessed products.

## Dashboard

```bash
python eco_monitor_v7.py --pred results_clean/final_predictions_long.csv \
    --s5p results_s5p_exact/s5p_aligned_30min.csv --fold 3
```

Produces `ECO_MONITOR_v7.html` (Ukrainian) and `ECO_MONITOR_v7_en.html` (English). Each frame shows the measured 30-min mean, the forecast with origin, issue time and target interval, a retrospective check, and the Sentinel-5P acquisition time, age and product version. The map cloud is indicative: a Gaussian plume from a virtual ground-level source at the station (Briggs coefficients, Pasquill class from wind speed and solar radiation, SO₂ dry deposition 0.8 cm/s). It is not an emission-based model or a concentration field. Forecasts are validated retrospectively for CO and SO₂ only; no threshold warnings or health advice are produced.

## Changes from the previous repository version

The earlier README (commit `c7a851a` and before) reported results of a run on the raw, uncleaned ground file (e.g. CO MAE reduction relative to Local 6.49 %, p = 0.0144). The current results follow from quality control R1–R4 and supersede those numbers. Per-fold and pooled values differ accordingly; the code of the model and the evaluation protocol are unchanged apart from the input file.

## Version

After uploading these files, create the tag that the paper refers to:

```bash
git add -A
git commit -m "Synchronise repository with SCIA-2026 manuscript (cleaned data, Tables 2-4)"
git tag -a v1.1-scia2026 -m "Code and results of the SCIA-2026 manuscript"
git push origin main --tags
```

## Limitations

Single station, single month (October 2019, pre-war conditions), two informative pollutant channels, retrospective alignment of satellite products. Sensitivity to the mixing coefficient α and to the bootstrap block length was not analysed.

## Citation

```
V. Kashtan, V. Hnatushenko, Intelligent Environmental Information System for Short-Term
Air-Pollutant Forecasting, in: SCIA-2026: 5th International Workshop on Social Communication
and Information Activity in Digital Humanities, Lviv, Ukraine, 2026.
```

## Funding

Ministry of Education and Science of Ukraine, grant No. 0126U000995, "Intelligent technologies for analysing spatiotemporal changes in aerospace imagery to support decision-making under conditions of armed aggression".

## License

See `LICENSE`.
