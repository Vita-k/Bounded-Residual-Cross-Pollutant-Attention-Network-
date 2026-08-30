# BR-CPA-RNet

**Air Pollutant Forecasting Using Bounded Residual Cross-Pollutant Attention and Sentinel-5P Data**

This repository provides a **single Python file** reproducing the final BR-CPA-RNet experiment reported in the manuscript.

## Scientific idea

BR-CPA-RNet preserves the pollutant-specific local temporal representation as the forecasting anchor and introduces cross-pollutant information only as a bounded residual correction:

\[
h^{BR}_{p,t}
=
h^{loc}_{p,t}
+
\alpha
\left(
h^{CPA}_{p,t}-h^{loc}_{p,t}
\right),
\qquad \alpha=0.30.
\]

The fixed coefficient limits unrestricted information transfer between pollutant channels.

Sentinel-5P/TROPOMI data are used as **regional atmospheric context**, not as direct substitutes for surface concentrations. For CO, NO₂, and SO₂, the model uses the quality-filtered mean column value, observation age, and availability.

## Final experimental setting

- monitoring station: Efir 1.3, Nikopol, Ukraine;
- study period: October 2019;
- raw observations: 44,640 one-minute records;
- modeling resolution: 30 min;
- historical context: 2 h (`T=4`);
- forecast horizon: 1 h (`H=2`);
- quantitatively evaluated pollutants: CO and SO₂;
- rolling-origin folds: 3;
- independently initialized neural models per fold: 5;
- bounded cross-pollutant coefficient: `alpha = 0.30`;
- Sentinel-5P maximum observation age: 48 h;
- paired moving-block bootstrap: 5,000 repetitions, 12-step (6 h) blocks.

## Repository contents

```text
README.md
BR_CPA_RNet_reproducible.py
requirements.txt
```

`BR_CPA_RNet_reproducible.py` is intentionally self-contained: the internal compact temporal model and BR-CPA-RNet v2 implementation are embedded in the same source file.

## Required input data

The script expects:

```text
2019-10-efir_1-3.xlsx
results_s5p_exact/
    s5p_aligned_30min.csv
```

The ground dataset used in the study is available from the official Ukrainian open-data portal:

https://data.gov.ua/dataset/b319ac30-f404-4e5d-bd60-6880c046d730

The Sentinel-5P table must be prepared using the manuscript-aligned causal preprocessing workflow: CO/SO₂ QA ≥ 50%, NO₂ QA ≥ 75%, approximately 50 km spatial context, 32×32 raster, acquisition-time matching, and a 48 h maximum observation age.

Raw monitoring data should only be redistributed when permitted by the original data provider's terms.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Place the two required input files in the paths shown above and run:

```bash
python BR_CPA_RNet_reproducible.py
```

Optional arguments are available through:

```bash
python BR_CPA_RNet_reproducible.py --help
```

The script writes the final fold-level metrics, summary tables, predictions, bootstrap significance results, satellite-age sensitivity results, gate diagnostics, and model-complexity information to:

```text
results_br_cpa_final/
```

## Main manuscript results

Across three rolling-origin folds, the final confirmatory experiment produced:

| Model | CO MAE | SO₂ MAE |
|---|---:|---:|
| Persistence | 24.638 | 0.754 |
| HistGBResidual | **23.950** | 0.800 |
| Local-S5P-mean | 26.032 | 0.778 |
| **BR-CPA-RNet** | **24.341** | 0.757 |
| Adaptive-gated variant | 24.882 | **0.748** |

Relative to the architecturally matched `Local-S5P-mean` configuration, BR-CPA-RNet reduced MAE by:

- **6.49% for CO**;
- **2.61% for SO₂**.

For CO, this improvement was statistically significant under the paired moving-block bootstrap:

- 95% CI: **[0.253, 3.460]**
- **p = 0.0144**

The SO₂ improvement was not statistically significant.

## Sentinel-5P contribution

In a matched ablation, adding Sentinel-5P context reduced MAE by:

- **1.56% for CO**;
- **0.67% for SO₂**.

These differences were not statistically significant for the available one-month dataset. Sentinel-5P should therefore be interpreted as complementary regional context rather than the dominant short-horizon predictor.

## Reproducibility note

For each temporal fold, predictions from five independently initialized neural models are averaged first. The reported mean ± standard deviation in the manuscript is then calculated across the three fold-level metrics.

## Security

Do **not** commit:

- Copernicus Data Space client secrets;
- OAuth tokens;
- `.env` files;
- passwords or API credentials.

## Citation

Kashtan, V., Hnatushenko, V.: **BR-CPA-RNet: Bounded Residual Cross-Pollutant Attention Network for Short-Term Air Pollutant Forecasting.** GitHub repository (2026).

A DOI-based citation can be added after archiving a tagged GitHub release in Zenodo.

## License

Add the authors' selected software license before publication. Any redistributed source data must follow the original provider's license and terms.
