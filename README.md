# ClearSky-AI

ClearSky-AI reconstructs cloud-obscured Sentinel-2 imagery using aligned
Sentinel-1 VV/VH radar data.

The project provides:

- A Streamlit reconstruction interface
- A 13-band GeoTIFF reconstruction output
- A prepared SEN12MS-CR sample gallery
- A held-out model benchmark
- An experimental LISS-IV application

## Benchmark results

The held-out benchmark contains **964 prepared SEN12MS-CR patches**.

| Metric | Cloudy input | Reconstruction |
| --- | ---: | ---: |
| Mean PSNR | 17.25 dB | 27.65 dB |
| Mean MAE | 0.11962 | 0.03130 |
| Mean SSIM | 0.56137 | 0.82366 |
| Mean SAM | 14.045 degrees | 8.970 degrees |
| Mean PSNR gain | - | +10.40 dB |

The reconstruction improved PSNR in **960 of 964 patches**.

The **Benchmark** page in the Streamlit app contains the charts, metric
explanations, and downloadable benchmark files.

## Dataset use

- **Training material:** SEN12MS-CR and AllClear, with approximately 90 GB of
  source data used for the project training work.
- **Held-out test benchmark:** All 964 official test-split patches are included.


## Quick start

Install Git LFS to retrieve the checkpoint and gallery assets.

```bash
git lfs install
git lfs pull
python -m pip install -r requirements.txt
streamlit run app.py
```

Use the **Reconstruction** page to select a paired gallery sample, upload an
aligned raster pair, or enter local raster paths.

Expected inputs:

- Sentinel-2: 13 bands in B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09,
  B10, B11, B12 order
- Sentinel-1: two bands in VV, VH order
- Both rasters must use the same pixel grid, CRS, and affine transform

## Outputs

- 13-band reconstructed Sentinel-2 GeoTIFF
- Source raster georeferencing
- Preview images and per-sample paired-reference metrics

## Experimental LISS-IV app

A separate optical-only LISS-IV interface is available:

```bash
streamlit run app_liss4.py
```

It uses different inputs and checkpoints and is not part of the Sentinel-2
held-out benchmark.

## Limitations

- Cloud-obscured surface detail is estimated and may be inaccurate.
- Results may differ outside the training and prepared demo domains.
- Reference images may have temporal differences.
- The held-out benchmark does not represent independent external validation.
