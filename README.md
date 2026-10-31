<div align="center">

# Forecasting Landslide Impact Probability in Nepal using AI and Earth Observation

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-installation">Installation</a></b>
|
<b><a href="#-quickstart">Quickstart</a></b>
|
<b><a href="#-data">Data</a></b>
|
<b><a href="#-data-processing-pipeline">Data Processing</a></b>
|
<b><a href="#-code-organization">Code Organization</a></b>
|
<b><a href="#-citation">Citation</a></b>
</p>

</div>

## 📄 Description
This repository contains the code for dataset curation, processing, model training, testing and analysis for predicting landslide impacts in Nepal: landslides that damage infrastructure and/or cause loss of life, over a 14-day horizon at district level. Statistical summaries of precipitation and terrain over each district's boundary are combined with satellite embeddings extracted from our LandslideNet model as inputs to a final ML classifier. All landslide impact records are taken from the Building Information Platform Against Disaster (BIPAD) portal, owned by the National Disaster Risk Reduction and Management Authority (NDRRMA).

## 🛠 Installation
```bash
git clone https://github.com/kelsdoerksen/landslide-forecast-nepal.git
cd landslide-forecast-nepal
conda create -n landslides python=3.14
conda activate landslides
pip install -r requirements.txt
```
The UNet / embedding pipeline in `unet/` needs a few extra packages, listed in `unet/requirements.txt` (GDAL is easiest to install from conda-forge).

Experiments are logged with [Weights & Biases](https://wandb.ai). To run without an account, set `WANDB_MODE=disabled` (or `offline`).

## 🚀 Quickstart
The `Sample` folder holds pre-processed data to train and test the classifier for 2016 and 2017, with UKMO ensemble member 0 forecasts and GPM IMERG v07 hindcasts. To train a Random Forest on 2016 and test on 2017:
```bash
mkdir -p results
WANDB_MODE=disabled python run_classifier.py \
    --root_dir Sample --results_dir results \
    --model rf --test_year 2017 \
    --hindcast_model GPMv07 --forecast_model UKMO --ensemble_num 0 \
    --experiment_type full --wandb_setting offline --tag quickstart
```
This takes < 1min on a laptop and writes metrics, predictions and feature importances to a run folder inside `results/`.

## 💧 Data

### What is in this repository
| Path | Contents |
|---|---|
| `Sample/LabelledData_GPMv07/UKMO/ensemble_0/{year}_windowsize14_district.csv` | Labelled samples (one row per district per day) for 2016-2017 |
| `Sample/bipad_records_2016-2024_july172025download.csv` | Landslide incident records from the BIPAD portal used for labels. Data downloaded from BIPAD on July 17th, 2025 |
| `techdemo_files/District_Labels.tif` | Raster of Nepal's 77 districts. Pixel values are district IDs |
| `techdemo_files/{dem,slope,aspect}_stats_all_district.csv` | District-level terrain statistics from the ALOS 30 m DEM |
| `techdemo_files/MODIS_District_2015-2021/` | District-level MODIS land cover per year |

### Sample file format
Each row is one `district` on one `date` (the forecast issue date). `label` is 1 if a landslide impact was recorded in that district in the following 14 days.

| Columns | Description |
|---|---|
| `precip_rate_tminus_{0..13}` | Hindcast (observed) daily precipitation for the 14 days before `date` |
| `GPMv07_{mean,max,min}_precip_rate`, `GPMv07_total_cumulative_precipitation` | Summaries of the 14-day hindcast window |
| `UKMO_ens_0_precip_rate_tplus_{1..14}` | Forecast daily precipitation for the 14 days after `date` |
| `UKMO_ens_0_precip_{mean,max,min}_precip_rate`, `UKMO_ens_0_precip_total_cumulative_precipitation` | Summaries of the 14-day forecast window |
| `dem_*`, `slope_*`, `aspect_*` | District terrain statistics (mean, median, stdev, min, max, range, variance) |
| `lc_mode`, `lc_var` | MODIS land cover mode and variance in the district |


### Raw data sources
To regenerate the dataset from scratch, you need:
| Data | Source |
|---|---|
| Precipitation hindcast | NASA GPM IMERG v07, via Google Earth Engine (`NASA/GPM_L3/IMERG_V07`) |
| Precipitation forecasts | Subseasonal ensemble forecasts from UKMO, NCEP, KMA and ECMWF ([S2S project](https://www.s2sprediction.net)) |
| Elevation, slope, aspect | ALOS World 3D 30 m DEM (JAXA) |
| Land cover | MODIS MCD12Q1, via Google Earth Engine (`MODIS/061/MCD12Q1`) |
| Landslide incidents (labels) | Nepal [BIPAD portal](https://bipadportal.gov.np) |

## 👩‍🔬 Data Processing Pipeline
The preprocessing scripts read from and write to a project data folder, set with the `LANDSLIDE_PROJECT_DIR` environment variable. The Earth Engine scripts use your own GEE Cloud project:
```bash
export LANDSLIDE_PROJECT_DIR=/path/to/your/data
export EE_PROJECT=your-gee-cloud-project
earthengine authenticate
```

Tabular dataset, in order (scripts in `Preprocessing/`):
1. `gee_processing.py --gee_data <collection> --query_year <year>`: download daily hindcast precipitation from Earth Engine.
2. `nc_to_npy.py` (UKMO/NCEP/KMA) and `ecmwf_processing.py` (ECMWF): convert forecast NetCDF files to per-day arrays.
3. `forecast_precip_model_prep.py --precip_source <model> --ens_member <n>` and `npy_precip_to_csv.py`: aggregate gridded precipitation to district level using `District_Labels.tif`.
4. `generate_samples_hindcast_forecast.py --year <year> --hindcast_source <src> --forecast_source <model> --ens_member <n> --incidents_file <incidents.csv>`: join hindcast, forecast, terrain, land cover and landslide labels into `$LANDSLIDE_PROJECT_DIR/LabelledData_<src>/<model>/ensemble_<n>/{year}_windowsize14_district.csv`.
   - `--incidents_file` is a landslide incident CSV with `District` and `Incident on` columns, such as a BIPAD portal download (e.g. `Sample/bipad_records_2016-2024_july172025download.csv`). Set `--incident_date_format` if its dates aren't `YYYY-MM-DD`.
   - It reads its inputs directly from `$LANDSLIDE_PROJECT_DIR`: `DEM-Derived_District/`, `<src>_Mean_Pixelwise/Subseasonal/DistrictLevel/`, `PrecipitationModel_Forecast_Data/Subseasonal/<model>/DistrictLevel/` and the MODIS folder (`--modis_dir`).

Array (UNet) dataset: `unet/data_processing/` (`gee_processing_unet.py`, `label_generation.py`, `data_generation.py`). Satellite embeddings for the classifier are generated with `unet/embedding_extraction.py`.

## 📚 Code Organization
```
├── run_classifier.py     # ML classifier pipeline: train / val / test
├── Preprocessing/        # Tabular dataset generation
├── Analysis/             # Plotting and result aggregation (F1, PR curves, feature importance, maps)
├── unet/                 # LandslideNet (UNet) training, prediction and embedding extraction
├── Sample/               # Small pre-processed sample dataset
└── techdemo_files/       # Static district-level inputs (district raster, terrain, land cover)
```

### ML classifier: `run_classifier.py`
| Argument | Description |
|---|---|
| `--root_dir` | Root data directory (e.g. `Sample`) |
| `--results_dir` | Directory where the run's results folder is created (must already exist) |
| `--model` | `rf`, `gb`, `xgb` or `mlp` |
| `--test_year` | Held-out test year (2017-2024) |
| `--hindcast_model` | Hindcast precipitation source, e.g. `GPMv07` |
| `--forecast_model` | Precipitation forecast model: `UKMO`, `NCEP`, `KMA`, `ECMWF` |
| `--ensemble_num` | Forecast ensemble member ID |
| `--experiment_type` | `full` (standard), `no_hindcast`, `no_forecast`, an embedding experiment name, `ukmo_trained`, or `monsoon_tool` |
| `--wandb_setting` | `offline` or `online` |
| `--test_forecast` | Only used with `--experiment_type ukmo_trained`: forecast model to test on |
| `--parameter_tuning` | Set to tune model hyperparameters (default: none) |
| `--tag` | wandb run tag. Also sets the train/test split: by default only years before `--test_year` are used for training; with `temporal-cv`, all other years are |

### LandslideNet: `unet/run_pipeline.py`
Run from the `unet` folder: `python run_pipeline.py --root_dir <data> --save_dir <results> ...`

| Argument | Description |
|---|---|
| `--epochs`, `--batch_size`, `--learning-rate`, `--optimizer`, `--loss`, `--dropout` | Training settings |
| `--classes` | Number of classes |
| `--test_year` | Test year (held out from training) |
| `--root_dir`, `--save_dir` | Data and results directories |
| `--val_percent` | Validation percentage |
| `--ensemble`, `--ensemble_member` | Forecast model (`KMA`, `NCEP`, `UKMO`) and member |
| `--tags` | wandb tag |

Run `python run_pipeline.py --help` for the remaining options (augmentation, channel dropout, normalisation).

## 📝 Citation
If you use this code or data, please cite:
<!-- TODO: add BibTeX for the paper -->
```bibtex
Doerksen, K., Li, S., Dunant, A. et al. AI-powered sub-seasonal forecasting of monsoon-triggered landslide impacts in Nepal. npj Nat. Hazards (2026). https://doi.org/10.1038/s44304-026-00282-9
```

## ⚖️ License
Code is released under the [MIT License](LICENSE). Data from third-party sources (BIPAD, NASA, JAXA, S2S) remains subject to the original providers' terms.

## ⚙️ Operational Model Demo
An example of the operational demo hosted on Google Colab can be found: [Operational Demo](https://colab.research.google.com/drive/1vBtqaSwEa8aSpxFIXuMS7m7sXKsuW6pX?usp=sharing)
