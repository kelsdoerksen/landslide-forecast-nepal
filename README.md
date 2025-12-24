<div align="center">

# Forecasting Landslide Probability in Nepal using AI and Earth Observation

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-dataset">Dataset</a></b>
|
<b><a href="#-data-processing">Data Processing</a></b>
|
<b><a href="#-code-organization">Code Organization</a></b>
</p>

</div>

## 📄 Description
This work presents the dataset curation, processing, model training, testing and analysis for the purposes of predicting landslide impacts on a 14-day temporal scale in Nepal that cause impact to infrastructure and/or loss of life. We present our methodology to utilize statistical summary information over a district's boundary in-combination with satellite embeddings extracted from our LandslideNet model as inputs into a final ML classifier.

## 💧 Dataset
Both the tabular-ML dataset and array dataset are composed of precipitation hindcast features, precipitation forecast featueres, and topographical feature extracted from the ALOS 30 meter resolution Digital Elevation model of Nepal. 

## 👩‍🔬 Data Processing
`/Preprocessing` folder contains the code to pre-process data for the tabular dataset.
`unet/data_processing` folder contains the code to pre-process data for the array dataset.

## 📚 Code Organization
Code is split into two main sections, that to run the ML classifiers and that to run the Deep Learning model. All code for the Deep Learning unet is can be found in the `unet` folder.

The ML pipeline (train, val, test) can be run via the command:
```python run_pipeline.py ```
With the arguments:
- --model: ML Model. Currently supports rf, gb, and xgb
- --forecast_model: Precipitation Forecast Model Used
- --ensemble_num: Ensemble member id used from precipitation forecast model
- --hindcast_model: Hindcast precipitation model used
- --experiment_type: Type of experiment. no_hindcast, no_forecast refers to removing those features respectively, full is standard
- --wandb_setting: Wandb experiment setting, offline or online
- --test_forecast: [OLD - set to None]
- --parameter_tuning: Specify if tuning model hyperparameters', default=None

The DL pipeline can be run via the command in the `unet` folder:
```python run_pipeline.py```
With the arguments:
- --epochs: Number of epochs
- --batch_size: Batch size
- --learning-rate: Learning rate
- --optimizer: Model optimizer
- --classes: Number of classes
- --test-year: Test year for analysis (sets out of training)
- --root_dir: Specify root directory
- --save_dir: Specify root save directory
- --val_percent: Validation percentage
- --ensemble: Ensemble Model. Must be one of KMA, NCEP, UKMO
- --ensemble_member: Ensemble Member
- --tags: wandb tag
