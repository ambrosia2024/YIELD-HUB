# A Python Toolkit for Machine Learning based Crop Yield Analysis, Prediction, and Benchmarking
This repository is evolving into an [dashboard](https://ambrosia2024.github.io/YIELD-HUB/main.html), SDK, and CLI for crop-yield inference on CY-BENCH-compatible data. It can also still be used for model training and evaluation.

# Quickstart

This is the recommended end-user flow today.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Add your Hugging Face token

The model repositories are private for now. Put a read token in `.env`:

```env
HF_TOKEN=your_token_here
HUGGINGFACE_HUB_TOKEN=your_token_here
```

You can use the same token value for both variables.

## 3. Place your data under `data_root`

For local development, the simplest setup is:

```text
YIELD-HUB/
├── data/
│   ├── maize/
│   │   └── NL/
│   └── wheat/
```

For a given `crop` and `country`, the target folder must contain:

```text
data/<crop>/<country>/
├── crop_calendar_<crop>_<country>.csv
├── fpar_<crop>_<country>.csv
├── location_<crop>_<country>.csv
├── meteo_<crop>_<country>.csv
├── ndvi_<crop>_<country>.csv
├── soil_<crop>_<country>.csv
├── soil_moisture_<crop>_<country>.csv
└── yield_<crop>_<country>.csv
```

Example:

```text
data/maize/NL/yield_maize_NL.csv
```

## 4. Validate the dataset

```bash
yield-hub-predict validate-data --crop maize --country NL --data-root ./data
```

## 5. List, fetch, and run a model

```bash
yield-hub-predict list-models
yield-hub-predict fetch-model --model-type rlinear --crop maize --country NL
yield-hub-predict predict --model-type rlinear --crop maize --country NL --data-root ./data
```

Predictions are written to `wrappers/data/` by default.

## 6. Troubleshoot quickly

- `Missing Hugging Face token`: add `HF_TOKEN` in `.env`.
- `No data found for <crop>-<country>`: check `data/<crop>/<country>/...` and run `validate-data`.
- `Missing columns` or `invalid_dates`: fix the CSV schema before running inference.
- `ModuleNotFoundError: cybench`: install and point the project at a valid `AgML-CY-BENCH` checkout for now.

# Current Scope

Current focus:

- SDK and Python API
- CLI for validation, artifact fetch, and prediction
- private Hugging Face model access

Later stage:

- dashboard
- public model access

# SDK Usage

The intended SDK workflow is:

1. install the package
2. provide CY-BENCH-compatible data via a `data_root`
3. list or fetch the model you want
4. run prediction

Python:

```python
from yield_hub import ModelRegistry, Predictor, validate_data

validate_data(crop="maize", country="NL", data_root="./data")

registry = ModelRegistry()
models = registry.list_models()
checkpoint_path = registry.fetch_model(model_type="rlinear", crop="maize", country="NL")

predictor = Predictor(data_root="./data")
df = predictor.predict(model_type="rlinear", crop="maize", country="NL")
```

CLI:

```bash
yield-hub-predict list-models
yield-hub-predict fetch-model --model-type rlinear --crop maize --country NL
yield-hub-predict validate-data --crop maize --country NL --data-root ./data
yield-hub-predict predict --model-type rlinear --crop maize --country NL --data-root ./data
```

The CLI also keeps backward compatibility with the older direct form:

```bash
yield-hub-predict --model-type rlinear --crop maize --country NL --data-root ./data
```

# Data Contract

The SDK currently validates against data contract version `1.0`.

`data_root` must resolve to a directory containing crop and country folders:

```text
data_root/
├── maize/
│   └── NL/
└── wheat/
```

For each `crop/country` pair, these files are mandatory:

```text
crop_calendar_<crop>_<country>.csv
fpar_<crop>_<country>.csv
location_<crop>_<country>.csv
meteo_<crop>_<country>.csv
ndvi_<crop>_<country>.csv
soil_<crop>_<country>.csv
soil_moisture_<crop>_<country>.csv
yield_<crop>_<country>.csv
```

Required columns by file:

- `yield`: `adm_id`, `harvest_year`, `yield`
- `soil`: `adm_id`, `awc`, `bulk_density`
- `crop_calendar`: `adm_id`, `sos`, `eos`
- `meteo`: `adm_id`, `date`, `tmin`, `tmax`, `prec`, `rad`, `tavg`, `cwb`
- `fpar`: `adm_id`, `date`, `fpar`
- `ndvi`: `adm_id`, `date`, `ndvi`
- `soil_moisture`: `adm_id`, `date`, `ssm`
- `location`: `adm_id`, `latitude`, `longitude`

Current validation rules:

- every required file must exist
- every required file must be non-empty
- all required columns must be present
- duplicate key rows are rejected for `yield`, `soil`, `crop_calendar`, and `location`
- temporal files must have parseable `date` values
- `yield` years are checked for gaps between min and max detected years
- numeric sanity checks are enforced for:
  - `yield.yield >= 0`
  - `soil.awc >= 0`
  - `soil.bulk_density >= 0`
  - `crop_calendar.sos` and `crop_calendar.eos` in `1..366`
  - `crop_calendar.eos >= crop_calendar.sos`
  - `fpar.fpar` in `0..100`, with `0..1` treated as the preferred normalized scale
  - `ndvi.ndvi` in `-1..1`
  - `soil_moisture.ssm` in `0..100`, with `0..1` treated as the preferred normalized scale
  - `location.latitude` in `-90..90`
  - `location.longitude` in `-180..180`
- temporal files are checked for year overlap with the detected yield years

What is currently mandatory:

- the full file set above
- CY-BENCH-compatible file names
- CY-BENCH-compatible required columns

What is currently optional:

- extra columns beyond the required schema
- local storage location, as long as `--data-root` points to it correctly

Validation example:

```bash
yield-hub-predict validate-data --crop maize --country NL --data-root ./data
```

Python example:

```python
from yield_hub import DATA_CONTRACT_VERSION, validate_data

print(DATA_CONTRACT_VERSION)
report = validate_data(data_root="./data", crop="maize", country="NL")
print(report["ok"])
```

# Packaging

This repository now includes a minimal `pyproject.toml` for editable installs and
package structure for wrapper tooling and future app/backend work:

```bash
pip install -e .
yield-hub-predict --model-type patchtst --country DE --crop maize --data-root ./data
```

Note: `cybench` still comes from the external `AgML-CY-BENCH` codebase and is not
published on PyPI, so that prerequisite remains separate for now.

The package is organized around:

- `yield_hub.settings`: environment loading and path resolution
- `yield_hub.artifacts`: Hugging Face artifact lookup, listing, and downloads
- `yield_hub.data`: dataset loading, local `data_root` support, and split setup
- `yield_hub.predictor`: importable prediction API
- `yield_hub.validation`: dataset validation
- `yield_hub.cli`: command-line entrypoint

# Examples

Python prediction example:

```python
from yield_hub import ModelRegistry, Predictor, validate_data

report = validate_data(data_root="./data", crop="maize", country="NL")
if not report["ok"]:
    raise ValueError(report)

registry = ModelRegistry()
registry.fetch_model(model_type="rlinear", crop="maize", country="NL")

predictor = Predictor(data_root="./data")
df = predictor.predict(model_type="rlinear", crop="maize", country="NL")
print(df.head())
```

CLI prediction example:

```bash
yield-hub-predict validate-data --crop maize --country NL --data-root ./data
yield-hub-predict fetch-model --model-type rlinear --crop maize --country NL
yield-hub-predict predict --model-type rlinear --crop maize --country NL --data-root ./data
```

# Troubleshooting

- `validate-data` returns `"target_exists": false`
  Point `--data-root` at the folder containing `maize/` and `wheat/`, not directly at the country folder.
- `missing_files` is non-empty
  Add the missing CSVs with the exact expected names.
- `missing_columns` is non-empty
  Rename or add the required columns listed in the validation report.
- `invalid_dates` is non-zero
  Normalize the `date` column to a parseable format such as `YYYYMMDD` or ISO date strings.
- `warnings` contains scale messages for `fpar` or `ssm`
  The file is accepted, but the values are not on the preferred normalized `0..1` scale.
- `duplicate_key_rows` is non-zero
  Deduplicate the file on the reported logical key before inference.
- `year_overlap_with_yield` is empty
  Your temporal files do not cover the same years as the yield file.
- checkpoint download fails
  Confirm the token has read access to the private `Ambrosia2024` repositories.
- `cybench` import fails
  This package still depends on an external `AgML-CY-BENCH` checkout for now.

# Model training and evaluation

Depending upon the setup of interest:
1. Temporal Split with the last 5 years as test dataset, the 2 years before for the validation data, and rest as training data (setups/temporal)
2. Performing Hyperparameter Optimization using Optuna (setups/hpo)
3. Walk Forward Setup with the last 5 years as the test dataset (setups/walk-forward)

Run

```
cd setups/temporal
python statisticalBaselines.py --model mlp --country DE --crop wheat --seed 1111 --save_dir ../output/saved_models/ --output_dir ../output/trained_models/
python tstBaselines.py --crop maize --country NL --model_type tst --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily --include_spatial_features
python linearBaselines.py --crop maize --country NL --model_type xlinear --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily --include_spatial_features
```

Alternatively, feel free to use the bash script if you are working with SLURM to train all the baselines together:
```
cd setups/temporal
sbatch run_baselines.sh
```

# Model Checkpoints (Currently a private repository)

All the trained checkpoints are available under the [Ambrosia2024/yield-hub](https://huggingface.co/collections/Ambrosia2024/yield-hub) collection on Hugging Face. Individual model repositories include [yield-autoformer-cybench](https://huggingface.co/Ambrosia2024/yield-autoformer-cybench) and the bundled `yield-transformers-cybench` and `yield-linear-cybench` repos.

The SDK downloads checkpoints on demand into the local Hugging Face cache. You do not need to manually copy checkpoints into this repository.

# Private Hugging Face Access

For private model repositories, create a token with read access to the `Ambrosia2024` organization repos and store it in `.env`:

```
HF_TOKEN=your_token_here
HUGGINGFACE_HUB_TOKEN=your_token_here
```

The repository includes CLI and Python helpers that download `config-and-results.csv` and model checkpoints from Hugging Face using that token.

# Legacy Research Setup

If you want to use this repository in the older integrated research layout with `AgML-CY-BENCH`, keep using the external CY-BENCH checkout and its installation instructions. That path is still relevant for training and benchmark development, but the recommended inference path is the SDK/CLI workflow above.
