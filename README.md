# Mollusk Age Estimation with a Neural Network

This project uses simulated mollusk measurements from a classroom exercise inspired by Abalone. It compares a multilayer perceptron (MLP) with linear regression and a baseline that predicts the mean age in the training set.

## Run locally

From the repository folder on Linux or WSL, run the following commands with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model.py train
python model.py predict --data data/prediction_example.csv
```

Training reads `data/training.csv` by default and writes `artifacts/model.joblib` and `artifacts/metrics.json`. Batch prediction writes `artifacts/predictions.csv` with the input columns and an `estimated_age_years` column. Use `--data path/to/file.csv` to supply your own data and `--help` after either subcommand to see its options.

Only load Joblib files you trust. Loading a Joblib file can execute code.

## Data and evaluation

Both CSVs contain simulated data. The prediction example uses rows from the demonstration dataset to illustrate the input format. Use `length_cm`, `width_cm`, `height_cm`, and `mass_g` as finite, positive inputs. Training also requires a finite, nonnegative `age_years` target and at least 20 rows.

The pipeline saves feature scaling and the MLP together. The MLP uses an internal validation split for early stopping, and all three models are evaluated on the same held-out test split.

## Demonstration results

The test split contains 750 of the 3,000 simulated records.

| Model | MAE (years) | RMSE (years) | R² |
| --- | ---: | ---: | ---: |
| Training-set mean | 1.938 | 2.120 | −0.001 |
| Linear regression | 0.709 | 0.861 | 0.835 |
| Neural network | 0.378 | 0.486 | 0.947 |

The MLP reduced mean absolute error by about 47% relative to linear regression on this split. The simulation includes nonlinear relationships between the measurements and age. Applying the model to real mollusks would require observed data, external validation, and an analysis of errors across age ranges.

## Notebook

Open [analysis.ipynb](analysis.ipynb) in VS Code with the Jupyter extension, select the virtual environment as the kernel, and run all cells. The notebook generates the simulated data and plots predicted versus simulated ages for both regression models.
