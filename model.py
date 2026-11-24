"""Estimate mollusk age from physical measurements and compare an MLP with linear regression."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = "mollusk-age-mlp"
FEATURES = ["length_cm", "width_cm", "height_cm", "mass_g"]
TARGET = "age_years"


def validate(data, training):
    measurements = data[FEATURES].to_numpy()
    if not np.isfinite(measurements).all() or (measurements <= 0).any():
        raise ValueError("Measurements must be finite and positive.")
    if training:
        age = data[TARGET].to_numpy()
        if not np.isfinite(age).all() or (age < 0).any():
            raise ValueError("age_years must be finite and nonnegative.")
        if len(data) < 20:
            raise ValueError("Provide at least 20 rows for training, validation, and testing.")


def fit_models(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, random_state=42
    )
    neural_network = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 16), early_stopping=True,
                     validation_fraction=0.15, max_iter=600, n_iter_no_change=25,
                     random_state=42),
    )
    models = {"training_mean": DummyRegressor(), "linear": LinearRegression(), "neural_network": neural_network}
    metrics = {"test_rows": len(X_test)}
    for name, model in models.items():
        predicted = model.fit(X_train, y_train).predict(X_test)
        metrics[name] = {
            "mae_years": mean_absolute_error(y_test, predicted),
            "rmse_years": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "r2": r2_score(y_test, predicted),
        }
    metrics["neural_network_epochs"] = neural_network[-1].n_iter_
    return neural_network, metrics


def predict(model, data):
    return {"estimated_age_years": model.predict(data[FEATURES])}


def read_csv(path, training=False):
    data = pd.read_csv(path)
    required = FEATURES + ([TARGET] if training else [])
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if data.empty or data[required].isna().any().any():
        raise ValueError("The CSV must contain rows and values in all required columns.")
    data[required] = data[required].astype(float)
    validate(data, training)
    return data


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train, evaluate, and save the model")
    train.add_argument("--data", type=Path, default=root / "data" / "training.csv")
    train.add_argument("--output", type=Path, default=root / "artifacts")
    prediction = commands.add_parser("predict", help="Generate predictions from a CSV")
    prediction.add_argument("--data", type=Path, required=True)
    prediction.add_argument("--model", type=Path, default=root / "artifacts" / "model.joblib")
    prediction.add_argument("--output", type=Path, default=root / "artifacts" / "predictions.csv")
    args = parser.parse_args()

    try:
        if args.command == "train":
            model, metrics = fit_models(read_csv(args.data, training=True))
            args.output.mkdir(parents=True, exist_ok=True)
            joblib.dump({"project": PROJECT, "model": model}, args.output / "model.joblib")
            report = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False)
            (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
            print(report)
            print(f"Saved model and metrics to {args.output}")
        else:
            data = read_csv(args.data)
            bundle = joblib.load(args.model)
            if bundle["project"] != PROJECT:
                raise ValueError("The model belongs to another project.")
            result = data.assign(**predict(bundle["model"], data))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"{len(result)} predictions saved to {args.output}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
