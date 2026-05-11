from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "restaurant_revenue.csv"
MODEL_DIR = ROOT / "models"
TARGET = "Monthly_Revenue"


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "name": name,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "pipeline": pipeline,
    }
    return metrics


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    candidates = [
        ("linear_regression", LinearRegression()),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=2,
            ),
        ),
    ]

    results = [
        evaluate_model(name, model, x_train, x_test, y_train, y_test)
        for name, model in candidates
    ]
    best = max(results, key=lambda item: item["r2"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_DIR / "best_restaurant_revenue_model.joblib")

    print("Dataset rows:", len(df))
    for result in results:
        print(
            f"{result['name']}: MAE={result['mae']:.3f}, "
            f"R2={result['r2']:.3f}"
        )
    print(
        "Saved best model to:",
        MODEL_DIR / "best_restaurant_revenue_model.joblib",
    )


if __name__ == "__main__":
    main()
