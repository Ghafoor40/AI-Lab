from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "restaurant_revenue.csv"
MODEL_PATH = ROOT / "models" / "best_restaurant_revenue_model.joblib"

FEATURES = [
    "Number_of_Customers",
    "Menu_Price",
    "Marketing_Spend",
    "Cuisine_Type",
    "Average_Customer_Spending",
    "Promotions",
    "Reviews",
]

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)

dataset = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)


def to_model_input(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Number_of_Customers": int(payload["number_of_customers"]),
                "Menu_Price": float(payload["menu_price"]),
                "Marketing_Spend": float(payload["marketing_spend"]),
                "Cuisine_Type": str(payload["cuisine_type"]),
                "Average_Customer_Spending": float(
                    payload["average_customer_spending"]
                ),
                "Promotions": int(payload["promotions"]),
                "Reviews": int(payload["reviews"]),
            }
        ]
    )


@app.get("/")
def index():
    return render_template(
        "index.html",
        cuisines=sorted(dataset["Cuisine_Type"].dropna().unique().tolist()),
        feature_names=FEATURES,
        total_rows=len(dataset),
    )


@app.get("/api/meta")
def meta():
    return jsonify(
        {
            "rows": len(dataset),
            "columns": dataset.columns.tolist(),
            "cuisines": sorted(dataset["Cuisine_Type"].dropna().unique().tolist()),
            "model_path": str(MODEL_PATH),
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    missing = [key for key in [
        "number_of_customers",
        "menu_price",
        "marketing_spend",
        "cuisine_type",
        "average_customer_spending",
        "promotions",
        "reviews",
    ] if key not in payload or str(payload[key]).strip() == ""]

    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        model_input = to_model_input(payload)
        prediction = float(model.predict(model_input)[0])
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    return jsonify(
        {
            "predicted_monthly_revenue": round(prediction, 2),
            "model_input": model_input.to_dict(orient="records")[0],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
