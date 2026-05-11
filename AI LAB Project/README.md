# Restaurant Revenue Prediction Project

This project uses the provided restaurant dataset to train a machine learning model that predicts `Monthly_Revenue`.

## Project structure

- `data/raw/restaurant_revenue.csv`: source dataset from the zip file
- `src/train.py`: training script with preprocessing and model comparison
- `src/app.py`: Flask backend and frontend entry point
- `templates/index.html`: browser UI
- `static/styles.css`: frontend styling
- `models/`: saved trained model output
- `requirements.txt`: Python dependencies

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run training

```powershell
python src/train.py
```

## Run frontend and backend

```powershell
.venv\Scripts\Activate.ps1
python src/app.py
```

Open this in browser:

`http://127.0.0.1:5000`

The script compares:

- Linear Regression
- Random Forest Regressor

It then saves the best model in `models/best_restaurant_revenue_model.joblib`.

## Target column

`Monthly_Revenue`
