# AI‑Driven Multi‑Target Weather Predictor

![Weather](https://img.shields.io/badge/forecast-multi--target-blue)

**Forecast temperature, feels‑like, humidity, precipitation & weather condition in one shot, with model interpretability baked in.**

---

## 🚀 Quick Start

```bash
# clone repo & enter
$ git clone <repo‑url>
$ cd weather‑predictor

# create env + install
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r Requirment.txt

# launch the notebook demo
$ jupyter lab b+f_weather_prediction_copy_2.ipynb
```

The notebook walks through data prep, feature engineering, training four models (Linear Regression, SVM, Random Forest, XGBoost), stacking them, and generating SHAP explanations.

---

## 🗂️ Project Structure

```
├── Dataset11-Weather-Data.csv   # raw data
├── b+f_weather_prediction_copy_2.ipynb  # full pipeline notebook
├── weather_model.pkl           # trained ensemble
├── weather_scaler.pkl          # feature scaler
├── Requirment.txt              # Python deps
└── README.md
```

---

## ✨ Features

* **Multi‑target regression & classification** in one pass.
* **Stacking ensemble** combining Linear Regression, SVM, Random Forest, XGBoost.
* **Custom features**: temperature range, humidity‑pressure interaction, more.
* **Confidence scores** on every prediction.
* **Explainability** via SHAP plots.

---

## 📊 Example Usage

```python
import pickle, pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = pickle.load(open('weather_scaler.pkl','rb'))
model  = pickle.load(open('weather_model.pkl','rb'))

sample = pd.DataFrame({
    'temp': [28], 'dew_point':[23], 'humidity':[78],
    'wind_speed':[3.4], 'pressure':[1005]
})
X = scaler.transform(sample)
print(model.predict(X))
```

---

## 🔬 Explainability

Run the last notebook cells to generate SHAP summary and dependence plots for full model transparency.

---

## 📜 License

MIT License — free to use, modify, and distribute.
