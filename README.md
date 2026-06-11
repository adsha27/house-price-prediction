# House Price Prediction

ML pipeline for predicting house prices on the Ames Housing dataset. Covers EDA, feature engineering, and comparison of multiple regression algorithms with SHAP-based interpretability.

## What it covers

- EDA with missing value heatmaps and correlation analysis
- Feature engineering pipeline (categorical encoding, skew correction, interaction features)
- Model comparison: Linear, Ridge, XGBoost, LightGBM, CatBoost, stacking ensembles
- SHAP values for feature importance and prediction explanation

## Setup

```bash
git clone https://github.com/adsha27/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

Data: download `train.csv` from the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place it in `data/raw/AmesHousing.csv`. California Housing (sklearn built-in) works without any download.

## Stack

Python, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, pandas, Matplotlib
