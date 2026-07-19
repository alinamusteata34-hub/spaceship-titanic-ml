# spaceship-titanic-ml
ML pipeline for Kaggle's Spaceship Titanic competition, data cleaning, feature engineering, and model comparison (Random Forest vs XGBoost)
# Spaceship Titanic — ML Classification Pipeline

Kaggle competition project predicting which passengers were "transported" 
to an alternate dimension, based on passenger data.

## What I did
- Cleaned missing data across 12+ features (numeric imputation, categorical mode-filling)
- Engineered new features from raw columns:
  - Split `Cabin` into `Deck`, `CabinNum`, and `Side`
  - Extracted `Group` and `GroupSize` from `PassengerId`
- Compared multiple models:
  - Random Forest (baseline): 77.6% validation accuracy
  - Random Forest + engineered features: 79.9% validation accuracy
  - XGBoost (tuned): 80.3% validation accuracy
- Submitted predictions to Kaggle leaderboard: **0.79424 public score**

  
## 🚀 Live Demo
Try the live prediction app here: https://spaceship-titanic-alina.streamlit.app/

## Tools
Python, pandas, scikit-learn, XGBoost

## Key takeaway
Feature engineering (especially extracting structured info from `Cabin`) 
had a bigger impact on performance than switching model types.
