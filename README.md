# Will This Visitor Buy? 🛒

Most website visitors leave without buying. This project predicts from a browsing session whether a visitor will purchase, so a site can show a coupon or chat only to visitors who need it.

## 🔗 Live Demo
https://purchase-intent-app-ax3qg7btznfatahrkmu5nj.streamlit.app

## Results (test set, 2,441 sessions)

| Model | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Baseline (majority class) | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.619 | 0.796 | 0.507 | 0.910 |
| Random Forest | 0.609 | 0.513 | 0.748 | 0.925 |
| Gradient Boosting | 0.673 | 0.639 | 0.711 | 0.936 |
| **Tuned Gradient Boosting + tuned threshold** | **0.70** | **0.73** | **0.67** | **0.937** |

The final model catches 73% of purchasing sessions (Purchase class).

## How it works
- Train/test split before any preprocessing (no leakage)
- scikit-learn Pipeline: feature engineering, scaling, one-hot encoding
- Baseline + 3 models, tuned with RandomizedSearchCV (5-fold CV) on the training set only
- Decision threshold chosen from cross-validated predictions on the training set
- Streamlit app returns the purchase probability and a suggested action

## Run locally
