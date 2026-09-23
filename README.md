# Customer Churn Prediction Using Ensemble Learning & SHAP Interpretability


An end-to-end customer churn prediction pipeline built on structured business
data (demographics, billing, service usage, and interaction history). The
project combines multiple models through **ensemble learning (voting and
stacking)** to outperform any single model baseline, and uses **SHAP** to
explain *why* the model predicts churn — surfacing the top churn drivers for
business stakeholders.


## Highlights

- Full pipeline: data cleaning → feature engineering → preprocessing →
  modeling → evaluation → interpretability
- 4 base models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- 2 ensemble strategies: **soft Voting Classifier** and **Stacking Classifier**
  (meta-learner: Logistic Regression on out-of-fold base predictions)
- Class imbalance handled with **SMOTE** on the training set
- Model comparison across Accuracy, Precision, Recall, F1, and ROC-AUC
- **SHAP** global summary plots, feature-importance ranking, and a per-customer
  waterfall plot explaining an individual high-risk prediction


## Repository structure

```
customer-churn-ensemble-shap/
├── Customer_Churn_Prediction_SHAP.ipynb   # main notebook (run this in Colab)
├── requirements.txt                       # for local / non-Colab runs
├── README.md
├── .gitignore

```

## Results

Model                  Accuracy  Precision  Recall   F1      ROC-AUC
-----------------------------------------------------------------------
Voting Ensemble          0.772     0.557     0.690   0.616    0.834
Gradient Boosting        0.768     0.552     0.684   0.611    0.834
Logistic Regression      0.732     0.497     0.778   0.607    0.833
XGBoost                  0.778     0.578     0.615   0.596    0.829
Random Forest            0.770     0.558     0.652   0.602    0.825
Stacking Ensemble        0.778     0.585     0.570   0.577    0.819
-----------------------------------------------------------------------


### Top churn drivers (SHAP)

- Month-to-month contracts are the single strongest churn driver
- High monthly charges combined with low tenure sharply increase churn risk
- Fiber-optic internet customers churn more than DSL customers
- Electronic check payment method correlates with higher churn

## How to run

git clone https://github.com/YOUR_USERNAME/customer-churn-ensemble-shap.git
cd customer-churn-ensemble-shap
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction_SHAP.ipynb
```

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `XGBoost` · `imbalanced-learn` (SMOTE) ·
`SHAP` · `matplotlib` / `seaborn` · Google Colab

## Dataset

[IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) —
7,043 customers, 21 features spanning demographics (gender, senior citizen,
partner, dependents), account info (tenure, contract, payment method,
billing), services subscribed (phone, internet, streaming, tech support), and
the churn label.

## License

MIT — feel free to fork and adapt for your own portfolio.
