# Energy Consumption Prediction – Regression Models

This repository contains a **machine learning workflow** for predicting monthly energy costs for residential and commercial buildings using a **synthetic dataset of 5,000 customers**. The notebook includes exploratory data analysis, preprocessing, model training, hyperparameter tuning, and evaluation.

---

## Dataset Overview

The dataset includes the following features:

* `customer_id`: Unique identifier for each customer (`CUSTOMER_0001` to `CUSTOMER_5000`)
* `customer_type`: Type of property (`residential` or `commercial`)
* `regions`: Geographic region (`North`, `Northeast`, `Midwest`, `Southeast`, `South`)
* `building_size_m2`: Building size in square meters (17, 24, 45, 52, or 77 m²)
* `occupants`: Number of occupants (1–4)
* `energy_cost_brl`: Monthly energy cost in BRL

---

## Notebook Contents

1. **Exploratory Data Analysis (EDA)**

   * Distributions, histograms, count plots, and boxplots
   * Correlation heatmaps
   * Insights into energy cost patterns across customer type, region, occupancy, and building size

2. **Data Preprocessing**

   * Standard scaling for numerical features (`building_size_m2`, `occupants`)
   * One-hot encoding for categorical features (`customer_type`, `regions`)
   * Train-test split (80/20)

3. **Predictive Modeling**

   * Linear Regression
   * Random Forest Regressor
   * Random Forest Regressor with GridSearchCV
   * Gradient Boosting Regressor
   * Gradient Boosting Regressor with GridSearchCV

4. **Evaluation**

   * Metrics: RMSE and R²
   * Scatter plots: predicted vs actual energy costs
   * Comparison table of all models

---

## Results

| Model             | RMSE  | R²    |
| ----------------- | ----- | ----- |
| Linear Regression | 20.15 | 0.308 |
| Random Forest     | 15.32 | 0.599 |
| RF (GridSearch)   | 14.96 | 0.618 |
| Gradient Boosting | 15.05 | 0.614 |
| GB (GridSearch)   | 15.01 | 0.616 |

> Considering RMSE, ensemble models outperform linear regression, with GridSearch slightly improving performance.

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

* Only **scikit-learn** models are used; no deep learning dependencies required.

---

## Optional Extensions

* Test additional regression models (KNN, SVR, etc.)
* Implement neural networks with TensorFlow or PyTorch
* Add feature engineering (interaction terms, polynomial features)

