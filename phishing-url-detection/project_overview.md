## 🔐  Phishing URL Detection — End-to-End Machine Learning Pipeline
📌 Overview

Phishing websites are a major cause of credential theft, financial fraud, and cyberattacks. This project builds a full machine learning pipeline that classifies phishing URLs using engineered lexical, structural, and domain-based features.
The final XGBoost model achieves 96% accuracy and strong generalization across validation splits.

This work was completed as part of the Boston University MS in Data Science program.

## 📊 Dataset Summary

11,055 URLs, labeled phishing vs. legitimate

88 engineered features, including:

URL length, entropy, digit/special character ratios

Subdomain depth, path complexity

Domain age, SSL certificate status

Suspicious token presence

Mild class imbalance

Required normalization, type enforcement, and feature validation

## Dataset Source 
https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset

## 🎯 Project Objectives

Perform comprehensive EDA on URL-based features

Engineer lexical, structural, and domain attributes

Test a wide range of ML models (linear, tree-based, ensemble)

Apply PCA for dimensionality reduction

Tune hyperparameters using GridSearchCV + Stratified K-Fold

Compare metrics and validate robustness

## 🔍 Exploratory Data Analysis

Key findings:

URL length, entropy, and subdomain depth strongly correlate with phishing behavior

Several features were highly skewed → scaling required

Outliers indicated manually crafted phishing URLs

PCA revealed clear separation between classes in first two components

Common visuals:

Heatmaps

Distribution plots

Correlation matrices

PCA 2-D projections

## 🛠 Feature Engineering
Lexical Features

Character entropy

Count of digits, hyphens, underscores

Suspicious keyword flags

Structural Features

Number of subdomains

URL path depth

Presence of IP address

Shortened URL indicator

Domain & Network Features

Domain age / registration length

DNS record validity

SSL verification

These features created a highly informative numerical dataset for modeling.

## 🤖 Modeling
Models Evaluated

Logistic Regression

SVM (linear & RBF kernel)

KNN

Random Forest

Gradient Boosting

XGBoost (Best Performer)

Why XGBoost Won

Handles complex nonlinear interactions

Robust to noisy / redundant features

Strong performance without scaling

Highly tunable with L1/L2 regularization

## ⚙️ Hyperparameter Optimization

Used GridSearchCV + Stratified 5-fold cross-validation.

Key tuned parameters:

max_depth

learning_rate

subsample

colsample_bytree

reg_lambda, reg_alpha

## 📉 PCA & Dimensionality Reduction

First 15 components explained ~92% variance

PCA improved interpretability

Slight accuracy decrease compared to full-feature model

Confirmed XGBoost performs best without dimensionality reduction

## 📈 Results
Final XGBoost Model Performance
Metric	Score
Accuracy	96%
AUC	0.98
Precision	0.95
Recall	0.97
F1-Score	0.96

Confusion matrix shows very few false positives and near-zero false negatives.

Top Feature Contributors

URL length

Entropy

Subdomain depth

Number of special characters

Domain age

HTTPS flag

📁 Project Notebooks
01_polynomial_interaction_terms.ipynb
02_lasso_ridge_elastic_net.ipynb
03_pcr_plsr_selection.ipynb
04_logistic_regression_feature_scaling.ipynb
05_svm_kernel_regularization.ipynb
06_decision_trees_random_forest.ipynb
07_kmeans_elbow_silhouette.ipynb
08_knn_classification.ipynb
09_gradient_boosting_xgboost.ipynb
10_pca_dimensionality_reduction.ipynb
11_dbscan_core_samples.ipynb

Each notebook documents the modeling process in detail with code, comments, and visualizations.

## 🏁 Conclusion

This project demonstrates a production-grade machine learning workflow for phishing detection and highlights strong capabilities in:

Feature engineering

Model selection

Hyperparameter tuning

PCA

Evaluation methodology

Interpretability

Domain knowledge in cybersecurity

## 🚀 Future Improvements

Add SHAP-based explainability

Deploy as a real-time API using FastAPI

Add live phishing URL scoring UI on GitHub Pages
