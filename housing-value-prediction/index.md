# Residential Property Tax Value Prediction
### *Fundamentals of Machine Learning — Boston University*

---

## 1. Project Overview

This project focuses on predicting **assessed residential property tax values** using a real housing dataset. The work was completed across two milestones:

- **Milestone 1:** Exploratory Data Analysis (EDA), data cleaning, baseline regression models  
- **Milestone 2:** Feature engineering, scaling, hyperparameter tuning, ensemble model comparison, final model selection  

The objective was to identify which features influence housing value and determine the regression model that provides the most accurate and stable predictions.

---

## 2. Dataset Description

The dataset contains structural and descriptive information about **residential properties**, including:

- Square footage & living area  
- Lot size  
- Bedrooms and bathrooms  
- Year built  
- Condition and quality indicators  
- Neighborhood or zoning categories  
- Historical tax value information  

**Target variable:** `taxvaluedollarcnt` (assessed property value in USD)

The target variable and key predictors were heavily **right-skewed**, impacting model choice and scaling strategies.

---

## 3. Exploratory Data Analysis (Milestone 1)

Major findings from EDA:

- **Strong correlations** between home value and physical features (square footage, bathrooms, condition).  
- **Right-skewed numeric distributions**, suggesting nonlinear models would perform better.  
- **Categorical variables** provided meaningful neighborhood-level variation.  
- **Outliers** (luxury homes) increased variance but did not significantly harm ensemble model performance.

EDA supported the hypothesis that **tree-based and ensemble models** would outperform linear models.

---

## 4. Feature Engineering

Multiple feature sets were constructed:

### Cleaned_NumScaled_CatUnscaled
- Numeric features scaled  
- Categorical features one-hot encoded  

### Cleaned_AllScaled
- All features scaled uniformly  
- Best suited for linear and regularized models  

### Best 15-Feature Subset (Final Feature Set)
Selected using:

- Mutual information  
- Correlation with the target  
- Early Gradient Boosting feature importance  

This subset produced the strongest overall performance and was used for the final model.

---

## 5. Modeling Approaches

The following models were trained and evaluated:

### Linear Models
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
Performance: RMSE ≈ **$480K–$510K**

### Tree-Based Models
- Decision Tree Regressor  
- Random Forest Regressor  
- Bagging Regressor  
Decision Trees overfit; Bagging and RF performed well but were computationally expensive to tune.

### Boosting Models
- **Gradient Boosting Regressor (final model)**

Gradient Boosting consistently demonstrated the best balance of accuracy and stability.

---

## 6. Hyperparameter Tuning

The tuning strategy focused on parameters with the biggest impact:

### Gradient Boosting
- `learning_rate`  
- `n_estimators`  
- `max_depth`  
- `max_features`  

### Random Forest
- `n_estimators`  
- `max_depth`  
- `max_features`  

### Bagging Regressor
- `n_estimators`  
- `max_samples`  
- `max_features`  

Random Forest and Bagging sweeps frequently stalled due to system resource limits, making Gradient Boosting the only model that could be reliably tuned end-to-end.

---

## 7. Model Performance

Approximate cross-validated RMSE values:

| Model | RMSE |
|-------|----------------|
| **Gradient Boosting (tuned)** | **$418,863** |
| Random Forest | $430K–$450K |
| Bagging Regressor | ~$440K |
| Ridge / Lasso / Linear | $480K–$510K |
| Decision Tree | ~$580K+ |

### Final Model Summary
**GradientBoostingRegressor** using the **15-feature subset**:

- **Mean CV RMSE:** $418,863  
- **Std Dev:** $27,101  
- **Num Features:** 15  

This model delivered the best combination of accuracy, consistency, and computational feasibility.

---

## 8. Key Takeaways

- Ensemble models significantly outperform linear models on nonlinear, skewed housing data.  
- A **carefully selected smaller feature subset** generalizes better than the full feature set.  
- Model complexity is not a substitute for high-quality features—“garbage in, garbage out.”  
- Practical compute limitations can influence feasible model selection.  
- Gradient Boosting offered the best accuracy with manageable training time.

---

## 9. Future Work

Potential improvements include:

- Testing **XGBoost, LightGBM, or CatBoost** with restricted parameter grids  
- Incorporating **geospatial and neighborhood-level data**  
- Using **SHAP values** for deeper interpretability  
- Applying **monotonic constraints** to enforce realistic pricing trends  
- Evaluating additional feature engineering approaches  

---

## 10. Notebooks

The full analysis and modeling workflow is documented in the following notebooks:

- **[Milestone 1 – Exploratory Data Analysis & Baseline Modeling](https://github.com/amarem07/Portfolio/blob/main/housing-value-prediction/Housing_01_EDA.ipynb)**  
  Covers data cleaning, exploratory analysis, distributional analysis, correlation studies, and initial regression models.

- **[Milestone 2 – Feature Engineering, Model Tuning & Final Selection](https://github.com/amarem07/Portfolio/blob/main/housing-value-prediction/Housing_02_Modeling.ipynb)**  
  Includes feature engineering strategies, scaling approaches, hyperparameter tuning, ensemble model comparison, and final model selection.

