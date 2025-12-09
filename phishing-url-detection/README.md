# 🛡️ Phishing URL Detection Using Machine Learning  
### **XGBoost · PCA · Feature Engineering · Model Optimization**

This project builds an end-to-end machine learning pipeline to classify phishing URLs using lexical features, domain attributes, and structural patterns. The dataset contains **11,055 URLs** with **88 engineered features** extracted directly from each URL string.

---

## 📌 Project Overview
Phishing websites impersonate legitimate businesses to steal information from users. This project uses machine learning to classify URLs as **phishing** or **legitimate** based purely on URL-level features—no HTML scraping required.

This makes the model fast, lightweight, and deployable in real-time.

---

## 📊 Dataset Description

**Dataset Size:** 11,055 URLs  
**Features:** 88 engineered attributes including:
- URL length, digit count, special character count  
- HTTPS presence  
- TLD analysis  
- Subdomain depth  
- Suspicious patterns (e.g., “@”, “//”, “-”)  
- Domain age, expiration, registrar fields  

**Target:**  
- `0` = legitimate  
- `1` = phishing  

---

## 🧹 Data Preprocessing & EDA
The preprocessing pipeline included:
- Handling missing values  
- Removing non-informative features  
- Scaling numerical features (StandardScaler)  
- Correlation analysis  
- Outlier detection  
- PCA to reduce dimensionality and noise  

**Exploratory analysis included:**
- Distribution analysis  
- Correlation heatmaps  
- Feature importance  
- Class balance visualization  

---

## 🔧 Machine Learning Pipeline

### **Models Trained**
| Algorithm | Purpose |
|----------|----------|
| Logistic Regression | Baseline, interpretable |
| SVM (RBF/Linear) | Strong non-linear classifier |
| KNN | Distance-based comparison |
| Random Forest | Tree-based baseline |
| XGBoost | Main optimized model |

---

## 🔬 PCA Dimensionality Reduction
PCA was applied to:
- Reduce noise  
- Remove multicollinearity  
- Speed up training  
- Improve generalization  

Tested multiple `n_components` values to benchmark performance.

---

## 🏆 Model Performance (Final Model: XGBoost)

**Accuracy:** **96%**  
**AUC:** 0.98  
**Precision/Recall:** Balanced, low false negatives  
**Confusion matrix:** Clean diagonal with minimal error  
**Key XGBoost parameters tuned:**
- `max_depth`  
- `learning_rate`  
- `n_estimators`  
- `subsample`  
- `colsample_bytree`  
- `reg_alpha`, `reg_lambda`  

Used **GridSearchCV + Stratified K-Fold**.

---

## 📈 Evaluation & Visuals
Included:
- ROC curve  
- Precision–Recall curve  
- Feature importance plots  
- Confusion matrix  
- PCA component visualization  

(Add plots to `/projects/phishing-url-detection/images/`)

---

## 🚀 Key Outcomes
- End-to-end ML pipeline built from scratch  
- Strong performance without needing HTML scraping  
- XGBoost significantly outperformed all baselines  
- PCA improved model stability and reduced noise  
- Ready for deployment as an API endpoint

---

## 🧠 Skills Demonstrated
- Feature engineering  
- ML classification  
- PCA & dimensionality reduction  
- Model tuning & optimization  
- EDA, visualization  
- Pipeline design  
- XGBoost mastery  
- Bias/variance and error analysis  

---

## 📎 Files in This Folder
