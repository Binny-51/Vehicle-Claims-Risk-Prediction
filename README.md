# 🚗 Car Insurance Claims Prediction  

This project predicts the **likelihood of car insurance claims** using **machine learning models**, with a special focus on the **XGBoost algorithm**.  

The goal is to assist insurance companies in **risk assessment**, **fraud detection**, and **premium optimization** by analyzing customer and vehicle data.  

---

## 📌 Features
- End-to-end ML pipeline with **data preprocessing, feature engineering, and model training**  
- Extensive **Exploratory Data Analysis (EDA)** for insights into claim patterns  
- Handling **missing values & outliers** using `SimpleImputer`, `KNNImputer`, and VIF analysis  
- Implementation of multiple ML models for comparison:
  - Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, AdaBoost, Bagging  
  - KNN, SVM, SGD Classifier  
  - **XGBoost (main algorithm)**  
  - CatBoost  
- **Hyperparameter tuning** with `GridSearchCV` & `RandomizedSearchCV`  
- Evaluation using:
  - **Classification** → F1-score, Confusion Matrix  
  - **Regression** → Mean Squared Error (MSE), Mean Absolute Error (MAE)  

---

## 📂 Project Workflow
1. **Data Loading & Cleaning**
   - Missing value treatment (Simple & KNN Imputation)  
   - Outlier detection & removal using **Variance Inflation Factor (VIF)**  
   - Handling categorical & numerical columns separately  

2. **Exploratory Data Analysis (EDA)**
   - Visualizations with **Seaborn & Matplotlib**  
   - Claim frequency by driver profile, car type, and other features  
   - Correlation analysis  

3. **Feature Engineering**
   - One-Hot Encoding & Ordinal Encoding for categorical variables  
   - Standard Scaling for numerical variables  
   - Feature selection  

4. **Model Building**
   - Data split into **Train/Test sets**  
   - Model pipelines with preprocessing + classifier  
   - Training multiple models  

5. **XGBoost Algorithm**
   - Main classifier chosen for final prediction  
   - Gradient Boosted Decision Trees with regularization  
   - Tuned hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, etc.  

6. **Model Tuning & Evaluation**
   - Hyperparameter tuning with GridSearchCV & RandomizedSearchCV  
   - Metrics used:
     - F1-Score (for imbalanced claims dataset)  
     - Confusion Matrix for classification performance  
     - MSE & MAE for regression analysis  

---

## 🛠️ Tech Stack
- **Language**: Python 3.x  
- **Libraries**:  
  - Data Analysis → `pandas`, `numpy`  
  - Visualization → `matplotlib`, `seaborn`  
  - ML Models → `scikit-learn`, `xgboost`, `catboost`  
  - Stats → `scipy`, `statsmodels`  

---

## 🚀 Installation
Clone this repository:
```bash
git clone https://github.com/your-username/car-insurance-claims-prediction.git
cd car-insurance-claims-prediction
