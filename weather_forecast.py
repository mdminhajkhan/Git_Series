import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
)

def run_ml_pipeline():
    print("=== PHASE 2: ML Pipeline & Feature Engineering ===")
    
    # 1. Ingest Data
    if not os.path.exists("weather_data.csv"):
        raise FileNotFoundError("weather_data.csv not found. Run generate_data.py first.")
        
    df = pd.read_csv("weather_data.csv")
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Check for missing values and impute if any
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Imputing...")
        df = df.ffill().bfill()
    else:
        print("Data is clean. No missing values found.")
        
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Feature Engineering
    print("Performing feature engineering...")
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Targets (shifted backward by 1 day, meaning tomorrow's value mapped to today's features)
    df['RainTomorrow'] = df['RainToday'].shift(-1)
    df['MaxTempTomorrow'] = df['MaxTemp'].shift(-1)
    
    # Save a copy for plotting historical rainfall timeseries (includes Date before dropping)
    df_for_timeseries = df.copy()
    
    # Drop the final row (contains NaNs for targets)
    df_clean = df.dropna().copy()
    print(f"Dataset size after dropping the shifted final row: {df_clean.shape[0]} rows.")
    
    # Split features and targets
    # We do NOT include Rainfall and RainToday of today in features, to align with the frontend inputs
    features = ['MinTemp', 'MaxTemp', 'Humidity', 'Pressure', 'Month', 'DayOfYear']
    
    X = df_clean[features]
    y_class = df_clean['RainTomorrow'].astype(int)
    y_reg = df_clean['MaxTempTomorrow']
    
    # 3. Chronological Split (Train: 2021-2024, Test: 2025)
    train_mask = df_clean['Date'].dt.year < 2025
    test_mask = df_clean['Date'].dt.year == 2025
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train_class, y_test_class = y_class[train_mask], y_class[test_mask]
    y_train_reg, y_test_reg = y_reg[train_mask], y_reg[test_mask]
    
    print(f"Train size: {X_train.shape[0]} samples (Years 2021-2024)")
    print(f"Test size: {X_test.shape[0]} samples (Year 2025)")
    
    # Save Date series for plotting
    test_dates = df_clean[test_mask]['Date']
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, "scaler.joblib")
    print("Saved feature scaler to scaler.joblib")
    
    # 4. Model Training & Comparison (Classification)
    print("\n--- Classification Models Comparison ---")
    
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost Classifier": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    class_results = {}
    best_f1 = -1
    best_class_model_name = ""
    best_class_model = None
    
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train_class)
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test_class, y_pred)
        prec = precision_score(y_test_class, y_pred, zero_division=0)
        rec = recall_score(y_test_class, y_pred, zero_division=0)
        f1 = f1_score(y_test_class, y_pred, zero_division=0)
        
        class_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
        print(f"{name}:")
        print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_class_model_name = name
            best_class_model = clf
            
    print(f"\nWinning Classification Model (Optimized for F1-Score): {best_class_model_name}")
    joblib.dump(best_class_model, "classification_model.joblib")
    print("Saved classification model to classification_model.joblib")
    
    # Train Logistic Regression explicitly (even if it wasn't the winner) to extract weights for JS frontend
    lr_model = classifiers["Logistic Regression"]
    lr_coef = lr_model.coef_[0]
    lr_intercept = lr_model.intercept_[0]
    
    # Math transformation for raw unscaled values:
    # z = intercept + sum(coef_i * (x_i - mean_i) / std_i)
    # z = (intercept - sum(coef_i * mean_i / std_i)) + sum((coef_i / std_i) * x_i)
    mean = scaler.mean_
    scale = scaler.scale_
    
    adjusted_coef = lr_coef / scale
    adjusted_intercept = lr_intercept - np.sum(lr_coef * mean / scale)
    
    print("\n--- Logistic Regression Parameters for JavaScript (Unscaled Inputs) ---")
    print(f"Features: {features}")
    print(f"Original Coefs: {lr_coef}")
    print(f"Original Intercept: {lr_intercept}")
    print(f"Scaler Mean: {mean}")
    print(f"Scaler Scale (Std Dev): {scale}")
    print("\n>>> COPY THESE INTO demo.html JAVASCRIPT <<<")
    print(f"Adjusted Coefficients (MinTemp, MaxTemp, Humidity, Pressure, Month, DayOfYear):")
    print(f"  {list(adjusted_coef)}")
    print(f"Adjusted Intercept:")
    print(f"  {adjusted_intercept}")
    print("----------------------------------------------------------------------\n")
    
    # 5. Model Training & Comparison (Regression - MaxTempTomorrow)
    print("--- Regression Models Comparison ---")
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=100)
    }
    
    best_r2 = -999
    best_reg_model_name = ""
    best_reg_model = None
    
    for name, reg in regressors.items():
        # Scale features for regression as well (Random Forest doesn't strictly need it, but Linear Regression does)
        reg.fit(X_train_scaled, y_train_reg)
        y_pred_reg = reg.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, y_pred_reg)
        
        print(f"{name}:")
        print(f"  MSE: {mse:.4f} | RMSE: {rmse:.4f} | R² Score: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_reg_model_name = name
            best_reg_model = reg
            
    print(f"\nWinning Regression Model: {best_reg_model_name}")
    joblib.dump(best_reg_model, "regression_model.joblib")
    print("Saved regression model to regression_model.joblib")
    
    # Print regression parameters if Linear Regression won (for frontend use, or we can use a heuristic)
    if best_reg_model_name == "Linear Regression":
        reg_coef = best_reg_model.coef_
        reg_intercept = best_reg_model.intercept_
        adj_reg_coef = reg_coef / scale
        adj_reg_intercept = reg_intercept - np.sum(reg_coef * mean / scale)
        print("\nLinear Regression adjusted parameters for JS (Unscaled):")
        print(f"Coefficients: {list(adj_reg_coef)}")
        print(f"Intercept: {adj_reg_intercept}")
    else:
        # Also print Linear Regression parameters just in case we want to use them in the standalone html
        lin_reg = regressors["Linear Regression"]
        reg_coef = lin_reg.coef_
        reg_intercept = lin_reg.intercept_
        adj_reg_coef = reg_coef / scale
        adj_reg_intercept = reg_intercept - np.sum(reg_coef * mean / scale)
        print("\nLinear Regression adjusted parameters for JS (Unscaled) [Reference]:")
        print(f"Coefficients: {list(adj_reg_coef)}")
        print(f"Intercept: {adj_reg_intercept}")
        
    print("\n=== PHASE 3: Visualization Export ===")
    sns.set_theme(style="darkgrid")
    
    # 1. Historical Rainfall Timeseries
    plt.figure(figsize=(12, 5))
    plt.plot(df_for_timeseries['Date'], df_for_timeseries['Rainfall'], color='teal', alpha=0.4, label='Daily Rainfall')
    # Add a rolling average to show monsoon seasonality clearly
    df_for_timeseries['Rainfall_30d'] = df_for_timeseries['Rainfall'].rolling(30, min_periods=1).mean()
    plt.plot(df_for_timeseries['Date'], df_for_timeseries['Rainfall_30d'], color='darkcyan', linewidth=2, label='30-Day Moving Average')
    plt.title("Historical Rainfall Patterns Over Time (5-Year Daily)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Rainfall (mm)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rainfall_timeseries.png", dpi=150)
    plt.close()
    print("Saved rainfall_timeseries.png")
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ['MinTemp', 'MaxTemp', 'Humidity', 'Pressure', 'Rainfall', 'RainToday', 'Month', 'DayOfYear', 'RainTomorrow', 'MaxTempTomorrow']
    corr = df_clean[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap of Weather Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved correlation_heatmap.png")
    
    # 3. Class Balance Plot
    plt.figure(figsize=(6, 5))
    # Count of RainTomorrow
    sns.countplot(x=y_class, palette="Set2")
    plt.title("Class Balance: Rain vs No Rain Tomorrow", fontsize=14, fontweight='bold')
    plt.xlabel("Rain Tomorrow (0 = No Rain, 1 = Rain)", fontsize=12)
    plt.ylabel("Count of Days", fontsize=12)
    plt.xticks([0, 1], ["No Rain", "Rain"])
    plt.tight_layout()
    plt.savefig("class_balance.png", dpi=150)
    plt.close()
    print("Saved class_balance.png")
    
    # 4. Confusion Matrix (Winning Classification Model)
    plt.figure(figsize=(6, 5))
    y_pred_best = best_class_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test_class, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    plt.title(f"Confusion Matrix: {best_class_model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")
    
    # 5. Feature Importance (Winning Classification Model)
    plt.figure(figsize=(8, 5))
    if hasattr(best_class_model, "feature_importances_"):
        importances = best_class_model.feature_importances_
    else:
        # For Logistic Regression, use absolute coefficients
        importances = np.abs(best_class_model.coef_[0])
        
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
    feat_imp.plot(kind='barh', color='skyblue', edgecolor='navy')
    plt.title(f"Feature Importance: {best_class_model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Saved feature_importance.png")
    
    # 6. Regression Scatter Plot (Actual vs Predicted MaxTempTomorrow)
    plt.figure(figsize=(7, 7))
    y_pred_reg_best = best_reg_model.predict(X_test_scaled)
    plt.scatter(y_test_reg, y_pred_reg_best, alpha=0.5, color='coral', edgecolors='red')
    
    # Diagonal y = x line
    min_val = min(y_test_reg.min(), y_pred_reg_best.min())
    max_val = max(y_test_reg.max(), y_pred_reg_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="y = x (Perfect Fit)")
    
    plt.title(f"MaxTemp Tomorrow Actual vs Predicted ({best_reg_model_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Actual MaxTemp (°C)", fontsize=12)
    plt.ylabel("Predicted MaxTemp (°C)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("regression_scatter.png", dpi=150)
    plt.close()
    print("Saved regression_scatter.png")
    
    print("\nPipeline execution complete. All model artifacts and evaluation plots have been saved.")

if __name__ == "__main__":
    run_ml_pipeline()
