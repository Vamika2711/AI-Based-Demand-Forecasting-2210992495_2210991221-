"""
Research Paper Source Code:
A Comparative Study of AI-Based and Traditional ERP Forecasting Methods 
Using Marketplace Data Integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set seed for reproducibility
np.random.seed(42)

# =====================================================================
# 1. DATA SIMULATION LAYER (Simulating ERP & Marketplace Integration)
# =====================================================================
print("--- [1/5] Ingesting Data from ERP & Marketplace APIs ---")

# Generate 3 years of daily data (1095 days)
date_range = pd.date_range(start="2023-01-01", end="2025-12-30", freq="D")
n_days = len(date_range)

# Simulate Internal ERP Data: Base demand with weekly and annual seasonality
base_demand = 200
weekly_pattern = np.sin(date_range.dayofweek * (2 * np.pi / 7)) * 30
annual_pattern = np.sin(date_range.dayofyear * (2 * np.pi / 365)) * 120
noise = np.random.normal(0, 15, n_days)

internal_sales = base_demand + weekly_pattern + annual_pattern + noise
internal_sales = np.clip(internal_sales, 10, None)  # Ensure no negative sales
internal_price = np.random.uniform(45, 55, n_days)

# Simulate External Marketplace Indicators
# Competitor pricing introduces a price discrepancy feature
competitor_price = internal_price * np.random.uniform(0.90, 1.10, n_days)
# Google Trends indicator (scaled 0-100) representing product category volume
marketplace_trend = 50 + (annual_pattern * 0.3) + np.random.normal(0, 5, n_days)
marketplace_trend = np.clip(marketplace_trend, 0, 100)

# Combine into a unified Data Frame (The Hybrid Framework Fusion)
df = pd.DataFrame({
    'Date': date_range,
    'ERP_Sales': internal_sales,
    'ERP_Price': internal_price,
    'Market_Comp_Price': competitor_price,
    'Market_Trend_Score': marketplace_trend
})
df.set_index('Date', inplace=True)

# =====================================================================
# 2. FEATURE ENGINEERING LAYER
# =====================================================================
print("--- [2/5] Executing Feature Engineering Engine ---")

# Traditional Baseline Feature (Moving Average)
df['SMA_7'] = df['ERP_Sales'].rolling(window=7).mean()

# Advanced ML Features (Lags & Contextual Indices)
df['Sales_Lag_1'] = df['ERP_Sales'].shift(1)
df['Sales_Lag_7'] = df['ERP_Sales'].shift(7)
df['Sales_Lag_30'] = df['ERP_Sales'].shift(30)

# Price Discrepancy Index (Positive value means our price is higher than competitor)
df['Price_Elasticity_Delta'] = df['ERP_Price'] - df['Market_Comp_Price']

# Clean up rows that generated NaN values due to lagging/rolling windows
df.dropna(inplace=True)

# Define feature arrays and targets
features = ['ERP_Price', 'Market_Comp_Price', 'Market_Trend_Score', 
            'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_30', 'Price_Elasticity_Delta']
X = df[features]
y = df['ERP_Sales']

# =====================================================================
# 3. TEMPORAL DATA SPLITTING & SCALING
# =====================================================================
print("--- [3/5] Splitting and Normalizing Datasets Chronologically ---")

# Chronological Split (80% Train, 20% Test) to protect time series integrity
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Isolate the traditional baseline forecast for the test set period
traditional_forecast = df['SMA_7'].iloc[split_idx:]

# Standardize inputs using Min-Max Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 4. MODEL EXECUTION LAYER
# =====================================================================
print("--- [4/5] Training Statistical and Machine Learning Models ---")

# Model A: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)

# Model B: Random Forest Regressor (Ensemble Architecture)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

# =====================================================================
# 5. EXPERIMENTAL EVALUATION LAYER
# =====================================================================
print("--- [5/5] Compiling Performance Matrix Results ---\n")

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

metrics = {
    'Traditional (SMA_7)': {
        'MAE': mean_absolute_error(y_test, traditional_forecast),
        'RMSE': np.sqrt(mean_squared_error(y_test, traditional_forecast)),
        'MAPE (%)': calculate_mape(y_test, traditional_forecast),
        'R2 Score': r2_score(y_test, traditional_forecast)
    },
    'Linear Regression (AI Base)': {
        'MAE': mean_absolute_error(y_test, lr_predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_predictions)),
        'MAPE (%)': calculate_mape(y_test, lr_predictions),
        'R2 Score': r2_score(y_test, lr_predictions)
    },
    'Random Forest (Proposed Framework)': {
        'MAE': mean_absolute_error(y_test, rf_predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_predictions)),
        'MAPE (%)': calculate_mape(y_test, rf_predictions),
        'R2 Score': r2_score(y_test, rf_predictions)
    }
}

# Print metrics inside a cleanly structured evaluation table
results_df = pd.DataFrame(metrics).T
print(results_df.to_string(formatters={'MAE': '{:,.2f}'.format, 'RMSE': '{:,.2f}'.format, 'MAPE (%)': '{:,.2f}%'.format, 'R2 Score': '{:,.4f}'.format}))

# =====================================================================
# 6. GRAPHICAL VISUALIZATION GENERATION
# =====================================================================
# Graph 1: Actual Demand vs. Model Predictions over time
plt.figure(figsize=(14, 6))
plt.plot(y_test.index[-90:], y_test.values[-90:], label='Actual Historical Sales', color='black', linewidth=2)
plt.plot(y_test.index[-90:], traditional_forecast.values[-90:], label='Traditional Baseline (SMA)', color='red', linestyle='--')
plt.plot(y_test.index[-90:], lr_predictions[-90:], label='Linear Regression Forecast', color='orange', linestyle=':')
plt.plot(y_test.index[-90:], rf_predictions[-90:], label='Random Forest (AI-Integrated)', color='green', linewidth=2)
plt.title('Demand Prediction Evaluation Benchmark (Last 90 Days Variance Analysis)')
plt.xlabel('Timeline Date Block')
plt.ylabel('Quantities Demanded / Dispatched')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('demand_variance_chart.png', dpi=300)
print("\n[Visual Output Saved] 'demand_variance_chart.png' generated successfully.")

# Graph 2: Feature Importance Evaluation (Validating the Marketplace Indicators)
importances = rf_model.feature_importances__
indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in indices]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=sorted_features, palette='viridis')
plt.title('Random Forest Feature Contribution Weights (Marketplace Context Impact)')
plt.xlabel('Relative Feature Gini-Importance Structural Score')
plt.ylabel('Engineered Feature Fields')
plt.tight_layout()
plt.savefig('feature_importance_chart.png', dpi=300)
print("[Visual Output Saved] 'feature_importance_chart.png' generated successfully.")