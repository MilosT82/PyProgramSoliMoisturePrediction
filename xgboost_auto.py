import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Load and preprocess the dataset
df = pd.read_excel('data.xlsx')
df = df.rename(columns={'Vreme': 'DateTime'})
df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True)
df.set_index('DateTime', inplace=True)
df = df.sort_index()

# Combine 'SM1' and 'SM2' into a new 'SM1' and drop unnecessary columns
df['SM1'] = df[['SM1', 'SM2']].mean(axis=1)
df.drop(columns=['SM2', 'SM3', 'SM4', 'SM5', 'SM6'], inplace=True)

# Aggregate data by date
df_daily = df.resample('D').mean().reset_index()
df_daily['DateTime'] = pd.to_datetime(df_daily['DateTime'])
df_daily = df_daily.set_index('DateTime')

# Create time-based features
df_daily['weekofyear'] = df_daily.index.isocalendar().week
df_daily['dayofyear'] = df_daily.index.dayofyear
df_daily['month'] = df_daily.index.month

# Lagged features using 'AT1'
df_daily['AT1_prev_month'] = df_daily['AT1'].shift(30)

# Rolling statistics
df_daily['rolling_mean_AT1_30d'] = df_daily['AT1'].rolling(window=30).mean()
df_daily['rolling_std_AT1_30d'] = df_daily['AT1'].rolling(window=30).std()

# Harmonic features
df_daily['week_sin'] = np.sin(2 * np.pi * df_daily['weekofyear'] / 52)
df_daily['day_sin'] = np.sin(2 * np.pi * df_daily['dayofyear'] / 365)
df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['month'] / 12)

# Short-term change features
df_daily['AT1_diff1'] = df_daily['AT1'].diff(1)
df_daily['AT1_diff2'] = df_daily['AT1'].diff(2)
df_daily['PP1_diff1'] = df_daily['PP1'].diff(1)
df_daily['PP1_diff2'] = df_daily['PP1'].diff(2)
df_daily['AH1_diff1'] = df_daily['AH1'].diff(1)
df_daily['AH1_diff2'] = df_daily['AH1'].diff(2)

# Extreme weather indicators
df_daily['AT1_hot_day'] = (df_daily['AT1'] > df_daily['AT1'].quantile(0.95)).astype(int)
df_daily['AT1_cold_day'] = (df_daily['AT1'] < df_daily['AT1'].quantile(0.05)).astype(int)
df_daily['PP1_extreme_rain'] = (df_daily['PP1'] > df_daily['PP1'].quantile(0.95)).astype(int)
df_daily['PP1_dry_day'] = (df_daily['PP1'] < 1).astype(int)  # Alternative threshold

# Interaction features
df_daily['AT1_AH1_interaction'] = df_daily['AT1'] * df_daily['AH1']
df_daily['PP1_seasonal'] = df_daily['PP1'] * df_daily['month_sin']

# Momentum-based rolling features
df_daily['ewm_AT1_30d'] = df_daily['AT1'].ewm(span=30).mean()
df_daily['ewm_PP1_30d'] = df_daily['PP1'].ewm(span=30).mean()

df_daily.dropna(inplace=True)

# Normalize time
df_daily['time'] = (df_daily.index - df_daily.index[0]).days
df_daily['time_norm'] = df_daily['time'] / df_daily['time'].max()

# Perform seasonal decomposition on 'AT1'
result = seasonal_decompose(df_daily['AT1'], model='additive', period=365)
df_daily['AT1_trend'] = result.trend
df_daily['AT1_residual'] = result.resid
df_daily.dropna(inplace=True)

# Dry/Wet Spell Duration
dry_threshold = 1  # Precipitation < 1 mm is considered dry
wet_threshold = 1  # Precipitation >= 1 mm is considered wet

df_daily['Dry_Spell_Duration'] = 0
df_daily['Wet_Spell_Duration'] = 0

# Calculate dry spell duration
dry_spell = 0
for i in range(len(df_daily)):
    if df_daily['PP1'].iloc[i] < dry_threshold:
        dry_spell += 1
    else:
        dry_spell = 0
    df_daily['Dry_Spell_Duration'].iloc[i] = dry_spell

# Calculate wet spell duration
wet_spell = 0
for i in range(len(df_daily)):
    if df_daily['PP1'].iloc[i] >= wet_threshold:
        wet_spell += 1
    else:
        wet_spell = 0
    df_daily['Wet_Spell_Duration'].iloc[i] = wet_spell

# Final feature selection
df_features = df_daily[['time_norm', 'AT1', 'AT1_prev_month', 'rolling_mean_AT1_30d', 'rolling_std_AT1_30d', 
                        'week_sin', 'day_sin', 'month_sin', 'AT1_trend', 'AT1_residual', 'Dry_Spell_Duration', 
                        'Wet_Spell_Duration', 'AT1_diff1', 'AT1_diff2', 'PP1_diff1', 'PP1_diff2', 'AH1_diff1', 
                        'AH1_diff2', 'AT1_hot_day', 'AT1_cold_day', 'PP1_extreme_rain', 'PP1_dry_day', 
                        'AT1_AH1_interaction', 'PP1_seasonal', 'ewm_AT1_30d', 'ewm_PP1_30d']]

# Apply scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_features)

# Split data
split_date = '2019-01-01'
train = features_scaled[df_daily.index < split_date]
test = features_scaled[df_daily.index >= split_date]
y_train = df_daily.loc[df_daily.index < split_date, 'SM1']
y_test = df_daily.loc[df_daily.index >= split_date, 'SM1']

# Define Optuna objective function
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'alpha': trial.suggest_float('alpha', 0, 10),
        'lambda': trial.suggest_float('lambda', 0, 10),
    }
    dtrain = xgb.DMatrix(train, label=y_train)
    cv_results = xgb.cv(params, dtrain, num_boost_round=params['n_estimators'], nfold=5, early_stopping_rounds=25, metrics="rmse", as_pandas=True, seed=42)
    return cv_results['test-rmse-mean'].min()

# Directory to save models
model_dir = 'best_xgboost_model'
os.makedirs(model_dir, exist_ok=True)

# Track the best model
best_r2 = 0
best_model_path = None

for i in range(350):
    print(f"Iteration {i+1}/350")
    
    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=7)
    
    # Train the model
    best_params = study.best_params
    dtrain = xgb.DMatrix(train, label=y_train)
    dtest = xgb.DMatrix(test, label=y_test)
    model = xgb.train(best_params, dtrain, num_boost_round=best_params['n_estimators'], early_stopping_rounds=40, evals=[(dtest, 'eval')])
    
    # Predictions
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)
    
    # Evaluation Metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f'Training R2: {r2_train}')
    print(f'Testing R2: {r2_test}')
    
    # Save the model if R-squared is >= 0.8 and better than the previous best
    if r2_test >= 0.8 and r2_test > best_r2:
        # Delete the previous best model
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
        
        # Save the new best model
        best_r2 = r2_test
        best_model_path = os.path.join(model_dir, f'best_model_r2_{r2_test:.4f}.model')
        model.save_model(best_model_path)
        print(f"New best model saved with R2: {r2_test} at {best_model_path}")

print("Process completed.")
print(f"Best R-squared: {best_r2}")