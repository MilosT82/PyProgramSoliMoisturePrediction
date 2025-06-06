import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# Directory where models are saved
model_dir = 'best_xgboost_model'

# Load the preprocessed dataset (ensure it matches the training data format)
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

# Extract feature names
feature_names = df_features.columns.tolist()

# Apply scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_features)

# Split data
split_date = '2019-01-01'
train = features_scaled[df_daily.index < split_date]
test = features_scaled[df_daily.index >= split_date]
y_train = df_daily.loc[df_daily.index < split_date, 'SM1']
y_test = df_daily.loc[df_daily.index >= split_date, 'SM1']

# Create a feature map file for XGBoost visualization
fmap_file = 'xgb.fmap'
with open(fmap_file, 'w') as f:
    for i, feature_name in enumerate(feature_names):
        # Format: <feature_index>\t<feature_name>\t<feature_type>
        # 'q' denotes quantitative (numerical) feature
        f.write(f'{i}\t{feature_name}\tq\n')

print(f"Feature map file created: {fmap_file}")
print("Feature mapping:")
for i, name in enumerate(feature_names):
    print(f"f{i} -> {name}")

# Load and evaluate saved models
for model_file in os.listdir(model_dir):
    if model_file.endswith('.model'):
        print(f"\nLoading model: {model_file}")
        model_path = os.path.join(model_dir, model_file)
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Prepare DMatrix for predictions
        dtest = xgb.DMatrix(test, label=y_test)
        
        # Make predictions
        y_test_pred = model.predict(dtest)
        
        # Evaluate performance
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mse = mean_squared_error(y_test, y_test_pred)
        print(f"Model: {model_file}")
        print(f"Test R²: {r2_test:.4f}")
        print(f"Test RMSE: {rmse_test:.4f}")
        print(f"Test MSE: {mse:.4f}")       
        
        # Plot predictions vs actual
        plt.figure(figsize=(14, 7))
        plt.plot(y_test.index, y_test, label='Actual Test')
        plt.plot(y_test.index, y_test_pred, label='Predicted Test', linestyle='--')
        plt.legend()
        plt.title(f'Actual vs Predicted SM1 - {model_file}')
        plt.show()

        # Get the number of trees in the model
        num_trees = len(model.get_dump())
        if num_trees > 0:
            # Plot the last tree using the feature map file
            plt.figure(figsize=(15, 10))
            xgb.plot_tree(model, num_trees=num_trees-1, fmap=fmap_file)
            plt.title(f"Last Tree (Tree #{num_trees-1}) with Feature Names")
            plt.show()
            
            # Get feature importance
            importance = model.get_score(fmap=fmap_file, importance_type='weight')
            importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
            plt.figure(figsize=(12, 8))
            plt.barh(list(importance.keys()), list(importance.values()))
            plt.title('Feature Importance')
            plt.xlabel('F-score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.show()
        else:
            print("No trees found in the model")