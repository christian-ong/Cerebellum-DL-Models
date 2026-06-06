import pandas as pd
import numpy as np

# 1. Load the CSV
df = pd.read_csv("experiments/wandb/ml_dmd/best_runs.csv", low_memory=False)

# 2. Extract dt from data_path
df['dt'] = df['data_path'].apply(
    lambda x: '0.01' if 'dt_0.01' in str(x) else ('0.05' if 'dt_0.05' in str(x) else 'other')
)

# 3. Extract the comparable 1.0s physical time RMSE
def get_1s_rmse(row):
    if row['dt'] == '0.01':
        return row.get('best_val_rollout_rmse_h100', np.nan)
    elif row['dt'] == '0.05':
        return row.get('best_val_rollout_rmse_h20', np.nan)
    return np.nan

df['rmse_1s'] = df.apply(get_1s_rmse, axis=1)

# Filter out rows where we couldn't find the metric
df_valid = df.dropna(subset=['rmse_1s'])

# 4. Create a pivot table comparing the lowest RMSE for each dt, by system and expansion type
pivot = df_valid.pivot_table(
    index=['system_name', 'expansion_type'],
    columns='dt',
    values='rmse_1s',
    aggfunc='min'  # Take the absolute best run if there are multiple
)

# 5. Determine the winner for each combination (Fixed for missing data)
def determine_winner(row):
    val_01 = row.get('0.01', np.nan)
    val_05 = row.get('0.05', np.nan)
    
    if pd.isna(val_01) and pd.isna(val_05):
        return 'N/A'
    elif pd.isna(val_05):
        return 'dt=0.01 (Only)'
    elif pd.isna(val_01):
        return 'dt=0.05 (Only)'
    elif val_01 < val_05:
        return 'dt=0.01'
    elif val_05 < val_01:
        return 'dt=0.05'
    return 'Tie'

pivot['Winner'] = pivot.apply(determine_winner, axis=1)

# 6. Display the detailed results
print("=== 1.0s Rollout RMSE Comparison (Lower is Better) ===")
print(pivot.to_string(float_format=lambda x: f"{x:.4e}" if pd.notna(x) else "NaN"))

# 7. Summarize the winners by expansion type
print("\n=== Summary: Which dt wins per Expansion Type? ===")
summary = pivot.reset_index().groupby('expansion_type')['Winner'].value_counts().unstack(fill_value=0)
print(summary)