import numpy as np

# so here probably we are making a Cyclic Encoding
# in order to capture the cyclical nature of time features like hour and month
FEATURES = (
    [f'feature_{i}' for i in range(20)]
    + ['forecast_zephyr', 'forecast_boreas']
    + ['hour', 'dayofweek', 'month', 'dayofyear',
       'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
)

# here ther eis the function that generates and adds the extra features
def add_time_features(df):
    df = df.copy()
    df['hour']      = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month']     = df['timestamp'].dt.month
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def validate_input(df):
    required = [f'feature_{i}' for i in range(20)] + ['timestamp', 'forecast_zephyr', 'forecast_boreas']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df.isnull().any().any():
        raise ValueError('Input contains null values')
