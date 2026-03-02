import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from src.features import FEATURES, add_time_features, validate_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(message)s')
log = logging.getLogger(__name__)

class PowerForecastModel:
    def __init__(self):
        self.base_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
        }
        self.meta = Ridge()

    def fit(self, train_df):
        train_df = add_time_features(train_df)
        X = train_df[FEATURES].values
        y = train_df['actual_mw'].values

        kf = KFold(n_splits=5, shuffle=False)
        oof_preds = np.zeros((len(X), len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            log.info(f'Training {name}...')
            for tr_idx, val_idx in kf.split(X):
                model.fit(X[tr_idx], y[tr_idx])
                oof_preds[val_idx, i] = model.predict(X[val_idx])
            log.info(f'  OOF MAE: {mean_absolute_error(y, oof_preds[:, i]):.3f} MW')

        self.meta.fit(oof_preds, y)
        log.info(f'Stack MAE: {mean_absolute_error(y, self.meta.predict(oof_preds)):.3f} MW')
        return self

    def predict(self, df):
        validate_input(df)
        df = add_time_features(df)
        X  = df[FEATURES].values
        base_preds = np.column_stack([m.predict(X) for m in self.base_models.values()])
        return self.meta.predict(base_preds).tolist()

    def save(self, path='models/model.pkl'):
        joblib.dump({'base_models': self.base_models, 'meta': self.meta}, path)
        log.info(f'Model saved to {path}')

    @staticmethod
    def load(path='models/model.pkl'):
        data     = joblib.load(path)
        instance = PowerForecastModel.__new__(PowerForecastModel)
        instance.base_models = data['base_models']
        instance.meta        = data['meta']
        return instance


if __name__ == '__main__':
    train = pd.read_csv('../data/forecast_train.csv', parse_dates=['timestamp'])
    model = PowerForecastModel()
    model.fit(train)
    model.save()
