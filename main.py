import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

SEED = 322
np.random.seed(SEED)


def mean_iou(y_true_low, y_true_high, y_pred_low, y_pred_high, eps=1e-6):
    y_true_low = np.array(y_true_low)
    y_true_high = np.array(y_true_high)
    y_pred_low = np.array(y_pred_low)
    y_pred_high = np.array(y_pred_high)

    y_true_low_adj = y_true_low - eps / 2
    y_true_high_adj = y_true_high + eps / 2
    y_pred_low_adj = y_pred_low - eps / 2
    y_pred_high_adj = y_pred_high + eps / 2

    inter = np.maximum(0, np.minimum(y_true_high_adj, y_pred_high_adj) - np.maximum(y_true_low_adj, y_pred_low_adj))
    union = (y_true_high_adj - y_true_low_adj) + (y_pred_high_adj - y_pred_low_adj) - inter
    union = np.maximum(union, eps)
    return np.mean(inter / union)


def add_cyclic_features(df):
    df['dt'] = pd.to_datetime(df['dt'])
    for col, period in [('dow', 7), ('month', 12)]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df


class DynamicPricingModel:
    def __init__(self):
        self.models_p05 = []
        self.models_p95 = []
        self.feature_names = []
        self.product_last_values = {}
        self.global_p05_mean = 0.0
        self.global_p95_mean = 0.0
        self.global_width_mean = 0.0

    def _save_product_history(self, train_df):
        self.global_p05_mean = train_df['price_p05'].mean()
        self.global_p95_mean = train_df['price_p95'].mean()
        self.global_width_mean = (train_df['price_p95'] - train_df['price_p05']).mean()

        for pid, group in train_df.sort_values('dt').groupby('product_id'):
            p05_vals = group['price_p05'].tolist()
            p95_vals = group['price_p95'].tolist()
            self.product_last_values[pid] = (p05_vals, p95_vals)

    def fit(self, train_df):
        df = train_df.copy()
        df = add_cyclic_features(df)
        self._save_product_history(df)

        # Generate lag-based rolling features using .shift(1)
        df = df.sort_values(['product_id', 'dt']).reset_index(drop=True)
        grp = df.groupby('product_id')
        df['p05_lag1'] = grp['price_p05'].shift(1)
        df['p95_lag1'] = grp['price_p95'].shift(1)
        df['width_lag1'] = df['p95_lag1'] - df['p05_lag1']

        for w in [7, 14]:
            df[f'p05_rolling_{w}'] = grp['price_p05'].shift(1).rolling(w, min_periods=1).mean()
            df[f'p95_rolling_{w}'] = grp['price_p95'].shift(1).rolling(w, min_periods=1).mean()
            df[f'width_rolling_{w}'] = grp['price_p95'].shift(1).rolling(w, min_periods=1).mean() - \
                                       grp['price_p05'].shift(1).rolling(w, min_periods=1).mean()

        # Isolation Forest anomaly score (Lecture 12: Reconstruction-based via proximity)
        iso = IsolationForest(contamination=0.05, random_state=SEED)
        anomaly_feats = df[['p05_lag1', 'p95_lag1', 'n_stores', 'activity_flag', 'width_lag1']].fillna(0)
        df['is_anomaly'] = iso.fit_predict(anomaly_feats)
        df['is_anomaly'] = (df['is_anomaly'] == -1).astype(int)

        # Feature set
        exclude = {'dt', 'price_p05', 'price_p95', 'row_id', 'product_id', 'width_lag1'}
        self.feature_names = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
        X = df[self.feature_names].fillna(0)
        y05 = df['price_p05'].values
        y95 = df['price_p95'].values

        # Train with GroupKFold by product_id
        gkf = GroupKFold(n_splits=5)
        groups = df['product_id'].values

        for fold, (tr, val) in enumerate(gkf.split(X, groups=groups)):
            X_tr, X_val = X.iloc[tr], X.iloc[val]
            y05_tr, y05_val = y05[tr], y05[val]
            y95_tr, y95_val = y95[tr], y95[val]

            # Quantile models (Lecture 12-style direct interval modeling)
            model05 = CatBoostRegressor(
                iterations=1000, learning_rate=0.03, depth=6,
                loss_function='Quantile:alpha=0.05',
                random_seed=SEED + fold,
                verbose=0, early_stopping_rounds=150
            )
            model05.fit(X_tr, y05_tr)
            self.models_p05.append(model05)

            model95 = CatBoostRegressor(
                iterations=1000, learning_rate=0.03, depth=6,
                loss_function='Quantile:alpha=0.95',
                random_seed=SEED + fold + 100,
                verbose=0, early_stopping_rounds=150
            )
            model95.fit(X_tr, y95_tr)
            self.models_p95.append(model95)

        return self

    def predict(self, test_df):
        df = test_df.copy()
        df = add_cyclic_features(df)

        # Build lag and rolling features using saved history
        p05_lag1, p95_lag1 = [], []
        p05_r7, p05_r14 = [], []
        p95_r7, p95_r14 = [], []
        width_r7, width_r14 = [], []

        for _, row in df.iterrows():
            pid = row['product_id']
            if pid in self.product_last_values:
                hist_p05, hist_p95 = self.product_last_values[pid]
                last_p05 = hist_p05[-1]
                last_p95 = hist_p95[-1]
            else:
                last_p05 = self.global_p05_mean
                last_p95 = self.global_p95_mean

            p05_lag1.append(last_p05)
            p95_lag1.append(last_p95)

            # Rolling means
            p05_r7.append(np.mean(hist_p05[-7:]) if pid in self.product_last_values else self.global_p05_mean)
            p05_r14.append(np.mean(hist_p05[-14:]) if pid in self.product_last_values else self.global_p05_mean)
            p95_r7.append(np.mean(hist_p95[-7:]) if pid in self.product_last_values else self.global_p95_mean)
            p95_r14.append(np.mean(hist_p95[-14:]) if pid in self.product_last_values else self.global_p95_mean)
            width_r7.append(
                np.mean(np.array(hist_p95[-7:]) - np.array(hist_p05[-7:]))
                if pid in self.product_last_values else self.global_width_mean
            )
            width_r14.append(
                np.mean(np.array(hist_p95[-14:]) - np.array(hist_p05[-14:]))
                if pid in self.product_last_values else self.global_width_mean
            )

        df['p05_lag1'] = p05_lag1
        df['p95_lag1'] = p95_lag1
        df['p05_rolling_7'] = p05_r7
        df['p05_rolling_14'] = p05_r14
        df['p95_rolling_7'] = p95_r7
        df['p95_rolling_14'] = p95_r14
        df['width_rolling_7'] = width_r7
        df['width_rolling_14'] = width_r14
        df['is_anomaly'] = 0  # placeholder; no anomaly in test

        X = df[self.feature_names].fillna(0)

        preds_p05 = np.mean([m.predict(X) for m in self.models_p05], axis=0)
        preds_p95 = np.mean([m.predict(X) for m in self.models_p95], axis=0)

        # Enforce valid interval
        preds_p95 = np.maximum(preds_p95, preds_p05 + 1e-4)
        preds_p05 = np.maximum(preds_p05, 0.0)

        return preds_p05, preds_p95


def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    model = DynamicPricingModel()
    model.fit(train)

    p05, p95 = model.predict(test)

    submission = pd.DataFrame({
        'row_id': test['row_id'],
        'price_p05': p05,
        'price_p95': p95
    })
    submission.to_csv('results/submission.csv', index=False)
    print("✅ Submission saved to 'submission.csv'")


if __name__ == '__main__':
    main()