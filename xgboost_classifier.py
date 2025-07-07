import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class WeeklyPriceIncreasePredictor:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()

    def prepare_data(self, historical_data):
        # Convert to DataFrame
        df = pd.DataFrame(historical_data['history']['day'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Resample to Fridays (weekly)
        weekly = df['close'].resample('W-FRI').last().dropna()

        # Create features: returns, moving averages
        data = pd.DataFrame()
        data['close'] = weekly
        data['return_1w'] = weekly.pct_change(1)
        data['return_2w'] = weekly.pct_change(2)
        data['ma_3w'] = weekly.rolling(window=3).mean()
        data['ma_5w'] = weekly.rolling(window=5).mean()
        data = data.dropna()

        # Create labels: 1 if next week's close > this week's close else 0
        data['label'] = (data['close'].shift(-1) > data['close']).astype(int)
        data = data.dropna()

        X = data[['return_1w', 'return_2w', 'ma_3w', 'ma_5w']]
        y = data['label']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, data

    def train(self, historical_data):
        X, y, self.data = self.prepare_data(historical_data)
        self.model.fit(X, y)

    def predict_next_5_weeks(self):
        preds = []
        confs = []

        # Use last known data for recursive predictions
        last_features = self.data[['return_1w', 'return_2w', 'ma_3w', 'ma_5w']].iloc[-1].values.reshape(1, -1)
        last_close = self.data['close'].iloc[-1]

        for _ in range(5):
            last_features_scaled = self.scaler.transform(last_features)
            prob = self.model.predict_proba(last_features_scaled)[0][1]  # Probability of increase
            preds.append(1 if prob > 0.5 else 0)
            confs.append(int(prob * 100))  # scale to 1-100

            # Simulate next week's features assuming price change equal to predicted label * avg return
            avg_return = np.mean(self.data['return_1w'])
            next_return_1w = avg_return if preds[-1] == 1 else -avg_return
            next_return_2w = last_features[0][0]
            next_ma_3w = (last_features[0][2] * 2 + last_close * (1 + next_return_1w)) / 3
            next_ma_5w = (last_features[0][3] * 4 + last_close * (1 + next_return_1w)) / 5

            last_features = np.array([[next_return_1w, next_return_2w, next_ma_3w, next_ma_5w]])

        return preds, confs
