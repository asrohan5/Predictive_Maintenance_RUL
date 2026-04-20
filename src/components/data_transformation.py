import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.exception import CustomException


_LOW_VARIANCE_SENSORS = {
    'sensor_1', 'sensor_5', 'sensor_6',
    'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'}

_INFORMATIVE_SENSORS = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]


@dataclass
class DataTransformationConfig:
    processed_dir = os.path.join('artifacts', 'processed')
    train_raw_csv = 'train_raw.csv'
    test_raw_csv = 'test_raw.csv'
    rul_csv = 'rul.csv'
    train_out = 'train_transformed.csv'
    test_out = 'test_transformed.csv'
    scaler_path = 'scaler.pkl'
    meta_path = 'feature_meta.json'

    max_rul = 125
    roll_windows = (10, 30, 50)   
    lags = (1, 3, 5)
    ewma_alpha = 0.3         
    variance_threshold = 1e-4


class DataTransformation:

    def __init__(self, config: DataTransformationConfig = None):
        self.cfg = config or DataTransformationConfig()
        self.scaler = StandardScaler()
        Path(self.cfg.processed_dir).mkdir(parents=True, exist_ok=True)

    #Sensor Filtering
    def _drop_low_variance_sensors(self, train_df, test_df):
        all_sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
        var_series = train_df[all_sensor_cols].var()
        data_driven_low = set(var_series[var_series < self.cfg.variance_threshold].index)

        drop_set = (_LOW_VARIANCE_SENSORS | data_driven_low) & set(all_sensor_cols)
        if drop_set:
            logging.info(f'Dropping {len(drop_set)} low-variance sensors: {sorted(drop_set)}')
            train_df = train_df.drop(columns=sorted(drop_set))
            test_df = test_df.drop(columns=sorted(drop_set))

        remaining_sensors = [c for c in train_df.columns if c.startswith('sensor_')]
        return train_df, test_df, remaining_sensors



    @staticmethod
    def _compute_train_rul(df, max_rul):
        df = df.copy()
        max_cycle = df.groupby('unit_number')['time_in_cycles'].transform('max')
        df['RUL'] = (max_cycle - df['time_in_cycles']).clip(upper=max_rul)
        
        return df

  
    @staticmethod
    def _merge_test_rul(test_df, rul_df):

        test_df = test_df.copy()
        last_cycle = (
            test_df.groupby('unit_number')['time_in_cycles']
            .max()
            .reset_index()
            .rename(columns={'time_in_cycles': 'last_cycle'})
        )
      
        rul_end = rul_df[['unit_number', 'RUL']].copy()
        test_df = test_df.merge(last_cycle, on='unit_number')
        test_df = test_df.merge(rul_end,    on='unit_number')
        test_df['RUL'] = (
            test_df['RUL'] + (test_df['last_cycle'] - test_df['time_in_cycles'])
        )
        test_df = test_df.drop(columns=['last_cycle'])
        
      return test_df


    #Feature Building
    @staticmethod
    def _cycle_norm(df):
      
        max_cycle = df.groupby('unit_number')['time_in_cycles'].transform('max')
        return (df['time_in_cycles'] / max_cycle).rename('cycle_norm')

    @staticmethod
    def _within_unit_zscore(df, cols):
        g = df.groupby('unit_number')
        unit_mean = g[cols].transform('mean')
        unit_std = g[cols].transform('std').replace(0.0, 1.0)
        z = (df[cols] - unit_mean) / unit_std
        z.columns = [f'{c}_z' for c in cols]
      
        return z

    @staticmethod
    def _ewma_features(df, cols, alpha):

        out = []
        for col in cols:
            ewma = (
                df.groupby('unit_number')[col]
                .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
            )
            
            out.append(ewma.rename(f'{col}_ewma'))
          
        return pd.concat(out, axis=1)

    @staticmethod
    def _diff_features(df, cols):

        out = []
        for col in cols:
            diff = (
                df.groupby('unit_number')[col]
                .transform(lambda x: x.diff().fillna(0.0))
            )
            
            out.append(diff.rename(f'{col}_diff1'))
        
        return pd.concat(out, axis=1)

    @staticmethod
    def _rolling_slope_numpy(values, window):
        
        n = len(values)
        result = np.zeros(n, dtype=float)
        for i in range(n):
            lo = max(0, i - window + 1)
            xi = np.arange(i - lo + 1, dtype=float) 
            yi = values[lo:i + 1]
            xm = xi.mean()
            ym = yi.mean()
            denom = ((xi - xm) ** 2).sum()
            if denom > 1e-9:
                result[i] = ((xi - xm) * (yi - ym)).sum() / denom
        
        return result

    @staticmethod
    def _rolling_stats(df, cols, window):
      
        g = df.groupby('unit_number')[cols]
        roll = g.rolling(window=window, min_periods=1)
        mean_df = roll.mean().reset_index(level=0, drop=True)
        std_df = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        min_df = roll.min().reset_index(level=0, drop=True)
        max_df = roll.max().reset_index(level=0, drop=True)

        mean_df.columns = [f'{c}_rmean{window}' for c in cols]
        std_df.columns = [f'{c}_rstd{window}' for c in cols]
        min_df.columns = [f'{c}_rmin{window}' for c in cols]
        max_df.columns = [f'{c}_rmax{window}' for c in cols]


      
        slope_arrays = {col: np.empty(len(df), dtype=float) for col in cols}

        for uid, grp in df.groupby('unit_number'):
            idx = grp.index 
            for col in cols:
                slope_arrays[col][df.index.get_indexer(idx)] = (DataTransformation._rolling_slope_numpy(grp[col].values, window))

        slope_df = pd.DataFrame(
            {f'{col}_slope{window}': slope_arrays[col] for col in cols},
            index=df.index,)

        return pd.concat([mean_df, std_df, min_df, max_df, slope_df], axis=1)

    @staticmethod
    def _lag_features(df, cols, lags):

        out = []
        for lag in lags:
            lagged = df.groupby('unit_number')[cols].shift(lag)
            first_vals = df.groupby('unit_number')[cols].transform('first')
            lagged = lagged.fillna(first_vals)
            lagged.columns = [f'{c}_lag{lag}' for c in cols]
            out.append(lagged)
          
        return pd.concat(out, axis=1) if out else pd.DataFrame(index=df.index)


  

    def _build_features(self, df, sensor_cols):
      
        blocks = [
            self._cycle_norm(df).to_frame(),
            self._within_unit_zscore(df, sensor_cols),
            self._ewma_features(df, sensor_cols, self.cfg.ewma_alpha),   
            self._diff_features(df, sensor_cols)]
      
        for w in self.cfg.roll_windows:
            blocks.append(self._rolling_stats(df, sensor_cols, window=w))
          
        blocks.append(self._lag_features(df, sensor_cols, self.cfg.lags))

        return pd.concat(blocks, axis=1)


  
    def initiate_data_transformation(self, train_path, test_path, rul_path):
        try:
            logging.info('Data Transformation: Started')
            p = self.cfg.processed_dir
            train_path = train_path or os.path.join(p, self.cfg.train_raw_csv)
            test_path = test_path or os.path.join(p, self.cfg.test_raw_csv)
            rul_path = rul_path or os.path.join(p, self.cfg.rul_csv)

            train_raw = pd.read_csv(train_path)
            test_raw = pd.read_csv(test_path)
            rul_df = pd.read_csv(rul_path)

            for df_ in [train_raw, test_raw]:
                df_.sort_values(['unit_number', 'time_in_cycles'], inplace=True)
                df_.reset_index(drop=True, inplace=True)

            logging.info(f'Computing train RUL with cap={self.cfg.max_rul}')
            train_raw = self._compute_train_rul(train_raw, self.cfg.max_rul)

            logging.info('Attaching Ground Truth RUL to test set')
            test_raw = self._merge_test_rul(test_raw, rul_df)

            train_raw, test_raw, sensor_cols = self._drop_low_variance_sensors(train_raw, test_raw)
            logging.info(f'Using {len(sensor_cols)} sensors: {sensor_cols}')

            logging.info('Building Engineered Features (train)')
            train_feats = self._build_features(train_raw, sensor_cols)


            logging.info('Building Engineered Features (test)')
            test_feats = self._build_features(test_raw, sensor_cols)

            keep_base = ['unit_number', 'time_in_cycles'] + sensor_cols + ['RUL']
            train_df = pd.concat([train_raw[keep_base].reset_index(drop=True), train_feats.reset_index(drop=True)], axis=1)
            test_df = pd.concat([test_raw[keep_base].reset_index(drop=True), test_feats.reset_index(drop=True)], axis=1)


            id_cols = {'unit_number', 'RUL'}
            feature_cols = [c for c in train_df.columns if c not in id_cols]

            logging.info(f'Fitting StandardScaler on {len(feature_cols)} features (train only)')
            train_df[feature_cols] = self.scaler.fit_transform(train_df[feature_cols].astype(float))
          
            test_df[feature_cols] = self.scaler.transform(test_df[feature_cols].astype(float))

            train_out  = os.path.join(p, self.cfg.train_out)
            test_out   = os.path.join(p, self.cfg.test_out)
            scaler_out = os.path.join(p, self.cfg.scaler_path)
            meta_out   = os.path.join(p, self.cfg.meta_path)

            train_df.to_csv(train_out, index=False)
            test_df.to_csv(test_out,  index=False)

            joblib.dump(
                {
                    'scaler': self.scaler,
                    'feature_cols': feature_cols,
                    'sensor_cols': sensor_cols,
                },scaler_out)

            meta = {
                'sensor_cols': sensor_cols,
                'feature_cols': feature_cols,
                'max_rul': self.cfg.max_rul,
                'roll_windows': list(self.cfg.roll_windows),
                'lags': list(self.cfg.lags),
                'ewma_alpha': self.cfg.ewma_alpha,
                'n_train_rows': len(train_df),
                'n_test_rows': len(test_df),
                'n_features': len(feature_cols),
            }
          
            with open(meta_out, 'w') as f:
                json.dump(meta, f, indent=2)

            logging.info(f'Final dataset — Train: {train_df.shape}, Test: {test_df.shape}, 'f'Features: {len(feature_cols)}')
          
            logging.info(f'Train: {train_out}')
            logging.info(f'Test: {test_out}')
            logging.info(f'Scaler: {scaler_out}')
          
            logging.info('Data Transformation: Complete')

            return train_out, test_out

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == '__main__':
#     config = DataTransformationConfig()
#     obj    = DataTransformation(config)
#     train_path = os.path.join(config.processed_dir, config.train_raw_csv)
#     test_path  = os.path.join(config.processed_dir, config.test_raw_csv)
#     rul_path   = os.path.join(config.processed_dir, config.rul_csv)
#     obj.initiate_data_transformation(train_path, test_path, rul_path)
