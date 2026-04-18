import os 
import sys
from src.logger import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.exception import CustomException


_COLUMN_NAMES = (
    ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + 
    [f'sensor_{i}' for i in range(1, 22)]
)

@dataclass
class DataIngestionConfig:
    raw_dir:str = os.path.join('artifacts', 'raw')
    processed_dir:str = os.path.join('artifacts','processed')

    train_file = 'train_FD001.txt'
    test_file = 'test_FD001.txt'
    rul_file = 'RUL_FD001.txt'


class DataIngestion:


    def __init__(self, config:DataIngestionConfig = None):
        self.cfg = config or DataIngestionConfig()
        Path(self.cfg.processed_dir).mkdir(parents=True, exist_ok=True)
    
    def _read_raw(self, filename):
        path = os.path.join(self.cfg.raw_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Raw data file not found: {path}')
        
        df = pd.read_csv(path, sep=r"\s+", header=None, engine='python')

        if df.shape[1]>=28:
            df.iloc[:, :26]
        elif df.shape[1] != 26:
            raise ValueError(
                f'Unexpected column count {df.shape[1]} in {filename}.'
                "Expected 26 ot 28 (with trailing NaN cols)"
            )
        df.columns = _COLUMN_NAMES
        return df
    

    def _read_rul(self, filename):
        path = os.path.join(self.cfg.raw_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'RUL File Not Found: {path}')
        
        rul = pd.read_csv(path, header=None, sep = r"\s+", names=['RUL'])
        rul['unit_number'] = rul.index+1
        return rul
    

    def initiate_data_ingestion(self):
        try:
            logger.info("Data Ingestion: Started")

            train_df = self._read_raw(self.cfg.train_file)
            test_df = self._read_raw(self.cfg.test_file)
            rul_df = self._read_rul(self.cfg.rul_file)


            logger.info(
                f'\nTrain Shape: {train_df.shape}\nTest Shape: {test_df.shape}\nRUL rows: {rul_df.shape}'
            )


            n_test_units = test_df['unit_number'].nunique()
            if n_test_units != len(rul_df):
                raise ValueError(
                    f'Mismatch: {n_test_units} test units v {len(rul_df)} RUL entries'
                )
            

            train_path = os.path.join(self.cfg.processed_dir, "train_raw.csv")
            test_path = os.path.join(self.cfg.processed_dir, 'test_raw.csv')
            rul_path = os.path.join(self.cfg.processed_dir, "rul.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            rul_df.to_csv(rul_path, index=False)

            logger.info(f'Train, Test, RUL files saved.')
            logger.info(f'Data Ingestion: Complete')

            return train_path, test_path, rul_path


        
        except Exception as e:
            raise CustomException(e, sys)



# if __name__=='__main__':
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()
