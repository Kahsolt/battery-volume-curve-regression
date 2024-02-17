#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/17

from utils import *

DATA_FILES = {
  'train': [
    'M001',
    'M003',
    'M006',
    'M012',
    'M013',
    'M016',
    'M019',
  ],
  'test1': [
    'M005',
    'M007',
    'M008',
    'M011',
    'M015',
  ],
}
DATA_SPLIT_TYPE = {
  'train': 'train',
  'test1': 'test',
  'test2': 'test',
}


def _smooth_y(df_cyc:DataFrame, id:str) -> ndarray:
  # truncate known tailing abnormal
  if id == 'M001':
    df_cyc.drop(df_cyc.tail(1).index, inplace=True)   # y = 1.1

  # fix period location error (?)
  x = df_cyc['y'].to_numpy()
  for i in range(len(x)):
    if 8.4659 < x[i] < 11.4219:
      x[i] *= 10
  df_cyc['y'] = x

  local_mean = lambda i: (x[i-1] + x[i+1]) / 2

  # fix known abnormals
  if id == 'M012':
    x = df_cyc['y'].to_numpy()
    for i in range(len(x)):
      if 77.430 < x[i] < 77.450:  # 77.440
        x[i] = local_mean(i)
    df_cyc['y'] = x
  if id == 'M003':
    x = df_cyc['y'].to_numpy()
    for i in range(len(x)):
      if 1.048 < x[i] < 1.068:  # 1.058
        x[i] *= 100
      if 0.374 < x[i] < 0.394:  # 0.384
        x[i] = local_mean(i)
    df_cyc['y'] = x

  # mean filter for potential abnormals
  x = df_cyc['y'].to_numpy()
  for i in range(1, len(x)-1):
    mean = local_mean(i)
    diff = abs(x[i] - mean)
    if diff > 1:    # MAGIC: better way?
      x[i] = mean
  df_cyc['y'] = x

  return df_cyc

def _rpad_y(df_cyc:DataFrame) -> ndarray:
  # chage -1 to right end (for testset display only)
  x = df_cyc['y'].to_numpy()
  for i in range(1, len(x)):
    if x[i] == -1:
      x[i] = x[i - 1]
  df_cyc['y'] = x
  return df_cyc

def _log1p_ts(df_act:DataFrame) -> DataFrame:
  df_act['ts'] = np.log1p(df_act['ts'].to_numpy())
  return df_act


def load_data(split:str, id:str, rpad_test:bool=False) -> Tuple[DataFrame, DataFrame]:
  fp = DATA_PATH / split / f'{id}.pkl'
  with open(fp, 'rb') as fh:
    df_cyc, df_act = pkl.load(fh)
  if DATA_SPLIT_TYPE[split] == 'train':
    df_cyc = _smooth_y(df_cyc, id)
  if rpad_test:
    df_cyc = _rpad_y(df_cyc)
  df_act = _log1p_ts(df_act)
  return df_cyc, df_act
