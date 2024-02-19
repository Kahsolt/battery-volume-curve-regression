#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/14

import pickle as pkl
from pathlib import Path
from traceback import print_exc
from typing import *

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent.relative_to(Path.cwd())
DATA_PATH = BASE_PATH / 'data'
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
SUBMIT_PATH = LOG_PATH / 'submit.csv'

DATA_FILES = {
  'test1': [
    'M005',
    'M007',
    'M008',
    'M011',
    'M015',
  ],
  'train': [
    'M001',
    'M003',
    'M006',
    'M012',
    'M013',
    'M016',
    'M019',
  ],
}
DATA_SPLIT_TYPE = {
  'train': 'train',
  'test1': 'test',
  'test2': 'test',
}
# 一个循环: 恒流转恒压充电 -> [静置] -> 恒流放电 -> [静置] (1-0-2-0), 但忽略所有 静置-0
N_ACTION_CYCLE = 2
ACTION_STATUS = {
  '静置': 0,
  '恒流转恒压充电': 1,
  '恒流放电': 2,
}
N_WIN = 5
N_PREC = 3

SEED = 114514


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

  local_mean = lambda x, i: (x[i-1] + x[i+1]) / 2

  # fix known abnormals
  if id == 'M012':
    x = df_cyc['y'].to_numpy()
    for i in range(len(x)):
      if 77.430 < x[i] < 77.450:  # 77.440
        x[i] = local_mean(x, i)
    df_cyc['y'] = x
  if id == 'M003':
    x = df_cyc['y'].to_numpy()
    for i in range(len(x)):
      if 1.048 < x[i] < 1.068:  # 1.058
        x[i] *= 100
      if 0.374 < x[i] < 0.394:  # 0.384
        x[i] = local_mean(x, i)
    df_cyc['y'] = x

  # mean filter for potential abnormals
  x = df_cyc['y'].to_numpy()
  for i in range(1, len(x)-1):
    mean = local_mean(x, i)
    diff = abs(x[i] - mean)
    if diff > 0.5:    # MAGIC: better way?
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

def _smooth_ts(df_act:DataFrame, id:str) -> DataFrame:
  # truncate known tailing abnormal
  if id == 'M001':
    df_act = df_act[df_act['cid'] != 906]

  # now we focus on 'ts'
  ts = df_act['ts'].to_numpy()
  del df_act['ts']

  # pad right-end missing data
  if id == 'M005':
    def right_end_repeat_pad(x:ndarray) -> ndarray:
      # valid data
      xlen = len(x)
      i = 0
      while x[i] > 0: i += 1
      x_valid = x[:i]
      # cyclic repeat segment
      REP_LEN = 10        # MAGIC: I just counted it out manually :)
      nlen_pad = xlen - len(x_valid)
      npad_count = nlen_pad // REP_LEN
      npad_reminder = nlen_pad % REP_LEN
      x_rep = x_valid[-REP_LEN:].tolist()
      # padded data
      x_padded = np.concatenate([x_rep * npad_count, x_rep[:npad_reminder]], axis=0)
      # use linear regression to manually decay the padded data
      if 'LinearRegression':
        def func(x, k, b): return k * x + b

        x_linear = x[500:1000]    # MAGIC: I juts observe that this range is a linear decay...
        x_linear_medfilt = medfilt(x_linear, kernel_size=7)
        xdata = list(range(500, 1000))
        ydata = x_linear_medfilt
        popt, pcov = curve_fit(func, xdata, ydata)
        k, b = popt
        for i in range(len(x_padded)):
          x_padded[i] += k * (i + 1)
      # concat up
      x_refixed = np.concatenate([x_valid, x_padded], axis=0)
      assert len(x_refixed) == xlen
      return x_refixed

    x_original = ts
    x_even = right_end_repeat_pad(x_original[0::2])
    x_odd  = right_end_repeat_pad(x_original[1::2])
    x_refixed = np.empty_like(x_original)
    x_refixed[0::2] = x_even
    x_refixed[1::2] = x_odd
    ts = x_refixed

    if not 'plot':
      plt.clf()
      plt.subplot(211) ; plt.title('x_even') ; plt.plot(x_even)
      plt.subplot(212) ; plt.title('x_odd')  ; plt.plot(x_odd)
      plt.show()

  # left-end-fix and median-filter
  if 'smooth filter':
    def left_end_extra_lerp(x:ndarray) -> ndarray:
      from scipy.interpolate import interp1d
      line = interp1d([1, 2, 3], x[1:4], kind='slinear', fill_value='extrapolate')
      x[0] = line(0).item()
      return x

    x_original = ts
    # unzip
    x_even = x_original[0::2]
    x_odd  = x_original[1::2]
    # fix starting point abnormal
    x_even = left_end_extra_lerp(x_even)
    # fix single-point abnormal
    x_smoothed = np.empty_like(x_original)
    # zip
    x_smoothed[0::2] = medfilt(x_even, kernel_size=5)
    x_smoothed[1::2] = medfilt(x_odd,  kernel_size=5)

    x_diff = np.abs(x_smoothed - x_original)
    x_final = np.where(x_diff < 7.5 * 1000, x_original, x_smoothed)   # MAGIC: what's better
    ts = x_final

  if False:
    plt.clf()
    plt.subplot(211) ; plt.title('x_diff')  ; plt.plot(x_diff)
    plt.subplot(212) ; plt.title('x_final') ; plt.plot(df_act['ts'])
    plt.show()

  # vrng renorm
  #ts = np.log1p(ts)
  ts = ts / 1000

  # write back, avoid warning
  df_act_new = DataFrame()
  df_act_new['cid'] = df_act['cid']
  df_act_new['aid'] = df_act['aid']
  df_act_new['act'] = df_act['act']
  df_act_new['ts'] = np.asarray(ts).astype(np.float32)

  return df_act_new


def load_data(split:str, id:str, rpad_test:bool=False) -> Tuple[DataFrame, DataFrame]:
  fp = DATA_PATH / split / f'{id}.pkl'
  with open(fp, 'rb') as fh:
    df_cyc, df_act = pkl.load(fh)
  if DATA_SPLIT_TYPE[split] == 'train':
    df_cyc = _smooth_y(df_cyc, id)
  if rpad_test:
    df_cyc = _rpad_y(df_cyc)
  df_act = _smooth_ts(df_act, id)
  return df_cyc, df_act


if __name__ == '__main__':
  for split, ids in DATA_FILES.items():
    for id in ids:
      df_cyc, df_act = load_data(split, id)
