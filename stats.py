#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/14

from numpy import ndarray
import matplotlib.pyplot as plt

from utils import *


def smooth_train(x:ndarray) -> ndarray:
  local_mean = lambda i: (x[i-1] + x[i+1]) / 2

  # fix case M001 tailing abnormal
  if 1.0 < x[-1] < 1.2:   # 1.1
    x = x[:-1]
  # fix period location error (?)
  for i in range(len(x)):
    if 8.4659 < x[i] < 11.4219:
      x[i] *= 10
  # fix case M012 abnormal
  for i in range(len(x)):
    if 77.430 < x[i] < 77.450:  # 77.440
      x[i] = local_mean(i)
  # fix case M003 abnormal
  for i in range(len(x)):
    if 1.048 < x[i] < 1.068:  # 1.058
      x[i] *= 100
    if 0.374 < x[i] < 0.394:  # 0.384
      x[i] = local_mean(i)
  # mean filter
  for i in range(1, len(x)-1):
    mean = local_mean(i)
    diff = abs(x[i] - mean)
    if diff > 1:    # MAGIC: better way?
      x[i] = mean

  return x

def smooth_test1(x:ndarray) -> ndarray:
  # chage -1 to right end (for testset)
  for i in range(1, len(x)):
    if x[i] == -1:
      x[i] = x[i - 1]
  return x


def stats_single():
  for split in ['train', 'test1']:
    dp = DATA_PATH / split
    for fp in dp.iterdir():
      if not fp.suffix == '.pkl': continue

      with open(fp, 'rb') as fh:
        df_cyc, df_act = pkl.load(fh)
      print(df_cyc)
      #print(df_act)

      y  = globals()[f'smooth_{split}'](df_cyc['y'].to_numpy())
      ts = np.log1p(df_act['ts'].to_numpy())

      plt.clf()
      plt.subplot(211) ; plt.plot(y,  label='y')  ; plt.legend() ; plt.title(f'{y[0]:.5f} -> {y[-1]:.5f}')
      plt.subplot(212) ; plt.plot(ts, label='ts') ; plt.legend()
      plt.tight_layout()
      fp = IMG_PATH / f'{split}-{fp.stem}.png'
      print(f'>> savefig {fp}')
      plt.savefig(fp, dpi=600)


def stats_compare():
  lbls, ys = [], []
  for split in ['train', 'test1']:
    dp = DATA_PATH / split
    for fp in dp.iterdir():
      if not fp.suffix == '.pkl': continue
      lbls.append(f'{split}-{fp.stem}')

      with open(fp, 'rb') as fh:
        df_cyc, df_act = pkl.load(fh)
      y = globals()[f'smooth_{split}'](df_cyc['y'].to_numpy())
      ys.append(np.asarray(y))

  #maxlen = max([len(y) for y in ys])
  #ys_padded = [np.pad(y, (0, maxlen - len(y)), mode='edge') for y in ys]

  plt.clf()
  plt.figure(figsize=(8, 8))
  for i, y in enumerate(ys):
    plt.plot(y, label=lbls[i])
  plt.legend()
  plt.suptitle('y')
  plt.tight_layout()
  fp = IMG_PATH / 'y.png'
  print(f'>> savefig {fp}')
  plt.savefig(fp, dpi=600)

  plt.clf()
  plt.figure(figsize=(6, 6))
  for i, y in enumerate(ys):
    plt.plot(y[:5], label=lbls[i])
  plt.legend()
  plt.suptitle('y-5')
  plt.tight_layout()
  fp = IMG_PATH / 'y-5.png'
  print(f'>> savefig {fp}')
  plt.savefig(fp, dpi=600)

  ys_prime = [y[1:] - y[:-1] for y in ys]

  plt.clf()
  plt.figure(figsize=(8, 8))
  for i, y in enumerate(ys_prime):
    plt.plot(y, label=lbls[i])
  plt.legend()
  plt.suptitle('y_prime')
  plt.tight_layout()
  fp = IMG_PATH / 'y_prime.png'
  print(f'>> savefig {fp}')
  plt.savefig(fp, dpi=600)


if __name__ == '__main__':
  stats_single()
  stats_compare()
