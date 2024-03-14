#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/14

import json
import seaborn as sns

try:
  from numba import jit
except:
  def jit(fn):
    def wrapper(*args, **kwargs):
      return fn(*args, **kwargs)
    return wrapper

from utils import *


@jit
def pearsonr(y_hat:ndarray, y:ndarray) -> float:
  y_shift = y - y.mean()
  y_hat_shift = y_hat - y_hat.mean()
  return np.sum(y_shift * y_hat_shift) / np.sqrt(np.sum(np.power(y_shift, 2)) * np.sum(np.power(y_hat_shift, 2)))

@jit
def get_max_corr(x:ndarray, y:ndarray) -> ndarray:
  xlen, ylen = len(x), len(y)
  if xlen == ylen:
    return pearsonr(x, y)

  if xlen > ylen:   # assure len(x) < len(y)
    x, y = y, x
  xlen, ylen = len(x), len(y)

  max_cor = -1
  for cp in range(ylen - xlen):
    y_slice = y[cp:cp+xlen]
    cor = pearsonr(x, y_slice)
    if cor > max_cor:
      max_cor = cor
  return max_cor


if __name__ == '__main__':
  names: list[str] = []
  seqs: List[ndarray] = []
  for split, ids in DATA_FILES.items():
    for id in ids:
      names.append(f'{split}-{id}')
      df_cyc, df_act = load_data(split, id)
      seqs.append(df_act['ts'].to_numpy())

  n_seqs = len(seqs)
  cor = np.zeros([n_seqs, n_seqs], dtype=np.float32)
  for i, seq1 in enumerate(seqs):
    for j, seq2 in enumerate(seqs):
      if i == j:
        cor[i, j] = 1
      elif i < j:
        print(f'>> {names[i]} -> {names[j]}')
        cor[i, j] = get_max_corr(seq1, seq2)
      else:   # i > j
        cor[i, j] = cor[j, i]

  plt.clf()
  plt.figure(figsize=(6, 6))
  sns.heatmap(cor, vmin=cor.min(), vmax=1.0, cbar=True, annot=True, annot_kws={'size': 5}, cmap='coolwarm')
  plt.xticks(np.asarray(range(n_seqs)) + 0.5, names, rotation=90)
  plt.yticks(np.asarray(range(n_seqs)) + 0.5, names, rotation=0)
  plt.gca().invert_yaxis()
  plt.suptitle('correlation')
  plt.tight_layout()
  fp = IMG_PATH / 'cor.png'
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=600)

  fp = IMG_PATH / 'cor.json'
  print(f'>> save stats to {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    data = {
      'name': names,
      'cor': cor.tolist(),
    }
    json.dump(data, fh, ensure_ascii=False, indent=2)
