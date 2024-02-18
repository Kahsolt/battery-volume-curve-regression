#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/14

from data import *


def stats_single():
  for split, ids in DATA_FILES.items():
    for id in ids:
      df_cyc, df_act = load_data(split, id, rpad_test=True)
      #print(df_cyc)
      #print(df_act)

      y = df_cyc['y'].to_numpy()
      ts = df_act['ts'].to_numpy()

      plt.clf()
      plt.subplot(211) ; plt.plot(y,  label='y')  ; plt.legend() ; plt.title(f'{y[0]:.5f} -> {y[-1]:.5f}')
      plt.subplot(212) ; plt.plot(ts, label='ts') ; plt.legend()
      plt.suptitle(f'{split}-{id}')
      plt.tight_layout()
      fp = IMG_PATH / f'{split}-{id}.png'
      print(f'>> savefig {fp}')
      plt.savefig(fp, dpi=600)


def savefig(ys:List[ndarray], lbls:List[str], fp:Path, title:str='', figsize:Tuple[int, int]=(6, 8)):
  plt.clf()
  plt.figure(figsize=figsize)
  for i, y in enumerate(ys):
    plt.plot(y, label=lbls[i])
  plt.legend()
  plt.suptitle(title)
  plt.tight_layout()
  fp = IMG_PATH / f'{title}.png'
  print(f'>> savefig {fp}')
  plt.savefig(fp, dpi=600)

def stats_compare():
  lbls, ys = [], []
  for split, ids in DATA_FILES.items():
    for id in ids:
      lbls.append(f'{split}-{id}')
      df_cyc, _ = load_data(split, id, rpad_test=True)
      y = df_cyc['y'].to_numpy()
      ys.append(y)

  fp = IMG_PATH / 'y.png'
  savefig(ys, lbls, fp, 'y', figsize=(8, 8))
  
  ys_5 = [y[:5] for y in ys]
  fp = IMG_PATH / 'y-5.png'
  savefig(ys_5, lbls, fp, 'y-5', figsize=(6, 6))

  ys_prime = [y[1:] - y[:-1] for y in ys]
  fp = IMG_PATH / 'y_prime.png'
  savefig(ys_prime, lbls, fp, 'y_prime', figsize=(8, 8))


if __name__ == '__main__':
  stats_single()
  stats_compare()
