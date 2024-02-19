#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/18

from argparse import ArgumentParser

from utils import *

EXPACTED_LENTH = {
  'M005': 1646,
  'M007': 431,
  'M008': 585,
  'M011': 592,
  'M015': 221,
}


def run(args):
  dp: Path = LOG_PATH / args.model
  split: str = args.split
  
  dfs: List[DataFrame] = []
  for id in DATA_FILES[split]:
    fp = dp / f'{split}-{id}.txt'
    data = np.loadtxt(fp)
    nlen = len(data)
    if nlen != EXPACTED_LENTH[id]:
      print(f'>> [{id}] expect: {EXPACTED_LENTH[id]}, got: {nlen}')

    df = DataFrame()
    df['电池编号'] = [id] * nlen
    df['循环号'] = list(range(1, nlen+1))
    df['放电容量/Ah'] = data.round(N_PREC)
    dfs.append(df)
  df_all = pd.concat(dfs, axis=0).reset_index(drop=True)
  fp = LOG_PATH / f'submit_{args.model}.csv'
  print(f'>> write to {fp}')
  df_all.to_csv(fp, encoding='utf-8', float_format='%.3f', index=None)


if __name__ == '__main__':
  parser = ArgumentParser()
  args = parser.add_argument('-M', '--model', default='lgb')
  args = parser.add_argument('-D', '--split', default='test1')
  args = parser.parse_args()

  run(args)
