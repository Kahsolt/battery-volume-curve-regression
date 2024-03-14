#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/14

# linear model in a case-by-case manner

from linear import *

LOG_DP = LOG_PATH / 'linear_by_ref'


if __name__ == '__main__':
  LOG_DP.mkdir(exist_ok=True)

  train_ids_all = [
    # 这几个要寄 -_-||
    #'M001',
    #'M003',
    # 这几个看起来比较有规律 -_-||
    'M006',
    'M012',
    'M013',
    'M016',
    'M019',
  ]

  CASES = {
    # test1
    'M005': ['M006', 'M012', 'M013', 'M019'],
    'M007': ['M016'],
    'M008': ['M006', 'M012', 'M013', 'M019'],
    'M011': ['M019'],
    'M015': ['M016'],
    # test2
    'M009': ['M006', 'M012', 'M013', 'M019'],
    'M010': ['M006', 'M012', 'M013', 'M019'], 
    'M014': train_ids_all,  # TODO: 查看 test1-M015/M007 前半段
    'M017': ['M019'],   # 'M003'
    'M020': ['M019'],   # 'M003'
    'M021': ['M016'],
  }

  lbls, ys = [], []
  for split, ids in DATA_FILES.items():
    for id in ids:
      name = f'{split}-{id}'
      lbls.append(name)
      try:
        if DATA_SPLIT_TYPE[split] == 'train':
          df_cyc, _ = load_data(split, id, rpad_test=True)
          y = df_cyc['y'].to_numpy()
          ys.append(y)
        else:
          ref_ids = CASES.get(id)
          traindata = make_trainset(ref_ids)
          model: LinearRegression = run_train(traindata, n_fold=1)
          print('w:', model.coef_)
          print('b:', model.intercept_)
          y_pred = run_infer(model, split, id, LOG_DP)
          ys.append(y_pred)
      except KeyboardInterrupt:
        exit(-1)
      except:
        print_exc()
        print(f'>> failed: {name}')

  from stats import savefig
  fp = LOG_DP / 'y-test2_pred.png'
  savefig(ys, lbls, fp, 'test2_pred', figsize=(8, 8))
