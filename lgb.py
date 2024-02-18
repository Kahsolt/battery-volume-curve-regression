#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/17

from traceback import print_exc

import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.signal import medfilt
import matplotlib.pyplot as plt

from data import *

LOG_DP = LOG_PATH / 'lgb'


def make_feat_df(split:str, id:str) -> Tuple[DataFrame, List[str], str]:
  df_cyc, df_act = load_data(split, id)
  y = df_cyc['y'].to_numpy().round(N_PREC)
  df_joined = df_cyc.merge(df_act, how='inner', on='cid').sort_values(by=['cid', 'aid'])
  ts = df_joined['ts'].to_numpy().round(N_PREC)
  try:
    ts = ts.reshape(-1, N_ACTION_CYCLE)
    assert len(y) == len(ts)
  except Exception as e:
    print(f'>> length mismatch: len(ts) / len(y) = {len(ts)} / {len(y)} = {len(ts) / len(y)}')
    breakpoint()
    raise ValueError from e

  X, Y = [], []
  nlen = len(y)
  for i in range(N_WIN, nlen):
    # 当前步数 + 前5步容量 + 前5步容量差分 + 前五步充放电时长 -> 当前容量
    val = y[i-N_WIN:i]
    val_prime = val[1:] - val[:-1]
    acts = ts[i-N_WIN:i]
    tgt = y[i]
    X.append(np.concatenate([[i], val, val_prime, acts.flatten()], axis=0))
    Y.append(tgt)
  X = np.stack(X, axis=0)
  Y = np.stack(Y, axis=0)
  feat_names = [
    'i'
  ] + [
    f'v_{i+1}' for i in range(N_WIN)
  ] + [
    f'vp_{i+1}' for i in range(N_WIN-1)
  ] + [
    f'act_{i+1}_{j+1}' for i in range(N_WIN) for j in range(N_ACTION_CYCLE)
  ]
  assert len(feat_names) == X.shape[-1], f'{len(feat_names)} != {X.shape[-1]}'
  target_name = 'target'
  feat_df = DataFrame(X)
  feat_df.columns = feat_names
  feat_df[target_name] = Y
  return feat_df, feat_names, target_name


def make_trainset() -> Tuple[DataFrame, List[str], str]:
  ids = [
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

  feat_dfs = []
  for id in tqdm(ids):
    try:
      feat_df, feat_names, target_name = make_feat_df('train', id)
    except KeyboardInterrupt:
      exit(-1)
    except:
      print(f'>> failed: train-{id}')
    feat_dfs.append(feat_df)
  feat_df = pd.concat(feat_dfs, axis=0).reset_index(drop=True)
  feat_df = feat_df.sample(frac=1.0, random_state=SEED)   # shuffle
  return feat_df, feat_names, target_name


def run_train(n_fold:int=5, seed:int=114514) -> Booster:
  df_train, feats, target = make_trainset()
  print('df_train.shape:', df_train.shape)
  print('feat_names:', feats)
  print('target_name:', target)

  # https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
  params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'learning_rate': 0.05,
    'num_leaves': 31,         # 31
    'feature_fraction': 0.9,  # 0.9
    'bagging_fraction': 0.8,  # 0.8
    'bagging_freq': 5,
    'seed': seed,
    'verbose': -1,
    'n_jobs': -1,
  }
  callbacks = [
    lgb.early_stopping(100),
    lgb.log_evaluation(500),
  ]

  importance = 0
  pred_oof = np.zeros([len(df_train)])    # out-of-fold preds
  kfold = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
  for fold, (train_idx, valid_idx) in enumerate(kfold.split(df_train)):
    print(f'[Fold {fold}]')
    trainset = lgb.Dataset(df_train.loc[train_idx, feats], df_train.loc[train_idx, target])
    validset = lgb.Dataset(df_train.loc[valid_idx, feats], df_train.loc[valid_idx, target], reference=trainset)
    model = lgb.train(params, trainset, valid_sets=validset, num_boost_round=5000, callbacks=callbacks)
    pred_oof[valid_idx] = model.predict(df_train.loc[valid_idx, feats])
    importance += model.feature_importance(importance_type='gain')

  feats_importance = pd.DataFrame()
  feats_importance['name'] = feats
  feats_importance['importance'] = importance / n_fold
  feats_importance.sort_values('importance', ascending=False, inplace=True)
  print(feats_importance.iloc[:30])

  target = df_train[target]
  mse_err = mean_absolute_error(target, pred_oof)
  print('>> valid mse_err:', mse_err)
  r2_sc = r2_score(target, pred_oof)
  print('>> valid r2_sc:', r2_sc)
  mape_err = mean_absolute_percentage_error(target, pred_oof)
  print('>> valid mape_err:', mape_err)

  return model


def run_infer(model:Booster, split:str, id:str):
  df_cyc, df_act = load_data(split, id, rpad_test=False)
  y_true = df_cyc['y'].to_numpy()
  df_cyc, df_act = load_data(split, id, rpad_test=True)
  y_true_rpad = df_cyc['y'].to_numpy()

  max_cid = df_cyc['cid'].max().item()
  len_y = len(y_true)
  if max_cid != len_y:
    print('max_cid:', max_cid)
    print('len_y:', len_y)
    breakpoint()

  feat_df, feat_names, _ = make_feat_df(split, id)
  feat_df = feat_df[feat_names]   # ignore target value
  y_pred = []
  for i in range(N_WIN):
    y_pred.append(y_true[i])
  v5 = y_pred[-N_WIN:]    # init state
  vp4 = [y - x for x, y in zip(v5[:-1], v5[1:])]
  for k in tqdm(range(len(feat_df))):
    cur = feat_df.iloc[k]
    for i in range(0, N_WIN):
      cur[f'v_{i+1}'] = v5[i]
    for i in range(0, N_WIN - 1):
      cur[f'vp_{i+1}'] = vp4[i]
    pred = model.predict(cur).item()
    y_pred.append(pred)
    # shift history
    v5 = y_pred[-N_WIN:]
    vp4 = [y - x for x, y in zip(v5[:-1], v5[1:])]
  y_pred = np.asarray(y_pred).round(N_PREC)

  y_pred_fix = y_pred.copy()
  y_pred_fix = medfilt(y_pred_fix, kernel_size=9)
  for i in range(len(y_true)):
    if y_true[i] > 0:
      y_pred_fix[i] = y_true[i]
  y_pred_fix = medfilt(y_pred_fix, kernel_size=5)
  for i in range(len(y_true)):
    if y_true[i] > 0:
      y_pred_fix[i] = y_true[i]

  name = f'{split}-{id}'

  fp = LOG_DP / f'{name}.txt'
  print(f'>> save preds to {fp}')
  np.savetxt(fp, y_pred, fmt=f'%.{N_PREC}f')

  plt.clf()
  plt.plot(y_true_rpad, 'b')
  plt.plot(y_pred, 'r')
  plt.plot(y_pred_fix, 'g')
  plt.suptitle(name)
  fp = LOG_DP / f'{name}.png'
  print(f'>> savefig {fp}')
  plt.savefig(fp, dpi=600)


if __name__ == '__main__':
  LOG_DP.mkdir(exist_ok=True)
  model_fp = LOG_DP / 'model.txt'

  if True or not Path(model_fp).exists():
    model = run_train()
    model.save_model(model_fp)

  model = lgb.Booster(model_file=model_fp)
  for split, ids in DATA_FILES.items():
    for id in ids:
      try:
        run_infer(model, split, id)
      except KeyboardInterrupt:
        exit(-1)
      except:
        print_exc()
        print(f'>> failed: {split}-{id}')
