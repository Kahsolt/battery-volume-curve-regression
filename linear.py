#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/17

# model the rough contour: c_i = f(charge_i, uncharge_i) + g(i) ≈ lowfreq(v_i)

from traceback import print_exc

import joblib
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.signal import medfilt

from data import *

LOG_DP = LOG_PATH / 'linear'
N_WIN = 5

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
    # log1p(当前步数) + 前5步充放电时长 -> 当前容量
    acts = ts[i-N_WIN:i]
    tgt = y[i]
    X.append(np.concatenate([[np.log1p(i)], acts.flatten()], axis=0))
    Y.append(tgt)
  X = np.stack(X, axis=0)
  Y = np.stack(Y, axis=0)
  feat_names = [
    'i'
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
      print_exc()
      print(f'>> failed: train-{id}')
    feat_dfs.append(feat_df)
  feat_df = pd.concat(feat_dfs, axis=0).reset_index(drop=True)
  feat_df = feat_df.sample(frac=1.0, random_state=SEED)   # shuffle
  return feat_df, feat_names, target_name


def run_train(n_fold:int=5, seed:int=114514) -> LinearModel:
  df_train, feats, target = make_trainset()
  print('df_train.shape:', df_train.shape)
  print('feat_names:', feats)
  print('target_name:', target)

  pred_oof = np.zeros([len(df_train)])    # out-of-fold preds
  kfold = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
  for fold, (train_idx, valid_idx) in enumerate(kfold.split(df_train)):
    print(f'[Fold {fold}]')
    X_train, y_train = df_train.loc[train_idx, feats], df_train.loc[train_idx, target]
    X_valid, y_valid = df_train.loc[valid_idx, feats], df_train.loc[valid_idx, target]
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_oof[valid_idx] = model.predict(X_valid)

  target = df_train[target]
  mse_err = mean_absolute_error(target, pred_oof)
  print('>> valid mse_err:', mse_err)
  r2_sc = r2_score(target, pred_oof)
  print('>> valid r2_sc:', r2_sc)
  mape_err = mean_absolute_percentage_error(target, pred_oof)
  print('>> valid mape_err:', mape_err)

  return model


def run_infer(model:LinearModel, split:str, id:str):
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
  for k in tqdm(range(len(feat_df))):
    cur = feat_df.iloc[k]
    cur = DataFrame(cur).T
    pred = model.predict(cur).item()
    y_pred.append(pred)
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
  model_fp = LOG_DP / 'model.pkl'

  if True or not Path(model_fp).exists():
    model = run_train()
    joblib.dump(model, model_fp)

  model = joblib.load(model_fp)
  for split, ids in DATA_FILES.items():
    for id in ids:
      try:
        run_infer(model, split, id)
      except KeyboardInterrupt:
        exit(-1)
      except:
        print_exc()
        print(f'>> failed: {split}-{id}')
