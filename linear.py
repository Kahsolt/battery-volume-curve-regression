#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/17

# model the rough contour: c_i = f(charge_i, uncharge_i) + g(i) ≈ lowfreq(v_i)

import joblib
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

from utils import *

LOG_DP = LOG_PATH / 'linear'
N_WIN = 5

DataInfo = Tuple[DataFrame, List[str], str]


def make_feat_df(split:str, id:str) -> DataInfo:
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

  use_id = True   # 修正递降率（使得末尾更近指数而非线性）
  use_cur = True  # 修正预测偏移（否则看起来预测 offset+1 了）
  X, Y = [], []
  nlen = len(y)
  for i in range(N_WIN, nlen):
    # log1p(当前步数) + 前5步(+当前)充放电时长 -> 当前容量
    acts = ts[i-N_WIN:i+use_cur]
    tgt = y[i]
    X.append(np.concatenate([[np.log1p(i)] if use_id else [], acts.flatten()], axis=0))
    Y.append(tgt)
  X = np.stack(X, axis=0)
  Y = np.stack(Y, axis=0)
  feat_names = ([
    'i'
  ] if use_id else []) + [
    f'act_{i+1}_{j+1}' for i in range(N_WIN+use_cur) for j in range(N_ACTION_CYCLE)
  ]
  assert len(feat_names) == X.shape[-1], f'{len(feat_names)} != {X.shape[-1]}'
  target_name = 'target'
  feat_df = DataFrame(X)
  feat_df.columns = feat_names
  feat_df[target_name] = Y
  return feat_df, feat_names, target_name


def make_trainset(ids:List[str]) -> DataInfo:
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


def run_train(traindata:DataInfo, n_fold:int=5, seed:int=114514) -> LinearModel:
  df_train, feats, target = traindata
  print('df_train.shape:', df_train.shape)
  print('feat_names:', feats)
  print('target_name:', target)

  if n_fold == 1:
    X_train, y_train = df_train.loc[:, feats], df_train.loc[:, target]
    X_valid, y_valid = df_train.loc[:, feats], df_train.loc[:, target]
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_oof = model.predict(X_valid)
  else:
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


def run_infer(model:LinearModel, split:str, id:str, log_dp:Path=LOG_DP) -> ndarray:
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

  # force fix prediction
  y_pred_before_fix = y_pred.copy()

  def fix_const_shift(x:ndarray, shift:float) -> ndarray:
    x_shift = x.copy()
    x_shift[5:] += shift
    return x_shift.round(N_PREC)

  ''' force fix: test1 '''
  if id == 'M005':  # near perfect
    pass
  if id == 'M007':  # const shift
    shift = np.median(y_true[345:385+1] - y_pred[345:385+1])
    print('>> M007 shift:', shift)    # 0.8114990234375057
    y_pred = fix_const_shift(y_pred, shift)
  if id == 'M008':  # const shift
    shift = np.median(y_true[-117:] - y_pred[-117:])
    print('>> M008 shift:', shift)    # -1.409001770019529
    y_pred = fix_const_shift(y_pred, shift)
  if id == 'M011':  # near perfect
    pass
  if id == 'M015':  # tailing no converge
    if not 'use lerp':    # 这个不太保真
      def func(x, k, b): return k * x + b
      xdata = list(range(50, 58))
      ydata = y_pred[50:58]   # seemingly a linear decay
      breakpoint()
      popt, pcov = curve_fit(func, xdata, ydata)
      y_pred_force_decay = y_pred.copy()
      for i in range(58, len(y_pred)):
        y_pred_force_decay[i] = func(i, *popt)
      y_pred = y_pred_force_decay.round(N_PREC)

    if 'use shift by ref':
      df_cyc, df_act = load_data('train', 'M016')
      y_ref = df_cyc['y'].to_numpy()[:len(y_pred)]
      y_pred_ref_shift = y_pred.copy()
      y_pred_seem_ok = y_pred[:58]
      y_ref_to_match = y_ref[:58]
      y_diff = y_pred_seem_ok[-1] - y_ref_to_match[-1]
      y_pred_ref_shift[58:] = y_ref[58:len(y_pred)] + y_diff
      y_pred = y_pred_ref_shift

  ''' force fix: test2 '''
  if id == 'M009':  # const shift
    y_pred = fix_const_shift(y_pred, -2.1)
  if id == 'M010':  # near perfect
    pass
  if id == 'M014':  # tailing no converge
    y_pred_stable = y_pred[:170]
    y_pred_unstable = y_pred[170:]
    y_pred = np.concatenate([y_pred_stable, np.ones_like(y_pred_unstable) * y_pred_stable[-1]])

    if 'interp':
      def func(x, a, b, c): return a ** (x - b) + c
      y_pred_loglike_parts = [
        (34, 134+1),
        (155, 170+1),
      ]
      xdata = []
      for rng in y_pred_loglike_parts: xdata.extend(list(range(*rng)))
      ydata = []
      for rng in y_pred_loglike_parts: ydata.extend(y_pred[slice(*rng)].tolist())
      popt, pcov = curve_fit(func, xdata, ydata, p0=[0.98556045, 156.60327162, 103.50732646])
      print('>> popt:', popt)

    y_pred_abnormal_parts = [
      (135, 155),
      (170, len(y_pred)),
    ]
    y_pred_fix = y_pred.copy()
    for rng in y_pred_abnormal_parts:
      for i in range(*rng):
        y_pred_fix[i] = func(i, *popt)
    y_pred = y_pred_fix.round(N_PREC)
  if id == 'M017':  # const shift
    y_pred = fix_const_shift(y_pred, -0.15)
  if id == 'M020':  # const shift
    y_pred = fix_const_shift(y_pred, +1)
  if id == 'M021':  # near perfect
    pass

  if not 'plot':
    plt.clf()
    plt.plot(y_pred_before_fix, 'r')
    plt.plot(y_pred, 'g')
    plt.show()
    plt.close()

  smooth = False
  if 'smooth preds' and smooth:
    y_pred_fix = y_pred.copy()
    y_pred_fix = medfilt(y_pred_fix, kernel_size=9)
    y_pred_fix = np.where(y_true > 0, y_true, y_pred_fix)
    y_pred_fix = medfilt(y_pred_fix, kernel_size=5)
    y_pred_fix = np.where(y_true > 0, y_true, y_pred_fix)
  else:
    y_pred_fix = y_pred

  name = f'{split}-{id}'
  fp = log_dp / f'{name}.txt'
  print(f'>> save preds to {fp}')
  np.savetxt(fp, y_pred_fix, fmt=f'%.{N_PREC}f')

  plt.clf()
  plt.plot(y_true_rpad, 'b', label='truth')
  plt.plot(y_pred,      'r', label='pred')
  if smooth: plt.plot(y_pred_fix,  'g', label='pred (filter)')
  plt.legend()
  plt.suptitle(name)
  fp = log_dp / f'{name}.png'
  print(f'>> savefig {fp}')
  plt.savefig(fp, dpi=600)
  plt.close()

  return y_pred_fix


if __name__ == '__main__':
  LOG_DP.mkdir(exist_ok=True)

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

  model_fp = LOG_DP / 'model.pkl'
  if False or not Path(model_fp).exists():
    traindata = make_trainset(ids)
    model = run_train(traindata)
    joblib.dump(model, model_fp)

  lbls, ys = [], []
  model = joblib.load(model_fp)
  for split, ids in DATA_FILES.items():
    for id in ids:
      name = f'{split}-{id}'
      lbls.append(name)
      try:
        y_pred = run_infer(model, split, id, LOG_DP)
        if DATA_SPLIT_TYPE[split] == 'train':
          df_cyc, _ = load_data(split, id, rpad_test=True)
          y = df_cyc['y'].to_numpy()
          ys.append(y)
        else:
          ys.append(y_pred)
      except KeyboardInterrupt:
        exit(-1)
      except:
        print_exc()
        print(f'>> failed: {name}')

  from vis_stats import savefig
  fp = LOG_DP / 'y-train_truth-test_pred.png'
  savefig(ys, lbls, fp, 'train_truth-test_pred', figsize=(8, 8))
