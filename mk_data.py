#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/14

import pickle as pkl
from zipfile import ZipFile
from datetime import datetime, time
from traceback import print_exc

import pandas as pd
from pandas import DataFrame
from openpyxl import open as XlsxFile, Workbook
import matplotlib.pyplot as plt

from utils import *

DATA_FILES = {
  'train.zip': 'train',
  'test1.zip': 'test',
  'test2.zip': 'test',
}


def datetime_to_ts(dt_str:str) -> int:
  ''' 2020/09/28 04:25:50.530 '''
  dt = datetime.fromisoformat(dt_str)
  return int(datetime.timestamp(dt) * 100)

def time_to_ts(tm_str:str) -> int:
  ''' 29:12:25:00.080 or 00:02:39.600 '''
  day = 0
  if tm_str.count(':') == 3:
    idx = tm_str.find(':')
    day = int(tm_str[:idx])
    tm_str = tm_str[idx+1:]
  tm = time.fromisoformat(tm_str)
  return (day * 86400 + tm.hour * 3600 + tm.minute * 60 + tm.second) * 100 + tm.microsecond // 10000


def process_xlsx(wb:Workbook) -> List[DataFrame]:
  df_cycle = DataFrame()
  df_action = DataFrame()
  df_timing = DataFrame()
  for sheet in wb.worksheets:
    name = sheet.title
    df = DataFrame(sheet.values, dtype=str)   # avoid cast
    column_names = list(df.iloc[df.index[0]])
    df.drop(index=df.index[0], axis=0, inplace=True)  # ignore fist row for labels
    df.columns = column_names

    # de-dup (do not know why)
    df.drop_duplicates(inplace=True)

    if 'fix case M008':
      if df.loc[1, '循环号'] is None:
        df.loc[1, '循环号'] = 0
    df = df[~(df['循环号'] == '0')]     # ignore cid=0
    df['循环号'] = df['循环号'].astype(np.uint32)
    if '工步状态' in list(df.columns):
      df = df[~(df['工步状态'] == '静置')]  # ignore all 静置

    if name == '循环数据':              # 一系列工步组成一个循环，记录该过程的放电容量
      assert not len(df_cycle)

      # dedup (use hidden truth data :)
      cid_list = df['循环号'].to_numpy().tolist()
      if len(cid_list) != len(set(cid_list)):
        print('len(list(cid)):', len(cid_list))
        print('len(set(cid)):', len(set(cid_list)))
        rows = []
        for cid, grp in df.groupby(by='循环号'):
          if len(grp) == 1:
            rows.append(grp)
          elif len(grp) == 2:
            grp_sel = grp[~grp['放电容量/Ah'].isna()]
            assert len(grp_sel) == 1
            rows.append(grp_sel)
          else:
            print('len(grp):', len(grp))
            breakpoint()
        df = pd.concat(rows, axis=0)

      df_cycle['cid'] = df['循环号'].astype(np.uint32)
      df_cycle['y']   = df['放电容量/Ah'].astype(np.float32).fillna(-1).round(3)  # fillna for testset
    elif name == '工步数据':            # 一系列时刻组成一个工步，对应单一的操作状态
      assert not len(df_action)
      df_action['cid'] = df['循环号'].astype(np.uint32)
      df_action['aid'] = df['工步号'].astype(np.uint32)
      df_action['act'] = df['工步状态'].map(ACTION_STATUS).astype(np.uint8)
    elif name.startswith('详细数据'):   # 30s 均匀打点时刻记录，存储工步执行的时长信息
      df_timing_sub = DataFrame()
      df_timing_sub['cid'] = df['循环号'].astype(np.uint32)
      df_timing_sub['aid'] = df['工步号'].astype(np.uint32)
      df_timing_sub['act'] = df['工步状态'].map(ACTION_STATUS).astype(np.uint8)   # 应与"工步数据"表一致，仅作 sanity check
      #df_timing_sub['ts_test'] = df['测试时间'].map(datetime_to_ts).astype(np.uint64)    # this time sometimes be wired back :(
      df_timing_sub['ts_proc'] = df['流程时间'].map(time_to_ts).astype(np.uint32)
      df_timing_sub['ts_act']  = df['工步时间'].map(time_to_ts).astype(np.uint32)
      df_timing = pd.concat([df_timing, df_timing_sub], axis=0)
    else:
      print(f'>> unknown sheet: {name}')
      breakpoint()

  # de-dup (do not know why)
  df_cycle .drop_duplicates(inplace=True)
  df_action.drop_duplicates(inplace=True)
  df_timing.drop_duplicates(inplace=True)

  # treate cid-aid as a data unit
  for (cid, aid), grp in df_timing.groupby(['cid', 'aid']):
    # sanity check
    act_set = set(grp['act'])
    try: assert len(act_set) == 1
    except: breakpoint()
    row = df_action[(df_action['cid']==cid) & (df_action['aid']==aid)]
    if not len(row):  # action table data in-complete, just remove it
      idx = df_action[(df_action['cid']==cid)].index  
      df_action.drop(index=idx, inplace=True)
      continue
    try:
      assert len(row) == 1
      assert row['act'].item() == act_set.pop()
    except: breakpoint()
    # aggregate
    locator = (df_action['cid']==cid) & (df_action['aid']==aid)
    ts_proc = grp['ts_proc'].max() - grp['ts_proc'].min()
    ts_act  = grp['ts_act' ].max() - grp['ts_act' ].min()
    try: assert abs(ts_proc - ts_act) < 100   # tolerant ~1s error
    except: breakpoint()
    df_action.loc[locator, 'ts'] = ts_act
  df_action = df_action[df_action['ts'] != 0]   # ignore dummy 静置
  df_action['ts'] = df_action['ts'].fillna(-1).astype(np.int64)  # fillna for testset

  return df_cycle, df_action


def process_cache():
  for fn, kind in DATA_FILES.items():
    fp_in = DATA_PATH / fn
    if not fp_in.exists(): continue
    dp_out = fp_in.with_suffix('')
    dp_out.mkdir(exist_ok=True)

    print(f'>> processing split {fn}...')
    zf = ZipFile(fp_in)
    zinfos = zf.infolist()    # NOTE: 保持有序！
    zinfos.sort(key=lambda zinfo: Path(zinfo.filename).stem)
    for zinfo in tqdm(zinfos):
      if zinfo.is_dir(): continue
      fn = Path(zinfo.filename).stem
      print(f'>> processing sample {fn}...')
      fp_out = (dp_out / fn).with_suffix('.pkl')
      if fp_out.exists(): continue

      try:
        with zf.open(zinfo) as fh:
          wb = XlsxFile(fh, read_only=True, keep_vba=False, data_only=True)
          df_cycle, df_action = process_xlsx(wb)
          print(df_cycle)
          print(df_action)
          data = df_cycle, df_action
        with open(fp_out, 'wb') as fh:
          pkl.dump(data, fh)
      except KeyboardInterrupt:
        exit(-1)
      except:
        print_exc()


if __name__ == '__main__':
  print('[process_cache]')
  process_cache()
