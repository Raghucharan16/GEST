import argparse, json
from pathlib import Path
import pandas as pd, numpy as np
import pickle

def load_df(dataset, raw_dir):
    path = Path(raw_dir) / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {'session_id','timestamp','item_id'}
    if not need.issubset(df.columns):
        raise ValueError(f"Expected columns {need}, got {df.columns.tolist()}")
    if df['timestamp'].dtype == object:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            df['timestamp'] = df.groupby('session_id').cumcount()
    return df

def prepare_sequences(df, cfg):
    df = df.sort_values(['session_id','timestamp'])
    if cfg.get('dedup_consecutive', True):
        prev = df.groupby('session_id')['item_id'].shift(1)
        df = df[(prev.isna()) | (df['item_id'] != prev)]
    vc = df['item_id'].value_counts()
    keep_items = set(vc[vc >= cfg.get('min_item_freq',5)].index)
    df = df[df['item_id'].isin(keep_items)]
    lens = df.groupby('session_id').size()
    keep_sess = set(lens[lens >= cfg.get('min_session_len',2)].index)
    df = df[df['session_id'].isin(keep_sess)]
    items = sorted(df['item_id'].unique().tolist())
    item2id = {it: i+1 for i, it in enumerate(items)}  # 0 = PAD
    df['iid'] = df['item_id'].map(item2id)
    win = int(cfg.get('win_len', 50))
    seqs = []
    for _, g in df.groupby('session_id'):
        s = g['iid'].tolist()
        if len(s) < 2: continue
        seqs.append(s[-win:])
    return seqs, item2id

def chrono_split(seqs, ratios=(0.8,0.1,0.1)):
    n = len(seqs)
    tr = int(n * ratios[0]); va = int(n * (ratios[0] + ratios[1]))
    return list(range(0,tr)), list(range(tr,va)), list(range(va,n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['diginetica','cosmetics','yoochoose'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--raw_dir', default='data_raw')
    ap.add_argument('--out_root', default='data_proc')
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    df = load_df(args.dataset, args.raw_dir)
    seqs, item2id = prepare_sequences(df, cfg)

    out_dir = Path(args.out_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'sessions.pkl','wb') as f:
        pickle.dump(seqs, f)
    with open(out_dir/'mappings.json','w') as f:
        json.dump({'item2id': {str(k): int(v) for k,v in item2id.items()}}, f)
    tr, va, te = chrono_split(seqs)
    with open(out_dir/'splits.json','w') as f:
        json.dump({'train': tr, 'valid': va, 'test': te}, f)
    with open(out_dir/'meta.json','w') as f:
        json.dump({'num_items': int(len(item2id)+1)}, f)
    print(f"[OK] {args.dataset}: {len(seqs)} sessions -> {out_dir}")

if __name__ == '__main__':
    main()
