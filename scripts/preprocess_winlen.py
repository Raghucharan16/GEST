# scripts/preprocess_winlen.py
import os, argparse, pandas as pd
def window_and_split(in_csv, out_dir, win_len):
    df = pd.read_csv(in_csv).sort_values(["session_id","timestamp"])
    # split long sessions into fixed windows of size win_len; left-pad shorter ones at model time
    out = []
    for sid, g in df.groupby("session_id"):
        items = g["item_id"].tolist()
        cats  = g["category_id"].tolist()
        pr    = g["price"].tolist()
        ts    = g["timestamp"].tolist()
        # slice from the end in win_len chunks (paper example)
        while len(items) > 0:
            chunk = list(zip(items[-win_len:], cats[-win_len:], pr[-win_len:], ts[-win_len:]))
            out.append((sid, len(items), chunk))
            items = items[:-win_len]
            cats  = cats[:-win_len]
            pr    = pr[:-win_len]
            ts    = ts[:-win_len]
    # Rebuild flattened frame with a synthetic row_id for splitting
    rows=[]
    for sid, L, chunk in out:
        for i,(it,ct,price,t) in enumerate(chunk):
            rows.append([sid, t, it, ct, price])
    dff = pd.DataFrame(rows, columns=["session_id","timestamp","item_id","category_id","price"])
    # time-sorted sessions for 80/10/10 split by session end-time
    end = dff.groupby("session_id")["timestamp"].max().sort_values()
    n=len(end); train_ids=end.index[:int(0.8*n)]; val_ids=end.index[int(0.8*n):int(0.9*n)]; test_ids=end.index[int(0.9*n):]
    os.makedirs(out_dir, exist_ok=True)
    dff.to_parquet(os.path.join(out_dir,"interactions.parquet"))
    pd.DataFrame({"session_id":train_ids}).to_parquet(os.path.join(out_dir,"train_sessions.parquet"))
    pd.DataFrame({"session_id":val_ids}).to_parquet(os.path.join(out_dir,"valid_sessions.parquet"))
    pd.DataFrame({"session_id":test_ids}).to_parquet(os.path.join(out_dir,"test_sessions.parquet"))
    print("Wrote", out_dir)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--win_len", type=int, required=True)
    a=ap.parse_args(); window_and_split(a.in_csv, a.out_dir, a.win_len)
