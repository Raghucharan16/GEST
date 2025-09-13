# scripts/convert_cosmetics.py
import os, argparse, pandas as pd
def main(raw_dir, out_csv):
    # The Kaggle zip contains 2019-10..2020-02 events; load & filter Jan 2020
    # Many users unzip to a single parquet/csv; adjust as necessary:
    # Example: 2019-Oct.csv, 2019-Nov.csv, ..., 2020-Jan.csv
    # If you only have one big CSV, load once and filter where month == 2020-01
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    dfs=[]
    for f in files:
        df = pd.read_csv(os.path.join(raw_dir,f))
        # Expect columns: user_id, session_id, event_time/timestamp, product_id/item_id, category_id, brand, price
        # Harmonize names:
        for k,v in {"product_id":"item_id","event_time":"timestamp"}.items():
            if k in df.columns: df.rename(columns={k:v}, inplace=True)
        # keep Jan 2020 only (paper)
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[(ts.dt.year==2020) & (ts.dt.month==1)]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Required columns
    for c in ["session_id","timestamp","item_id","category_id"]: 
        assert c in df.columns, f"missing column {c}"
    if "price" not in df.columns: df["price"]=0.0

    # Filters per paper
    item_freq = df["item_id"].value_counts()
    df = df[df["item_id"].isin(item_freq[item_freq>=5].index)]
    sess_len = df["session_id"].value_counts()
    df = df[df["session_id"].isin(sess_len[sess_len>=3].index)]

    df.sort_values(["session_id","timestamp"], inplace=True)
    df[["session_id","timestamp","item_id","category_id","price"]].to_csv(out_csv, index=False)
    print("Saved", out_csv)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    a=ap.parse_args(); main(a.raw_dir, a.out_csv)
