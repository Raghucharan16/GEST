# scripts/convert_tmall.py
import os, argparse, pandas as pd
def sessionize(df, gap_min=30):
    df = df.sort_values(["user_id","timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # New session if gap > 30m or user changes
    new_sess = (df["user_id"]!=df["user_id"].shift(1)) | \
               ((df["timestamp"]-df["timestamp"].shift(1)).dt.total_seconds().fillna(0) > gap_min*60)
    df["session_id"] = new_sess.cumsum()
    return df

def main(raw_dir, out_csv):
    # Adjust column names to your dump: user_id, item_id, category_id, seller_id, brand_id, timestamp, action_type
    # Concatenate all months (Jan–Jun 2016)
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(os.path.join(raw_dir,f)) for f in files], ignore_index=True)
    df = df.rename(columns={"time":"timestamp"})
    for c in ["user_id","item_id","category_id","timestamp"]:
        assert c in df.columns, f"missing {c}"

    df = sessionize(df, 30)
    # Build price if present; else 0
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
