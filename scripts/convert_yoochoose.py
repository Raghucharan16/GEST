# scripts/convert_yoochoose.py
import os, argparse, pandas as pd, numpy as np
def main(raw_dir, out_csv):
    clicks = pd.read_csv(os.path.join(raw_dir,"yoochoose-clicks.dat"),
                         sep=",", header=None,
                         names=["session_id","timestamp","item_id","category_id"])
    # Some dumps are semicolon-separated; if so, use sep=";"
    buys = pd.read_csv(os.path.join(raw_dir,"yoochoose-buys.dat"),
                       sep=",", header=None,
                       names=["session_id","timestamp","item_id","price"])
    items = pd.read_csv(os.path.join(raw_dir,"yoochoose-items.dat"),
                        sep=",", header=None,
                        names=["item_id","category_id"])

    # clicks don’t have price; buys do. Merge buys into clicks for price signals.
    clicks = clicks.merge(items, on="item_id", how="left", suffixes=("","_items"))
    # Build a per-(item,category) price from buys
    buy_price = buys.groupby(["item_id"], as_index=False)["price"].mean()
    clicks = clicks.merge(buy_price, on="item_id", how="left")

    # Impute missing price by category mean (paper)
    cat_mean = clicks.groupby("category_id")["price"].mean()
    clicks["price"] = clicks.apply(
        lambda r: cat_mean.get(r["category_id"], np.nan) if pd.isna(r["price"]) else r["price"], axis=1
    )
    # Remaining NaNs → 0
    clicks["price"] = clicks["price"].fillna(0)

    # Filter: item support ≥5, session len ≥3
    item_freq = clicks["item_id"].value_counts()
    clicks = clicks[clicks["item_id"].isin(item_freq[item_freq>=5].index)]
    sess_len = clicks["session_id"].value_counts()
    clicks = clicks[clicks["session_id"].isin(sess_len[sess_len>=3].index)]

    clicks.sort_values(["session_id","timestamp"], inplace=True)
    clicks[["session_id","timestamp","item_id","category_id","price"]].to_csv(out_csv, index=False)
    print("Saved", out_csv)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    a=ap.parse_args(); main(a.raw_dir, a.out_csv)
