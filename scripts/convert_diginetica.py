# scripts/convert_diginetica.py
import os, argparse, pandas as pd
def main(raw_dir, out_csv):
    # adjust filename if your bundle differs
    df = pd.read_csv(os.path.join(raw_dir,"train-item-views.csv"), sep=";")
    # Common column names
    ren = {"sessionId":"session_id","eventdate":"timestamp",
           "itemId":"item_id","category":"category_id","price":"price"}
    for k,v in ren.items():
        if k in df.columns: df.rename(columns={k:v}, inplace=True)
    # If category is in another file, merge here (e.g., product_categories.csv)
    # If price absent, fill 0
    if "price" not in df.columns: df["price"]=0.0
    df = df[["session_id","timestamp","item_id","category_id","price"]].dropna()

    # Filters per paper
    item_freq = df["item_id"].value_counts()
    df = df[df["item_id"].isin(item_freq[item_freq>=5].index)]
    sess_len = df["session_id"].value_counts()
    df = df[df["session_id"].isin(sess_len[sess_len>=3].index)]

    df.sort_values(["session_id","timestamp"], inplace=True)
    df.to_csv(out_csv, index=False); print("Saved", out_csv)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    a=ap.parse_args(); main(a.raw_dir, a.out_csv)
