import argparse, json
from pathlib import Path
import torch
from dataset import SessionGraphDataset
from loader import make_loader
from model_sgnnhn import SGNNHN
from metrics import mrr_at_k, ndcg_at_k, hit_at_k

def run_split(model, loader, device, K_list=(10,20)):
    model.eval()
    ranks = []
    with torch.no_grad():
        for nodes, alias, alias_len, Ain, Aout, tgt in loader:
            nodes, alias, alias_len, Ain, Aout, tgt = \
                nodes.to(device), alias.to(device), alias_len.to(device), Ain.to(device), Aout.to(device), tgt.to(device)
            scores = model.predict(nodes, alias, alias_len, Ain, Aout)
            order = torch.argsort(scores, dim=1, descending=True)
            pos = (order == tgt.view(-1,1)).nonzero(as_tuple=False)
            r = torch.full((scores.size(0),), 1e9, dtype=torch.float32, device=device)
            r[pos[:,0]] = pos[:,1].float() + 1.0
            ranks.extend(r.cpu().tolist())
    res = {f"MRR@{k}": mrr_at_k(ranks, k) for k in K_list}
    res.update({f"NDCG@{k}": ndcg_at_k(ranks, k) for k in K_list})
    res.update({f"Hit@{k}": hit_at_k(ranks, k) for k in K_list})
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['diginetica','cosmetics','yoochoose'])
    ap.add_argument('--data_root', default='data_proc')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--k', type=int, nargs='+', default=[10,20])
    args = ap.parse_args()

    device = torch.device('cpu')
    ddir = Path(args.data_root)/args.dataset
    meta = json.load(open(ddir/'meta.json'))
    num_items = int(meta['num_items'])

    ck = torch.load(args.ckpt, map_location='cpu')
    model = SGNNHN(num_items=num_items, emb_dim=ck.get('meta',{}).get('emb_dim',128))
    model.load_state_dict(ck['state_dict'])
    model.eval()

    te_ds = SessionGraphDataset(ddir/'sessions.pkl', ddir/'splits.json', 'test', max_win=9999)
    te_loader = make_loader(te_ds, batch_size=256, shuffle=False)

    res = run_split(model, te_loader, device, args.k)
    print("Test:", res)

if __name__ == '__main__':
    main()
