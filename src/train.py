import argparse, json
from pathlib import Path
import torch, torch.nn as nn
from tqdm import tqdm
from dataset import SessionGraphDataset
from loader import make_loader
from model_sgnnhn import SGNNHN
from metrics import mrr_at_k, ndcg_at_k, hit_at_k

def evaluate(model, loader, device, K_list=(10,20)):
    model.eval()
    ranks = []
    with torch.no_grad():
        for nodes, alias, alias_len, Ain, Aout, tgt in loader:
            nodes, alias, alias_len, Ain, Aout, tgt = \
                nodes.to(device), alias.to(device), alias_len.to(device), Ain.to(device), Aout.to(device), tgt.to(device)
            scores = model.predict(nodes, alias, alias_len, Ain, Aout)
            scores[:, 0] = -1e9
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
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--emb_dim', type=int, default=128)
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    args = ap.parse_args()

    device = torch.device('cpu')
    ddir = Path(args.data_root) / args.dataset
    meta = json.load(open(ddir/'meta.json'))
    num_items = int(meta['num_items'])

    tr_ds = SessionGraphDataset(ddir/'sessions.pkl', ddir/'splits.json', 'train', max_win=9999)
    va_ds = SessionGraphDataset(ddir/'sessions.pkl', ddir/'splits.json', 'valid', max_win=9999)
    tr_loader = make_loader(tr_ds, args.batch_size, True)
    va_loader = make_loader(va_ds, args.batch_size, False)

    model = SGNNHN(num_items=num_items, emb_dim=args.emb_dim, step=args.step, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    best = None
    ckdir = Path('checkpoints')/args.dataset
    ckdir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckdir/'best.pt'

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")
        for nodes, alias, alias_len, Ain, Aout, tgt in pbar:
            nodes, alias, alias_len, Ain, Aout, tgt = \
                nodes.to(device), alias.to(device), alias_len.to(device), Ain.to(device), Aout.to(device), tgt.to(device)
            scores = model(nodes, alias, alias_len, Ain, Aout)
            scores[:, 0] = -1e9
            loss = crit(scores, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))
        val = evaluate(model, va_loader, device)
        print("Valid:", val)
        if (best is None) or (val["MRR@10"] > best["MRR@10"]):
            best = val
            torch.save({'state_dict': model.state_dict(), 'meta': {'num_items': num_items, 'emb_dim': args.emb_dim}},
                       best_ckpt)
            print(f"[✓] Saved {best_ckpt}")

    print("Best Valid:", best)
    print("Best checkpoint:", best_ckpt)

if __name__ == '__main__':
    main()
