import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    # nodes, alias, alias_len, A_in, A_out, tgt
    nodes, alias, alias_len, A_in, A_out, tgt = zip(*batch)

    maxN = max(x.size(0) for x in nodes)
    maxL = max(x.size(0) for x in alias)

    # pad nodes to maxN (pad value 0 = PAD/star)
    nodes_pad = []
    for nd in nodes:
        if nd.size(0) < maxN:
            pad = torch.full((maxN - nd.size(0),), 0, dtype=nd.dtype)
            nodes_pad.append(torch.cat([nd, pad], 0))
        else:
            nodes_pad.append(nd)
    nodes = torch.stack(nodes_pad, 0)

    # pad alias to maxL with 0; we’ll index with alias_len so padding is ignored
    def pad1d(xs, pad_val=0):
        out = []
        for x in xs:
            if x.size(0) < maxL:
                pad = torch.full((maxL - x.size(0),), pad_val, dtype=x.dtype)
                out.append(torch.cat([x, pad], 0))
            else:
                out.append(x)
        return torch.stack(out, 0)
    alias = pad1d(alias, 0)

    alias_len = torch.stack(alias_len, 0)

    B = len(A_in)
    Ain = torch.zeros(B, maxN, maxN, dtype=A_in[0].dtype)
    Aout= torch.zeros(B, maxN, maxN, dtype=A_out[0].dtype)
    for i in range(B):
        n = A_in[i].size(0)
        Ain[i, :n, :n] = A_in[i]
        Aout[i,:n, :n] = A_out[i]

    tgt = torch.stack(tgt, 0)
    return nodes, alias, alias_len, Ain, Aout, tgt

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0)
