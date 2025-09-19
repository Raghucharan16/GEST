import pickle, json, torch
from torch.utils.data import Dataset

class SessionGraphDataset(Dataset):
    """
    Per-sample session graph:
      nodes: LongTensor [n_nodes] unique item IDs (+ star node at end, id=0)
      alias: LongTensor [L-1] mapping each input position to node index
      alias_len: LongTensor [] (true length of alias)
      A_in/A_out: FloatTensor [n_nodes, n_nodes] (row-normalized)
      target: LongTensor [] (last item id)
    """
    def __init__(self, pkl_path, splits_path, split, max_win=9999):
        with open(pkl_path,'rb') as f:
            self.seqs = pickle.load(f)
        self.splits = json.load(open(splits_path))
        self.indices = self.splits[split]
        self.max_win = max_win

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        s = self.seqs[self.indices[i]]
        if len(s) > self.max_win:
            s = s[-self.max_win:]

        tgt = s[-1]; inp = s[:-1]
        nodes_list = list(dict.fromkeys(inp))
        node2idx = {n:i for i,n in enumerate(nodes_list)}
        alias = [node2idx[x] for x in inp]
        alias_len = len(alias)

        n = len(nodes_list) + 1                      # + star
        star_idx = n - 1
        A_in  = torch.zeros(n, n, dtype=torch.float32)
        A_out = torch.zeros(n, n, dtype=torch.float32)

        # consecutive transitions
        for a, b in zip(inp[:-1], inp[1:]):
            ia, ib = node2idx[a], node2idx[b]
            if ia != ib:
                A_out[ia, ib] += 1.0
                A_in [ib, ia] += 1.0

        # item self-loops
        for j in range(n-1):
            A_out[j, j] += 1.0
            A_in [j, j] += 1.0

        # star connections (down-weighted)
        star_w = 0.3
        for j in range(n-1):
            A_out[star_idx, j] += star_w; A_in[j, star_idx] += star_w
            A_out[j, star_idx] += star_w; A_in[star_idx, j] += star_w

        def row_norm(A):
            d = A.sum(dim=1, keepdim=True) + 1e-8
            return A / d

        A_in  = row_norm(A_in)
        A_out = row_norm(A_out)

        nodes = torch.tensor(nodes_list + [0], dtype=torch.long)  # star uses id 0 (PAD emb)
        alias = torch.tensor(alias, dtype=torch.long)
        alias_len = torch.tensor(alias_len, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        return nodes, alias, alias_len, A_in, A_out, tgt

