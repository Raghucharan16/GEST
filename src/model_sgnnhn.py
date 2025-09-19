import torch, torch.nn as nn

class Highway(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.proj = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
    def forward(self, x, y):
        t = torch.sigmoid(self.gate(x))
        h = self.proj(y)
        return t * h + (1 - t) * x

class SGNNHN(nn.Module):
    """
    Star Graph Neural Network with Highway Networks (CPU-friendly):
      - star node linked to all item nodes
      - in/out propagation (GGNN style)
      - highway gating
      - SR-GNN-style attention readout using last item
    Shapes:
      nodes: [B, N] (last node is the star, id=0)
      alias: [B, L-1]  (indices into nodes for each position in the input sequence)
      alias_len: [B]   (true length of alias before padding)
      A_in/A_out: [B, N, N]
    """
    def __init__(self, num_items, emb_dim=128, step=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.step = step
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=0)  # 0 = PAD + star
        self.lin_in  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.lin_out = nn.Linear(emb_dim, emb_dim, bias=False)
        self.gru = nn.GRUCell(2*emb_dim, emb_dim)
        self.highway = Highway(emb_dim)
        self.attn_w1 = nn.Linear(emb_dim, emb_dim)
        self.attn_w2 = nn.Linear(emb_dim, emb_dim)
        self.attn_q  = nn.Linear(emb_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def gnn_layer(self, h, A_in, A_out):
        """
        h:     [B, N, D]
        A_in:  [B, N, N]
        A_out: [B, N, N]
        returns updated h: [B, N, D]
        """
        # propagate
        h_in  = torch.bmm(A_in,  self.lin_in(h))
        h_out = torch.bmm(A_out, self.lin_out(h))
        a = torch.cat([h_in, h_out], dim=-1)          # [B, N, 2D]

        # GRUCell expects [B_flat, 2D] and [B_flat, D]
        B, N, D2 = a.shape
        D = D2 // 2
        a_flat = a.reshape(B * N, 2*D)
        h_flat = h.reshape(B * N, D)
        h_new = self.gru(a_flat, h_flat)              # [B*N, D]
        h_new = h_new.reshape(B, N, D)

        # highway gate (broadcast-safe)
        return self.highway(h, h_new)

    def forward(self, nodes, alias, alias_len, A_in, A_out):
        # embed nodes (star node id=0 -> padding embedding)
        h = self.item_emb(nodes)                       # [B, N, D]
        for _ in range(self.step):
            h = self.gnn_layer(h, A_in, A_out)
            h = self.dropout(h)

        # get the embedding of the last clicked item in the (unpadded) sequence
        B = alias.size(0)
        last_pos = alias_len - 1                       # [B]
        last_alias = alias[torch.arange(B), last_pos]  # [B]
        h_last = h[torch.arange(B), last_alias]        # [B, D]

        # attention over node embeddings
        q = self.attn_q(torch.tanh(self.attn_w1(h_last).unsqueeze(1) + self.attn_w2(h)))  # [B,N,1]
        attn = torch.softmax(q.squeeze(-1), dim=-1)                                       # [B,N]
        g = (attn.unsqueeze(-1) * h).sum(dim=1)                                           # [B,D]
        s = self.dropout(g + h_last)                                                      # fuse global + last
        return s @ self.item_emb.weight.T
    
    def predict(self, nodes, alias, alias_len, A_in, A_out):
        return self.forward(nodes, alias, alias_len, A_in, A_out)
