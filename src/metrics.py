import numpy as np

def hit_at_k(ranks, K):
    r = np.asarray(ranks)
    return float(np.mean((r <= K).astype(np.float32)))

def mrr_at_k(ranks, K):
    r = np.asarray(ranks, dtype=np.float32)
    return float(np.mean(np.where(r <= K, 1.0 / r, 0.0)))

def ndcg_at_k(ranks, K):
    r = np.asarray(ranks, dtype=np.float32)
    return float(np.mean(np.where(r <= K, 1.0 / np.log2(r + 2.0), 0.0)))
