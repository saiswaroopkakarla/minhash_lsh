import os
import time
import numpy as np
from itertools import combinations

def read_doc(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip()

def char_kgrams(text, k):
    grams = set()
    for i in range(len(text) - k + 1):
        grams.add(text[i:i+k])
    return grams

def build_vocab(gram_sets):
    vocab = {}
    idx = 0
    for gs in gram_sets.values():
        for g in gs:
            if g not in vocab:
                vocab[g] = idx
                idx += 1
    return vocab

def make_hash_params(t, m=99991):
    # generate t random hash functions: h(x) = (a*x + b) % m
    np.random.seed(42)
    a = np.random.randint(1, m, size=t)
    b = np.random.randint(0, m, size=t)
    return a, b, m

def minhash_signature(gram_set, vocab, a, b, m):
    t = len(a)
    sig = np.full(t, np.inf)
    for gram in gram_set:
        x = vocab[gram]
        hashes = (a * x + b) % m
        sig = np.minimum(sig, hashes)
    return sig

def approx_jaccard(sig1, sig2):
    return np.mean(sig1 == sig2)

# ── load docs ──────────────────────────────────────────
docs = {}
for i in range(1, 5):
    docs[f'D{i}'] = read_doc(os.path.join('minhash', f'D{i}.txt'))

# 3-grams only (as required by Q2)
gram3 = {name: char_kgrams(text, 3) for name, text in docs.items()}
vocab = build_vocab(gram3)

t_values = [20, 60, 150, 300, 600]
pairs = list(combinations(['D1','D2','D3','D4'], 2))

# ── Q2A: D1 vs D2 for each t ───────────────────────────
print("=" * 55)
print("QUESTION 2A — Approx Jaccard(D1, D2) for each t")
print("=" * 55)
for t in t_values:
    a, b, m = make_hash_params(t)
    sig1 = minhash_signature(gram3['D1'], vocab, a, b, m)
    sig2 = minhash_signature(gram3['D2'], vocab, a, b, m)
    j_approx = approx_jaccard(sig1, sig2)
    print(f"  t={t:>4d}  =>  Approx J(D1, D2) = {j_approx:.4f}")

# true jaccard for reference
true_j = len(gram3['D1'] & gram3['D2']) / len(gram3['D1'] | gram3['D2'])
print(f"\n  True Jaccard(D1, D2) = {true_j:.4f}  [for reference]")

# ── Q2B: best value of t ───────────────────────────────
print("\n" + "=" * 55)
print("QUESTION 2B — Finding best t (accuracy vs time)")
print("=" * 55)
extra_t = [20, 60, 150, 300, 600, 1000]
for t in extra_t:
    a, b, m = make_hash_params(t)
    start = time.time()
    sig1 = minhash_signature(gram3['D1'], vocab, a, b, m)
    sig2 = minhash_signature(gram3['D2'], vocab, a, b, m)
    elapsed = time.time() - start
    j_approx = approx_jaccard(sig1, sig2)
    error = abs(j_approx - true_j)
    print(f"  t={t:>5d}  approx={j_approx:.4f}  error={error:.4f}  time={elapsed:.4f}s")
