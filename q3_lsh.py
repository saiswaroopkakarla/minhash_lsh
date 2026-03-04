import os
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

def s_curve(s, r, b):
    return 1 - (1 - s**r)**b

def find_best_rb(t, tau):
    # try all valid r,b combos where r*b == t
    best = None
    best_score = -1
    for r in range(1, t+1):
        if t % r == 0:
            b = t // r
            # we want f(tau) close to 0.5 — that's the threshold point
            f_tau = s_curve(tau, r, b)
            # also want steep curve: f(0.8) high, f(0.6) low
            steepness = s_curve(0.8, r, b) - s_curve(0.6, r, b)
            score = steepness - abs(f_tau - 0.5)
            if score > best_score:
                best_score = score
                best = (r, b)
    return best

def lsh_candidate_pairs(signatures, r, b, doc_names):
    candidates = set()
    t = list(signatures.values())[0].shape[0]
    for band in range(b):
        buckets = {}
        for name in doc_names:
            band_sig = tuple(signatures[name][band*r:(band+1)*r])
            if band_sig not in buckets:
                buckets[band_sig] = []
            buckets[band_sig].append(name)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for pair in combinations(bucket, 2):
                    candidates.add(tuple(sorted(pair)))
    return candidates

def approx_jaccard(sig1, sig2):
    return np.mean(sig1 == sig2)

# ── load docs ──────────────────────────────────────────
docs = {}
for i in range(1, 5):
    docs[f'D{i}'] = read_doc(os.path.join('minhash', f'D{i}.txt'))

gram3 = {name: char_kgrams(text, 3) for name, text in docs.items()}
vocab = build_vocab(gram3)

t   = 160
tau = 0.7
doc_names = ['D1', 'D2', 'D3', 'D4']
pairs = list(combinations(doc_names, 2))

a, b_params, m = make_hash_params(t)
signatures = {
    name: minhash_signature(gram3[name], vocab, a, b_params, m)
    for name in doc_names
}

# ── Q3A: find best r and b ─────────────────────────────
print("=" * 55)
print("QUESTION 3A — Best r and b for tau=0.7, t=160")
print("=" * 55)
best_r, best_b = find_best_rb(t, tau)
print(f"  Best r = {best_r}  (rows per band)")
print(f"  Best b = {best_b}  (number of bands)")
print(f"  Check: r x b = {best_r * best_b} (should equal t={t})")
print(f"\n  S-curve values at key thresholds:")
for s in [0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f"    f({s}) = {s_curve(s, best_r, best_b):.4f}")

# ── Q3B: probability each pair is a candidate ──────────
print("\n" + "=" * 55)
print("QUESTION 3B — Probability each pair similarity > tau")
print("=" * 55)

# true jaccard for reference
true_j = {}
for d1, d2 in pairs:
    j = len(gram3[d1] & gram3[d2]) / len(gram3[d1] | gram3[d2])
    true_j[(d1, d2)] = j

# approx jaccard from minhash
approx_j = {}
for d1, d2 in pairs:
    approx_j[(d1, d2)] = approx_jaccard(signatures[d1], signatures[d2])

print(f"\n  Using r={best_r}, b={best_b}")
print(f"  {'Pair':<10} {'True J':>8} {'Approx J':>10} {'P(candidate)':>14}")
print(f"  {'-'*46}")
for d1, d2 in pairs:
    tj = true_j[(d1, d2)]
    aj = approx_j[(d1, d2)]
    prob = s_curve(aj, best_r, best_b)
    print(f"  ({d1},{d2})    {tj:>8.4f}   {aj:>8.4f}   {prob:>12.4f}")

# show candidate pairs from actual LSH
candidates = lsh_candidate_pairs(signatures, best_r, best_b, doc_names)
print(f"\n  Candidate pairs found by LSH (similarity > tau={tau}):")
if candidates:
    for pair in sorted(candidates):
        print(f"    {pair}")
else:
    print("    None found")
