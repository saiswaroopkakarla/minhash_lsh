import numpy as np
import pandas as pd
from itertools import combinations

# ── load data ──────────────────────────────────────────
df = pd.read_csv('ml-100k/u.data', sep='\t',
                 names=['user', 'movie', 'rating', 'timestamp'])

user_movies = df.groupby('user')['movie'].apply(set).to_dict()
users = sorted(user_movies.keys())
all_movies = sorted(df['movie'].unique())
movie_index = {m: i for i, m in enumerate(all_movies)}

# ── helpers ────────────────────────────────────────────
def exact_jaccard(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

def make_hash_params(t, m=199999):
    a = np.random.randint(1, m, size=t)
    b = np.random.randint(0, m, size=t)
    return a, b, m

def minhash_signature(movie_set, a, b, m):
    sig = np.full(len(a), np.inf)
    for movie in movie_set:
        x = movie_index[movie]
        hashes = (a * x + b) % m
        sig = np.minimum(sig, hashes)
    return sig

def lsh_candidate_pairs(signatures, r, b, users):
    candidates = set()
    for band in range(b):
        buckets = {}
        for u in users:
            band_sig = tuple(signatures[u][band*r:(band+1)*r])
            buckets.setdefault(band_sig, []).append(u)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for pair in combinations(bucket, 2):
                    candidates.add(tuple(sorted(pair)))
    return candidates

# ── precompute exact pairs for tau=0.6 and tau=0.8 ────
print("Precomputing exact Jaccard similarities...")
exact_06 = set()
exact_08 = set()
for u1, u2 in combinations(users, 2):
    j = exact_jaccard(user_movies[u1], user_movies[u2])
    if j >= 0.6:
        exact_06.add((u1, u2))
    if j >= 0.8:
        exact_08.add((u1, u2))

print(f"Exact pairs (J >= 0.6): {len(exact_06)}")
print(f"Exact pairs (J >= 0.8): {len(exact_08)}")

# ── LSH experiments ────────────────────────────────────
# configs: (t, r, b)
configs = [
    (50,  5, 10),
    (100, 5, 20),
    (200, 5, 40),
    (200, 10, 20),
]

N_RUNS = 5

def run_lsh_experiment(t, r, b, exact_set, tau_label):
    fp_list, fn_list = [], []
    for run in range(N_RUNS):
        a, b_p, m = make_hash_params(t)
        sigs = {u: minhash_signature(user_movies[u], a, b_p, m) for u in users}
        candidates = lsh_candidate_pairs(sigs, r, b, users)
        fp = len(candidates - exact_set)
        fn = len(exact_set - candidates)
        fp_list.append(fp)
        fn_list.append(fn)
    return np.mean(fp_list), np.mean(fn_list)

print("\n" + "=" * 65)
print("QUESTION 5 — LSH on MovieLens")
print("=" * 65)

for tau, exact_set, label in [(0.6, exact_06, "tau=0.6"), (0.8, exact_08, "tau=0.8")]:
    print(f"\n  Target similarity: {label}  (ground truth pairs: {len(exact_set)})")
    print(f"  {'Config':<18} {'Avg FP':>8} {'Avg FN':>8}")
    print(f"  {'-'*36}")
    for t, r, b in configs:
        avg_fp, avg_fn = run_lsh_experiment(t, r, b, exact_set, label)
        print(f"  t={t}, r={r}, b={b:<6}   {avg_fp:>8.2f} {avg_fn:>8.2f}")
