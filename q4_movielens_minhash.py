import numpy as np
import pandas as pd
from itertools import combinations

# ── load data ──────────────────────────────────────────
df = pd.read_csv('ml-100k/u.data', sep='\t',
                 names=['user', 'movie', 'rating', 'timestamp'])

# build a dict: user -> set of movies they rated
user_movies = df.groupby('user')['movie'].apply(set).to_dict()
users = sorted(user_movies.keys())
print(f"Total users: {len(users)}, Total movies: {df['movie'].nunique()}")

# ── exact jaccard ──────────────────────────────────────
def exact_jaccard(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

print("\nComputing exact Jaccard for all user pairs (this may take a minute)...")
similar_pairs_exact = []
for u1, u2 in combinations(users, 2):
    j = exact_jaccard(user_movies[u1], user_movies[u2])
    if j >= 0.5:
        similar_pairs_exact.append((u1, u2, round(j, 4)))

print(f"Pairs with exact Jaccard >= 0.5: {len(similar_pairs_exact)}")
for u1, u2, j in similar_pairs_exact[:10]:
    print(f"  User {u1} & User {u2}: J = {j}")
if len(similar_pairs_exact) > 10:
    print(f"  ... and {len(similar_pairs_exact) - 10} more")

# ── minhash helpers ────────────────────────────────────
all_movies = sorted(df['movie'].unique())
movie_index = {m: i for i, m in enumerate(all_movies)}
n_movies = len(all_movies)

def make_hash_params(t, m=199999):
    np.random.seed(None)  # fresh seed each run for 5-run averaging
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

def approx_jaccard_sig(sig1, sig2):
    return np.mean(sig1 == sig2)

# ── run experiments ────────────────────────────────────
t_values = [50, 100, 200]
exact_set = set((u1, u2) for u1, u2, _ in similar_pairs_exact)
N_RUNS = 5

print("\n" + "=" * 65)
print("QUESTION 4 — MinHash on MovieLens (averaged over 5 runs)")
print("=" * 65)

for t in t_values:
    fp_list, fn_list, found_list = [], [], []

    for run in range(N_RUNS):
        a, b_p, m = make_hash_params(t)

        # compute signatures for all users
        sigs = {u: minhash_signature(user_movies[u], a, b_p, m) for u in users}

        # find approximate similar pairs
        approx_set = set()
        for u1, u2 in combinations(users, 2):
            j = approx_jaccard_sig(sigs[u1], sigs[u2])
            if j >= 0.5:
                approx_set.add((u1, u2))

        fp = len(approx_set - exact_set)
        fn = len(exact_set - approx_set)
        fp_list.append(fp)
        fn_list.append(fn)
        found_list.append(len(approx_set))

    print(f"\n  t={t} hash functions:")
    print(f"    Avg pairs found      : {np.mean(found_list):.1f}")
    print(f"    Avg false positives  : {np.mean(fp_list):.2f}")
    print(f"    Avg false negatives  : {np.mean(fn_list):.2f}")

print(f"\n  Exact pairs (ground truth): {len(exact_set)}")
