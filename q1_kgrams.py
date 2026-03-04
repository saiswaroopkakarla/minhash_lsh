import os
from itertools import combinations

def read_doc(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip()

def char_kgrams(text, k):
    grams = set()
    for i in range(len(text) - k + 1):
        grams.add(text[i:i+k])
    return grams

def word_kgrams(text, k):
    words = text.split()
    grams = set()
    for i in range(len(words) - k + 1):
        grams.add(' '.join(words[i:i+k]))
    return grams

def jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union != 0 else 0.0

# Load all 4 documents
docs = {}
for i in range(1, 5):
    path = os.path.join('minhash', f'D{i}.txt')
    docs[f'D{i}'] = read_doc(path)

# Build k-gram sets
char2 = {name: char_kgrams(text, 2) for name, text in docs.items()}
char3 = {name: char_kgrams(text, 3) for name, text in docs.items()}
word2 = {name: word_kgrams(text, 2) for name, text in docs.items()}

gram_types = {
    'Character 2-grams': char2,
    'Character 3-grams': char3,
    'Word 2-grams':      word2,
}

pairs = list(combinations(['D1','D2','D3','D4'], 2))

print("=" * 55)
print("QUESTION 1 — K-GRAM SIZES")
print("=" * 55)
for gtype, gset in gram_types.items():
    print(f"\n{gtype}:")
    for name, s in gset.items():
        print(f"  {name}: {len(s)} unique grams")

print("\n" + "=" * 55)
print("QUESTION 1B — JACCARD SIMILARITY BETWEEN ALL PAIRS")
print("=" * 55)
for gtype, gset in gram_types.items():
    print(f"\n{gtype}:")
    for a, b in pairs:
        j = jaccard(gset[a], gset[b])
        print(f"  J({a}, {b}) = {j:.4f}")
