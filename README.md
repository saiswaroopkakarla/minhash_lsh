# CSL7110 Assignment 2 — MinHashing and LSH

## Overview
Implementation of MinHashing and Locality Sensitive Hashing (LSH) techniques
for document similarity detection, applied on small text documents and the
MovieLens 100k dataset.

## Repository Structure
```
minhash_lsh/
├── minhash/                    # D1.txt, D2.txt, D3.txt, D4.txt
├── ml-100k/                    # MovieLens 100k dataset
├── q1_kgrams.py                # K-gram construction + Jaccard similarity
├── q2_minhash.py               # MinHash signatures on text documents
├── q3_lsh.py                   # LSH on text documents
├── q4_movielens_minhash.py     # MinHash on MovieLens dataset
├── q5_movielens_lsh.py         # LSH on MovieLens dataset
├── requirements.txt            # Python dependencies
└── README.md
```

## Requirements
- Python 3.x
- numpy
- pandas
- scipy
- tqdm

Install all dependencies:
```
pip install -r requirements.txt
```

## How to Run

### Question 1 — K-Grams
```
python3 q1_kgrams.py
```

### Question 2 — MinHash
```
python3 q2_minhash.py
```

### Question 3 — LSH
```
python3 q3_lsh.py
```

### Question 4 — MinHash on MovieLens
```
python3 q4_movielens_minhash.py
```

### Question 5 — LSH on MovieLens
```
python3 q5_movielens_lsh.py
```

## Dataset
MovieLens 100k dataset — 943 users, 1682 movies.
Download from: https://files.grouplens.org/datasets/movielens/ml-100k.zip

## Author
Kakarla Sai Swaroop
M25DE1023
IIT Jodhpur — M.Tech Data Engineering
