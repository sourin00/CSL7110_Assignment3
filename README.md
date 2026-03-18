# CSL7110 Assignment 3: Recommender Systems — README

## Dependencies
```
pip install scikit-learn scipy pandas numpy scikit-surprise shap lime tensorflow
```

## Dataset Setup
1. Download the MovieLens dataset (ml-latest or ml-25m)
2. Unzip into a folder named `ml-latest/` in the same directory as the notebook
3. The folder must contain: `movies.csv`, `ratings.csv`, `tags.csv`

## Running the Notebook
```bash
jupyter notebook CSL7110_Assignment3_Recommender_Systems.ipynb
```
Run all cells top-to-bottom (`Kernel → Restart & Run All`).

## Structure
| Part | Tasks | Method |
|------|-------|--------|
| 1 | 1–2 | Content-Based Filtering (TF-IDF, User Profiles) |
| 2 | 3–4 | Collaborative Filtering (User-CF, Item-CF) |
| 3 | 5–6 | Matrix Factorization (SVD scratch, Surprise SVD) |
| 4 | 7   | Hybrid (Meta-Learning GBM) |
| 5 | 8–9 | Neural CF + Reinforcement Learning (MAB, Q-Learning) |
| 6 | 10–13| Explainability (SHAP, k-NN, LIME) |

## Scalability Design
- **Sparse matrices** (scipy CSR) for user-item storage — no dense N×N matrices
- **Chunked CSV loading** via `chunksize=500_000` for files >1 GB
- **Truncated SVD** (scipy `svds`) instead of full SVD — O(k) vs O(min(U,I))
- **linear_kernel** for TF-IDF similarity — sparse dot product, no N×N cosine matrix
- **Sampled evaluation** — RMSE/P@K computed on subsets for speed
