# AML Assignment — Sentiment Analysis and Face Alignment

Applied Machine Learning (G6061), University of Sussex, Spring 2026.

This repository contains the code, notebooks and trained-model artefacts for the end-of-module assignment. The compiled report (`report/report.pdf`) is the primary deliverable.

## Repository layout

```
.
├── report/                  LaTeX source, references and compiled PDF
├── task1_nlp/               Task 1 — sentiment classification with spam handling
│   ├── 01_explore.ipynb       data exploration
│   ├── 02_baseline.ipynb      TF-IDF + LR baseline on raw labels
│   ├── 03_spam_filter.ipynb   cosine-to-centroid + word-list spam filters
│   ├── 04_models.ipynb        MNB, LR-TFIDF, LR-W2V, word-list comparison
│   ├── 05_evaluate.ipynb      confusion matrices, ablation, failure cases
│   ├── 06_nltk_external.ipynb external evaluation on NLTK movie_reviews
│   ├── 07_submission.ipynb    test-set predictions
│   ├── src/preprocess.py
│   ├── models/                trained joblib bundles + BEST_MODEL.txt
│   ├── results/               CSVs, JSONs, LaTeX tables
│   ├── figures/
│   └── submission/task1_predictions.csv
└── task2_cv/                Task 2 — facial landmark alignment
    ├── 01_explore.ipynb
    ├── 02_baseline.ipynb      mean-face floor
    ├── 03_hog_linreg.ipynb    HOG + ridge regression
    ├── 04_cascaded.ipynb      cascaded pose-indexed HOG + ridge
    ├── 05_cnn.ipynb           4-block CNN, with and without augmentation
    ├── 06_robustness.ipynb    noise / rotation / brightness / scale sweeps
    ├── 07_submission.ipynb    test-set predictions, scaled to original resolution
    ├── src/{cnn,landmarks,preprocess_img}.py
    ├── models/                trained weights + BEST_MODEL_T2.txt
    ├── results/               CED data, JSON metrics, robustness CSV
    ├── figures/
    └── submission/task2_predictions.csv
```

## Headline results

**Task 1 — best model: Multinomial Naive Bayes on TF-IDF unigrams**

| Model | Val acc | Val macro-F1 |
|---|---:|---:|
| **MNB on TF-IDF** | **0.790** | **0.807** |
| LR on TF-IDF | 0.777 | 0.797 |
| LR on averaged Word2Vec | 0.694 | 0.725 |
| Word-list (top-K=500) | 0.671 | 0.675 |

**Task 2 — best model: Cascaded regression with pose-indexed HOG features**

| Approach | Mean NME ↓ | Median NME ↓ |
|---|---:|---:|
| Mean-face baseline | 0.1246 | 0.1117 |
| Raw-pixel ridge | 0.0745 | 0.0671 |
| HOG + ridge | 0.0564 | 0.0503 |
| **Cascaded HOG + ridge** | **0.0510** | **0.0442** |
| CNN (no augmentation) | 0.0758 | 0.0688 |
| CNN (with augmentation) | 0.0832 | 0.0781 |

NME = inter-ocular-normalised mean Euclidean landmark error (W09_L18).

## Reproducing the results

1. Place the original datasets in `data/` (Task 1 CSVs) and `task2_cv/data/` (face-alignment `.npz` files). These are **not** included in the repository.
2. Open any `05_*.ipynb` to regenerate metrics and figures from the bundled trained models — nothing needs to be retrained.
3. Run `07_submission.ipynb` in either task to regenerate the prediction CSV.

Developed with Python 3.11, scikit-learn 1.4, gensim 4, scikit-image 0.22, PyTorch 2.x.

## What's not in this repo

- Original assignment datasets (`data/sentiment_analysis_*.csv`, `task2_cv/data/face_alignment_*.npz`) — not redistributed.
- Submission build folder and zip (`submission_AML/`, `submission_AML.zip`) — redundant with the source files here.
