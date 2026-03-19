# ⚔️ Titanic Survival Prediction — Support Vector Machines (SVM)

SVM classifier predicting Titanic passenger survival, with visual exploration of the maximum margin concept, kernel trick, and hyperparameter tuning. Built as part of my AI Engineer learning journey.

---

## What This Project Covers

- **Maximum margin visualization** — plotting support vectors, margin boundaries, and decision regions on synthetic data to build intuition before touching real data
- **Effect of C** — side-by-side comparison of how regularization strength affects boundary width and generalization
- **Kernel trick** — demonstrating why linear kernels fail on non-linear data and how RBF solves it
- **Effect of gamma** — smooth vs wiggly boundaries and how to detect overfitting visually
- **Three kernels compared** — Linear, RBF, and Polynomial on Titanic data
- **GridSearchCV** — automated tuning of C and gamma with a heatmap of all combinations
- **SVM vs Logistic Regression** — head-to-head comparison using cross-validation

---

## Results

| Model | Test Accuracy | Test F1 | ROC-AUC | CV F1 |
|-------|-------------|---------|---------|-------|
| Logistic Regression | ~82% | ~0.75 | ~0.88 | ~0.75 |
| SVM (default RBF) | ~82% | ~0.74 | ~0.84 | ~0.74 |
| SVM (tuned C=10, γ=0.01) | ~82% | ~0.75 | ~0.88 | ~0.76 |

**Best configuration:** RBF kernel, C=10, gamma=0.01

---

## Key Findings

- **RBF kernel outperforms** linear and polynomial on all metrics — consistent with theory
- **Improvement over Logistic Regression is marginal** (~1-2% F1) on Titanic — expected because the dominant feature (Sex) is binary and linear, giving logistic regression a natural advantage
- **High C + high gamma** combination shows lower CV scores — both parameters push toward overfitting and cross-validation catches it
- **SVM's real strength** is on high-dimensional data like medical imaging or text classification, where the kernel trick captures patterns that linear models cannot

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/titanic-svm
cd titanic-svm

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch notebook
jupyter notebook svm_clean.ipynb
```

The notebook loads the Titanic dataset directly from a URL — no download needed.

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

---

## Part of My AI Engineer Roadmap

- ✅ Python & Pandas
- ✅ Data Preprocessing & Feature Engineering
- ✅ Linear Regression
- ✅ Logistic Regression
- ✅ Support Vector Machines ← you are here
- ⬜ Decision Trees & Random Forest
- ⬜ Gradient Boosting
- ⬜ Deep Learning
- ⬜ NLP & Generative AI
- ⬜ AI Engineering & MLOps
