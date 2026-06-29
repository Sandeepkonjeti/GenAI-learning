# Phase 2 — Classical ML Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase File](../03_Phase2_Machine_Learning.md)

---

## Algorithm Quick Reference

| Algorithm | Type | When to Use | Key Hyperparameter |
|-----------|------|-------------|-------------------|
| Linear Regression | Regression | Linear relationship, interpretability | L1/L2 regularization λ |
| Logistic Regression | Classification | Binary/multi-class baseline | C (inverse regularization) |
| Decision Tree | Both | Interpretable rules, mixed feature types | max_depth |
| Random Forest | Both | Robust baseline, handles missing | n_estimators, max_features |
| **XGBoost** | Both | **Best tabular data performance** | learning_rate, n_estimators, max_depth |
| SVM | Both | Small high-dimensional data | C, kernel, gamma |
| K-Means | Clustering | Group unlabeled data | k (elbow method) |
| PCA | Dim reduction | Visualize, reduce noise | n_components |

---

## Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias   → Underfitting → Model too simple → fix: more features, more complex model
High Variance → Overfitting → Model memorizes training set → fix: more data, regularization, dropout
```

| Sign | Means | Fix |
|------|-------|-----|
| Train error high, test error high | High bias | More capacity |
| Train error low, test error high | High variance | Regularization, more data |
| Both errors low | 🎉 Good fit | Ship it |

---

## Evaluation Metrics

### Classification

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | TP+TN / total | Balanced classes only |
| Precision | TP / (TP+FP) | Minimize false positives (spam) |
| Recall | TP / (TP+FN) | Minimize false negatives (cancer) |
| F1 | 2·P·R/(P+R) | Balanced precision/recall |
| AUC-ROC | Area under ROC curve | Ranking, imbalanced classes |
| Log-loss | $-\frac{1}{n}\sum y\log\hat p$ | When probabilities matter |

### Regression

| Metric | Sensitive to Outliers? | Use When |
|--------|:---------------------:|----------|
| MAE | No | Robust evaluation |
| RMSE | Yes | Penalize large errors |
| R² | — | Explained variance (1 = perfect) |

---

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## XGBoost Key Concepts

| Concept | Explanation |
|---------|-------------|
| Boosting | Sequential ensemble: each tree corrects previous errors |
| Gradient boosting | Trees fit the negative gradient of loss |
| Regularization | L1 (`reg_alpha`), L2 (`reg_lambda`) on leaf weights |
| Feature importance | `plot_importance(model)` — gain, weight, cover |
| Early stopping | `eval_set`, `early_stopping_rounds=50` — prevents overfit |
| SHAP | Game-theoretic feature attribution; `shap.TreeExplainer` |

---

## Feature Engineering Quick Reference

```python
# Missing values
df['col'].fillna(df['col'].median())        # numerical
df['col'].fillna('Unknown')                  # categorical

# Encoding
pd.get_dummies(df['cat_col'])                # one-hot (low cardinality)
from sklearn.preprocessing import OrdinalEncoder  # high cardinality

# Scaling (REQUIRED for SVM, KNN, logistic reg — NOT for trees)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)      # fit on train only!
X_test = scaler.transform(X_test)            # transform only
```

---

## Common Mistakes

- **Fitting scaler on test data** — data leakage; always fit on train, transform test
- **Using accuracy on imbalanced classes** — misleading; use F1 or AUC
- **Not stratifying CV splits** for classification — use `StratifiedKFold`
- **XGBoost default learning_rate=0.3** — too high; use 0.01–0.1 with more estimators
- **Comparing models without confidence intervals** — use CV std

---

## Interview Quick-Hits

**Q: How does XGBoost differ from Random Forest?**  
A: RF builds trees in parallel (bagging); XGBoost builds sequentially (boosting) — each tree fixes the residual errors of the ensemble so far. XGBoost generally outperforms RF on tabular data.

**Q: What is data leakage and how do you prevent it?**  
A: When information from the test set (or future) influences training. Prevention: always fit preprocessing on train set only; use time-based splits for temporal data.

**Q: When would you prefer logistic regression over XGBoost?**  
A: When interpretability is required, data is sparse/high-dimensional (NLP bag-of-words), or dataset is very small (<1000 rows).

**Q: Explain the bias-variance tradeoff.**  
A: High bias = underfitting (model too simple). High variance = overfitting (model too complex). Goal: find the sweet spot. Regularization, cross-validation, and ensemble methods help navigate this.

**Q: What is SHAP and why use it over feature importance?**  
A: SHAP (SHapley Additive exPlanations) provides consistent, game-theoretically grounded attribution per sample. Built-in feature importance uses gain/frequency which can be misleading; SHAP is more trustworthy.
