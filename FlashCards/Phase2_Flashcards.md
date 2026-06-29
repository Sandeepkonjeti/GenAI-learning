# Phase 2 Flashcards — Classical Machine Learning

[← Flashcards Index](./README.md) | [Phase 2 File](../03_Phase2_Machine_Learning.md) | [Cheat Sheet](../CheatSheets/Phase2_ML.md)

---

**ID**: P2-001
**Front**: What is the bias-variance tradeoff?
**Back**: Total error = Bias² + Variance + Irreducible noise. High bias = model too simple (underfitting). High variance = model too complex (overfitting). Goal: minimize both. Techniques: regularization ↓ variance, more features ↓ bias, more data ↓ variance.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase2 #ml #bias-variance

---

**ID**: P2-002
**Front**: What is data leakage in ML and give two examples?
**Back**: When information from outside the training set contaminates the model. Examples: (1) Fitting a scaler on the full dataset then splitting — test stats leak into train. (2) Including future features for time-series prediction. (3) Target leakage: feature that encodes the label (e.g., "diagnosis" field when predicting disease). Prevention: always fit preprocessing on ONLY the training fold.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #data-leakage

---

**ID**: P2-003
**Front**: What is stratified k-fold cross-validation and when is it required?
**Back**: Splits data into k folds maintaining the same class distribution in each fold as the full dataset. Required for: imbalanced classification (if not stratified, some folds may have no positive samples). Always use `StratifiedKFold` for classification. Use `KFold` for regression. `cross_val_score` uses stratified by default for classifiers.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase2 #ml #cross-validation

---

**ID**: P2-004
**Front**: How does XGBoost differ from Random Forest?
**Back**: Random Forest: parallel ensemble (bagging) — each tree trained independently on bootstrap sample. XGBoost: sequential ensemble (boosting) — each tree corrects residuals of the previous ensemble. Result: XGBoost generally outperforms RF on tabular data with proper tuning. RF is harder to overfit. XGBoost needs careful hyperparameter tuning (learning_rate, n_estimators, max_depth).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #xgboost #random-forest #ensemble

---

**ID**: P2-005
**Front**: What is the ROC-AUC metric and what does AUC=0.5 mean?
**Back**: ROC curve: TPR vs FPR at all thresholds. AUC = area under ROC = probability that model ranks a random positive higher than a random negative. AUC=1.0: perfect. AUC=0.5: random classifier (diagonal line). AUC=0.0: perfectly wrong. Good for imbalanced classes where accuracy is misleading.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #evaluation #AUC

---

**ID**: P2-006
**Front**: When would you prefer precision over recall, and vice versa?
**Back**: Precision = TP/(TP+FP) — among predicted positives, how many are actually positive. Recall = TP/(TP+FN) — among actual positives, how many did we catch. Prefer precision when FP is costly: spam filter (don't want to block real emails). Prefer recall when FN is costly: cancer detection (don't want to miss cases). F1 = harmonic mean when both matter equally.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase2 #ml #evaluation #precision #recall

---

**ID**: P2-007
**Front**: What does `StandardScaler.fit_transform(X_train)` do and why must you NOT use it on X_test?
**Back**: `fit`: computes mean and std from X_train. `transform`: applies (x - mean)/std. `fit_transform` = fit + transform in one step. On X_test: ONLY use `transform(X_test)` (using train stats). If you `fit_transform(X_test)`, you compute test mean/std separately — data leakage. Models like SVM, KNN, logistic regression require scaling. Trees (XGBoost, RF) do NOT require it.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase2 #ml #preprocessing #scaler

---

**ID**: P2-008
**Front**: What is SHAP and why is it more reliable than built-in feature importance?
**Back**: SHAP (SHapley Additive exPlanations): each feature gets a value = its average marginal contribution across all possible orderings. Properties: consistent, local (per-sample), game-theoretically grounded. Built-in importance (gain/frequency): can be misleading for correlated features. SHAP is model-agnostic and provides per-sample explanations. `shap.TreeExplainer(model)` for trees.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #interpretability #SHAP

---

**ID**: P2-009
**Front**: What is the difference between L1 (Lasso) and L2 (Ridge) regularization?
**Back**: L1 (Lasso): penalty = λ||w||₁ → promotes sparsity (many weights = exactly 0) → feature selection. L2 (Ridge/weight decay): penalty = λ||w||₂² → shrinks all weights toward 0, rarely zeros. Elastic Net: combines both. Use L1 when you believe many features are irrelevant. Use L2 as default regularizer. In PyTorch AdamW: weight_decay implements L2.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #regularization #L1 #L2

---

**ID**: P2-010
**Front**: How does a decision tree choose where to split?
**Back**: At each node, tries all features and all split thresholds. Picks the split that maximizes information gain (= reduction in entropy) or minimizes Gini impurity. Gini = 1 - Σ p_i². After split, recursively builds subtrees. Stopping criteria: max_depth reached, min_samples_leaf, no further gain. Key: greedy — local optimum at each node, not global.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #decision-tree #splitting

---

**ID**: P2-011
**Front**: What is gradient boosting in XGBoost? What does "boosting" mean?
**Back**: Boosting: sequential ensemble — each model trained on the residuals (errors) of the previous ensemble. In gradient boosting: trees fit the NEGATIVE GRADIENT of the loss. Equivalent to gradient descent in function space. Each new tree corrects what the previous ensemble got wrong. Result: very powerful on tabular data. `learning_rate` scales contribution of each tree (smaller = slower but more accurate).
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase2 #ml #xgboost #gradient-boosting

---

**ID**: P2-012
**Front**: What are the XGBoost hyperparameters you must always tune?
**Back**: (1) `n_estimators` + `learning_rate`: more trees at lower LR is usually better. (2) `max_depth` (3-6): controls tree complexity. (3) `subsample` (0.8): row sampling per tree. (4) `colsample_bytree` (0.8): feature sampling. (5) `min_child_weight`: minimum sum of weights in leaf. Use early stopping: `eval_set` + `early_stopping_rounds=50`.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase2 #ml #xgboost #hyperparameters

---

**ID**: P2-013
**Front**: What is k-means clustering? What is the elbow method?
**Back**: k-means: (1) Initialize k centroids randomly. (2) Assign each point to nearest centroid. (3) Update centroids = mean of assigned points. (4) Repeat until convergence. Elbow method: plot inertia (sum of squared distances to centroids) vs k. Choose k at the "elbow" where adding more clusters gives diminishing returns. Also: silhouette score for cluster quality.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase2 #ml #clustering #kmeans

---

**ID**: P2-014
**Front**: What is the difference between parametric and non-parametric ML models?
**Back**: Parametric: fixed number of parameters regardless of dataset size (linear regression, neural networks, SVM with kernel). Non-parametric: number of parameters grows with data (k-NN, kernel SVM, Gaussian processes, decision trees with no depth limit). Trade-off: parametric = faster inference, non-parametric = more flexible but memory-intensive.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #parametric

---

**ID**: P2-015
**Front**: What is class imbalance and what techniques address it?
**Back**: When one class appears much more than another (e.g., fraud: 0.1% positive). Techniques: (1) Resampling: SMOTE (oversample minority), undersample majority. (2) Class weights: `class_weight='balanced'` in sklearn. (3) Threshold tuning: adjust decision threshold from 0.5. (4) Appropriate metrics: AUC, F1, precision-recall curve instead of accuracy. (5) Ensemble methods with balanced bootstrap.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase2 #ml #imbalance

---

**ID**: P2-016
**Front**: What is feature engineering and why can it matter more than model choice?
**Back**: Feature engineering: creating new features from raw data (ratios, logs, interactions, date parts, aggregations). Often matters more than algorithm: a well-engineered linear model can beat XGBoost with raw features. Domain knowledge is crucial. Examples: log(price) instead of price, user_age² for non-linear effects, lag features for time series, TF-IDF for text. Deep learning aims to automate feature engineering.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #feature-engineering

---

**ID**: P2-017
**Front**: What is a confusion matrix? Compute TP, FP, TN, FN for a given matrix.
**Back**: 2×2 table: rows = actual, columns = predicted. TN FP / FN TP (for binary). From it: Accuracy = (TP+TN)/total. Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2TP/(2TP+FP+FN). Example: [[900, 50], [30, 20]] → TP=20, FP=50, FN=30, TN=900.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase2 #ml #evaluation #confusion-matrix

---

**ID**: P2-018
**Front**: What is holdout validation vs cross-validation and when use each?
**Back**: Holdout: single train/test split (80/20). Fast but high variance — result depends on which samples end up in test. Cross-validation (k-fold): k different splits, average results. Lower variance, more reliable, but k× slower. Use CV for: model selection, hyperparameter tuning. Use holdout for: final evaluation, large datasets where CV is expensive.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase2 #ml #validation

---

**ID**: P2-019
**Front**: How does logistic regression compute probabilities? What is the log-odds?
**Back**: logistic regression: P(y=1|x) = σ(w^T x + b) = 1/(1+exp(-(w^T x + b))). Log-odds (logit): log(P(y=1)/P(y=0)) = w^T x + b. Linear model in log-odds space. Decision boundary: w^T x + b = 0 (hyperplane). Coefficients interpretable as change in log-odds per unit feature change.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase2 #ml #logistic-regression

---

**ID**: P2-020
**Front**: What is the curse of dimensionality and how does it affect k-NN?
**Back**: In high dimensions: data becomes sparse, distances become meaningless (all points equidistant). k-NN in high dimensions: hard to find meaningful nearest neighbors — every point is "equally far" from every other. Solutions: dimensionality reduction (PCA), approximate nearest neighbors (FAISS), use cosine similarity. Also affects: density estimation, sampling, visualization.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #dimensionality #knn

---

**ID**: P2-021
**Front**: What is regularization and why does it help with overfitting?
**Back**: Regularization adds a penalty term to the loss: L_reg = L + λ·Ω(w). Forces the model to use smaller, simpler weights. Intuition: model must "justify" each unit of weight complexity with sufficient reduction in training loss. Prevents memorizing noise. L2 keeps all features; L1 selects features. λ (lambda/alpha) controls strength — tune via cross-validation.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase2 #ml #regularization #overfitting

---

**ID**: P2-022
**Front**: What is early stopping in boosting models?
**Back**: Monitor validation loss during training. Stop when validation loss starts increasing (model overfitting). XGBoost: `early_stopping_rounds=50` — stop if val metric doesn't improve for 50 rounds. Saves best model from the training run. Also applies to neural networks. Prevents overfitting without needing to fix n_estimators — just set it high and let early stopping find the optimum.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase2 #ml #early-stopping #xgboost

---

**ID**: P2-023
**Front**: What is a learning curve and what does it tell you?
**Back**: Plot: training score and validation score vs training set size. High bias: both scores are low and converge early — more data won't help, need better model. High variance: train score high, val score low, large gap — more data helps, or reduce model complexity. Plateau: adding data no longer helps — model/data has reached its limit.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #learning-curves #diagnosis

---

**ID**: P2-024
**Front**: What is the difference between bagging and boosting?
**Back**: Bagging (Bootstrap AGGregatING): parallel — each model trained on bootstrap sample (with replacement). Reduces variance. Example: Random Forest. Boosting: sequential — each model trained on residuals of previous. Reduces bias. Example: XGBoost, AdaBoost. Bagging: models are independent. Boosting: models are dependent (each corrects the previous). Bagging harder to overfit; boosting generally higher accuracy.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #ensemble #bagging #boosting

---

**ID**: P2-025
**Front**: What is hyperparameter tuning? Compare grid search vs random search.
**Back**: Grid search: exhaustive over all combinations — 2 params × 5 values each = 25 × k_folds evaluations. Slow but thorough. Random search: sample combinations randomly — same budget explores more diverse hyperparameter space. Empirically: random search finds good solutions ~10× fewer evaluations for high-dim hyperparameter spaces. Bayesian optimization (Optuna): uses past results to suggest next point.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase2 #ml #hyperparameters #tuning

---

**ID**: P2-026
**Front**: What is the difference between generative and discriminative models?
**Back**: Discriminative: model P(Y|X) directly — learn decision boundary. Examples: logistic regression, SVM, neural networks. Generative: model joint P(X,Y) or P(X|Y) — can generate new samples. Examples: Naive Bayes, VAE, GANs, LLMs. Discriminative generally better for classification. Generative needed for: sampling, density estimation, handling missing features. LLMs are generative: model P(next_token | previous_tokens).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #generative #discriminative

---

**ID**: P2-027
**Front**: What is one-hot encoding and when is it problematic?
**Back**: Converts categorical feature with k categories to k binary features. Problem: (1) High cardinality (1000+ categories → 1000 binary columns). (2) Doesn't capture ordinal relationships. (3) Increases dimensionality. Alternatives: ordinal encoding (for ordered), target encoding (replace with mean of target), embedding (deep learning — learns dense representations). Trees handle high cardinality directly.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase2 #ml #encoding #categorical

---

**ID**: P2-028
**Front**: Why can't you use test set performance to select models?
**Back**: If you evaluate 100 models on the same test set and pick the best, you're overfitting to the test set. The test set should be touched ONLY ONCE at the very end. Use cross-validation for model selection and hyperparameter tuning. Use a separate validation set (or CV) for all intermediate decisions. Final test set report is your unbiased estimate of generalization.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #evaluation #test-set

---

**ID**: P2-029
**Front**: What is the Naive Bayes assumption and why is it "naive"?
**Back**: Assumes all features are conditionally independent given the class: P(x₁,...,xₙ|y) = Π P(xᵢ|y). "Naive" because this is almost never true (word frequencies in text are correlated). Yet NB often works well in practice for text classification despite violated assumptions. Why: classification only needs correct ranking of P(y|x), not accurate probabilities.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #naive-bayes #independence

---

**ID**: P2-030
**Front**: What is the difference between classification and regression? Give ML metrics for each.
**Back**: Classification: predict discrete category. Metrics: accuracy, precision, recall, F1, AUC-ROC, log-loss. Regression: predict continuous value. Metrics: MAE (mean absolute error), RMSE (root mean squared error), R² (explained variance, 1=perfect). Key: RMSE penalizes outliers more (squared); MAE more robust. R²<0 means model worse than predicting the mean.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase2 #ml #classification #regression #metrics

---

**ID**: P2-031
**Front**: What is the "no free lunch" theorem in ML?
**Back**: No single algorithm performs best on all possible problems. Every algorithm has assumptions (inductive bias) — it works well when those assumptions match the data. Example: linear regression assumes linear relationship; if true, it's optimal; if not, worse than alternatives. Practical implication: always try multiple algorithms; domain knowledge about data structure guides algorithm choice.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #no-free-lunch

---

**ID**: P2-032
**Front**: What is PCA used for in ML and what are its limitations?
**Back**: PCA: linear dimensionality reduction. Uses: (1) Visualization (reduce to 2D/3D). (2) Remove noise (keep top components). (3) Speed up downstream ML. (4) Remove multicollinearity for linear models. Limitations: (1) Linear only — can't capture complex manifolds (use UMAP/t-SNE). (2) Loses interpretability. (3) Doesn't consider class labels (unsupervised). (4) Sensitive to feature scaling — always standardize first.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #PCA #dimensionality-reduction

---

**ID**: P2-033
**Front**: What is a pipeline in sklearn and why should you always use one?
**Back**: `Pipeline([('scaler', StandardScaler()), ('model', XGBClassifier())])`. Benefits: (1) Prevents data leakage — fit called only on train fold inside CV. (2) Convenient: single `.fit()/.predict()` call. (3) Serializable: save whole pipeline with joblib. (4) Works with GridSearchCV. Never manually separate preprocessing from model — use Pipeline.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase2 #ml #sklearn #pipeline

---

**ID**: P2-034
**Front**: What is t-SNE and when should you use it vs PCA?
**Back**: t-SNE: non-linear dimensionality reduction for visualization. Good at revealing cluster structure. PCA: linear, fast, deterministic, preserves global structure. t-SNE: non-linear, slow, non-deterministic, preserves local structure (clusters). Use PCA for: preprocessing before ML, speed. Use t-SNE for: visualization of high-dim data. UMAP: faster than t-SNE, preserves more global structure — generally preferred now.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase2 #ml #tsne #visualization

---

**ID**: P2-035
**Front**: How would you handle missing values in a dataset? What should you NOT do?
**Back**: Numerical: median imputation (robust to outliers), or MICE (multivariate imputation). Categorical: mode imputation, or separate "Unknown" category. Tree models: handle NaN natively (XGBoost has native missing value handling). Do NOT: (1) Drop rows with missing values if missing is informative. (2) Impute before splitting (leakage). (3) Use mean if outliers present. Always: add a binary "was_missing" feature — missingness itself is often predictive.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase2 #ml #missing-values #preprocessing
