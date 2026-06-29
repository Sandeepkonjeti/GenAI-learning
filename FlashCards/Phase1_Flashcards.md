# Phase 1 Flashcards — Mathematics for ML

[← Flashcards Index](./README.md) | [Phase 1 File](../02_Phase1_Mathematics_for_ML.md) | [Cheat Sheet](../CheatSheets/Phase1_Math.md)

---

**ID**: P1-001
**Front**: What is the result of multiplying a (3×4) matrix by a (4×2) matrix?
**Back**: (3×2) matrix. Inner dimensions must match (4=4); output shape = (outer dims). Rule: (m×k) @ (k×n) = (m×n). Matrix multiply is NOT commutative: AB ≠ BA in general.
**Difficulty**: Beginner
**Category**: Math
**Tags**: #phase1 #linear-algebra #matrix

---

**ID**: P1-002
**Front**: What is an eigenvector and eigenvalue? Write the definition.
**Back**: For square matrix A, eigenvector v and eigenvalue λ satisfy: Av = λv. Meaning: A only scales v (by λ), doesn't change its direction. Intuition: eigenvectors are the "natural axes" of a linear transformation.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #linear-algebra #eigenvalues

---

**ID**: P1-003
**Front**: Write the chain rule for backpropagation. If L = f(g(x)), what is dL/dx?
**Back**: Chain rule: dL/dx = (dL/df) · (df/dg) · (dg/dx). In neural networks: if z = Wx and a = σ(z) and L = loss(a), then dL/dW = dL/da · da/dz · dz/dW = dL/da · σ'(z) · x^T.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #calculus #backprop #chain-rule

---

**ID**: P1-004
**Front**: What is SVD? Write it out and explain the three matrices.
**Back**: Singular Value Decomposition: A = UΣV^T. U (m×m): left singular vectors (orthonormal). Σ (m×n): diagonal matrix of singular values (≥0, sorted descending). V^T (n×n): right singular vectors (orthonormal). Every matrix has SVD (unlike eigendecomposition). Key: low-rank approximation using top-k singular values.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #linear-algebra #SVD

---

**ID**: P1-005
**Front**: What is KL divergence? Is it symmetric? Where is it used in LLMs?
**Back**: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x)) ≥ 0. NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P). In LLMs: RLHF adds KL penalty β·D_KL(π||π_ref) to prevent fine-tuned model from drifting too far from base model. "Forward KL" (P||Q) vs "reverse KL" (Q||P) have different optimization behaviors.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase1 #probability #KL-divergence #RLHF

---

**ID**: P1-006
**Front**: What is the softmax function? What does the temperature parameter T do?
**Back**: softmax(z_i) = exp(z_i) / Σ exp(z_j). Converts logits to probabilities summing to 1. With temperature: softmax(z/T). T→0: distribution sharpens → argmax (greedy). T→∞: distribution flattens → uniform. T=1: standard. T<1: more deterministic; T>1: more random.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #probability #softmax #temperature

---

**ID**: P1-007
**Front**: What is Bayes' theorem? Write it and give an ML use case.
**Back**: P(A|B) = P(B|A)·P(A) / P(B). In ML: Naive Bayes classifier uses it to compute P(class|features) = P(features|class)·P(class) / P(features). Also fundamental to Bayesian inference: posterior ∝ likelihood × prior.
**Difficulty**: Beginner
**Category**: Math
**Tags**: #phase1 #probability #bayes

---

**ID**: P1-008
**Front**: What is Maximum Likelihood Estimation (MLE)?
**Back**: Find parameters θ that maximize the probability of observing the training data: θ̂ = argmax_θ Π P(x_i; θ) = argmax_θ Σ log P(x_i; θ). Maximizing log-likelihood of a Gaussian = minimizing MSE. Maximizing log-likelihood of categorical = minimizing cross-entropy.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #probability #MLE

---

**ID**: P1-009
**Front**: What is cross-entropy loss and how does it relate to KL divergence?
**Back**: H(P, Q) = -Σ P(x) log Q(x). Relationship: H(P,Q) = H(P) + D_KL(P||Q). When P is one-hot (true labels), H(P) = 0, so minimizing cross-entropy = minimizing KL divergence from true distribution. For binary: -[y·log(ŷ) + (1-y)·log(1-ŷ)].
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #probability #cross-entropy #loss

---

**ID**: P1-010
**Front**: What is gradient descent? Write the update rule.
**Back**: θ ← θ - α·∇_θ L. α = learning rate. ∇_θ L = gradient of loss w.r.t. parameters. Intuition: step downhill in parameter space. Variants: SGD (one sample), mini-batch (small batch), full-batch (all data). Adam uses adaptive learning rates per parameter.
**Difficulty**: Beginner
**Category**: Math
**Tags**: #phase1 #calculus #gradient-descent #optimization

---

**ID**: P1-011
**Front**: What is cosine similarity and when is it preferred over Euclidean distance?
**Back**: cos(θ) = (a·b) / (||a|| · ||b||). Range: -1 to 1 (1 = identical direction). Preferred for high-dimensional embeddings where magnitude is uninformative (e.g., text embeddings). Euclidean distance preferred when magnitude matters (spatial distance). Most vector DBs use cosine similarity for embedding search.
**Difficulty**: Beginner
**Category**: Math
**Tags**: #phase1 #linear-algebra #similarity #embeddings

---

**ID**: P1-012
**Front**: What is the Jacobian matrix and when does it appear in backprop?
**Back**: The Jacobian J_ij = ∂f_i/∂x_j — matrix of all partial derivatives of a vector-valued function. Appears when computing gradients through layers with vector outputs: dL/dx = J^T · dL/df. For the softmax layer, the Jacobian has the form s_i(δ_ij - s_j) — why softmax gradient is a matrix, not a vector.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase1 #calculus #jacobian #backprop

---

**ID**: P1-013
**Front**: What does it mean for two vectors to be orthogonal? Why does orthogonality matter in ML?
**Back**: Orthogonal: a·b = 0 (perpendicular). Orthonormal: orthogonal AND unit length. In ML: (1) PCA components are orthogonal — each captures independent variance. (2) Attention Q,K,V are projected to orthogonal subspaces. (3) Orthogonal weight init (deep network training stability). (4) QR decomposition uses orthonormalization.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #linear-algebra #orthogonality

---

**ID**: P1-014
**Front**: What is the L1 vs L2 norm and what regularization does each correspond to?
**Back**: L1: ||v||₁ = Σ|v_i|. L2: ||v||₂ = √(Σv_i²). L1 regularization (Lasso): adds λ||W||₁ to loss → promotes sparsity (many weights → 0). L2 regularization (Ridge/weight decay): adds λ||W||₂² → shrinks all weights toward 0, doesn't zero them. L2 corresponds to Gaussian prior; L1 to Laplace prior.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #linear-algebra #regularization

---

**ID**: P1-015
**Front**: What is entropy in information theory? What does it measure?
**Back**: H(X) = -Σ p(x) log₂ p(x). Measures uncertainty/surprise in a distribution. Maximum entropy = uniform distribution (maximum uncertainty). Minimum entropy = deterministic (one outcome probability 1). In LLMs: perplexity = 2^H — measures how "surprised" the model is by test data. Lower perplexity = better model.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #probability #entropy #information-theory

---

**ID**: P1-016
**Front**: What is PCA and how does SVD enable it?
**Back**: PCA finds orthogonal directions of maximum variance in data. Algorithm: (1) center data: X_c = X - mean(X). (2) Compute SVD: X_c = UΣV^T. (3) Principal components = columns of V. (4) Project: X_pca = X_c @ V[:, :k]. Singular values σ_i = variance explained by each component. Low-rank SVD truncation = approximate PCA.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #linear-algebra #PCA #SVD

---

**ID**: P1-017
**Front**: What is the normal distribution and why is it so common in ML?
**Back**: N(μ, σ²): bell curve centered at μ, spread σ. Common because: (1) Central Limit Theorem — sum of many independent r.v.s → Gaussian. (2) Maximum entropy distribution given mean and variance. (3) Mathematical tractability (conjugate prior). In ML: weight initialization (Xavier/He), noise injection, VAE latent space.
**Difficulty**: Beginner
**Category**: Math
**Tags**: #phase1 #probability #gaussian

---

**ID**: P1-018
**Front**: What is the difference between variance and standard deviation? Which is additive?
**Back**: Variance: σ² = E[(X - μ)²] — average squared deviation. Std: σ = √(variance) — same units as X. Variance is additive for independent random variables: Var(X+Y) = Var(X) + Var(Y). Std is NOT additive. In ML: variance of gradients is what we control via learning rate, batch size, initialization.
**Difficulty**: Beginner
**Category**: Math
**Tags**: #phase1 #probability #statistics

---

**ID**: P1-019
**Front**: What is the derivative of the sigmoid function?
**Back**: σ(x) = 1/(1+e^(-x)). σ'(x) = σ(x)·(1-σ(x)). Maximum value = 0.25 (at x=0). For large |x|: σ'(x) → 0 — this is the vanishing gradient problem. Why ReLU replaced sigmoid in hidden layers: ReLU gradient = 1 for x>0 (no vanishing).
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #calculus #sigmoid #vanishing-gradient

---

**ID**: P1-020
**Front**: What is meant by "the curse of dimensionality"?
**Back**: In high dimensions: (1) Volume concentrates near surface of hypersphere, not center. (2) Most points are far from each other — distance metrics become less meaningful. (3) Exponentially more data needed to cover space. (4) Nearest neighbor search becomes expensive. Relevance: embedding spaces are 1536-dim; cosine similarity works better than Euclidean in high-D.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase1 #probability #dimensionality

---

**ID**: P1-021
**Front**: Why do we use log probabilities instead of probabilities in ML training?
**Back**: (1) Numerical stability: probabilities of sequences become extremely small (e.g., 0.3^100 → underflow). Log converts products to sums: log(Π p_i) = Σ log(p_i). (2) Mathematically equivalent: argmax log(p) = argmax p (log is monotone). (3) Cross-entropy loss = -Σ y_i log(ŷ_i) — uses log probabilities directly.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #probability #numerical-stability

---

**ID**: P1-022
**Front**: What is matrix rank and why does it matter for LoRA?
**Back**: Rank of matrix = number of linearly independent rows (= columns). Full rank: rank = min(m,n). Low rank: rank < min(m,n). For LoRA: pretrained weight updates ΔW ≈ BA where rank(BA) ≤ r ≪ min(d,k). Hypothesis: fine-tuning only needs to modify a low-dimensional subspace of weight space.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase1 #linear-algebra #rank #LoRA

---

**ID**: P1-023
**Front**: What is the geometric interpretation of the dot product?
**Back**: a·b = ||a||·||b||·cos(θ) where θ = angle between vectors. Properties: a·b > 0 → angle < 90° (similar direction). a·b = 0 → orthogonal. a·b < 0 → angle > 90° (opposite direction). In attention: Q·K^T measures "how aligned" each query is with each key.
**Difficulty**: Beginner
**Category**: Math
**Tags**: #phase1 #linear-algebra #dot-product

---

**ID**: P1-024
**Front**: What is a Markov chain and how does it relate to language modeling?
**Back**: Markov chain: sequence where P(X_t | X_{t-1}, ..., X_0) = P(X_t | X_{t-1}) — only previous state matters. Language modeling: n-gram models are Markov (order n-1). Transformers are NOT Markov — they attend to ALL previous tokens (full context). This is a key advantage of transformers over traditional n-gram LMs.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase1 #probability #markov #language-modeling

---

**ID**: P1-025
**Front**: What does "independent and identically distributed" (i.i.d.) mean and when does ML violate it?
**Back**: i.i.d.: each sample drawn from same distribution (identical) and is statistically independent of others (independent). ML often violates this: (1) Time series — temporal correlation. (2) Text sequences — tokens are highly correlated. (3) Train/test distribution shift. (4) Data collected in batches (batch effects). i.i.d. assumption is required for most theoretical ML guarantees.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase1 #probability #statistics #iid

---

**ID**: P1-026
**Front**: What is the trace of a matrix and what property makes it useful?
**Back**: tr(A) = Σ A_ii (sum of diagonal elements). Key property: tr(AB) = tr(BA) — cyclic property. Also: tr(A) = Σ λ_i (sum of eigenvalues). Used in: matrix norms, regularization (Frobenius norm² = tr(A^T A)), gradient computation through trace expressions.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase1 #linear-algebra #trace

---

**ID**: P1-027
**Front**: What is numerical stability and why does it matter for softmax?
**Back**: Floating point has limited precision — very large or very small numbers overflow/underflow. Naive softmax: exp(1000) → inf. Stable softmax: subtract max before exp: softmax(z_i - max(z)) = softmax(z_i). Same mathematical result, no overflow. This is why frameworks implement "numerically stable softmax" and log-softmax separately.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #math #numerical-stability #softmax

---

**ID**: P1-028
**Front**: What is the relationship between MAP estimation and regularization?
**Back**: MAP (Maximum A Posteriori) = MLE + prior. θ̂_MAP = argmax P(θ|data) = argmax [log P(data|θ) + log P(θ)]. Gaussian prior P(θ) ~ N(0, σ²) → log P(θ) ∝ -||θ||² → L2 regularization. Laplace prior → L1 regularization. Regularization is Bayesian inference with a prior!
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase1 #probability #MAP #regularization #Bayesian

---

**ID**: P1-029
**Front**: What is the Hessian matrix and when does it appear in ML?
**Back**: Hessian H_ij = ∂²L/∂θ_i ∂θ_j — matrix of second derivatives. Positive definite Hessian → local minimum. Negative definite → local maximum. Indefinite → saddle point. In ML: (1) Newton's method uses inverse Hessian for faster convergence. (2) Saddle points (not local minima) are the main challenge in deep learning. (3) Too expensive to compute exactly for large models.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase1 #calculus #hessian #optimization

---

**ID**: P1-030
**Front**: What is mutual information and how does it differ from correlation?
**Back**: MI(X;Y) = H(X) + H(Y) - H(X,Y) — measures how much knowing Y reduces uncertainty about X. Properties: MI ≥ 0; MI = 0 iff X,Y independent. Unlike correlation (linear only), MI captures ANY statistical dependence (nonlinear). Used in: feature selection, information bottleneck theory of deep learning.
**Difficulty**: Advanced
**Category**: Math
**Tags**: #phase1 #probability #information-theory

---

**ID**: P1-031
**Front**: What is the gradient of a scalar loss with respect to a weight matrix?
**Back**: dL/dW has the SAME shape as W. For linear layer z = Wx + b: dL/dW = dL/dz · x^T (outer product). This is a rank-1 matrix update. In practice: sum over batch: dL/dW = (1/B) Σ_b dL/dz_b · x_b^T. Shape check: if W is (d_out, d_in), dL/dW must also be (d_out, d_in).
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #calculus #backprop #gradients

---

**ID**: P1-032
**Front**: What is the difference between a parameter and a hyperparameter?
**Back**: Parameter: learned from data during training (weights W, biases b). Hyperparameter: set before training, controls the learning process (learning rate α, batch size B, number of layers L, rank r in LoRA). Parameters are optimized by gradient descent; hyperparameters by grid search, random search, or Bayesian optimization.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase1 #math #fundamentals

---

**ID**: P1-033
**Front**: What is convexity and why do we care about it in ML?
**Back**: A function is convex if the line segment between any two points lies above the function. Convex optimization: any local minimum = global minimum (gradient descent guaranteed to find it). Most ML loss surfaces are non-convex (neural networks). Exceptions: logistic regression, SVM, linear regression — convex losses. Why care: understanding non-convexity explains why deep learning training is tricky.
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #calculus #optimization #convexity

---

**ID**: P1-034
**Front**: What is the softmax temperature trick for improving sampling quality?
**Back**: Sample from softmax(logits/T) instead of argmax (T→0) or uniform (T→∞). T < 1: focus on high-probability tokens — more coherent, less creative. T > 1: spread probability mass — more diverse, potentially incoherent. In production: T=0.0–0.2 for factual tasks, T=0.7–1.0 for creative tasks. Also: "top-p" (nucleus) sampling: sample from smallest set summing to probability p.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase1 #probability #sampling #temperature

---

**ID**: P1-035
**Front**: Explain the attention score formula and what each component does.
**Back**: score(Q, K) = QK^T / √d_k. Q (query): "what am I looking for?" K (key): "what do I contain?" QK^T: dot product measures query-key similarity. √d_k: scaling — without it, dot products grow large in high dimensions, pushing softmax into saturation region (near-zero gradients). Then softmax → weights. Then weighted sum of V (values = "what information to pass").
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase1 #linear-algebra #attention #transformers
