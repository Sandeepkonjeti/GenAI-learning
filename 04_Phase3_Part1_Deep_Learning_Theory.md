# Phase 3a: Deep Learning Theory
**Months 6–8 | Difficulty: 7/10 | Label: 🔴 Must Learn**

> **Previous Phase**: [Phase 2 — Machine Learning](./03_Phase2_Machine_Learning.md)  
> **Next Phase**: [Phase 3b — PyTorch](./04_Phase3_Part2_PyTorch.md)  
> **Parallel Reading**: Begin [Phase 3b](./04_Phase3_Part2_PyTorch.md) in Week 5 alongside this phase

---

## Table of Contents

- [Phase Overview](#phase-overview)
- [Why Deep Learning?](#why-deep-learning)
- [Topic 1: The Neural Network](#topic-1-the-neural-network)
  - [Neurons and Layers](#neurons-and-layers)
  - [Activation Functions](#activation-functions)
  - [Universal Approximation Theorem](#universal-approximation-theorem)
- [Topic 2: Backpropagation — The Full Derivation](#topic-2-backpropagation--the-full-derivation)
- [Topic 3: Optimizers](#topic-3-optimizers)
  - [SGD and Momentum](#sgd-and-momentum)
  - [Adam and AdamW](#adam-and-adamw)
  - [Learning Rate Schedules](#learning-rate-schedules)
- [Topic 4: Normalization](#topic-4-normalization)
  - [Batch Normalization](#batch-normalization)
  - [Layer Normalization](#layer-normalization)
- [Topic 5: Regularization for Deep Learning](#topic-5-regularization-for-deep-learning)
- [Topic 6: The Training Loop in Depth](#topic-6-the-training-loop-in-depth)
- [Topic 7: Convolutional Neural Networks](#topic-7-convolutional-neural-networks)
- [Topic 8: Recurrent Networks and Why They Were Replaced](#topic-8-recurrent-networks-and-why-they-were-replaced)
- [Resources](#resources)
- [Common Mistakes](#common-mistakes)
- [Week-by-Week Plan](#week-by-week-plan)
- [Mastery Checklist](#mastery-checklist)

---

## Phase Overview

| Attribute | Details |
|-----------|---------|
| **Duration** | Months 6–8 (6 weeks theory) |
| **Daily Time** | 1.5–2 hours |
| **Difficulty** | 7/10 |
| **Label** | 🔴 Must Learn |
| **Prerequisites** | Phase 1 (all math), Phase 2 (implemented backprop for 2-layer net) |
| **Outcome** | Deep understanding of how neural networks learn; can debug training problems |

---

## Why Deep Learning?

Classical ML requires **hand-crafted features**. You must decide: use word frequency, sentence length, punctuation count — and you encode human intuitions about what matters.

Deep learning learns **features from raw data** automatically. Given enough data and compute, a deep network discovers representations that are often better than anything a human would engineer.

The breakthrough was realizing that stacking many layers of nonlinear transformations enables learning **hierarchical representations**:

```
Raw Input → Layer 1 (edges) → Layer 2 (shapes) → Layer 3 (objects) → Output
```

For text:
```
Characters → Tokens → Words → Phrases → Sentences → Meaning → Output
```

This hierarchy emerges from the data — not from human engineering. This is why transformers can understand context, sentiment, code, math, and reasoning from the same pretraining objective.

---

## Topic 1: The Neural Network

### Neurons and Layers

A single neuron computes:

$$z = \sum_i w_i x_i + b = \mathbf{w}^T\mathbf{x} + b$$
$$a = f(z)$$

Where $f$ is an activation function (nonlinearity).

A layer of neurons computes this in parallel:

$$\mathbf{z} = X\mathbf{W} + \mathbf{b}, \quad \mathbf{a} = f(\mathbf{z})$$

This is just **matrix multiplication + bias + nonlinearity**.

```python
import numpy as np

class DenseLayer:
    """
    A single fully-connected layer — the building block of every neural network.
    """
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        # Xavier/Glorot initialization: scale by sqrt(2 / fan_in)
        # Prevents vanishing/exploding gradients at initialization
        scale = np.sqrt(2.0 / input_size)
        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros(output_size)
        self.activation = activation
        
        # Cache for backward pass
        self.input = None
        self.z = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.z = x @ self.W + self.b   # Linear transformation
        return self._activate(self.z)
    
    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'gelu':
            return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activate_grad(self, z: np.ndarray) -> np.ndarray:
        """Derivative of activation function."""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2
        elif self.activation == 'linear':
            return np.ones_like(z)
        # GELU gradient is complex; use numerical approx for now
    
    def backward(self, grad_output: np.ndarray) -> tuple:
        """
        grad_output: gradient flowing from the layer above
        Returns: gradient to pass to layer below, gradients for W and b
        """
        # Gradient through activation
        grad_z = grad_output * self._activate_grad(self.z)
        
        # Gradient for weights: dL/dW = input.T @ grad_z
        grad_W = self.input.T @ grad_z
        
        # Gradient for bias: dL/db = sum over batch
        grad_b = grad_z.sum(axis=0)
        
        # Gradient to pass backward: dL/d(input) = grad_z @ W.T
        grad_input = grad_z @ self.W.T
        
        return grad_input, grad_W, grad_b
```

---

### Activation Functions

**Why nonlinearities are needed**: Without activation functions, stacking linear layers is equivalent to a single linear layer. The entire network could be collapsed to $y = (W_n \cdot W_{n-1} \cdots W_1)x + b$ — still just linear. Activation functions break linearity and enable learning complex patterns.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 200)

activations = {
    'Sigmoid\nσ(x) = 1/(1+e^{-x})': {
        'fn': lambda x: 1 / (1 + np.exp(-x)),
        'grad': lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x))),
        'use': 'Binary output, gates in LSTMs',
        'issue': 'Saturates → vanishing gradients'
    },
    'Tanh\ntanh(x)': {
        'fn': np.tanh,
        'grad': lambda x: 1 - np.tanh(x)**2,
        'use': 'RNNs, bounded outputs',
        'issue': 'Still saturates at extremes'
    },
    'ReLU\nmax(0, x)': {
        'fn': lambda x: np.maximum(0, x),
        'grad': lambda x: (x > 0).astype(float),
        'use': 'CNNs, MLPs — most common',
        'issue': 'Dying ReLU: negative inputs always get 0 gradient'
    },
    'GELU\n≈ x·Φ(x)': {
        'fn': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
        'grad': None,  # complex
        'use': 'Transformers (GPT, BERT) — standard now',
        'issue': 'More expensive than ReLU'
    },
    'SwiGLU\n(Gated Linear Unit)': {
        'fn': lambda x: x * (1 / (1 + np.exp(-x))),  # Swish = x * sigmoid(x)
        'grad': None,
        'use': 'LLaMA, modern LLMs',
        'issue': 'Requires 2x parameters in FFN'
    }
}

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (name, info) in zip(axes, activations.items()):
    y = info['fn'](x)
    ax.plot(x, y, 'b-', linewidth=2)
    if info['grad']:
        ax.plot(x, info['grad'](x), 'r--', linewidth=1.5, label='gradient', alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(name, fontsize=9)
    ax.set_ylim(-1.5, 1.5)
plt.tight_layout()
```

**Summary table**:

| Activation | Used In | Key Property | Avoid When |
|-----------|---------|--------------|-----------|
| Sigmoid | Output (binary), LSTM gates | Maps to (0,1) | Hidden layers — vanishing gradients |
| Tanh | RNNs, some hidden layers | Maps to (-1,1), zero-centered | Deep networks |
| ReLU | CNNs, MLPs | Fast, simple | Might have dying neurons |
| GELU | Transformers (BERT, GPT) | Smooth, works with attention | Rarely a problem |
| SwiGLU | LLaMA, modern LLMs | Best empirical performance | Budget-constrained (2x FFN params) |

---

### Universal Approximation Theorem

**Theorem**: A neural network with a single hidden layer of sufficient width can approximate any continuous function to arbitrary precision (given enough neurons).

**What this means for you**: Deep networks are not magic. They're flexible function approximators. Given enough parameters and diverse training data, they can learn virtually any input-output mapping. This is why pretraining on vast text enables LLMs to "learn" grammar, math, code, and reasoning — all from the same next-token prediction objective.

**Practical implication**: The question is never "can the network represent this function?" It's always "do we have enough data and compute, and is the optimization effective?"

---

## Topic 2: Backpropagation — The Full Derivation

You implemented backprop for a 2-layer network in Phase 1. Now generalize to arbitrary depth.

The key insight: **gradient flows backward through the network by repeated application of the chain rule.**

For a network with layers $L_1, L_2, ..., L_n$ followed by loss $\mathcal{L}$:

$$\frac{\partial \mathcal{L}}{\partial W_k} = \frac{\partial \mathcal{L}}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_{k+1}}{\partial a_k} \cdot \frac{\partial a_k}{\partial W_k}$$

```python
import numpy as np

class MLP:
    """
    Multi-layer perceptron with full backpropagation.
    This is the core building block of all deep learning.
    """
    def __init__(self, layer_sizes: list[int], activation: str = 'relu'):
        """
        layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
        Example: [784, 256, 128, 10] for MNIST
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            # Last layer has linear activation (we apply softmax separately)
            act = activation if i < len(layer_sizes) - 2 else 'linear'
            self.layers.append(DenseLayer(in_size, out_size, act))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        current = X
        for layer in self.layers:
            current = layer.forward(current)
        return current  # logits (before softmax)
    
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def cross_entropy_loss(self, logits: np.ndarray, y_true: np.ndarray) -> tuple:
        """
        Combined softmax + cross-entropy (numerically stable).
        Returns: (loss, gradient w.r.t. logits)
        
        The beautiful gradient: dL/d(logits) = (probs - one_hot_y) / N
        This is why softmax + cross-entropy is the standard output.
        """
        N = y_true.shape[0]
        probs = self.softmax(logits)
        
        # Log-sum-exp trick for stable log(softmax)
        log_probs = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs -= np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
        
        # Loss: -mean of log probability of true class
        correct_log_probs = log_probs[np.arange(N), y_true]
        loss = -np.mean(correct_log_probs)
        
        # Gradient w.r.t. logits: (softmax_output - one_hot) / N
        grad = probs.copy()
        grad[np.arange(N), y_true] -= 1
        grad /= N
        
        return loss, grad
    
    def backward(self, grad_logits: np.ndarray) -> dict:
        """
        Backward pass through all layers.
        Returns dict of all parameter gradients.
        """
        gradients = {}
        current_grad = grad_logits
        
        for i, layer in enumerate(reversed(self.layers)):
            layer_idx = len(self.layers) - 1 - i
            current_grad, grad_W, grad_b = layer.backward(current_grad)
            gradients[f'W{layer_idx}'] = grad_W
            gradients[f'b{layer_idx}'] = grad_b
        
        return gradients
    
    def update(self, gradients: dict, lr: float = 0.01):
        for i, layer in enumerate(self.layers):
            layer.W -= lr * gradients[f'W{i}']
            layer.b -= lr * gradients[f'b{i}']

# Train on MNIST-like data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target  # 1797 samples, 64 features, 10 classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLP([64, 128, 64, 10], activation='relu')

batch_size = 32
n_epochs = 50
learning_rate = 0.01

for epoch in range(n_epochs):
    # Mini-batch gradient descent
    indices = np.random.permutation(len(X_train))
    epoch_loss = 0
    
    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        logits = model.forward(X_batch)
        loss, grad = model.cross_entropy_loss(logits, y_batch)
        gradients = model.backward(grad)
        model.update(gradients, lr=learning_rate)
        epoch_loss += loss
    
    if epoch % 10 == 0:
        # Evaluate
        test_logits = model.forward(X_test)
        y_pred = np.argmax(test_logits, axis=1)
        acc = (y_pred == y_test).mean()
        print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | Test Acc: {acc:.4f}")
```

---

## Topic 3: Optimizers

Gradient descent has a problem: it's slow and the single learning rate doesn't adapt to different parameters. Optimizers solve this.

### SGD and Momentum

```python
import numpy as np

class SGD:
    """Basic Stochastic Gradient Descent."""
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params: dict, grads: dict):
        for key in params:
            params[key] -= self.lr * grads[key]

class SGDWithMomentum:
    """
    SGD with momentum — adds a "velocity" term.
    
    v = β * v - lr * grad     (accumulate history)
    w = w + v                 (apply velocity)
    
    Momentum helps in two ways:
    1. Accelerates through consistent gradients (ball rolling downhill)
    2. Dampens oscillations across inconsistent gradients
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params: dict, grads: dict):
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
            params[key] += self.velocity[key]
```

### Adam and AdamW

Adam (Adaptive Moment Estimation) is the default optimizer for most deep learning, including transformers.

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment — mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment — variance)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}$$
$$w_t = w_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
import numpy as np

class Adam:
    """
    Adam optimizer — standard for deep learning.
    
    Key insight: each parameter gets its own adaptive learning rate.
    Parameters with large gradients get small updates (already moving fast).
    Parameters with small gradients get large updates (need more exploration).
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # first moment (mean)
        self.v = {}  # second moment (uncentered variance)
        self.t = 0   # timestep
    
    def update(self, params: dict, grads: dict):
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            
            # Bias-corrected moments
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # Parameter update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class AdamW(Adam):
    """
    AdamW = Adam + decoupled weight decay.
    
    Standard Adam applies L2 regularization through the gradient,
    but this interacts badly with adaptive learning rates.
    AdamW applies weight decay directly to parameters, decoupled from gradient.
    
    AdamW is the standard optimizer for training transformers.
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay
    
    def update(self, params: dict, grads: dict):
        # Apply weight decay BEFORE Adam update (decoupled)
        for key in params:
            if 'bias' not in key:  # typically don't decay biases
                params[key] *= (1 - self.lr * self.weight_decay)
        
        # Regular Adam update
        super().update(params, grads)
```

### Learning Rate Schedules

```python
import numpy as np
import matplotlib.pyplot as plt

class CosineAnnealingWithWarmup:
    """
    The standard learning rate schedule for training transformers.
    
    Phase 1 (warmup): LR linearly increases from 0 to max_lr
    Phase 2 (cosine): LR smoothly decreases following cosine curve
    
    Why warmup? Early in training, gradients are noisy and large LR causes divergence.
    Warmup stabilizes initial training.
    """
    def __init__(self, max_lr: float, warmup_steps: int, total_steps: int):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.max_lr * 0.5 * (1 + np.cos(np.pi * progress))

# Visualize different schedules
steps = np.arange(10000)
scheduler = CosineAnnealingWithWarmup(max_lr=3e-4, warmup_steps=500, total_steps=10000)
lrs = [scheduler.get_lr(s) for s in steps]

plt.figure(figsize=(10, 4))
plt.plot(steps, lrs)
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing with Linear Warmup\n(Standard for Transformer Training)')
plt.axvline(500, color='r', linestyle='--', label='End of warmup')
plt.legend(); plt.show()
```

---

## Topic 4: Normalization

### Batch Normalization

**Problem**: As gradients flow through many layers, the distribution of activations shifts constantly. This makes training unstable and requires very small learning rates.

**Solution**: After each layer, normalize activations to have zero mean and unit variance, then apply learnable scale and shift:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

```python
import numpy as np

class BatchNorm:
    """
    Batch Normalization.
    
    Used in: CNNs, ResNets, many older architectures.
    Note: DOES NOT work well for small batch sizes or recurrent networks.
    This is why Layer Norm replaced it in transformers.
    """
    def __init__(self, n_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = np.ones(n_features)   # learnable scale
        self.beta = np.zeros(n_features)   # learnable shift
        self.eps = eps
        self.momentum = momentum
        
        # Running statistics for inference
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        
        self.cache = {}
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mean = x.mean(axis=0)    # Mean over batch dimension
            var = x.var(axis=0)
            
            # Normalize
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            self.cache = {'x': x, 'x_hat': x_hat, 'mean': mean, 'var': var}
        else:
            # Inference: use running statistics (batch size may be 1)
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        return self.gamma * x_hat + self.beta
```

### Layer Normalization

**Why Layer Norm replaced Batch Norm for transformers**:

| | Batch Norm | Layer Norm |
|--|--|--|
| Normalizes over | Batch (per feature) | Features (per sample) |
| Batch size 1 | ❌ Fails | ✓ Works fine |
| Sequence models | ❌ Variable lengths | ✓ Works perfectly |
| Parallel training | ❌ All-reduce needed | ✓ Independent |
| Used in | CNNs, older MLPs | **All transformers** |

```python
class LayerNorm:
    """
    Layer Normalization — the normalization method used in ALL transformers.
    
    Normalizes over the feature dimension (not batch), so it works 
    with batch size = 1 and variable sequence lengths.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.gamma = np.ones(normalized_shape)   # learnable scale
        self.beta = np.zeros(normalized_shape)   # learnable shift
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (..., normalized_shape)
        Normalizes over the last dimension.
        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# Example: transformer layer uses LayerNorm
# x shape: (batch_size, seq_len, d_model)
batch_size, seq_len, d_model = 8, 512, 768
x = np.random.randn(batch_size, seq_len, d_model)
ln = LayerNorm(d_model)
out = ln.forward(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")          # Same shape
print(f"Output mean (should ≈ 0): {out[0, 0].mean():.6f}")
print(f"Output std  (should ≈ 1): {out[0, 0].std():.6f}")
```

---

## Topic 5: Regularization for Deep Learning

### Dropout

```python
import numpy as np

class Dropout:
    """
    Dropout: randomly zeroes neurons during training.
    
    Why it works:
    1. Forces the network not to rely on any single neuron
    2. Approximates training an ensemble of 2^n networks
    3. Is a Bayesian approximation to inference
    
    During training: randomly zero neurons with probability p
    During inference: scale by (1-p) to maintain expected values
    (OR use inverted dropout: scale by 1/(1-p) during training, no scaling at inference)
    """
    def __init__(self, p: float = 0.1):
        """p = probability of DROPPING (zeroing) a neuron"""
        assert 0 <= p < 1
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return x
        
        # Inverted dropout (standard): scale during training
        scale = 1.0 / (1.0 - self.p)
        self.mask = (np.random.rand(*x.shape) > self.p).astype(float) * scale
        return x * self.mask

# Typical dropout rates:
# - 0.1 in transformers (attention dropout, residual dropout)
# - 0.2-0.5 in MLPs
# - 0.5 in classification heads
```

### Early Stopping

```python
class EarlyStopping:
    """
    Monitor validation loss and stop training when it stops improving.
    The most practical regularization technique.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch}. Best was epoch {self.best_epoch}.")
                self.should_stop = True
        return self.should_stop
```

---

## Topic 6: The Training Loop in Depth

This is the pattern you'll use for every neural network you ever train. Master it now.

```python
import numpy as np
from typing import Callable

def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
) -> dict:
    """
    Complete training loop with:
    - Mini-batch gradient descent
    - Learning rate scheduling
    - Validation monitoring
    - Early stopping
    - Training history
    
    This exact structure translates directly to PyTorch.
    """
    optimizer = AdamW(lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWithWarmup(
        max_lr=learning_rate,
        warmup_steps=100,
        total_steps=n_epochs * (len(X_train) // batch_size)
    )
    early_stopping = EarlyStopping(patience=patience)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    global_step = 0
    
    for epoch in range(n_epochs):
        model.set_training(True)
        
        # Shuffle training data each epoch
        indices = np.random.permutation(len(X_train))
        epoch_losses = []
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            # ── Forward Pass ──
            logits = model.forward(X_batch)
            
            # ── Compute Loss ──
            loss, grad_logits = model.cross_entropy_loss(logits, y_batch)
            
            # ── Backward Pass ──
            gradients = model.backward(grad_logits)
            
            # ── Get Current LR ──
            current_lr = scheduler.get_lr(global_step)
            optimizer.lr = current_lr
            
            # ── Parameter Update ──
            # (simplified — real version passes params dict)
            model.step(gradients, optimizer)
            
            epoch_losses.append(loss)
            global_step += 1
        
        # ── Validation ──
        model.set_training(False)
        val_logits = model.forward(X_val)
        val_loss, _ = model.cross_entropy_loss(val_logits, y_val)
        val_acc = (np.argmax(val_logits, axis=1) == y_val).mean()
        
        # ── Record History ──
        history['train_loss'].append(np.mean(epoch_losses))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        # ── Early Stopping ──
        if early_stopping(val_loss, epoch):
            break
    
    return history
```

---

## Topic 7: Convolutional Neural Networks

**Why study CNNs** even though you're targeting LLMs: CNNs introduced the concept of **parameter sharing** and **local connectivity** — ideas that influenced transformers. Vision transformers (ViT) and multimodal models (LLaVA) use CNNs or CNN-like encoders. Understanding CNNs also deepens your understanding of what features neural networks learn.

```python
import numpy as np

def convolve2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    2D convolution from scratch.
    
    Key insight: convolution is parameter sharing.
    The SAME filter is applied at every spatial location.
    This exploits the translational invariance of images.
    
    image shape: (H, W)
    kernel shape: (k_h, k_w)
    """
    H, W = image.shape
    k_h, k_w = kernel.shape
    out_h = (H - k_h) // stride + 1
    out_w = (W - k_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    for i in range(0, out_h):
        for j in range(0, out_w):
            patch = image[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            output[i, j] = np.sum(patch * kernel)
    
    return output

# Edge detection kernel (Sobel filter)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# CNNs LEARN these filters from data
# Early layers learn edges, middle layers learn shapes, later layers learn objects
```

**Skip connections (ResNets)** — critically important:

```python
# ResNet skip connection — enables training very deep networks (50, 100, 1000+ layers)
# Without skip connections: gradients vanish over many layers
# With skip connections: gradients have a direct path to early layers

def residual_block_forward(x, W1, b1, W2, b2, activation):
    """
    y = F(x) + x
    
    If the gradients want to pass through completely unchanged,
    F(x) just needs to learn to output zero.
    This makes deep networks easy to train: start from identity,
    learn small corrections.
    """
    identity = x
    
    # Two layers
    z1 = x @ W1 + b1
    a1 = activation(z1)
    z2 = a1 @ W2 + b2
    
    # Skip connection: add input directly to output
    output = z2 + identity   # ← This is the key innovation of ResNets
    
    return output
```

**Why skip connections work**: They create direct gradient paths from the loss back to early layers. Without them, gradients must flow through every nonlinearity and can vanish to zero.

---

## Topic 8: Recurrent Networks and Why They Were Replaced

**Study this to understand the problem that transformers solve.**

```python
import numpy as np

class SimpleRNN:
    """
    Vanilla RNN — processes sequences step by step.
    
    h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
    
    Problem: to compute gradient at step t=0 in a 512-step sequence,
    the gradient must flow through 512 tanh operations.
    tanh'(x) ≤ 1 everywhere, often << 1.
    Multiply 512 numbers ≤ 1: result approaches 0 (vanishing gradient).
    """
    def __init__(self, input_size: int, hidden_size: int):
        scale = 0.01
        self.W_x = np.random.randn(input_size, hidden_size) * scale
        self.W_h = np.random.randn(hidden_size, hidden_size) * scale
        self.b = np.zeros(hidden_size)
    
    def forward(self, sequence: np.ndarray) -> tuple:
        """
        sequence shape: (seq_len, input_size)
        Returns: hidden states, final hidden state
        """
        seq_len = sequence.shape[0]
        h = np.zeros(self.W_h.shape[0])
        hidden_states = []
        
        for t in range(seq_len):
            h = np.tanh(sequence[t] @ self.W_x + h @ self.W_h + self.b)
            hidden_states.append(h)
        
        return np.array(hidden_states), h

# The problem in numbers:
# Sequence length = 512
# tanh'(0) = 1.0 (maximum)
# But tanh'(large x) ≈ 0 (saturated)
# After 50 steps: gradient ≈ 0.99^50 ≈ 0.6
# After 200 steps: gradient ≈ 0.99^200 ≈ 0.13  
# After 512 steps: gradient ≈ 0.99^512 ≈ 0.006  ← nearly gone!

# LSTMs solve this with gates but still struggle with very long sequences.
# Transformers solve this completely: every token attends directly to every other token.
# There is NO sequential dependency in the gradient flow.
# This is the core architectural advantage of transformers.
```

---

## Resources

| Rank | Resource | Type | Cost | Why |
|------|----------|------|------|-----|
| 1 | [Andrej Karpathy — micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | YouTube | Free | **Must watch.** Build backpropagation from scratch in 2 hours. |
| 2 | [fast.ai Practical Deep Learning for Coders](https://course.fast.ai/) | Course | Free | Best top-down DL course. You build before theory. Follow alongside this guide. |
| 3 | [Deep Learning Specialization (DeepLearning.AI)](https://www.coursera.org/specializations/deep-learning) | Course | ~$50/mo | Systematic, rigorous. Andrew Ng explains everything from first principles. |
| 4 | [Dive into Deep Learning (d2l.ai)](https://d2l.ai/) | Book | Free | Interactive; theory and code side-by-side. Excellent reference. |
| 5 | Deep Learning (Goodfellow, Bengio, Courville) | Book | Free PDF | Theoretical bible. Use as reference for proofs. |
| 6 | [Karpathy — makemore series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | YouTube | Free | Build language models from bigrams to transformers. Bridges Phase 3→4. |

**Non-Negotiable Videos**:
- Karpathy micrograd (2 hours) — do this first
- DeepLearning.AI Course 1: Neural Networks and Deep Learning (all weeks)
- Karpathy makemore parts 1–5 (start in Week 5)

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Using PyTorch before understanding backprop | Can't debug gradient issues | Implement backprop in NumPy first |
| Not doing gradient checking | Wrong gradients = silent failure | Always gradient check your implementations |
| Treating Adam as a black box | Can't tune it or debug training | Implement Adam from scratch (above) |
| Not implementing batch norm & layer norm | Don't understand normalization | Code both; understand the difference |
| Skipping skip connections | Miss key intuition for transformers | Implement residual blocks |
| Skipping RNNs entirely | Don't understand what problem transformers solve | Study briefly; don't go deep |

---

## Week-by-Week Plan

| Week | Focus | Resource | Output |
|------|-------|----------|--------|
| **Week 1** | Neurons, layers, activation functions | DL Spec Course 1 Week 1 + d2l.ai Ch 3 | Implement MLP with 5 activation functions |
| **Week 2** | Backpropagation derivation | Karpathy micrograd + DL Spec Week 2 | Full backprop MLP with gradient checking |
| **Week 3** | Optimizers: SGD, momentum, Adam, AdamW | DL Spec Course 2 Week 2 + MML book | Implement all optimizers; visualize convergence |
| **Week 4** | Normalization + regularization | d2l.ai Ch 8 + DL Spec | Implement BatchNorm and LayerNorm |
| **Week 5** | Full training loop + CNNs | fast.ai Lesson 1–3 + DL Spec Course 4 | Train MLP on digits; train simple CNN |
| **Week 6** | RNNs/LSTMs + skip connections + review | Karpathy makemore 1–3 + DL Spec Course 5 | RNN from scratch; understand why transformers replace them |

---

## Mastery Checklist

### Theory
- [ ] Can explain why activation functions are needed (linearity argument)
- [ ] Can derive the gradient of cross-entropy + softmax output layer
- [ ] Can trace the chain rule through a 3-layer network by hand
- [ ] Understands why Adam adapts learning rate per parameter
- [ ] Knows the difference between Batch Norm and Layer Norm (and why transformers use LayerNorm)
- [ ] Can explain why skip connections enable training very deep networks
- [ ] Can explain why RNNs have vanishing gradients and why transformers don't

### Implementation
- [ ] Implemented backpropagation for a deep MLP from scratch
- [ ] Implemented Adam optimizer from scratch
- [ ] Implemented LayerNorm and BatchNorm from scratch
- [ ] Implemented Dropout (inverted) from scratch
- [ ] Trained an MLP on MNIST-like data achieving >90% accuracy (without PyTorch)
- [ ] Watched Karpathy micrograd video and re-implemented it

### Ready for PyTorch
- [ ] Has watched Karpathy makemore episodes 1–3
- [ ] Understands the training loop pattern (forward → loss → backward → step)
- [ ] Can explain what `model.zero_grad()` does and why it's needed

---

## Moving to Phase 3b

**Before proceeding to [Phase 3b: PyTorch](./04_Phase3_Part2_PyTorch.md), confirm:**

- [ ] Implemented backpropagation for a multi-layer network from scratch
- [ ] Implemented Adam, BatchNorm, LayerNorm, Dropout from scratch
- [ ] Trained a network on real data to a good accuracy
- [ ] Watched Karpathy micrograd video
- [ ] Understands why transformers replaced RNNs

**Why PyTorch comes next**: You now understand exactly what PyTorch is doing under the hood. When you call `loss.backward()`, you know it's running the chain rule on the computational graph. When you use `nn.LayerNorm`, you know how it normalizes. This knowledge makes you a 10x more effective PyTorch user.

---

## Phase Completion & Readiness Assessment

> Complete this assessment **before** moving to Phase 3b (PyTorch). Understanding the theory deeply means you'll use PyTorch as a tool rather than magic.

---

### 1. Knowledge Checklist

**Neural Network Fundamentals**
- [ ] Universal Approximation Theorem — what it means and its limits
- [ ] Architecture: input layer, hidden layers, output layer, width vs. depth
- [ ] Forward pass: layer-by-layer computation, matrix multiply + activation
- [ ] Loss functions: MSE for regression, cross-entropy for classification
- [ ] Gradient descent: how it applies to neural networks

**Backpropagation**
- [ ] Chain rule: how it applies to composed functions
- [ ] Computational graph: nodes, edges, forward pass, backward pass
- [ ] Gradient flow: how gradients accumulate from output to input
- [ ] Vanishing gradient problem: why deep sigmoid networks fail to train
- [ ] Exploding gradient problem: why gradient clipping is needed

**Activation Functions**
- [ ] Sigmoid: formula, derivative, saturation problem
- [ ] Tanh: formula, derivative, zero-centred vs. sigmoid
- [ ] ReLU: formula, derivative, dying ReLU problem
- [ ] Leaky ReLU, PReLU, ELU — when and why
- [ ] GELU (used in BERT/GPT) — smooth approximation of ReLU

**Weight Initialisation**
- [ ] Why zero initialisation fails (symmetry breaking problem)
- [ ] Xavier/Glorot initialisation — derivation from variance preservation
- [ ] He initialisation — why it's better for ReLU networks
- [ ] Kaiming initialisation in PyTorch

**Normalisation**
- [ ] Batch Normalisation: formula, training vs. inference behaviour, learnable γ/β
- [ ] Layer Normalisation: formula, why it's used in transformers
- [ ] Group Normalisation: when batch statistics are unreliable
- [ ] Why normalisation stabilises and speeds up training

**Regularisation**
- [ ] L1/L2 weight decay applied to neural networks
- [ ] Dropout: training mode (mask) vs. inference mode (scale), inverted dropout
- [ ] Data augmentation as regularisation
- [ ] Early stopping

---

### 2. Practical Skills Checklist

- [ ] Implement forward and backward pass for a single dense layer from scratch using only NumPy
- [ ] Implement backpropagation through a 2-layer network by hand on paper
- [ ] Implement all five activation functions with their derivatives
- [ ] Implement batch normalisation (both forward and backward) from scratch
- [ ] Implement dropout (inverted dropout style) from scratch
- [ ] Identify from a learning curve whether a network has vanishing gradients
- [ ] Choose the correct weight initialisation for a given activation function
- [ ] Explain what happens if you use zero initialisation (demonstrate symmetry breaking)

---

### 3. Coding Challenges

**Challenge A — Backprop from Scratch**
```python
# Build a 2-layer neural network with NumPy ONLY:
# Architecture: Input(784) → Dense(256) → ReLU → Dense(10) → Softmax
# Implement:
#   - forward(X): returns output probabilities
#   - backward(X, y, output): computes all gradients
#   - update(lr): applies gradient descent
# Train on flattened MNIST images
# Achieve > 95% test accuracy
# Plot: loss and accuracy per epoch
```

**Challenge B — Activation Function Comparison**
```python
# Train the same architecture with 5 different activations: Sigmoid, Tanh, ReLU, GELU, ELU
# Use the SAME weight initialisation for all (He init)
# Plot: training loss curves for all 5 on the same chart
# Show that sigmoid/tanh converge slower or not at all for deep networks
# Discuss in 3 sentences: why ReLU/GELU win
```

**Challenge C — Gradient Visualisation**
```python
# Build a deep network (10 layers) and:
# 1. Log the gradient norm at each layer during the first 100 steps
# 2. Plot: gradient magnitude vs. layer depth for sigmoid vs. ReLU
# 3. Demonstrate vanishing gradients visually
# 4. Show that He initialisation with ReLU keeps gradient norms roughly equal across layers
```

---

### 4. Mini Project

**Micrograd Extension**: Extend Karpathy's micrograd (Project 6) to support:
- `ReLU`, `Tanh`, `Sigmoid` activation classes
- `BatchNorm1d` layer class (forward only is sufficient)
- `Dropout` layer class with a `training` flag
- Train on the full Iris dataset (4 features, 3 classes)
- Plot the computation graph for a 2-layer network using Graphviz

---

### 5. Capstone Project

**Neural Network Library "NanoNet"**: Build a mini deep learning framework with:
- `Layer` base class with `forward(X)` and `backward(dout)` interface
- `Linear(in_features, out_features)` layer
- `ReLU()`, `Sigmoid()`, `Tanh()` activation layers
- `BatchNorm1d()` layer
- `Dropout(p)` layer
- `Sequential([...layers...])` container
- `CrossEntropyLoss()` and `MSELoss()` classes
- `SGD(lr)` and `Adam(lr)` optimizers
- Full training loop that achieves >97% on MNIST

---

### 6. Interview Questions

**Beginner**

1. **Q: What is backpropagation?**
   A: Backpropagation is the algorithm that computes gradients of the loss with respect to all parameters in a neural network. It applies the chain rule, propagating error signals from output to input. The gradients are then used by an optimizer to update the weights.

2. **Q: What is an activation function and why do neural networks need them?**
   A: Without activation functions, a neural network is just a sequence of matrix multiplications — equivalent to a single linear layer. Activation functions introduce non-linearity, allowing the network to learn complex, non-linear decision boundaries.

3. **Q: What is the vanishing gradient problem?**
   A: In deep networks with sigmoid/tanh activations, gradients are multiplied by the activation derivative at each layer. Since max sigmoid derivative is 0.25, after 10 layers the gradient is (0.25)^10 ≈ 10^-6 — effectively zero. Early layers receive no gradient signal and don't learn.

4. **Q: Why is ReLU preferred over sigmoid for hidden layers?**
   A: ReLU derivative is either 0 (inactive) or 1 (active). No saturation in the positive region means gradients don't vanish for active neurons. ReLU is also computationally cheaper (threshold vs. exp).

5. **Q: What is dropout and how does it prevent overfitting?**
   A: Dropout randomly zeroes out a fraction p of neurons during training. This prevents co-adaptation — neurons can't rely on specific other neurons. At inference, no neurons are dropped but outputs are scaled by (1-p). Effectively trains an ensemble of 2^N different subnetworks.

6. **Q: What is the dying ReLU problem?**
   A: If a neuron receives a large negative input, ReLU outputs 0 and the gradient is also 0 — the neuron can never recover. With large learning rates or bad initialisation, many neurons can "die" permanently. Leaky ReLU (small slope for x<0) mitigates this.

7. **Q: What is batch normalisation and what problem does it solve?**
   A: BatchNorm normalises layer inputs to zero mean and unit variance, then applies learnable scale (γ) and shift (β). It solves *internal covariate shift* — the distribution of layer inputs changes during training, making training unstable and slow. BatchNorm allows much higher learning rates and is less sensitive to initialisation.

**Intermediate**

8. **Q: Derive why Xavier initialisation uses std = sqrt(1/fan_in)?**
   A: For a layer y = Wx with x ~ N(0,1), Var(y_i) = n_in * Var(w) * Var(x). To preserve variance across layers, we want Var(y) = Var(x), so Var(w) = 1/n_in, giving std(w) = sqrt(1/n_in). For sigmoid/tanh, use sqrt(2/(fan_in+fan_out)) (Glorot). For ReLU, use sqrt(2/fan_in) (He) because ReLU kills half the units.

9. **Q: What is the difference between batch norm and layer norm?**
   A: BatchNorm normalises across the batch dimension (mean/var computed over N examples for each feature). LayerNorm normalises across the feature dimension for each example independently. LayerNorm is preferred in NLP/transformers because: (1) batch size can be 1, (2) variable sequence lengths make batch stats unreliable.

10. **Q: How does the chain rule apply to backpropagation in a multi-layer network?**
    A: For loss L through layers f₃(f₂(f₁(x))), the chain rule gives: dL/dx = dL/df₃ · df₃/df₂ · df₂/df₁ · df₁/dx. In a neural network, this is computed layer by layer in reverse (backprop), reusing intermediate activations from the forward pass.

11. **Q: What is gradient clipping and when is it necessary?**
    A: Gradient clipping limits gradient norms above a threshold: if ||g|| > max_norm, scale g by max_norm/||g||. Required for RNNs (exploding gradients through long sequences) and any network where gradients can become very large. PyTorch: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`.

12. **Q: What is inverted dropout and why is it preferred over standard dropout?**
    A: Standard dropout: zero out with probability p at training, scale outputs by (1-p) at inference. Inverted dropout: scale by 1/(1-p) during training, do nothing at inference. Inverted is preferred because inference code stays identical to training code with `model.eval()`.

13. **Q: Explain the computational graph and how PyTorch builds it dynamically.**
    A: A computational graph records all operations applied to tensors. Each node is an operation; edges are tensors. PyTorch builds this graph dynamically during the forward pass (define-by-run). When `.backward()` is called, it traverses the graph in reverse to compute gradients via the chain rule.

**Advanced**

14. **Q: Why does batch normalisation act as a regularizer?**
    A: BatchNorm adds noise to each layer's activations (the batch statistics are estimates of the true statistics). This noise acts similarly to dropout, regularising the network. Empirically, networks with BatchNorm often don't need dropout. The regularisation effect disappears at inference (exact running statistics used).

15. **Q: Derive the backward pass for a fully connected layer.**
    A: Forward: Y = XW + b. Backward: given dL/dY, compute: dL/dX = dL/dY @ W^T; dL/dW = X^T @ dL/dY; dL/db = sum(dL/dY, axis=0). This is the heart of backprop — memorise these formulas.

16. **Q: What is the difference between a computational graph and a static graph (TensorFlow 1.x style)?**
    A: Static graph (TF1): graph defined before running; must compile before execution; harder to debug but optimisable. Dynamic graph (PyTorch): graph built on-the-fly during forward pass; easier to debug (standard Python debugging); slightly less optimisable. Modern TF (TF2) uses dynamic graphs by default with optional tracing.

17. **Q: Why does deep learning require normalisation but shallow ML usually doesn't?**
    A: In shallow ML, features are preprocessed once. In deep networks, the distribution of each layer's output changes as parameters update — this is the internal covariate shift problem. Without normalisation, later layers must constantly adapt to shifting input distributions, destabilising training.

18. **Q: What is weight sharing and give an example of where it's used?**
    A: Weight sharing is when the same weights are used at multiple positions. CNNs share convolutional filter weights across all spatial positions (translation invariance). RNNs share recurrent weights across all time steps. Transformers share the embedding matrix between input and output. Weight sharing reduces parameters and introduces useful inductive biases.

19. **Q: What is the Hessian and why can't we use it in deep learning?**
    A: The Hessian is the N×N matrix of second derivatives. For a model with N parameters, storing the Hessian requires O(N²) memory and inverting it (Newton's method) requires O(N³) time. For a 100M parameter model: 10^16 entries — physically impossible. This is why all practical deep learning uses first-order gradient methods.

20. **Q: How do residual connections (skip connections) solve the vanishing gradient problem?**
    A: ResNet: output = F(x) + x. Gradient through the skip connection is exactly 1 — it doesn't vanish. The full gradient is: dL/dx = dL/d(F(x)+x) = dL/dF(x) · dF(x)/dx + dL/dx (the identity). Even if the residual block gradient vanishes, the identity path preserves gradient flow to early layers.

---

### 7. Self-Assessment Quiz

- [ ] Write the cross-entropy loss for a 3-class problem with softmax outputs.
- [ ] What is the derivative of ReLU?
- [ ] What is the derivative of sigmoid, expressed in terms of sigmoid(x)?
- [ ] Why does zero initialisation fail for neural networks?
- [ ] Write the batch normalisation formula: BN(x) = ?
- [ ] What is the difference between model.train() and model.eval() in PyTorch?
- [ ] What is weight decay in the context of neural networks?
- [ ] What activation function does GPT-2 use in its MLP layers?
- [ ] What is the purpose of the learnable parameters γ and β in batch normalisation?
- [ ] Name three causes of training instability in deep networks.
- [ ] What is gradient descent with momentum? Write the update rule.
- [ ] What is the difference between shallow and deep networks theoretically?
- [ ] Why does GELU outperform ReLU in transformers?
- [ ] What is the dying ReLU problem and which activation solves it?
- [ ] Draw the computational graph for y = relu(W2 * relu(W1 * x + b1) + b2).
- [ ] What is the symmetry-breaking problem in neural networks?
- [ ] If your training loss is decreasing but validation loss is increasing, what is happening?
- [ ] What does gradient clipping do to the gradient vector?
- [ ] What is spectral normalisation and when is it used?
- [ ] What is the difference between L-layer networks and skip connections?
- [ ] What is the Universal Approximation Theorem?
- [ ] What is overfitting in a neural network context and what causes it?
- [ ] How does data augmentation help prevent overfitting?
- [ ] What is batch size and how does it affect training dynamics?
- [ ] What is a hyperparameter vs. a learnable parameter?

**Scoring**: 22–25 ✅ = Ready. 17–21 = Review weak areas. Below 17 = Spend more time on Phase 3a.

---

### 8. Common Mistakes

| Mistake | Why It Happens | How to Avoid |
|---------|---------------|--------------|
| Not implementing backprop from scratch | PyTorch does it automatically | Force yourself to implement it manually once — you'll understand autograd much better |
| Using sigmoid in hidden layers of deep networks | Old habit from tutorials | Use ReLU/GELU by default; sigmoid only in output layers for binary classification |
| Zero initialisation for all layers | Seems safe | Always use He (ReLU) or Glorot (tanh/sigmoid) init |
| Forgetting to set model.train() / model.eval() | Easy to miss | BatchNorm and Dropout behave differently in train vs. eval; always switch modes |
| Using BatchNorm but not understanding its training vs. inference difference | Black-box usage | Know that during training it uses batch stats; during eval it uses running stats |
| Not normalising inputs | Works fine in tutorials | Unnormalised inputs cause large gradients and training instability in deep networks |
| Large learning rates with no gradient clipping | Causes NaN loss | Monitor gradient norms in the first few batches; clip if they exceed 1.0 |

---

### 9. Readiness Criteria

You are ready for Phase 3b (PyTorch) when **all** of the following are true:

- [ ] I implemented a 2-layer neural network with forward AND backward pass from scratch in NumPy
- [ ] I can derive the backward pass formulas for a dense layer on paper without reference
- [ ] I can explain vanishing gradients and why ResNets solve them
- [ ] I completed the Micrograd extension mini project
- [ ] I scored 22/25 or higher on the Self-Assessment Quiz
- [ ] I can answer at least 16/20 Interview Questions correctly
- [ ] I understand why PyTorch's `.backward()` is just the chain rule on a computational graph

---

### 10. Revision Summary

```
BACKPROPAGATION
─────────────────────────────────────────────────────
Forward:  compute activations layer by layer: a = activation(Wx + b)
Backward: compute gradients using chain rule in reverse
Dense:    dL/dX = dL/dY @ Wᵀ  |  dL/dW = Xᵀ @ dL/dY  |  dL/db = Σ dL/dY

ACTIVATIONS
─────────────────────────────────────────────────────
Sigmoid:  σ(x) = 1/(1+e⁻ˣ)    derivative: σ(x)(1-σ(x))  MAX = 0.25
Tanh:     tanh(x)               derivative: 1 - tanh²(x)   MAX = 1
ReLU:     max(0, x)             derivative: 0 or 1          NO saturation
GELU:     x·Φ(x)               smooth, used in GPT/BERT

INITIALISATION
─────────────────────────────────────────────────────
Xavier:   std = sqrt(1/fan_in)     best for sigmoid/tanh
He:       std = sqrt(2/fan_in)     best for ReLU (accounts for dead neurons)

NORMALISATION
─────────────────────────────────────────────────────
BatchNorm: normalise across batch dim → stable training, acts as regulariser
LayerNorm: normalise across feature dim → better for NLP/variable-length sequences

REGULARISATION
─────────────────────────────────────────────────────
Dropout:  zero out p fraction of neurons during training; scale by 1/(1-p)
L2:       add λ||w||² to loss; gradients shrink weights toward zero
Early stop: monitor val loss; stop when it stops improving
```

---

### 11. Next Phase Prerequisites

**What Phase 3b (PyTorch) requires from Phase 3a:**

| Phase 3a Concept | How Phase 3b Uses It |
|-----------------|----------------------|
| Computational graph | PyTorch builds exactly this graph during forward pass |
| Chain rule / backprop | When you call `.backward()`, this is what runs |
| BatchNorm/LayerNorm | You'll configure these in `nn.Module` |
| Dropout | `nn.Dropout(p)` — you'll know exactly what it does |
| Weight initialisation | `nn.Linear` default init vs. He init with `nn.init.kaiming_normal_` |
| Vanishing gradient | You'll use residual connections in PyTorch and understand why |

**The critical dependency**: When you call `loss.backward()` in PyTorch, you should be able to mentally simulate what it's computing. Phase 3a gives you that mental model. Without it, PyTorch is a black box — you'll copy code but not know how to debug it.

---

*Phase 3a | Part of the [GenAI Engineer Roadmap](./00_README.md)*
