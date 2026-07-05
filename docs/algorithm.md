# Algorithm Notes

MiniXGBoost implements regularized, second-order gradient tree boosting for
squared-error regression and binary logistic classification.

## Additive Model

At boosting round `t`, the raw prediction is updated by a new tree:

```text
y_hat_i^(t) = y_hat_i^(t-1) + eta * f_t(x_i)
```

`eta` is the learning rate. Regression returns the raw score; binary
classification passes it through a numerically stable sigmoid.

## Gradients And Hessians

For squared error:

```text
g_i = y_hat_i - y_i
h_i = 1
```

For binary logistic loss, with `p_i = sigmoid(y_hat_i)`:

```text
g_i = p_i - y_i
h_i = p_i * (1 - p_i)
```

The implementation floors logistic Hessians to avoid numerical collapse for
very confident predictions.

## Regularized Leaf Weight

For a leaf containing gradient sum `G` and Hessian sum `H`, the optimal weight
with L1 (`alpha`) and L2 (`lambda`) regularization is:

```text
w* = -soft_threshold(G, alpha) / (H + lambda)
```

where:

```text
soft_threshold(G, alpha) = sign(G) * max(abs(G) - alpha, 0)
```

## Split Gain

For left and right child statistics `(G_L, H_L)` and `(G_R, H_R)`, the split
gain is:

```text
gain = 0.5 * [score(L) + score(R) - score(parent)] - gamma
score(G, H) = soft_threshold(G, alpha)^2 / (H + lambda)
```

Splits must also satisfy `min_samples_leaf` and `min_child_weight`.

## Exact Greedy Search

For each sampled feature, non-missing values are sorted. The algorithm scans
adjacent distinct values, updates prefix gradient/Hessian sums, and evaluates a
threshold at their midpoint. Complexity for one node is approximately:

```text
O(number_of_features * number_of_rows * log(number_of_rows))
```

Feature searches are distributed across at most `n_jobs` asynchronous workers.
Tie-breaking is deterministic, so serial and parallel training produce the same
model for the same seed.

## Missing Values

Missing feature values are represented by `NaN`. Every candidate threshold is
evaluated twice: once with missing rows sent left and once with them sent right.
The better direction is stored in the node and reused during prediction.

## Early Stopping

The model monitors validation loss when a validation set is supplied, otherwise
training loss. After `early_stopping_rounds` without improvement, trees and
metric histories are restored to the best iteration.

## Model Format

The text format stores a magic header and version, prediction-time parameters,
the base score, and every tree node. The loader rejects unsupported versions and
truncated tree data. The format is intentionally readable and portable rather
than optimized for size.
