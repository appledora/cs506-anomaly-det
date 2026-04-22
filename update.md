The web search was not relevant here -- this is purely based on your codebase. Here is the complete guide:

***

## Feature Engineering Guide Per Hypothesis

***

## H1: Geometry Features

### Drop These
- Raw Mahalanobis distance (global covariance -- misleading across classes)
- Raw local density (no per-class normalization)
- Cosine similarity scalar (too coarse)

### Add These

**Step 1.** Extract logits (pre-softmax) from the clean ViT for every sample alongside the CLS embedding.

**Step 2.** Compute **prediction margin**:
```python
logits_sorted = np.sort(logits, axis=1)[:, ::-1]
margin = logits_sorted[:, 0] - logits_sorted[:, 1]  # top1 minus top2
```

**Step 3.** Compute **per-class normalized Mahalanobis**. For each predicted class `c`, compute the mean and covariance of clean embeddings only, then score each sample against its predicted class prototype:
```python
from scipy.spatial.distance import mahalanobis
for c in classes:
    clean_embs_c = clean_embeddings[clean_labels == c]
    mu = clean_embs_c.mean(axis=0)
    cov = np.cov(clean_embs_c.T) + 1e-6 * np.eye(dim)
    VI = np.linalg.inv(cov)
    scores[i] = mahalanobis(emb[i], mu, VI)
```

**Step 4.** Compute **KNN distance ratio**: for each sample, compute its distance to its k=10 nearest clean neighbors in the same predicted class, divided by the median such distance over all clean samples in that class.
```python
ratio = sample_knn_dist / median_clean_knn_dist_for_predicted_class
```

**Final H1 feature vector:** `[margin, per_class_mahal, knn_ratio, pca_reconstruction_error]` -- 4 features.

***

## H2: Uncertainty Features

### Drop These
- Raw softmax entropy (redundant with confidence)
- KL divergence from uniform (monotone transform of entropy)
- Margin (already in H1)

Keep only `max_confidence` as the single base uncertainty scalar, then add genuinely distinct features.

### Add These

**Step 1.** Compute **top-3 entropy** (entropy restricted to top-3 logits only, renormalized):
```python
top3 = np.sort(probs, axis=1)[:, -3:]
top3 = top3 / top3.sum(axis=1, keepdims=True)
top3_entropy = -np.sum(top3 * np.log(top3 + 1e-9), axis=1)
```

**Step 2.** Compute **MC Dropout uncertainty** using the existing noisy ViT -- no new model needed. Enable dropout at inference and run 20 forward passes:
```python
model.train()  # enables dropout
with torch.no_grad():
    preds = np.stack([softmax(model(x)) for _ in range(20)], axis=0)
mc_mean = preds.mean(axis=0)
mc_variance = preds.var(axis=0).sum(axis=1)   # total predictive variance
mc_entropy = -np.sum(mc_mean * np.log(mc_mean + 1e-9), axis=1)
```

**Step 3.** Compute **confidence gap relative to class baseline**: for each sample with predicted class `c`, subtract the mean clean confidence for class `c`:
```python
class_mean_conf = {c: clean_probs[clean_labels==c, c].mean() for c in classes}
conf_gap = probs[i, pred_class[i]] - class_mean_conf[pred_class[i]]
```

**Final H2 feature vector:** `[max_confidence, top3_entropy, mc_variance, mc_entropy, conf_gap]` -- 5 features, genuinely non-redundant.

***

## H3: Training Dynamics

### The Blocker to Fix First

The noisy ViT ran for only 1 effective epoch. **Before changing any features**, fix the training loop:

```python
# In trainNoisyViTWithDynamics():
# Option A (preferred): separate early stopping from dynamics logging
#   - run for fixed 30 epochs, always log dynamics
#   - use early stopping only to select best weights for inference, not to halt logging

# Option B: increase patience dramatically
early_stopping = EarlyStopping(patience=25, min_delta=1e-4)
num_epochs = 50  # must run long enough for forgetting events to occur
```

### Drop These (currently degenerate with 1 epoch)
- `loss_slope` (requires at least 5 epochs to fit a slope)
- `forgetting_count` (requires at least 2 epochs to observe a flip)
- `correctness_count` (trivially 0 or 1 with 1 epoch)

### Add These (after fixing epoch count)

**Step 1.** Compute **variability** (std of correctness across epochs -- the core Dataset Cartography signal):
```python
variability = correct_trajectories.std(axis=1)  # shape: [n_samples]
```

**Step 2.** Compute **confidence mean and std** across epochs:
```python
conf_mean = conf_trajectories.mean(axis=1)
conf_std = conf_trajectories.std(axis=1)
```

**Step 3.** Compute **forgetting count** properly (number of times a sample flips from correct to incorrect across consecutive epochs):
```python
flips = np.diff(correct_trajectories.astype(int), axis=1)
forgetting_count = (flips == -1).sum(axis=1)
```

**Step 4.** Compute **loss AUC** (area under loss curve -- high for near-miss, low and decreasing for clean-hard):
```python
from numpy import trapz
loss_auc = np.array([trapz(loss_trajectories[i]) for i in range(n)])
```

**Final H3 feature vector:** `[conf_mean, conf_std, variability, forgetting_count, loss_auc]` -- 5 features. Near-miss samples should show high variability + high forgetting; clean-hard should show low variability + consistently low confidence.

***

## H4a: Image Quality (Corrupted)

### Drop These
- `iq_blur` computed at 32x32 (Laplacian variance is near-zero at this resolution)
- `iq_contrast` as a raw std (not normalized)
- `hf_ratio` (inverts direction for noise vs. blur corruptions)

### Add These

**Step 1.** Upsample to 224x224 before computing all quality features:
```python
img_224 = cv2.resize(img_32, (224, 224), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img_224, cv2.COLOR_RGB2GRAY).astype(float)
```

**Step 2.** Compute **multi-scale Laplacian variance** at 3 scales:
```python
scales = [gray, cv2.pyrDown(gray), cv2.pyrDown(cv2.pyrDown(gray))]
lap_vars = [cv2.Laplacian(s, cv2.CV_64F).var() for s in scales]
# features: lap_var_scale0, lap_var_scale1, lap_var_scale2
```

**Step 3.** Compute **noise residual** (image minus median-filtered version):
```python
from scipy.ndimage import median_filter
noise_residual = gray - median_filter(gray, size=3)
noise_energy = noise_residual.var()
```

**Step 4.** Compute **MSCN kurtosis** (standard no-reference IQA -- sensitive to noise and blur):
```python
from scipy.stats import kurtosis
mu = cv2.GaussianBlur(gray, (7,7), 1.5)
sigma = np.sqrt(cv2.GaussianBlur((gray - mu)**2, (7,7), 1.5)) + 1e-6
mscn = (gray - mu) / sigma
mscn_kurtosis = kurtosis(mscn.ravel())
```

**Final H4a feature vector:** `[lap_var_scale0, lap_var_scale1, lap_var_scale2, noise_energy, mscn_kurtosis]` -- 5 features.

***

## H4b: Label Agreement (Ambiguous)

### Keep As-Is (with a caveat)
`label_human_confidence` and `label_conf_gap` are correct and work (F1=0.964). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/28463064/e77ebabe-a814-4c0e-a3aa-0ce1020aeae9/write.txt)

### One Change
Drop `label_human_available` -- it is a binary flag that is `True` for all ambiguous samples by construction and carries zero information beyond class membership. It is a data leakage artifact.

```python
# Remove from LABEL_COLS:
LABEL_COLS = ['label_human_confidence', 'label_conf_gap']  # drop label_human_available
```

***

## Consolidated Feature Column Reference

| Hypothesis | Final Features | Count |
|---|---|---|
| H1 Geometry | `margin`, `per_class_mahal`, `knn_ratio`, `pca_recon_err` | 4 |
| H2 Uncertainty | `max_conf`, `top3_entropy`, `mc_variance`, `mc_entropy`, `conf_gap` | 5 |
| H3 Dynamics | `conf_mean`, `conf_std`, `variability`, `forgetting_count`, `loss_auc` | 5 |
| H4a Quality | `lap_var_s0`, `lap_var_s1`, `lap_var_s2`, `noise_energy`, `mscn_kurtosis` | 5 |
| H4b Label | `label_human_confidence`, `label_conf_gap` | 2 |


## Section 2.2: Fix the Noisy ViT Training Loop (H3 blocker)

This is the most critical change. Find `trainNoisyViTWithDynamics` and make these edits:

**1. Decouple early stopping from dynamics logging.**
```python
# CHANGE: increase patience so dynamics run long enough
early_stopping = EarlyStopping(patience=25, min_delta=1e-4, verbose=True)
num_epochs = 50  # was 50 but early stopping killed it at epoch 1 -- now patience=25 ensures ~25+ epochs run

# CHANGE: never trim trajectories based on early stopping
# REMOVE these lines entirely:
# loss_trajectories = loss_trajectories[:, :actual_best]
# conf_trajectories = conf_trajectories[:, :actual_best]
# correct_trajectories = correct_trajectories[:, :actual_best]
# Keep full trajectories regardless of when early stopping triggers
```

**2. Initialize trajectory arrays for the full epoch count, not just best epoch.**
```python
# CHANGE: always allocate for full num_epochs
loss_trajectories = np.zeros((n_train, num_epochs))
conf_trajectories = np.zeros((n_train, num_epochs))
correct_trajectories = np.zeros((n_train, num_epochs), dtype=bool)
```

**3. Save the full dynamics, not trimmed.**
```python
# CHANGE: save after training loop ends regardless of early stopping trigger
np.savez(DYNAMICS_PATH,
         loss=loss_trajectories,
         conf=conf_trajectories,
         correct=correct_trajectories)
```

***

## Section 2.3: Replace / Add Feature Extraction

### H1 Geometry -- replace `extract_geometry_features()`

```python
# REMOVE: raw global mahalanobis, raw local density, raw cosine scalar

# ADD: per-class normalized Mahalanobis
def per_class_mahal(emb, pred_class, clean_embs_by_class):
    mu = clean_embs_by_class[pred_class]['mean']
    VI = clean_embs_by_class[pred_class]['inv_cov']
    return mahalanobis(emb, mu, VI)

# ADD: prediction margin from logits (requires saving logits, not just embeddings)
# In your ViT forward pass, also return logits:
margin = logits_sorted[:, 0] - logits_sorted[:, 1]

# ADD: KNN distance ratio
from sklearn.neighbors import NearestNeighbors
# fit NearestNeighbors on clean embeddings per class, compute ratio at inference

# KEEP: pca_reconstruction_error (it is fine)
```

You will need to modify your ViT forward pass to **return both CLS embeddings AND logits**. Currently you only extract the CLS token. Add one line:

```python
# In your embedding extraction loop:
with torch.no_grad():
    features = model.forward_features(imgs)       # CLS embeddings
    logits = model.head(features[:, 0])            # logits from head
```

### H2 Uncertainty -- replace `extract_uncertainty_features()`

```python
# REMOVE: entropy (redundant with max_conf)
# REMOVE: kl_divergence_from_uniform (monotone transform of entropy)
# REMOVE: margin (already in H1 geo features)

# KEEP: max_confidence

# ADD: top3_entropy
top3 = np.sort(probs, axis=1)[:, -3:]
top3 = top3 / top3.sum(axis=1, keepdims=True)
top3_entropy = -np.sum(top3 * np.log(top3 + 1e-9), axis=1)

# ADD: MC Dropout -- run at inference using existing noisy ViT
model.train()  # enables dropout
mc_preds = []
with torch.no_grad():
    for _ in range(20):
        mc_preds.append(torch.softmax(model(imgs), dim=1).cpu().numpy())
mc_preds = np.stack(mc_preds, axis=0)  # (20, N, C)
mc_variance = mc_preds.var(axis=0).sum(axis=1)
mc_entropy = -np.sum(mc_preds.mean(axis=0) * np.log(mc_preds.mean(axis=0) + 1e-9), axis=1)
model.eval()

# ADD: confidence gap relative to class baseline
# Compute mean clean confidence per class first, then subtract
```

### H3 Dynamics -- replace `extract_dynamics_features()`

```python
# REMOVE: loss_slope (was meaningless with 1 epoch)
# REMOVE: correctness_count (trivially 0 or 1 with 1 epoch)
# REMOVE: forgetting_count as previously computed (needs recomputing after fix above)

# After the training fix, recompute all dynamics from the full trajectories:
conf_mean = conf_trajectories.mean(axis=1)
conf_std  = conf_trajectories.std(axis=1)
variability = correct_trajectories.std(axis=1)   # Dataset Cartography signal
flips = np.diff(correct_trajectories.astype(int), axis=1)
forgetting_count = (flips == -1).sum(axis=1)
loss_auc = np.trapz(loss_trajectories, axis=1)

# KEEP: loss_auc (concept is right, just needs real trajectory data)
```

### H4a Image Quality -- replace `extract_image_quality_features()`

```python
# REMOVE: iq_blur at 32x32
# REMOVE: iq_contrast (raw std)
# REMOVE: hf_ratio

# ADD: upsample first, then compute all features at 224x224
img_224 = cv2.resize(img_32, (224, 224), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img_224, cv2.COLOR_RGB2GRAY).astype(np.float64)

# Multi-scale Laplacian
lap_vars = []
s = gray.copy()
for _ in range(3):
    lap_vars.append(cv2.Laplacian(s, cv2.CV_64F).var())
    s = cv2.pyrDown(s)

# Noise residual
from scipy.ndimage import median_filter
noise_energy = (gray - median_filter(gray, size=3)).var()

# MSCN kurtosis
from scipy.stats import kurtosis as sp_kurtosis
mu_g = cv2.GaussianBlur(gray, (7,7), 1.5)
sigma_g = np.sqrt(cv2.GaussianBlur((gray - mu_g)**2, (7,7), 1.5)) + 1e-6
mscn = (gray - mu_g) / sigma_g
mscn_kurtosis = sp_kurtosis(mscn.ravel())
```

### H4b Label Agreement -- one-line change

```python
# REMOVE: label_human_available from LABEL_COLS
LABEL_COLS = ['label_human_confidence', 'label_conf_gap']
```
