# Anomaly Type Detection in Noisy Image Classification Datasets

## Overview

Modern image classifiers are trained on datasets containing heterogeneous problematic examples: truly clean images, hard-but-correct samples, near-miss label errors, gross label errors, and out-of-distribution (OOD) images. These distinct "anomaly types" are typically treated as a homogeneous "noisy" category, despite having different signatures in model behavior. This project attempts to disentangle these anomaly types by generating a synthetic dataset with controlled proportions of each type, extracting a comprehensive set of features (geometric, uncertainty, training dynamics, image quality, and human annotation metadata), and testing hypotheses about which features distinguish which anomaly types. 

**Dataset Size**: 100,000 samples (50k CIFAR-10, 50k CIFAR-100)  
**Models**: Two ViT-B/16 models (Clean ViT, Noisy ViT)  
**Features**: 20 features across 5 families (Geometry, Image Quality, Uncertainty, Dynamics, Label Agreement)  




---

## Dataset Generation

### Data Sources

- **CIFAR-10/100**: Original images and labels from CIFAR repository
- **CIFAR-N**: Human reannotation data (CIFAR-10-N, CIFAR-100-N) for ambiguous samples
- **MIT Indoor Scenes**: 67 indoor scene categories for OOD injection
- **Pretrained ViT-B/16**: Feature extractor for semantic similarity computations

![alt text](image-5.png)
## Synthetic Dataset Composition

![alt text](image-7.png)
### 1. Clean (35%)
Correctly labeled samples serving as the ground truth baseline. These represent ideal training data with aligned image content and assigned label.
![alt text](image.png)
### 2. Near-Miss Label Errors (15%)
Semantically plausible but incorrect labels.
![alt text](data_vit/near_miss_samples.png)
For each sample, extract ViT embedding and compute cosine similarity to all class centroids. Replace true label with one of the k=3 nearest classes (excluding true class).

**Examples**: dog → cat, truck → automobile, deer → horse

**Rationale**: Simulates annotator confusion between visually similar categories. These errors are the hardest to detect because the mislabeled class is a reasonable alternative interpretation.

### 3. Gross Label Errors (10%)
Semantically implausible label flips.
![alt text](data_vit/gross_samples.png)
For each sample, select from the k=3 furthest classes in ViT embedding space.

**Examples**: frog → truck, bird → automobile, cat → ship

**Rationale**: Represents catastrophic annotation failures (copy-paste errors, wrong image-label pairing, database corruption). Should be easily detectable via geometric features.

### 4. Out-of-Distribution (5%)
Images that do not belong to the CIFAR domain at all.
![alt text](data_vit/ood_samples.png)
Sample images from MIT Indoor Scenes dataset (67 categories: corridor, restaurant, bedroom, etc.) and randomly assign CIFAR labels.

**Examples**: Indoor scene labeled as "airplane", office labeled as "dog"

**Rationale**: Tests whether models can detect domain shift. These samples should have embeddings far from all CIFAR class centroids.

### 5. Clean-Hard (15%)
Correctly labeled but atypically difficult samples.
![alt text](data_vit/clean_hard_samples.png)
Compute cosine distance from each sample to its true class centroid in ViT space. Select the top 15% most distant samples within each class while preserving correct labels.

**Examples**: Occluded objects, unusual viewpoints, edge cases within a category

**Rationale**: Represents legitimate boundary cases. Critical for testing whether features can distinguish inherent difficulty from label errors. Clean-Hard and Near-Miss are the key challenge: both are near decision boundaries, but only one is mislabeled.

### 6. Random Flip (10%)
Uniform random label assignment, independent of image content.

Replace true label with a random class sampled uniformly from the label set.

**Examples**: Any class → any other class with equal probability

**Rationale**: Represents annotation noise with no semantic structure. Should produce high uncertainty across all samples.

### 7. Ambiguous (5%)
Samples where human annotators disagree.
![alt text](data_vit/human_disagreements.png)
Identify samples from CIFAR-N datasets where multiple annotators provided different labels. Assign one of the human-provided labels (not necessarily the original CIFAR label).

**Examples**: Images that could reasonably be labeled as multiple classes

**Rationale**: Captures genuine label uncertainty. These samples may not be "errors" but rather reflect intrinsic ambiguity. Should have low human confidence scores and high model uncertainty.

### 8. Corrupted (5%)
Visually degraded images with correct labels.
![alt text](data_vit/corrupted_samples.png)
Apply one of four corruptions randomly:
- Gaussian noise: σ=25
- Gaussian blur: σ=1.5
- Contrast reduction: 0.3× original
- Brightness reduction: 0.3× original

**Examples**: Blurry cat (labeled cat), noisy airplane (labeled airplane)

**Rationale**: Tests whether models conflate low image quality with label errors. Should be distinguishable via pixel-level quality metrics independent of semantic features.
![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)
![alt text](image-4.png)
---

## Proposed Hypotheses

The study tests four hypotheses about which feature families distinguish which anomaly types:

### H1: Geometric Features Detect Domain and Magnitude Errors
Geometric features (embedding space structure) separate OOD and Gross errors from the rest, but fail to distinguish Near-Miss from Clean-Hard. The experiments would validate that embedding space captures domain shift and large semantic gaps, but semantic plausibility makes Near-Miss indistinguishable from Clean-Hard in geometry alone.

### H2: Uncertainty Features Separate Medium-Tier Anomalies

Model confidence and entropy distinguish Random Flip and Gross errors from Clean and Near-Miss, but confuse Near-Miss with Clean-Hard. The experiments would show that uncertainty captures the model's lack of confidence on implausible labels (Random Flip, Gross), but cannot resolve the ambiguity between mislabeled but plausible samples (Near-Miss) and legitimately hard samples (Clean-Hard).

### H3: Training Dynamics Break Near-Miss/Clean-Hard Degeneracy

Temporal learning patterns distinguish Clean-Hard (initially learned, then stable) from Near-Miss (learned then forgotten, or never stably learned). The experiments would analyze loss and confidence trajectories during training to show that Clean-Hard samples are consistently learned and retained, while Near-Miss samples exhibit unstable learning patterns (initial memorization followed by rejection, or oscillating correctness). This would confirm that training dynamics provide a unique signal for mislabeling that geometry and uncertainty cannot resolve.


### H4: Dedicated Signals for Corrupted and Ambiguous

Image quality metrics separate Corrupted; human confidence separates Ambiguous. The experiments would validate that pixel-level features (blur, contrast) effectively identify Corrupted samples regardless of label correctness, while human annotation metadata (confidence scores, disagreement rates) uniquely identify Ambiguous samples that are inherently uncertain even to humans.

---

## Feature Extraction

**Clean ViT**: Trained only on clean CIFAR samples. Provides reference geometry in a "ground truth" embedding space
**Noisy ViT**: Trained on all 8 anomaly types. Provides uncertainty estimates shaped by label noise exposure


## Features

### Geometric Features (Clean ViT) - 5 features

#### 1. `geo_cos_assigned`
Cosine similarity between sample embedding and assigned class centroid.


- **High values** (>0.8): Sample is semantically coherent with assigned label (likely Clean or Clean-Hard)
- **Medium values** (0.5-0.8): Sample is somewhat related to assigned label (Near-Miss candidate)
- **Low values** (<0.5): Sample is unrelated to assigned label (Gross or OOD)



#### 2. `geo_rank_assigned`
Rank of assigned class among all classes when sorted by centroid similarity (1 = closest, 10 or 100 = furthest).

- **Rank 1**: Assigned label is the most semantically similar class → Clean or Clean-Hard
- **Rank 2-3**: Assigned label is close but not closest → Near-Miss candidate (true class might be rank 1)
- **Rank > 5**: Assigned label is far from semantically reasonable → Gross or Random Flip


#### 3. `geo_mahal_assigned`
Mahalanobis distance to assigned class distribution (accounts for class covariance, not just centroid).

- **Low values**: Sample is typical for assigned class (Clean)
- **Medium values**: Sample is atypical but within class distribution (Clean-Hard)
- **High values**: Sample is outside class distribution (Near-Miss, Gross, OOD)



#### 4. `geo_ratio_assigned_neighbor`
Ratio of similarity to assigned class vs. nearest other class.

- **Ratio > 1**: Assigned class is closer than any other class → Clean
- **Ratio ≈ 1**: Assigned class and nearest neighbor are equally close → Near-Miss (boundary case)
- **Ratio < 1**: Another class is closer than assigned class → Mislabeled


#### 5. `geo_local_density`
Number of samples within radius threshold in embedding space (k-nearest neighbor density).

- **High density**: Sample is in a dense region of embedding space → Typical sample (Clean, Clean-Hard)
- **Low density**: Sample is in a sparse region → Atypical (OOD, outlier within class)

OOD samples from MIT Scenes should have very low density because they're isolated from CIFAR clusters.


---

### Image Quality Features (Pixel-level) - 4 features

Detect visual corruption independent of label correctness. Tests H4.

#### 1. `iq_blur`
Laplacian variance (edge sharpness).

- **High values** (>2000): Sharp image → Clean, Near-Miss, Gross, etc.
- **Low values** (<500): Blurry image → Corrupted (blur corruption applied)

Laplacian variance directly measures high-frequency content. Blurred images lose edge detail.


#### 2. `iq_contrast`
Standard deviation of pixel intensities.

- **High values** (>40): High contrast → Normal image
- **Low values** (<20): Low contrast → Corrupted (contrast reduction applied)

Low-contrast images have pixel values concentrated in a narrow range.


#### 3. `iq_hf_ratio`
Ratio of high-frequency to low-frequency energy in FFT spectrum.

- **High values** (>0.4): Rich high-frequency content → Sharp, detailed image
- **Low values** (<0.2): Suppressed high-frequencies → Blurred or smoothed image



#### 4. `iq_brightness`
Mean pixel intensity.

**Why it matters**:
- **High values** (>150): Bright image → Normal
- **Low values** (<80): Dark image → Corrupted (brightness reduction applied)

---

### Uncertainty Features (Noisy ViT) - 6 features

**Purpose**: Capture model confidence and prediction stability. Tests H2.

#### 1. `unc_confidence`
Maximum softmax probability (model's certainty in its prediction).

- **High values** (>0.9): Model is confident → Clean, Clean-Hard
- **Medium values** (0.5-0.9): Model is uncertain → Near-Miss, Ambiguous
- **Low values** (<0.5): Model is very uncertain → Random Flip, Gross

Models trained on noisy data learn to be less confident on mislabeled samples (though this can also occur for legitimately hard samples).


#### 2. `unc_entropy`
Shannon entropy of prediction distribution (spread of probability mass).

- **Low values** (0-0.2): Peaked distribution (confident prediction) → Clean
- **High values** (>1.0): Flat distribution (uniform uncertainty) → Random Flip, very hard samples

Entropy measures how "spread out" the probability distribution is. Random Flip samples should have near-uniform distributions.

#### 3. `unc_margin`
Difference between top-1 and top-2 predicted class probabilities.

- **High margin** (>0.7): Clear winner (confident) → Clean
- **Low margin** (<0.3): Top-2 classes have similar probabilities → Near-Miss, Ambiguous

Near-Miss samples often have the true class as the second-highest prediction, resulting in low margin.


#### 4. `unc_model_disagree`
Binary indicator if Clean ViT and Noisy ViT predictions differ.

- **Disagree (1)**: Clean model (trained on clean data) predicts differently than Noisy model → Sample is likely noisy
- **Agree (0)**: Both models agree → Sample is likely clean

Clean ViT hasn't seen label noise, so it predicts based on true semantic structure. Noisy ViT has adapted to noise. Disagreement suggests the sample is in a noisy region.


#### 5. `unc_kl_divergence`
KL divergence from Clean ViT distribution to Noisy ViT distribution.

- **Low KL** (<0.2): Similar distributions → Models agree on semantics → Clean or Clean-Hard
- **High KL** (>1.0): Divergent distributions → Models disagree → Noisy sample

Continuous version of `unc_model_disagree`. Captures degree of disagreement, not just binary mismatch.


#### 6. `unc_topk_entropy`
Entropy computed over only top-k classes (k=3).

 Regular entropy can be high simply because there are many classes. Top-k entropy focuses on whether the model is uncertain among the most likely classes specifically. Near-Miss samples should have high top-k entropy because the true class and assigned class are both in top-k.

---

### Training Dynamics Features (Noisy ViT) - 4 features

Capture temporal learning patterns during training. Tests H3.

Clean-Hard samples are learned early and retained (stable). Near-Miss samples are either learned then forgotten (memorization followed by rejection), or never stably learned (oscillating correctness).

#### 1. `dyn_forgetting_events`
Number of times a sample transitions from correct → incorrect prediction during training.

- **Zero forgetting**: Sample is either always correct (Clean) or always wrong (very hard/mislabeled)
- **High forgetting** (≥2): Sample is learned then forgotten repeatedly → Mislabeled

Models initially memorize all samples (including mislabeled ones), but as training progresses, they "forget" samples that don't fit the learned patterns. Near-Miss samples are semantically plausible enough to be memorized, but eventually rejected as inconsistent.


#### 2. `dyn_late_loss`
Cross-entropy loss at final epoch (or mean over last 10 epochs).

- **Low late loss** (<0.5): Model has learned this sample → Clean, Clean-Hard
- **High late loss** (>2.0): Model never learned this sample → Mislabeled or extremely hard

Samples with high late loss are those the model "gave up on" after many epochs. These are strong candidates for mislabeling.


#### 3. `dyn_learning_improvement`
Reduction in loss from early epochs to late epochs.

- **Positive improvement**: Sample was learned → Clean, Clean-Hard
- **Zero/negative improvement**: No learning progress → Mislabeled

Near-Miss samples may show initial improvement (memorization) followed by increase (forgetting), resulting in low net improvement.

#### 4. `dyn_cartography_confidence`
Mean model confidence across all training epochs (from Dataset Cartography paper).

- **High mean confidence** (>0.8): "Easy to learn" samples → Clean
- **Medium mean confidence** (0.5-0.8): "Ambiguous" samples → Near-Miss, Ambiguous
- **Low mean confidence** (<0.5): "Hard to learn" samples → Random Flip, Gross

Complements `unc_confidence` by averaging over training history rather than using final-epoch value.

---

### Label Agreement Features (Metadata) - 2 features

 Leverage external signals orthogonal to model behavior.

#### 1. `label_human_confidence`
Agreement rate among human annotators from CIFAR-N.

- **High confidence** (>0.8): Humans agree → Clear label → Clean
- **Medium confidence** (0.5-0.8): Some disagreement → Ambiguous
- **Low confidence** (<0.5): High disagreement → Ambiguous or very difficult

Only available for samples in CIFAR-N. Tests H4 directly.

#### 2. `label_conf_gap`
Difference between model confidence on synthetic label vs. original CIFAR label.

- **Positive gap**: Model prefers synthetic label → Clean or Near-Miss (if synthetic is correct)
- **Negative gap**: Model prefers original CIFAR label → Gross or Random Flip (synthetic is implausible)

This feature is the single strongest discriminator in practice because it directly compares model preference for the assigned label vs. the ground truth.


---
# Hypothesis Testing and Results


**Metrics**: Per-class F1-score , macro F1, confusion matrix , feature importance 

Classifiers:  Random Forest, XGBoost, Logistic Regression, SVM, KNN. All models were evaluated using 5-fold stratified cross-validation with consistent preprocessing (StandardScaler normalization) and random seed (42) for reproducibility.


### H1: Geometric Features Detect Domain-Level and Magnitude Errors

**Original Hypothesis**: Geometric features (embedding space structure from Clean ViT) should detect OOD  and Gross errors, but fail on Near-Miss and Clean-Hard.

**Actual Results**:
- **OOD**: F1 = **0.008**  → **CATASTROPHIC FAILURE**
- **Gross**: F1 = **0.444**  → **MODERATE FAILURE**
- **Near-Miss**: F1 = 0.601  → Better than predicted
- **Clean-Hard**: F1 = 0.046  → Consistent with hypothesis

The classifier cannot distinguish "low similarity because image is from wrong domain" from "low similarity because label is wrong but image is still CIFAR."

### H2: Uncertainty Features Separate Medium-Tier Anomalies

**Original Hypothesis**: Model confidence and entropy distinguish Random Flip  and Gross, but confuse Near-Miss with Clean-Hard.

**Actual Results**:
- **Random Flip**: F1 = **0.243**→ **UNDERPERFORMED**
- **Gross**: F1 = **0.444** → **UNDERPERFORMED**
- **Near-Miss**: F1 = **0.601**  → **OUTPERFORMED**
- **Clean-Hard**: F1 = **0.046** → Consistent

**Verdict**: ✓ **PARTIALLY SUPPORTED**


Uncertainty features performed better on Near-Miss than predicted, suggesting they **can** resolve some semantic ambiguity when combined with other features. However, they underperformed on Random Flip and Gross, likely because:

**Feature Importance**:
```
Top uncertainty features:
  unc_kl_divergence:    5.3% importance
  unc_topk_entropy:     5.1%
  unc_entropy:          4.7%
  unc_margin:           4.6%
  unc_confidence:       4.4%
```


---

### H3: Training Dynamics Break Near-Miss/Clean-Hard Degeneracy

**Original Hypothesis**: Temporal learning patterns (forgetting events, late loss) distinguish Near-Miss (unstable learning, F1 gap > 0.20 vs Clean-Hard) from Clean-Hard (stable learning).

**Actual Results**:
- **Near-Miss**: F1 = **0.601** (multiclass)
- **Clean-Hard**: F1 = **0.046** (multiclass)
- **Gap**: 0.555→ **GAP EXISTS BUT...**

**Feature Importance**:
```
dyn_forgetting_events:       0.000% 
dyn_late_loss:               2.5%
dyn_learning_improvement:    0.000%
dyn_cartography_confidence:  1.3%
```

**Verdict**: ❌ **HYPOTHESIS REJECTED**


The Noisy ViT training was terminated at **epoch 9** via early stopping (patience=7, best epoch=2). This created two critical problems:

1. **No forgetting events recorded**: With only 9 epochs, samples didn't have time to transition from correct → incorrect → correct, so `dyn_forgetting_events` is zero for all samples.

2. **Dynamics features are constants**: Features like `dyn_learning_improvement` and `dyn_forgetting_events` show zero variance across samples, making them useless for classification.

### H4: Specialized Features for Corrupted and Ambiguous

#### H4a: Corrupted Detection via Image Quality

**Original Hypothesis**: Image quality metrics (blur, contrast, high-frequency ratio) detect Corrupted samples with F1 > 0.80.

**Actual Results**:
- **Corrupted**: F1 = **0.001**→ **COMPLETE FAILURE**

**Feature Importance**:
```
iq_blur:       5.0%
iq_contrast:   5.0%
iq_hf_ratio:   5.0%
iq_brightness: (not in top features)
```

**Verdict**: ❌ **HYPOTHESIS REJECTED**


**histogram overlap analysis** between Clean and Corrupted samples for the four image quality features revealed near-complete distribution overlap :

| Feature | Clean μ±σ | Corrupted μ±σ | Overlap % |
|---------|-----------|---------------|-----------|
| iq_blur | 2377±1519 | 2342±1491 | **90.1%** |
| iq_contrast | 49.9±15.6 | 49.2±15.5 | **89.8%** |
| iq_hf_ratio | 0.47±0.11 | 0.47±0.11 | **90.1%** | 
| iq_brightness | 123.6±34.3 | 123.2±34.8 | **89.3%** | 

The applied corruptions (σ=25 for blur, 0.3× contrast/brightness) were too mild to create a distinguishable signal at 32×32 resolution. The distributions of image quality features for Corrupted samples are almost identical to Clean samples, leading to near-zero F1.
![alt text](image-8.png)
#### H4b: Ambiguous Detection via Human Confidence

**Original Hypothesis**: Human annotation disagreement (`label_human_confidence`) detects Ambiguous samples with F1 > 0.70.

**Actual Results**:
- **Ambiguous**: F1 = **0.715** (expected >0.70) → ✓ **SUCCESS**

**Feature Importance**:
```
label_human_confidence: 6.6% (rank 2)
```

**Verdict**: ✓ **HYPOTHESIS CONFIRMED**

This is the **only hypothesis that succeeded as predicted**. Human annotation disagreement is a strong signal. External metadata (human annotations) provides orthogonal information to model-derived features, validating the value of CIFAR-N datasets for anomaly detection research.

---

## Combined Model Performance

### Overall Results

**Macro F1**: **0.3518**

**Per-Class F1 (Actual vs Expected)**:

| Anomaly Type | Actual F1 | 
|--------------|-----------|
| Clean | **0.756** |
| Ambiguous | **0.715** | 
| Near-Miss | **0.601** | 
| Gross | **0.444** |
| Random Flip | **0.243** | 
| Clean-Hard | **0.046** |
| OOD | **0.008** | 
| Corrupted | **0.001** |

## Conclusion

We tested a 4-hypothesis framework for detecting 8 anomaly types in noisy image classification datasets. Of the 4 hypotheses, **only 1 fully succeeded** (H4b - Ambiguous via human confidence), while the other 3 failed due to systematic design flaws:

- **H1 (Geometry)**: Failed due to label dependency in features
- **H3 (Dynamics)**: Failed due to insufficient training epochs  
- **H4a (Corrupted)**: Failed due to imperceptible corruption at 32×32 resolution


