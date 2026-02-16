## A Multi-Signal Study of Data and Label Anomalies in Deep Classification

Modern image classifiers are trained on datasets that contain many kinds of problematic examples: truly clean images, hard-but-correct images, near-miss label errors (e.g., dog labeled as cat), gross label errors (e.g., frog labeled as truck), and out-of-distribution (OOD) images that do not belong in the dataset at all. These different “anomaly types” are usually treated as one homogeneous category (“noisy” or “outlier”), even though they likely have very different signatures in model behavior. In this project, the goal is to have empirical and theoratical findings on how different anomaly types look from the perspective of a trained model, and which features are most useful for distinguishing between them.

### Project Outline 

1. Dataset: Adapt a labeled dataset on CIFAR‑10/100 (or utilize existing anomaly dataset like AlleNoise) where each training image is categorized into one of several anomaly types:
    - Clean: A clean image with a correct label.
    - Near-miss label noise : Flip label to semantically similar class based on predefined pairs or embedding distance (e.g., dog→cat, truck→car)
    - Gross label noise: Flip label to semantically distant class based on predefined pairs or embedding distance (e.g: Frog image → originally "frog" → flip to farthest class → now labeled "airplane")
    - Out-of-Distribution (OOD): Replace image with an OOD image (e.g., from a different dataset like SVHN or random noise) and assign it a random label from the original dataset. 
    - Clean but Hard: A clean image that is difficult for the model to classify correctly (e.g., low confidence, high loss, or misclassified by a baseline model).
    - Random Flip: Randomly flip the label to any other class with equal probability, regardless of semantic similarity.

    > Data sources: 
    - Allenoise: https://github.com/allegro/AlleNoise 
    - CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html
    - CIFAR-10-100-N: https://github.com/UCSC-REAL/cifar-10-100n


2. **Feature Extraction:**  To extract all feature types, I will train two separate ViT models. 
    - Clean ViT: Train a ViT model on a clean subset of the dataset (e.g., only clean images) to extract features that are not influenced by anomalies. Provides "clean" class prototypes and embeddings for geometry features

    - Noisy ViT: Train a ViT model on the full dataset (including all anomaly types) to extract features that capture the influence of anomalies on model behavior. Provides uncertainty predictions and training dynamics logs

3. **Potential Features:**
    - Geometry-based features: distance to class prototypes, local density, and embedding space clustering metrics (e.g., silhouette score) based on the clean ViT embeddings.
    - Uncertainty-based features: predictive entropy, margin confidence, and variance across multiple stochastic forward passes (e.g., using dropout) from the noisy ViT.
    - Training dynamics features: loss trajectory, confidence trajectory, and forgetting events (number of times an example transitions from correct to incorrect classification during training) from the noisy ViT.

4. **Hypotheses:**
    - H1: Gross label noise and out-of-distribution samples are geometrically far from their assigned class prototypes in the learned embedding space, while near-miss errors and clean-hard examples remain close to their assigned prototypes. This implies and corroborates existing findings that we can use simple distance measurements to catch egregious errors - like a frog labeled as an airplane, because the frog's features will place it nowhere near the airplane cluster. However, this approach will fail for subtle mistakes: a dog mislabeled as "cat" will be close to the cat cluster (since dogs and cats are similar), making it indistinguishable from a genuinely difficult cat image. 
    - H2: Prediction uncertainty (entropy, confidence, margin) is similarly high for both clean-but-hard examples and near-miss mislabeled examples, making uncertainty alone insufficient to distinguish label errors from inherent difficulty. A clean-but-hard dog image (occluded, unusual angle) naturally produces low confidence because the visual features are weak. A dog mislabeled as "cat" during training also produces low confidence because the model learned conflicting information—its visual features say "dog" but the training label said "cat." Both converge to similar entropy and margin values despite having fundamentally different causes.
    - H3:  Training dynamics—specifically, per-sample loss trajectories over epochs—provide discriminative signals that separate clean-but-hard examples (which show eventual learning) from near-miss mislabeled examples (which show persistent high loss or erratic patterns), even when final uncertainty is identical.  While two images might look equally "uncertain" to a trained model, the journey they took during training tells different stories. A genuinely hard but correctly-labeled image will show steady improvement, with loss gradually decreasing as the model learns to handle its complexity. In contrast, a near-miss mislabeled image will show a more erratic loss trajectory, with high loss that fails to improve over time, reflecting the model's struggle to reconcile conflicting signals. By analyzing these trajectories, we can uncover patterns that are invisible in static uncertainty measures, providing a powerful tool for distinguishing between different anomaly types.

5. **Model Building:**
    - In H1, we test whether geometric features can distinguish gross noise and OOD samples from clean data, but fail to distinguish near-miss from clean-hard.
        - Models: Logistic regression, SVM, or simple thresholding based on distance to prototypes.
        - Evaluation: Precision, recall, and F1-score for identifying gross noise and OOD samples, and failure to distinguish near-miss from clean-hard.
    - In H2, we test whether uncertainty features alone can distinguish between clean-but-hard and near-miss mislabeled examples.
        - Models: Logistic regression or SVM using only uncertainty features.
        - Evaluation: Precision, recall, and F1-score for distinguishing clean-but-hard from near-miss mislabeled examples, expecting poor performance.
    - In H3, we test whether training dynamics features can successfully distinguish between clean-but-hard and near-miss mislabeled examples, even when uncertainty is similar.
        - Models: Logistic regression, SVM, or a simple feedforward neural network using training dynamics features (e.g., loss trajectory statistics).
        - Evaluation: Precision, recall, and F1-score for distinguishing clean-but-hard from near-miss mislabeled examples, expecting significantly improved performance compared to H2.



6. **Proposed Timeline**

| Week  | Main Tasks                                              |
| ----- | ------------------------------------------------------- | 
| 1     | Dataset preparation, baseline model, noise injection    | 
| 2     | Train clean ViT/ResNet, extract geometry features       | 
| 3     | Train noisy ViT/ResNet with dynamics logging            | 
| 4     | Extract uncertainty & dynamics features, merge datasets | 
| 5     | Test H1 (geometry) and H2 (uncertainty)                 | 
| 6     | Test H3 (dynamics) and relevant ablation study                   | 
| 7     | Final visualizations and analysis                       | 
| 8     | Documentation, testing, presentation                    | 
