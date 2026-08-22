"""
hazard_classifier.py
=====================
Three-layer food contamination triage system:
  1. Rule-Based Heuristic     — keyword/regex + pathogen taxonomy (fast, interpretable, brittle)
  2. Morphological Classifier — Latin/English root matching (handles unknown pathogens)
  3. TF-IDF + LogReg Model    — learned from training data (robust to phrasing, needs data)
  
Optionally augmented by GPT-2 semantic embedding classifier (see semantic_matcher.py).
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from scipy.stats import bootstrap

from src.semantic_matcher import morphological_classify, get_semantic_classifier


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: Rule-Based Heuristic
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedHeuristic:
    """
    Keyword and regex heuristic for food hazard classification.
    
    Strengths: fast, zero training data, interpretable.
    Known weaknesses: false positives on negated clauses (e.g., "tested negative for
    Salmonella"), false positives on product certifications (e.g., "peanut-free").
    """

    def __init__(self):
        self.NEGATION_PATTERN = re.compile(
            r'(tested?\s+(negative|clear)\s+for|free\s+of|free\s+from|'
            r'certified\s+\S+-free|does\s+not\s+contain|no\s+\w+\s+(found|detected|present))',
            re.IGNORECASE
        )
        self.BIO_PATTERN = re.compile(
            r'\b(listeria|salmonella|e\.?\s*coli|botulinum|campylobacter|norovirus|'
            r'vibrio|shigella|yersinia|brucella|cryptosporidium|giardia|toxoplasma|'
            r'trichinella|ascaris|taenia|legionella|staphylococcus|streptococcus|'
            r'bacillus|cronobacter|hepatitis\s*a|cyclospora|aspergillus|fusarium|'
            r'clostridium|pathogen|bacteria|virus|parasite|mold|spore|microorganism)\b',
            re.IGNORECASE
        )
        self.ALLERGEN_PATTERN = re.compile(
            r'\b(undeclared|allergen|allergy|anaphylaxis|peanut|milk|dairy|soy|'
            r'wheat|gluten|crustacean|shellfish|tree\s+nut|almond|walnut|cashew|'
            r'pecan|hazelnut|pistachio|macadamia|egg|sesame|fish|lupin|mustard|'
            r'celery|sulfite|casein|whey|lecithin)\b',
            re.IGNORECASE
        )
        self.PHYSICAL_PATTERN = re.compile(
            r'\b(plastic|glass|metal|wood|splinter|rubber|foreign\s+(material|object)|'
            r'stone|gravel|bone\s+fragment|silica|foil|cardboard|fiberglass|staple|nail|'
            r'screw|ceramic|shard|debris|wire|shaving|fragment|piece[s]?|'
            r'tungsten|titanium|graphite|zirconium|copper\s+wire|solder)\b',
            re.IGNORECASE
        )
        self.CHEMICAL_PATTERN = re.compile(
            r'\b(lead|mercury|cadmium|arsenic|dioxin|pcb|pesticide|herbicide|'
            r'fungicide|aflatoxin|mycotoxin|patulin|melamine|bisphenol|bpa|'
            r'acrylamide|benzene|nitrate|nitrite|antibiotic|chloramphenicol|'
            r'malachite|sudan\s+red|fumonisin|ochratoxin|zearalenone|'
            r'deoxynivalenol|phthalate|aldehyde|histamine|hydrogen\s+cyanide|'
            r'ethylene\s+oxide|tetracycline|chemical|residue|toxin)\b',
            re.IGNORECASE
        )

    def _strip_negations(self, text: str) -> str:
        """Remove negated contaminant clauses to reduce false positives."""
        # Remove patterns like "tested negative for Salmonella"
        return self.NEGATION_PATTERN.sub('', text)

    def predict_single(self, text: str) -> str:
        clean = self._strip_negations(text)
        # Biological checked first (highest health urgency signal)
        if self.BIO_PATTERN.search(clean):
            return "Biological"
        if self.ALLERGEN_PATTERN.search(clean):
            return "Allergen"
        if self.PHYSICAL_PATTERN.search(clean):
            return "Physical"
        if self.CHEMICAL_PATTERN.search(clean):
            return "Chemical"
        return "Unknown"

    def predict(self, texts):
        return [self.predict_single(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: TF-IDF + Logistic Regression ML Model
# ─────────────────────────────────────────────────────────────────────────────

class TinyMLModel:
    """
    TF-IDF + Logistic Regression classifier.
    - TF-IDF with bigrams captures "undeclared peanuts", "metal shavings", etc.
    - class_weight='balanced' compensates for any class imbalance in training data.
    - Outputs calibrated class probabilities for confidence scoring.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            stop_words='english',
            sublinear_tf=True   # log(1 + tf) — reduces dominance of common terms
        )
        self.classifier = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            C=1.0               # regularization; tune higher for more expressive model
        )
        self.classes = []
        self.is_trained = False

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes = list(self.classifier.classes_)
        self.is_trained = True

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def get_top_features(self, category: str, n: int = 10):
        """Returns top n TF-IDF features for a given category."""
        if category not in self.classes:
            return []
        idx = self.classes.index(category)
        coef = self.classifier.coef_[idx]
        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = np.argsort(coef)[-n:][::-1]
        return [(feature_names[i], round(coef[i], 3)) for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION with Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, metric_fn, n_resamples=500, confidence=0.95):
    """
    Compute bootstrap confidence interval for a metric (e.g., accuracy, F1).
    Returns (lower, upper) bounds.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n = len(y_true_arr)

    metric_values = []
    rng = np.random.default_rng(42)
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        val = metric_fn(y_true_arr[indices], y_pred_arr[indices])
        metric_values.append(val)

    alpha = (1 - confidence) / 2
    lower = np.percentile(metric_values, alpha * 100)
    upper = np.percentile(metric_values, (1 - alpha) * 100)
    return round(lower, 4), round(upper, 4)


def _macro_f1(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return f1


def cross_validate_ml(texts, labels, n_splits=5):
    """
    Stratified K-Fold cross-validation of the TF-IDF + LogReg model.

    Returns per-fold accuracy and F1 scores plus the mean/std, and a simple
    overfitting verdict comparing in-sample vs held-out performance.

    Because the dataset is small (~100-200 rows), CV is the only reliable way
    to detect whether the model has memorised the training examples rather than
    learned generalizable patterns. A large gap (in-sample >> CV mean) is the
    primary overfitting signal.
    """
    from sklearn.model_selection import StratifiedKFold

    texts_arr = np.array(texts)
    labels_arr = np.array(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_acc, fold_f1 = [], []

    for train_idx, test_idx in skf.split(texts_arr, labels_arr):
        X_train, X_test = texts_arr[train_idx], texts_arr[test_idx]
        y_train, y_test = labels_arr[train_idx], labels_arr[test_idx]

        fold_model = TinyMLModel()
        fold_model.train(X_train.tolist(), y_train.tolist())
        preds = fold_model.predict(X_test.tolist())

        fold_acc.append(accuracy_score(y_test, preds))
        fold_f1.append(_macro_f1(y_test, preds))

    # In-sample (full fit) for gap calculation
    full_model = TinyMLModel()
    full_model.train(texts, labels)
    in_sample_preds = full_model.predict(texts)
    in_sample_acc = accuracy_score(labels, in_sample_preds)
    in_sample_f1 = _macro_f1(labels, in_sample_preds)

    cv_acc_mean = float(np.mean(fold_acc))
    cv_f1_mean  = float(np.mean(fold_f1))
    acc_gap     = round(in_sample_acc - cv_acc_mean, 4)
    f1_gap      = round(in_sample_f1  - cv_f1_mean,  4)

    # Verdict: gap > 0.15 is a clear overfit warning for a small dataset
    if acc_gap > 0.20:
        verdict = "⚠️  HIGH overfit risk — in-sample accuracy is significantly above CV accuracy."
    elif acc_gap > 0.10:
        verdict = "🟡 MODERATE overfit signal — model may be partially memorising training examples."
    else:
        verdict = "✅ LOW overfit risk — in-sample and CV accuracy are close."

    return {
        "n_splits":        n_splits,
        "fold_acc":        [round(v, 4) for v in fold_acc],
        "fold_f1":         [round(v, 4) for v in fold_f1],
        "cv_acc_mean":     round(cv_acc_mean, 4),
        "cv_acc_std":      round(float(np.std(fold_acc)), 4),
        "cv_f1_mean":      round(cv_f1_mean, 4),
        "cv_f1_std":       round(float(np.std(fold_f1)), 4),
        "in_sample_acc":   round(in_sample_acc, 4),
        "in_sample_f1":    round(in_sample_f1, 4),
        "acc_gap":         acc_gap,
        "f1_gap":          f1_gap,
        "verdict":         verdict,
    }


def evaluate_models(df: pd.DataFrame) -> dict:
    """
    Full evaluation of all three classification layers:
      - Rule-Based Heuristic
      - Morphological Root Classifier
      - TF-IDF + LogReg ML Model (with cross-validation overfitting check)

    Returns ONLY primitive / pickle-serializable values (dicts, DataFrames,
    lists, floats). Model objects are intentionally excluded — the app builds
    them separately with st.cache_resource to avoid pickle errors.
    """
    texts = df['text'].tolist()
    labels = df['true_category'].tolist()

    # --- Layer 1: Heuristic ---
    heuristic_preds = RuleBasedHeuristic().predict(texts)

    # --- Layer 2: Morphological ---
    morpho_preds = [morphological_classify(t) or "Unknown" for t in texts]

    # --- Layer 3: ML Model (full-fit for in-sample metrics) ---
    ml_model = TinyMLModel()
    ml_model.train(texts, labels)
    ml_preds = ml_model.predict(texts)

    # --- Cross-validation overfitting check ---
    cv_results = cross_validate_ml(texts, labels, n_splits=5)

    # --- Optional: GPT-2 Semantic ---
    sem_classifier = get_semantic_classifier()
    semantic_available = sem_classifier.available
    if semantic_available:
        semantic_preds = []
        for t in texts:
            result = sem_classifier.predict_single(t)
            semantic_preds.append(result["prediction"] if result else "Unknown")
    else:
        semantic_preds = ["N/A"] * len(texts)

    def get_metrics(y_true, y_pred, name=""):
        valid = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp not in ("Unknown", "N/A")]
        if not valid:
            return {
                "name": name, "accuracy": 0, "precision": 0,
                "recall": 0, "f1": 0, "n_classified": 0,
                "acc_ci": (0, 0), "f1_ci": (0, 0)
            }
        yt_valid, yp_valid = zip(*valid)
        labels_set = sorted(set(yt_valid))
        acc = accuracy_score(yt_valid, yp_valid)
        p, r, f1, _ = precision_recall_fscore_support(
            yt_valid, yp_valid, labels=labels_set, average='macro', zero_division=0
        )
        cm = confusion_matrix(yt_valid, yp_valid, labels=labels_set)
        return {
            "name":          name,
            "accuracy":      round(acc, 4),
            "precision":     round(p, 4),
            "recall":        round(r, 4),
            "f1":            round(f1, 4),
            "n_classified":  len(yt_valid),
            "acc_ci":        bootstrap_ci(yt_valid, yp_valid, accuracy_score),
            "f1_ci":         bootstrap_ci(yt_valid, yp_valid, _macro_f1),
            "confusion_matrix": {
                "labels": labels_set,
                "matrix": cm.tolist()
            }
        }

    df_out = df.copy()
    df_out['heuristic_pred'] = heuristic_preds
    df_out['morpho_pred']    = morpho_preds
    df_out['ml_pred']        = ml_preds
    df_out['semantic_pred']  = semantic_preds

    # ml_top_features: serializable list of (category → [(feature, weight)])
    ml_top_features = {
        cat: ml_model.get_top_features(cat, n=12)
        for cat in ml_model.classes
    }

    return {
        "heuristic_metrics":  get_metrics(labels, heuristic_preds, "Heuristic"),
        "morpho_metrics":     get_metrics(labels, morpho_preds,    "Morphological"),
        "ml_metrics":         get_metrics(labels, ml_preds,        "ML (TF-IDF + LogReg)"),
        "cv_results":         cv_results,
        "ml_top_features":    ml_top_features,
        "ml_classes":         list(ml_model.classes),
        "df":                 df_out,
        "semantic_available": semantic_available,
    }
