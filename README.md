# Food Hazard Triage & Benchmarking System

**Track C — Tiny Model / Eval**

A Streamlit benchmarking tool that compares lightweight NLP approaches for classifying food-safety incident reports into four dominant hazard categories: **Biological, Allergen, Physical,** and **Chemical**.

## What I Built

The dashboard evaluates four complementary approaches:

1. **Rule-based heuristic:** keyword taxonomies, regex, and negation handling for fast, interpretable first-pass triage.
2. **Morphological classifier:** Latin/English root patterns and material rules for zero-shot handling of unfamiliar pathogens, compounds, and industrial debris.
3. **Supervised tiny ML model:** TF-IDF features with calibrated logistic regression for learned recall-notice language patterns.
4. **Optional GPT-2 semantic baseline:** embedding-centroid similarity for semantic refinement when transformer dependencies are enabled.

Users can submit an incident description, compare method predictions and confidence, inspect benchmark metrics, and explore representative failure cases.

## Data and Evaluation

The project uses a synthetic, balanced dataset of 139 short reports modeled on publicly available FDA and USDA recall-notice language. It includes deliberately difficult examples: negated hazards, dual hazards, certification contradictions, rare industrial materials, and novel pathogen-like terms.

Because this is a small synthetic dataset, performance metrics are prototype benchmarks—not real-world food-safety claims. The app reports stratified 5-fold cross-validation and 500-resample bootstrap confidence intervals for accuracy and macro F1 to make overfitting and uncertainty visible.

## Assumptions and Limits

The system returns one **dominant** hazard class. Real incidents may involve multiple hazards, incomplete evidence, and regulatory context requiring subject-matter review. This is an educational evaluation artifact, not a production, clinical, regulatory, or operational decision system.

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

See [USER_GUIDE.md](USER_GUIDE.md) for installation details, dashboard walkthroughs, architecture, assumptions, limitations, troubleshooting, and extension ideas. See [AI_NOTE.md](AI_NOTE.md) for AI-collaboration notes.
