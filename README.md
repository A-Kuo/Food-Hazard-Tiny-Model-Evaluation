# Food Hazard Triage & Benchmarking System

Streamlit: https://d4mlrwzgvgu5yds5y7qiqx.streamlit.app/

**Track:** C (Tiny Model / Eval)

> For deeper insights into the design decisions, architectural evolution, and human-AI collaboration that shaped this project, read **`AI_NOTE.md`** alongside this README. The two documents are designed to be explored in parallel for a complete understanding.

---

## The Inspiration

During June–August 2026, the United States experienced a notable spike in reported food supply contamination incidents tied to recent regulatory relaxations and reduced quality-control oversight. This project explores the **triage methodologies** used by Food Safety Quality Assurance (QA) Specialists and Regulatory Compliance Officers at agencies like the FDA and USDA to rapidly categorize and prioritize incoming contamination reports—an essential skill when thousands of incidents arrive weekly.

*This project does not solve the contamination crisis; it is a learning exercise in understanding the decision-making frameworks and NLP techniques that safety inspectors use daily.*

---

## What I Built

A production-ready Streamlit dashboard that **evaluates and benchmarks three distinct NLP triage architectures** for food recall and contamination alert classification:

### Why Three Architectures?

Food safety triage is hard because:
1. **Speed matters**: Inspectors must categorize hundreds of incident reports daily. A fast heuristic baseline is essential.
2. **Unknown hazards appear**: New pathogens and chemical compounds emerge constantly—zero-shot morphological patterns help classify them without retraining.
3. **Phrasing varies widely**: A supervised ML model learns from real incident text patterns, but only with small, balanced datasets.

Each layer trades off **speed**, **interpretability**, and **generalization**:

### Layer 1: Rule-Based Heuristic Parser
- **How it works:** Fast regex, pathogen/allergen keyword taxonomies, and negation clause stripping.
- **Strength:** Interpretable, no training data needed, runs instantly.
- **Weakness:** Brittle on negated clauses (`"tested negative for Salmonella but recalled for..."`) and over-triggers on product certifications.
- **Use case:** Quick triage for obvious cases.

### Layer 2: Morphological Root Classifier (Zero-Shot)
- **How it works:** Detects Latin and English morphological patterns (`-ella`, `-coccus`, `-bacter`, `-ium`, `-virus`, `-phthalate`, `-aldehyde`) to classify unknown pathogens and compounds without training data.
- **Strength:** Handles novel bacteria names (e.g., `Pseudomonella virulens`) and industrial metals (tungsten, titanium, graphite).
- **Weakness:** Less reliable for allergens (vernacular food names don't follow Latin roots).
- **Use case:** Classify emerging or rare hazards.

### Layer 3: Supervised Tiny ML Classifier
- **How it works:** TF-IDF vectorizer (unigrams + bigrams, sublinear scaling) + Logistic Regression with calibrated probability outputs.
- **Strength:** Learns real incident phrasing; high accuracy on seen categories with confidence scores.
- **Weakness:** Requires balanced training data (~140 rows) and can overfit on small datasets.
- **Use case:** High-precision classification on well-known hazard types.

### Layer 4 (Optional): GPT-2 Semantic Centroid Classifier
- **How it works:** Mean-pooled GPT-2 embeddings build class centroids; unknown text is classified by cosine similarity.
- **Strength:** Semantic understanding; robust to phrasing variation.
- **Weakness:** Requires 500MB transformer download; slower inference.
- **Use case:** Semantic refinement when transformers are available.

### Built-In Evaluation Suite
- **Bootstrap 95% Confidence Intervals** — 500 resamples for accuracy and macro F1 to quantify model uncertainty.
- **5-Fold Stratified Cross-Validation** — Reveals in-sample overfitting on the small dataset and estimates true generalization ability.

---

## Who It Is For

* **Food Safety Inspectors & QA Specialists** — Categorize high-volume incident reports (recalls, contamination alerts, supplier notifications) into actionable hazard classes in real time.
* **Regulatory Compliance Officers (FDA/USDA)** — Audit triage decisions with transparent, auditable classifications across four hazard categories (Biological, Allergen, Physical, Chemical) with visible confidence scores and failure-mode analysis.
* **Supply Chain Risk Analysts** — Understand which triage methods catch which hazard types, and where the system is most brittle.

---

## Dataset & Evaluation Data

### Synthetic Incident Dataset (139 balanced rows)

Because real FDA/USDA incident reports are not uniformly available in public datasets, I synthesized a balanced training and test set that reflects **genuine recall notice language** from published FDA Enforcement Reports and USDA FSIS alerts.

**Class Distribution:**
- **Biological** (39 incidents): Listeria, Salmonella, E. coli, Norovirus, Clostridium, etc.
- **Allergen** (31 incidents): Undeclared peanuts, milk, wheat, shellfish, tree nuts, sesame.
- **Physical** (35 incidents): Glass shards, plastic fragments, metal filings, wood splinters, tungsten carbide, ceramic chips.
- **Chemical** (34 incidents): Phthalates, heavy metals (lead, mercury), pesticide residues, industrial compounds.

### Adversarial Edge Cases

The dataset intentionally includes **stress tests** that real-world triage systems struggle with:

| Trap Type | Example | Why It Matters |
|-----------|---------|---|
| **Negation** | *"Tested negative for Salmonella, but recalled for undeclared milk"* | Heuristic rule-based systems often trigger on pathogen keywords regardless of negation. |
| **Dual Hazard** | *"Lead solder contamination from equipment failure"* | Physical foreign objects (solder) combined with chemical toxicity (lead). |
| **Facility Paradox** | *"Certified peanut-free facility, recalling for metal shavings from broken equipment"* | Facility certification claims contrast sharply with the actual hazard. |
| **Rare Metals** | *"Tungsten carbide filings from precision grinding media"* | Unusual industrial materials not in typical allergen/pathogen lists. |
| **Novel Pathogen** | *"Contamination with Clostriferens toxicus detected in canned goods"* | Fictional Latin-root pathogen name to test morphological zero-shot classification. |
| **Predicate-First** | *"Glass, metal shavings, and unidentified debris found during routine sampling"* | Sentence structure leading with the hazard, not the food product. |

These cases help identify **where each triage method fails**—essential for building auditable compliance workflows.

---

## Design Assumptions & Rationale

These assumptions shape how the triage system interprets food safety language:

### 1. Predicate-First Focus
**Assumption:** In food safety notices, the *food product* is the grammatical subject, while the *contaminant* resides in the predicate clause.

**Why it matters:** Incident reports often read: *"Frozen strawberries recalled for Listeria contamination"* (subject: berries, predicate: Listeria). The triage algorithms prioritize contaminant noun phrases, not the food type, because the *hazard class* is what safety inspectors must act on immediately.

**eg.:**
- Correctly identifies "Listeria" → Biological (even if food is not mentioned)
- Handles "undeclared peanuts" → Allergen (even if product name is generic)

### 2. Latin Morphological Reliability for Pathogens
**Assumption:** Pathogen binomial nomenclature (genus + species, e.g., *Salmonella typhimurium*) follows predictable Latin morphological patterns that enable zero-shot classification.

** Reasoning*:** Unknown or emerging pathogens (e.g., *Clostriferens toxicus*, *Pseudomonella virulens*) can be classified without retraining because they follow the same Latin suffixes as known bacteria.

** Example patterns:**
- `-ella` (Salmonella, Shigella, Brucella)
- `-coccus` (Staphylococcus, Streptococcus)
- `-bacter` (Campylobacter)
- `-ium` (Clostridium)
- `-virus` (Norovirus, Hepatitis A virus)

**The important caveat here** is that allergens do NOT follow Latin patterns (peanuts, milk, wheat are vernacular), so morphological matching is unreliable for such cases.

### 3. Chemical Invariance + Allergen Guard
**Assumption:** Chemical compounds are classified as Chemical hazards *unless* they are food allergens explicitly flagged in regulations (e.g., sulfites, which are chemical additives but managed under Allergen protocols).

**Why it matters:** Prevents false-positive chemical classifications on allergen proteins like casein, lecithin, and gluten (which have chemical morphology but are regulated as allergens).

**Example:**
- `Diethylhexylphthalate` → Chemical (plasticizer)
- `Casein` → Allergen (not Chemical, despite `-in` suffix)

### 4. Stratified 5-Fold CV as Overfitting Safeguard
**Assumption:** On small datasets (~140 rows), in-sample model accuracy is inherently optimistic. Stratified 5-fold cross-validation provides the honest generalization estimate.

**Why it matters:** A 95% in-sample accuracy on 140 rows is misleading—the model may be memorizing. By splitting the data into 5 folds and evaluating per-fold, we expose whether the model **generalizes** or just **memorizes**. A large gap (e.g., 92% in-sample, 75% CV) signals overfitting and justifies caution before deployment.

**Decision rule:** If CV accuracy is more than 5–10 percentage points below in-sample accuracy, the model is memorizing and needs more data or regularization.

---

## Known Limitations & Boundary Cases

### 1. Allergen vs. Chemical Morphology Conflict
**The Problem:**
- Suffixes like `-in`, `-ine`, `-ate` appear in both chemical compounds AND allergen proteins.
- Example: `Casein` (milk allergen) and `Penicillin` (chemical antibiotic) both end in `-in`.
- Without a guard rule, the morphological classifier would incorrectly flag casein as a chemical compound.

**The Fix:**
- Implemented an **allergen guard heuristic**: if strong allergen keyword evidence is present (e.g., "milk", "peanuts", "wheat"), chemical morphology scores are suppressed.

**Remaining Risk:**
- Edge cases like sulfites (both chemical AND allergen) remain ambiguous and require context.

### 2. Dual-Category Boundary Overlap
**The Problem:**
- Some incidents involve both physical AND chemical hazards.
- Example: *"Lead solder fragments from broken thermometer"* is a Physical foreign object (solder debris) AND a Chemical hazard (lead toxicity).
- Example: *"Mercury droplets from broken thermometer"* (physical + chemical).

**Current Handling:**
- The triage system classifies based on the **dominant language** in the report.
- If "lead solder" appears, the system prioritizes the chemical aspect (lead toxicity).
- If "glass shards" dominates, it classifies as Physical.

**Best Practice:**
- In production, dual-category incidents should trigger a **secondary review** flag for inspectors.

### 3. Morphological Fragility on Allergens
**The Problem:**
- Unlike pathogens, allergen names are **vernacular food names** (*peanuts*, *milk*, *wheat*, *shellfish*) and do NOT follow Latin roots.
- Morphological matching (Latin suffixes) is unreliable for allergens—it has ~50% accuracy on unseen allergen data.

**Current Handling:**
- The system relies on **keyword matching** (heuristic layer) + **ML learned patterns** (ML layer) for allergen classification.
- Morphological classification is intentionally de-emphasized for the Allergen class.

**Impact:**
- Heuristic and ML models are more reliable for allergens than morphological matching.

### 4. Rare Industrial Materials
**The Problem:**
- Modern manufacturing uses exotic metals (tungsten, titanium, zirconium, graphite) that may not appear in training data.
- A classic keyword-based heuristic might misclassify them as "unknown" or even as chemical compounds.

**Current Handling:**
- The morphological classifier contains a **rare metals list** (tungsten, titanium, graphite, zirconium) that triggers Physical classification.
- Ensures zero-shot handling of industrial materials not in the training set.

### 5. Small Dataset Overfitting Risk
**The Problem:**
- With only 139 training examples, the ML model risks memorizing the training set.
- A 100% in-sample accuracy on 140 rows is a red flag, not a success.

**Current Handling:**
- **5-fold Stratified Cross-Validation** provides the honest generalization benchmark.
- **Bootstrap 95% confidence intervals** quantify uncertainty.
- The dashboard explicitly displays both in-sample and CV accuracy, with an automated **overfitting risk verdict** (Low / Moderate / High).

---

## Set up

```bash
# 1. Navigate to this directory
cd food_hazard_triage

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
python -m streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Ready to triage incidents!

---

## Installation & Getting Started

### Prerequisites
- Python 3.9 or later
- pip package manager

### Step-by-Step Installation

1. **Ensure this directory is in your workspace:**
   ```
   your_project/
   └── food_hazard_triage/  ← You are here
       ├── README.md
       ├── AI_NOTE.md
       ├── app.py
       ├── requirements.txt
       ├── data/
       └── src/
   ```

2. **Install dependencies:**
   ```bash
   cd food_hazard_triage
   pip install -r requirements.txt
   ```
   
   This installs: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `scipy`, and `altair`.

3. **Run the Streamlit dashboard:**
   ```bash
   python -m streamlit run app.py
   ```

   **Output:**
   ```
   You can now view your Streamlit app in your browser.
   
     Local URL: http://localhost:8501
     Network URL: http://<your-ip>:8501
   ```

4. **Open in your browser** and start triaging incidents!

### Optional: Enable GPT-2 Semantic Classifier

The dashboard includes an optional 4th layer using GPT-2 embeddings. To enable:

1. **Uncomment these lines** in `requirements.txt`:
   ```
   # transformers>=4.38.0
   # torch>=2.0.0
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: First run will download ~500MB of GPT-2 model weights.

3. **Restart the app:**
   ```bash
   python -m streamlit run app.py
   ```

---

## Project Structure

```
food_hazard_triage/
├── README.md                          # This file: full documentation
├── AI_NOTE.md                         # Developer notes: read alongside README
├── requirements.txt                   # Python dependencies
├── app.py                             # Main Streamlit dashboard
├── data/
│   └── food_recalls_sample.csv       # Synthetic 139-row training dataset
└── src/
    ├── __init__.py
    ├── generate_data.py              # Dataset generation script
    ├── hazard_classifier.py           # Layer 1 (Heuristic) + Layer 3 (ML) + Evaluation suite
    └── semantic_matcher.py            # Layer 2 (Morphological) + Layer 4 (GPT-2 optional)
```

---

## Dashboard Walkthrough

### Page 1: Live Triage Dashboard
- Paste or select an incident report text.
- The system runs all four layers concurrently and displays results with confidence scores.
- Useful for real-time incident triage.

### Page 2: Benchmark Analytics & Overfitting Diagnostics
- View in-sample accuracy, macro F1, and bootstrap 95% confidence intervals for all three layers.
- Stratified 5-fold CV metrics reveal whether the ML model is overfitting.
- Confusion matrices show per-class performance.

### Page 3: Failure Mode Explorer
- Filter misclassifications by architecture (Heuristic, Morphological, ML).
- Drill into specific failure cases to understand where the system struggles.
- Identifies unclassified or low-confidence predictions.

### Page 4: Model Diagnostics & Taxonomies
- View the top TF-IDF feature weights per class (what the ML model learned).
- Reference table: Morphological root patterns and their target hazard classes.
- Allergen caveat: Explains why morphological matching fails on allergens.

---

## Understanding the Results

### Accuracy Metrics
- **Heuristic Accuracy:** Fast baseline. Expect 70–80% on well-formed reports.
- **Morphological Accuracy:** Zero-shot approach. Lower on allergens (~50–60%), higher on biology (~85–90%).
- **ML Model Accuracy:** Highest on known categories but vulnerable to overfitting on small datasets.

### Confidence Intervals
- Bootstrap 95% CI gives an honest range of performance uncertainty.
- A wide CI (e.g., [65%, 85%]) suggests unstable predictions; a narrow CI (e.g., [78%, 82%]) is more reliable.

### Overfitting Diagnostics
- **In-Sample Accuracy:** How well the ML model memorized the training set.
- **5-Fold CV Accuracy:** Honest generalization estimate on held-out data.
- **Gap (Δ):** If Δ > 10%, the model is overfitting. Use CV accuracy for real-world expectations.

### Per-Fold Variance
- Shows CV accuracy per fold. High variance (e.g., Fold 1: 85%, Fold 3: 65%) suggests the dataset is imbalanced or contains outliers.

---

## Extending the System

### Add More Training Data
1. **Edit `src/generate_data.py`** to include real or synthetic incident reports.
2. **Rerun the app:** The ML model automatically retrains on the larger dataset.
3. **Check CV diagnostics:** More data should close the in-sample/CV gap and reduce overfitting risk.

### Improve Morphological Patterns
1. **Edit `src/semantic_matcher.py`** to add new Latin roots or rare materials.
2. Example: Add `-bacillus` for additional bacillus species, or extend the rare metals list.

### Swap the ML Model
1. **Edit `src/hazard_classifier.py`** in the `TinyMLModel` class.
2. Replace `LogisticRegression` with `RandomForestClassifier`, `SVC`, or `XGBClassifier`.
3. Retrain and compare CV metrics.

---

## Developer Notes

**Read `AI_NOTE.md` alongside this README** for insights into:
- Why each layer was chosen.
- Human-in-the-loop AI collaboration during development.
- Specific architectural decisions and debugging moments.
- Edge cases and their solutions.

The two documents are designed to be read **side-by-side** for a complete understanding of the project's rationale and implementation.

---

## Implementation Timeline & Milestones
* **8:18 PM UTC** — Timer started. Track C scoping, domain selection (food safety/recall triage), and directory scaffolding.
* **8:43 PM UTC** — Estimated 30% baseline training data constructed; initial regex heuristics and TF-IDF pipeline verified.
* **8:50 PM UTC** — Project scaled up to 139 balanced records with deep taxonomy, unusual physical materials, Latin root morphology parser, and GPT-2 semantic dictionary design.
* **9:09 PM UTC** — Changes to user-facing markdown files completed (rubrics and detailed AI notes).
* **9:18 PM UTC** — UI fixes implemented: FDA-style dashboard layout added, navigation moved to sidebar, loading screen distraction meters implemented, and cache serialization (`@st.cache_data` vs `@st.cache_resource`) resolved.
* *Stepped away for 10 minutes to run*
*  **9:33 PM UTC** — Final evaluation suite completed: bootstrap 95% CI, 5-fold CV, and confusion matrices for all three layers.
* 

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
- **Solution:** Run `pip install -r requirements.txt` again, ensuring you are in the `food_hazard_triage/` directory.

### "Port 8501 is already in use"
- **Solution:** Streamlit will automatically try the next port (8502, 8503, etc.). Or kill the existing Streamlit process and restart.

### "GPT-2 semantic classifier is unavailable"
- **Expected:** If `transformers` and `torch` are not uncommented in `requirements.txt`, the optional 4th layer is disabled.
- **To enable:** Uncomment those lines, run `pip install -r requirements.txt`, and restart the app.

### "App seems slow on first run"
- **Expected:** On first run, GPT-2 weights are downloaded (~500MB). Subsequent runs are cached and faster.

### "Data file not found"
- **Solution:** The app auto-generates `data/food_recalls_sample.csv` on first run. If it fails, ensure the `data/` directory exists: `mkdir -p data/`

---

## What I Would Do Next With More Time
* **Live openFDA API Integration** — Ingest streaming real-world recall notices directly from the openFDA food enforcement endpoint.
* **Sentence-Transformers Embeddings** — Upgrade semantic centroid matching using a lightweight sentence-transformer (`all-MiniLM-L6-v2`).
* **Trie-Based Morphological Decomposer** — Implement a complete Latin root morpheme parser for biological and chemical zero-shot triage.

---
