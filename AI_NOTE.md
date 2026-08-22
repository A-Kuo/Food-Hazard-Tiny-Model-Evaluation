# AI Collaboration Note: Food Hazard Triage & Eval

## Did You Use AI?
AI assistants (Gemini, Antigravity AI, and Ollama) for build sprint testing dataset curation.

---

## How You Used It & Human-in-the-Loop Direction
The development workflow was **Human-in-the-Loop (HITL)**: I directed every architectural decision, prompt strategy, and validation checkpoint, while using AI as a rapid coding copilot:

* **Domain & Architectural Direction** — I selected Track C (Tiny Model / Eval) and scoped the food contamination hazard triage problem, defining the 4 target categories (Biological, Allergen, Physical, Chemical) and the persona (Food Safety QA / Compliance Analyst).
* **Multi-Layer System Design** — I directed the progression from a baseline Rule-Based Heuristic to a Latin/English Morphological Root Classifier, a supervised TF-IDF + Logistic Regression ML pipeline, and an optional GPT-2 Semantic Centroid Dictionary.
* **Domain & Linguistic Logic** — I formulated the predicate-first principle (food = subject, contaminant = predicate) and instructed the AI on specific taxonomy rules (e.g., treating unusual metals like tungsten, titanium, and gold as Physical debris; matching Latin pathogen roots like `-ella`, `-coccus`, `-bacter`, `-rium`, `-virus`).
* **Allergen/Chemical Boundary Management** — I identified the risk where chemical morphological rules (`-in`/`-ine`) cause false positives on allergen proteins (*casein*, *lecithin*) and instructed the implementation of an allergen guard suppression heuristic.
* **Overfitting & Statistical Rigor** — I directed the inclusion of 95% bootstrap confidence intervals and mandated an explicit 5-fold Stratified Cross-Validation check to expose in-sample memorization risk on our small dataset.
* **Automated Workflow & Model Integrity** — Gemini and Ollama were leveraged during the automated workflow to generate edge-case incident reports and verify model integrity under adversarial test phrasing.
* **Debugging & Architecture Fixes** — When Streamlit encountered a pickle serialization error (`UnserializableReturnValueError`), I directed the architectural separation of data caching (`@st.cache_data` for DataFrames and metrics) from object caching (`@st.cache_resource` for sklearn and custom model instances) and requested the initialization progress meter.


---

## One Prompt, Workflow, Or Moment That Helped
The most impactful AI workflow was generating a multi-class synthetic dataset of 139 recall reports featuring deliberately adversarial edge cases (negations like *"tested negative for Listeria, recalled for undeclared peanuts"*, certification contradictions like *"certified peanut-free facility, recalled for metal shavings"*, and Latin root stress tests with fictional bacteria names). Generating these in code saved hours of manual data collection and enabled immediate model failure-mode analysis.

---

## One Thing You Verified Or Decided Yourself
I verified and decided the **cross-validation overfitting diagnostic design**: recognizing that on a ~140-row dataset, a 100% in-sample ML accuracy is misleading. I decided to explicitly display the in-sample vs. 5-fold cross-validation accuracy gap directly in the Streamlit benchmark UI, alongside an automated overfitting risk verdict (Low / Moderate / High) and per-fold variance metrics, to ensure complete transparency for the evaluating teammate.
