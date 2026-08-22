"""
app.py — Food Contamination Hazard Triage & Eval
Track C: Tiny Model / Eval | DS Intern Build Challenge

Run from the food_hazard_triage/ directory:
    python -m streamlit run app.py

Requires Python 3.9+. Install dependencies with:
    pip install -r requirements.txt

Optional (for GPT-2 semantic classifier):
    Uncomment transformers and torch in requirements.txt, then reinstall.
"""

import sys
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FDA Hazard Triage Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64

CATEGORY_COLORS = {
    "Biological": "🦠",
    "Allergen":   "⚠️",
    "Physical":   "🔩",
    "Chemical":   "🧪",
    "Unknown":    "❓",
    "N/A":        "—",
    "No signal":  "—",
}

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

CATEGORY_IMAGE_FILES = {
    "Biological": "biological.png",
    "Allergen":   "allergen.png",
    "Physical":   "physical.png",
    "Chemical":   "chemical.png",
}

@st.cache_data
def get_image_as_base64(file_path: str) -> str:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
                return f"data:image/png;base64,{encoded}"
        except Exception:
            pass
    return ""

def get_category_icon_html(category: str, width: int = 20) -> str:
    filename = CATEGORY_IMAGE_FILES.get(category)
    if filename:
        file_path = os.path.join(ASSETS_DIR, filename)
        base64_str = get_image_as_base64(file_path)
        if base64_str:
            return f'<img src="{base64_str}" style="width: {width}px; height: {width}px; vertical-align: middle; margin-right: 6px;" />'
    
    # Fallback to emoji
    emoji = CATEGORY_COLORS.get(category, "❓")
    return f'<span style="font-size: {width}px; vertical-align: middle; margin-right: 6px;">{emoji}</span>'

def render_styled_result(category: str, label: str, alert_type: str = "info", subtitle: str = ""):
    icon_html = get_category_icon_html(category, width=24)
    colors = {
        "info":    {"bg": "#EFF6FF", "border": "#3B82F6", "text": "#1E3A8A"},
        "warning": {"bg": "#FFFBEB", "border": "#F59E0B", "text": "#78350F"},
        "success": {"bg": "#ECFDF5", "border": "#10B981", "text": "#064E3B"},
    }.get(alert_type, {"bg": "#F3F4F6", "border": "#9CA3AF", "text": "#1F2937"})
    
    sub_html = f"<div style='font-size: 13px; color: {colors['text']}bb; margin-top: 2px;'>{subtitle}</div>" if subtitle else ""
    html = f"""
    <div style="
        display: flex; 
        flex-direction: column;
        justify-content: center;
        padding: 12px 16px; 
        background-color: {colors['bg']}; 
        border-left: 5px solid {colors['border']}; 
        border-radius: 6px;
        margin-bottom: 8px;
    ">
        <div style="display: flex; align-items: center;">
            {icon_html}
            <span style="font-weight: bold; font-size: 16px; color: {colors['text']}; margin-left: 4px;">{category}</span>
        </div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "food_recalls_sample.csv")

# ─────────────────────────────────────────────────────────────────────────────
# CACHING STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        from src.generate_data import generate_dataset
        return generate_dataset()
    return pd.read_csv(DATA_PATH)

@st.cache_data(show_spinner=False)
def run_evaluation(_df_hash, df_json: str) -> dict:
    from src.hazard_classifier import evaluate_models
    df = pd.read_json(df_json)
    return evaluate_models(df)

@st.cache_resource(show_spinner=False)
def get_live_models():
    from src.hazard_classifier import RuleBasedHeuristic, TinyMLModel
    from src.semantic_matcher import morphological_classify, get_semantic_classifier
    df = load_data()
    texts = df['text'].tolist()
    labels = df['true_category'].tolist()

    heuristic = RuleBasedHeuristic()
    ml_model  = TinyMLModel()
    ml_model.train(texts, labels)
    sem = get_semantic_classifier()

    return {
        "heuristic":            heuristic,
        "ml_model":             ml_model,
        "morphological_classify": morphological_classify,
        "sem":                  sem,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP PROGRESS METER (Dashboard Style)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_with_progress():
    bar_placeholder = st.empty()
    
    with bar_placeholder.container():
        st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🛡️ Initializing FDA Triage Analytics...</h2>", unsafe_allow_html=True)
        bar = st.progress(0, text="Establishing connection to FDA data standards...")
        time.sleep(0.3)
        
        # Step 1
        bar.progress(15, text="Loading historical recall notices dataset...")
        df = load_data()
        time.sleep(0.3)
        
        # Step 2
        bar.progress(35, text=f"Dataset loaded ({len(df)} records). Bootstrapping evaluation layers...")
        df_json = df.to_json()
        
        # We don't have hooks inside run_evaluation, so this jump happens during calculation.
        bar.progress(45, text="Running K-Fold Cross-Validation and building ML models (This may take a moment)...")
        results = run_evaluation(id(df), df_json)
        
        # Step 3
        bar.progress(80, text="Validation complete. Compiling inference endpoints...")
        models = get_live_models()
        time.sleep(0.3)
        
        bar.progress(95, text="Rendering dashboard UI...")
        time.sleep(0.2)
        bar.progress(100, text="System Ready.")
        time.sleep(0.3)
        
    bar_placeholder.empty()
    return df, results, models

df, results, models = initialize_with_progress()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION & LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/U.S._Food_and_Drug_Administration_logo.svg/320px-U.S._Food_and_Drug_Administration_logo.svg.png", width=150)
    st.markdown("### Hazard Triage System")
    st.caption("v2.1.0 | Regulatory Compliance Tools")
    st.divider()
    
    st.markdown("**NAVIGATION**")
    page = st.radio(
        "Select Module:",
        ["Live Triage Dashboard", "Benchmark Analytics", "Failure Mode Explorer", "Model Diagnostics"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.caption("Active Config: Track C (Tiny ML)")
    st.caption(f"Database: {len(df)} Records loaded")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: LIVE TRIAGE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "Live Triage Dashboard":
    st.title("Live Incident Triage")
    st.markdown("### Process Incoming Recall & Contamination Reports")
    st.markdown(
        "Use this module to process incoming FDA/USDA incident reports. The system evaluates "
        "reports across **three distinct NLP architectures** concurrently to establish a consensus classification."
    )

    examples = {
        "Select an example…": "",
        "[Bio] Norovirus in frozen berries":
            "Routine sampling revealed Norovirus contamination in the frozen mixed berry blend.",
        "[Allergen] Undeclared peanuts (negation trap)":
            "Product tested negative for Salmonella and E. coli, but is recalled for containing undeclared peanuts.",
        "[Physical] Tungsten filings (unusual metal)":
            "Tungsten carbide filings from grinding media found in imported spice blend.",
        "[Chemical] Complex compound name":
            "Elevated concentrations of diethylhexylphthalate detected migrating from packaging into dairy product.",
        "[Bio - Unknown Pathogen] Latin root stress test":
            "Contamination with Clostriferens toxicus, an emerging pathogen, detected in canned goods.",
        "[Edge] Peanut-free facility but glass recall":
            "Certified peanut-free facility. Recalling product due to glass fragments from broken equipment.",
    }

    selected = st.selectbox("Quick Load Template:", list(examples.keys()))
    user_input = st.text_area("Incident Report Text:", value=examples[selected], height=120)

    if st.button("🔍 Run Triage Protocol", type="primary") and user_input.strip():
        
        # Distraction / detailed loading meter
        with st.status("Initiating Triage Protocol...", expanded=True) as status:
            time.sleep(0.2)
            st.write("⚙️ Parsing text entities...")
            time.sleep(0.3)
            
            from src.semantic_matcher import morphological_classify
            heuristic = models["heuristic"]
            ml_model  = models["ml_model"]
            sem       = models["sem"]

            st.write("🔍 Running Layer 1: Rule-Based Heuristic Matcher...")
            h_pred   = heuristic.predict_single(user_input)
            time.sleep(0.2)
            
            st.write("🧬 Running Layer 2: Latin/English Morphological Root Parser...")
            m_pred   = morphological_classify(user_input) or "No signal"
            time.sleep(0.2)
            
            st.write("🤖 Running Layer 3: TF-IDF + Logistic Regression ML Inference...")
            ml_pred  = ml_model.predict([user_input])[0]
            ml_probs = ml_model.predict_proba([user_input])[0]
            time.sleep(0.2)
            
            if sem.available:
                st.write("🧠 Querying GPT-2 Semantic Vector Space...")
                sem_res = sem.predict_single(user_input)
            else:
                sem_res = None
            
            status.update(label="Triage Protocol Complete", state="complete", expanded=False)

        # Layout for results
        st.divider()
        st.subheader("Classification Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Layer 1: Heuristic**")
            render_styled_result(h_pred, "heuristic", alert_type="info")
            st.caption("Regex + pathogen taxonomy. Fast but brittle on negations.")

        with col2:
            st.markdown("**Layer 2: Morphological**")
            render_styled_result(m_pred, "morphological", alert_type="warning")
            st.caption("Latin root matching. Predicts unknown pathogens/metals.")

        with col3:
            st.markdown("**Layer 3: ML Model**")
            render_styled_result(ml_pred, "ml_model", alert_type="success")
            
            probs_df = pd.DataFrame({
                "Category": ml_model.classes,
                "Confidence": ml_probs,
            }).sort_values("Confidence", ascending=False)
            
            # Altair bar chart for confidence
            chart = (
                alt.Chart(probs_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X("Confidence:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1]), title=None),
                    y=alt.Y("Category:N", sort="-x", title=None),
                    color=alt.Color("Category:N", legend=None, scale=alt.Scale(
                        domain=["Biological", "Allergen", "Physical", "Chemical"],
                        range=["#10B981", "#F59E0B", "#6366F1", "#EC4899"]
                    )),
                    tooltip=["Category", alt.Tooltip("Confidence:Q", format=".1%")]
                )
                .properties(height=140)
            )
            st.altair_chart(chart, use_container_width=True)

        if sem_res:
            st.divider()
            st.subheader("🧠 GPT-2 Semantic Centroid Match")
            pred = sem_res["prediction"]
            render_styled_result(pred, "sem", alert_type="info", subtitle="Cosine similarity to class centroids")
            sem_df = pd.DataFrame({
                "Category": list(sem_res["scores"].keys()),
                "Cosine Similarity": list(sem_res["scores"].values()),
            }).sort_values("Cosine Similarity", ascending=False)
            st.dataframe(
                sem_df.style.format({"Cosine Similarity": "{:.4f}"}),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption(
                "ℹ️ GPT-2 semantic matching is optional. "
                "Uncomment `transformers` and `torch` in requirements.txt to enable."
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: BENCHMARK ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Benchmark Analytics":
    st.title("Benchmark Analytics & Overfitting Diagnostics")
    st.markdown(
        "Metrics are computed on the full dataset with **bootstrap confidence intervals** "
        "(500 resamples, 95%). ML in-sample metrics reflect the full-fit model. "
        "Cross-validation accuracy reveals true generalisation ability."
    )

    h_m  = results['heuristic_metrics']
    mo_m = results['morpho_metrics']
    ml_m = results['ml_metrics']
    cv   = results['cv_results']

    def fmt_ci(ci):
        return f"[{ci[0]:.1%} – {ci[1]:.1%}]"

    # KPI Layout
    st.markdown("### Top-Line Model Performance (In-Sample)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Heuristic Accuracy:** {h_m['accuracy']:.1%} \n\n (95% CI: {fmt_ci(h_m['acc_ci'])})")
    with col2:
        st.warning(f"**Morphological Accuracy:** {mo_m['accuracy']:.1%} \n\n (95% CI: {fmt_ci(mo_m['acc_ci'])})")
    with col3:
        st.success(f"**ML Model Accuracy:** {ml_m['accuracy']:.1%} \n\n (95% CI: {fmt_ci(ml_m['acc_ci'])})")

    # ── Confusion Matrices ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Confusion Matrix Breakdown")
    cm_cols = st.columns(3)
    
    for cm_col, m_dict, title in zip(
        cm_cols, 
        [h_m, mo_m, ml_m], 
        ["Heuristic", "Morphological", "ML (LogReg)"]
    ):
        with cm_col:
            st.markdown(f"**{title}**")
            cm_data = m_dict.get("confusion_matrix", {})
            if cm_data and "matrix" in cm_data:
                labels = cm_data["labels"]
                cm_df = pd.DataFrame(cm_data["matrix"], index=[f"T: {l}" for l in labels], columns=[f"P: {l}" for l in labels])
                st.dataframe(cm_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

    # ── Overfitting Check ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🧪 Overfitting Diagnostic: Stratified 5-Fold CV")
    st.markdown(
        "Because the ML model is trained on a small dataset, in-sample accuracy is inherently optimistic. "
        "The **5-fold CV accuracy** is the honest estimate of generalizability. "
        "A large gap (Δ) between in-sample and CV accuracy indicates memorization (overfitting)."
    )

    st.markdown(cv['verdict'])

    ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
    ov_col1.metric("In-Sample Accuracy", f"{cv['in_sample_acc']:.1%}")
    ov_col2.metric(
        "CV Accuracy (5-fold mean)",
        f"{cv['cv_acc_mean']:.1%}",
        delta=f"Δ gap: {cv['acc_gap']:.1%}",
        delta_color="inverse",
    )
    ov_col3.metric("In-Sample Macro F1", f"{cv['in_sample_f1']:.1%}")
    ov_col4.metric(
        "CV Macro F1 (5-fold mean)",
        f"{cv['cv_f1_mean']:.1%}",
        delta=f"Δ gap: {cv['f1_gap']:.1%}",
        delta_color="inverse",
    )

    fold_df = pd.DataFrame({
        "Fold Phase": [f"Validation Fold {i+1}" for i in range(cv['n_splits'])],
        "Accuracy": cv['fold_acc'],
        "Macro F1": cv['fold_f1'],
    })
    st.dataframe(
        fold_df.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: FAILURE MODE EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Failure Mode Explorer":
    st.title("Failure Mode & Edge Case Explorer")
    st.markdown("Filter and drill down into specific misclassifications and unclassified stress-test scenarios.")

    eval_df = results['df']

    filter_c1, filter_c2 = st.columns([1, 1])
    with filter_c1:
        classifier_choice = st.selectbox(
            "Target Architecture:",
            ["Heuristic", "Morphological", "ML Model"]
        )
    with filter_c2:
        risk_filter = st.selectbox(
            "Filter by Risk Severity:",
            ["All", "High", "Moderate", "Low"]
        )

    pred_col = {
        "Heuristic":     "heuristic_pred",
        "Morphological": "morpho_pred",
        "ML Model":      "ml_pred",
    }[classifier_choice]

    filtered_df = eval_df if risk_filter == "All" else eval_df[eval_df['risk_level'] == risk_filter]

    failures  = filtered_df[
        (filtered_df['true_category'] != filtered_df[pred_col]) &
        (~filtered_df[pred_col].isin(["Unknown", "N/A", "No signal"]))
    ]
    unknowns  = filtered_df[filtered_df[pred_col].isin(["Unknown", "N/A", "No signal"])]

    fcol1, fcol2 = st.columns(2)
    fcol1.metric(f"Misclassified by {classifier_choice}", len(failures))
    fcol2.metric(f"Unclassified (no signal)", len(unknowns))

    st.divider()

    if not failures.empty:
        st.subheader(f"⚠️ Known Misclassifications ({len(failures)})")
        for _, row in failures.iterrows():
            true_icon = CATEGORY_COLORS.get(row['true_category'], '')
            pred_icon = CATEGORY_COLORS.get(row[pred_col], '')
            label = (
                f"True: {true_icon} {row['true_category']} "
                f"| Predicted: {pred_icon} {row[pred_col]} | [{row['risk_level']} Risk]"
            )
            with st.expander(label):
                st.write(f"**Report Details:** {row['text']}")
                st.caption(f"Record ID: `{row['id']}`")

    if not unknowns.empty:
        st.subheader(f"❓ Unclassified / No Signal ({len(unknowns)})")
        st.dataframe(
            unknowns[['id', 'text', 'true_category', 'risk_level', pred_col]],
            use_container_width=True,
            hide_index=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: MODEL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Diagnostics":
    st.title("Model Diagnostics & Taxonomies")
    
    st.subheader("TF-IDF Feature Importance Matrix")
    st.markdown(
        "Top weighted bigrams and unigrams learned by the Logistic Regression model per class. "
        "These are the features actively driving ML predictions."
    )

    ml_top_features = results['ml_top_features']
    ml_classes      = results['ml_classes']

    cols = st.columns(len(ml_classes))
    for col, cat in zip(cols, ml_classes):
        with col:
            icon_html = get_category_icon_html(cat, width=24)
            st.markdown(f"<div style='display: flex; align-items: center;'>{icon_html}<strong style='font-size: 16px;'>{cat}</strong></div>", unsafe_allow_html=True)
            top = ml_top_features.get(cat, [])
            if top:
                feat_df = pd.DataFrame(top, columns=["Feature Token", "Reg. Weight"])
                st.dataframe(
                    feat_df.style.format({"Reg. Weight": "{:.3f}"}),
                    hide_index=True,
                )

    st.divider()
    st.subheader("Morphological Root Taxonomy Reference")
    st.markdown("""
| Pattern Type | Morphology Suffix/Prefix | Target Hazard Class |
|---|---|---|
| Biological (Bacteria) | `-ella` (Salmonella, Shigella) | 🦠 Biological |
| Biological (Bacteria) | `-coccus` (Staphylococcus) | 🦠 Biological |
| Biological (Bacteria) | `-bacter` (Campylobacter) | 🦠 Biological |
| Biological (Bacteria) | `-rium` (Clostridium) | 🦠 Biological |
| Biological (Virus) | `-virus` (Norovirus) | 🦠 Biological |
| Chemical (Plastics) | `-phthalate` (Diethylhexylphthalate) | 🧪 Chemical |
| Chemical (Organics) | `-aldehyde` (Acetaldehyde) | 🧪 Chemical |
| Chemical (Metals) | Heavy metals (Lead, Mercury) | 🧪 Chemical |
| Physical (Industrial) | Tungsten, Titanium, Graphite | 🔩 Physical |
| Physical (Debris) | Glass, Metal shavings, Ceramic | 🔩 Physical |

> **Allergen Caveat**: Morphological matching is *least reliable* for allergens.  
> Suffixes like `-in` (casein, penicillin) trigger false Chemical hits.  
> The system includes a morphological guard: if strong allergen keyword evidence is present, Chemical morphology scores are suppressed.
    """)
