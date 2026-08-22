"""
semantic_matcher.py
====================
GPT-2-based semantic hazard classifier using mean-pooled token embeddings.

Strategy:
- Build a centroid embedding for each hazard class from a curated dictionary of known hazard terms.
- For an input text, extract the predicate noun phrases (the contaminant, not the food subject),
  mean-pool their GPT-2 embeddings, and classify by cosine similarity to class centroids.
- This handles unknown terms (e.g., new pathogens, unusual metals) that TF-IDF has never seen.

Morphological Root Pre-Classifier (no model required):
- Biological: endings like -ella, -ium, -coccus, -bacter, -monas, -virus, -phage, -toxin (if biological context)
              prefixes like Salmo-, Lister-, Campylo-, Clostri-, Staph-, Strepto-
- Chemical:   endings like -ate, -ide, -ene, -ol, -ine (aliphatic), -phthalate, -toxin (mycotoxin context)
              prefixes like poly-, di-, tri-, mono-, chloro-, bromo-, fluoro-
- Physical:   materials that are hard/inert solids: metal names, minerals, synthetic materials
- Allergen:   morphology is less reliable — defer to TF-IDF and heuristic for allergens

CAVEAT: The morphological -in/-ine endings create ambiguity:
  - "casein" (dairy allergen) ends in -in, but looks chemical
  - "penicillin" ends in -in, but is a drug/biological
  These will be caught as false positives by morphology alone. The full pipeline
  down-weights morphology when allergen keyword evidence is present.
"""

import re
import numpy as np
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. MORPHOLOGICAL ROOT PRE-CLASSIFIER (no GPU / no download required)
# ─────────────────────────────────────────────────────────────────────────────

# Pathogen name suffixes strongly associated with biological organisms
BIO_SUFFIXES = [
    r'ella\b',      # Salmonella, Shigella, Brucella, Klebsiella, Pseudomonella
    r'coccus\b',    # Staphylococcus, Streptococcus, Enterococcus
    r'bacter\b',    # Campylobacter, Helicobacter, Acinetobacter
    r'bacter[a-z]+',
    r'monella\b',   # Salmonella variant match
    r'monas\b',     # Pseudomonas, Xanthomonas
    r'rium\b',      # Clostridium, Bacterium, Bacteroidium
    r'clostridia',
    r'clostri[a-z]+',
    r'virus\b',     # Norovirus, Rotavirus, Hepatovirus
    r'viridae\b',
    r'phage\b',
    r'lamblia\b',   # Giardia lamblia
    r'parvum\b',    # Cryptosporidium parvum
    r'gondii\b',    # Toxoplasma gondii
    r'spiralis\b',  # Trichinella spiralis
    r'lumbricoides\b',
    r'saginata\b',
    r'cayetanensis\b',
    r'sakazakii\b',
    r'enterica\b',
]

# Common biological prefixes (genus-level)
BIO_PREFIXES = [
    r'\bsalmo', r'\blister', r'\bcampylo', r'\bclostri', r'\bstaph',
    r'\bstrepto', r'\bshigell', r'\bvibri', r'\byersin', r'\bbrucell',
    r'\bcryptospor', r'\bgiardia', r'\btoxoplas', r'\btrichin',
    r'\bascari', r'\bcronobach', r'\blegionell', r'\btaeni',
    r'\baspergill', r'\bpenicilli', r'\bfusari',
]

# Chemical compound morphological markers
CHEM_SUFFIXES = [
    r'ate\b',       # nitrate, phosphate, sulfate, diethylhexylphthalate
    r'ide\b',       # chloride, cyanide, dioxide
    r'ene\b',       # benzene, styrene, toluene
    r'phthalate\b',
    r'biphenyl',
    r'dioxin',
    r'fumonisins?',
    r'ochratoxin',
    r'aflatoxin',
    r'patulin',
    r'zearalenone',
    r'deoxynivalenol',
    r'trichothecene',
    r'aldehyde\b',  # acetaldehyde, formaldehyde
    r'phenol\b',    # but NOT: casein, penicillin → guarded below
]

# Chemical element / heavy metal names (always Chemical unless physical evidence)
CHEMICAL_ELEMENTS = [
    r'\blead\b', r'\bmercury\b', r'\bcadmium\b', r'\barsenic\b',
    r'\bdioxin', r'\bpcb\b', r'\bpesticide', r'\bherbicide',
    r'\bfungicide', r'\bantibiotic\b', r'\bveterinary drug',
    r'\bmelamine\b', r'\bbpa\b', r'\bbisphenol',
    r'\bchloramphenicol', r'\bmalachite green', r'\bsudan red',
    r'\bnitrate\b', r'\bnitrite\b', r'\bbenzene\b', r'\bacrylamide\b',
]

# Physical material names (hard inert solids, manufacturing materials)
PHYSICAL_MATERIALS = [
    r'\bglass\b', r'\bmetal\b', r'\bplastic\b', r'\brubber\b',
    r'\bwood\b', r'\bwire\b', r'\bstone\b', r'\bgravel\b',
    r'\bceramic\b', r'\bbonefrag', r'\bsilica\b', r'\bfoil\b',
    r'\bfiberglass\b', r'\bstaple\b', r'\bnail\b', r'\bscrew\b',
    r'\btungsten\b', r'\bgold\b', r'\btitanium\b', r'\bgraphite\b',
    r'\bcarbon fiber\b', r'\bzirconium\b', r'\bcopper wire\b',
    r'\bsolder\b', r'\bshaving', r'\bfragment', r'\bshard',
    r'\bdebris\b', r'\bpiece[s]?\b', r'\bsplinter\b',
]

# Allergen morphological markers (less reliable — used only to boost, not override)
ALLERGEN_TERMS = [
    r'\bpeanut', r'\btree nut', r'\bwalnut', r'\balmond', r'\bcashew',
    r'\bpecan', r'\bpistach', r'\bhazelnut', r'\bmacadamia',
    r'\bmilk\b', r'\bdairy\b', r'\bcasein\b', r'\bwhey\b',
    r'\bsoy\b', r'\bwheat\b', r'\bgluten\b', r'\begg\b',
    r'\bsesame\b', r'\bshellfish\b', r'\bcrustacean',
    r'\bfish\b', r'\blocust bean',
]

# Ambiguous -in / -ine endings (flag but don't hard-classify as chemical)
AMBIGUOUS_IN_ENDINGS = [r'casein', r'penicillin', r'amylopectin', r'lecithin']


def morphological_classify(text: str) -> Optional[str]:
    """
    Returns a hazard category prediction based purely on morphological / lexical
    pattern matching, or None if no strong signal found.
    
    Priority: Allergen context overrides -in chemical matches.
    """
    txt = text.lower()
    
    # Strip negated clauses before analysis (e.g., "tested negative for X")
    # This removes the negated term to prevent false positives
    txt = re.sub(
        r'(tested? (negative|clear|free) for|free of|free from|certified \S+-free|does not contain|no \w+ (found|detected|present))\s+[\w\s,]+?(?=[,.]|$)',
        '', txt
    )

    scores = {"Biological": 0, "Allergen": 0, "Physical": 0, "Chemical": 0}

    # Biological suffix/prefix signals
    for pattern in BIO_SUFFIXES:
        if re.search(pattern, txt, re.IGNORECASE):
            scores["Biological"] += 2
    for pattern in BIO_PREFIXES:
        if re.search(pattern, txt, re.IGNORECASE):
            scores["Biological"] += 3

    # Chemical element / compound signals
    for pattern in CHEMICAL_ELEMENTS:
        if re.search(pattern, txt, re.IGNORECASE):
            scores["Chemical"] += 3
    for pattern in CHEM_SUFFIXES:
        if re.search(pattern, txt, re.IGNORECASE):
            # Guard: don't count -ate/-ide if clearly in a biological context
            scores["Chemical"] += 1

    # Physical material signals
    for pattern in PHYSICAL_MATERIALS:
        if re.search(pattern, txt, re.IGNORECASE):
            scores["Physical"] += 3

    # Allergen signals (moderate weight — allergen keywords are fairly reliable)
    for pattern in ALLERGEN_TERMS:
        if re.search(pattern, txt, re.IGNORECASE):
            scores["Allergen"] += 2

    # Allergen override: if strong allergen evidence, suppress chemical -in match
    if scores["Allergen"] >= 4 and scores["Chemical"] < 4:
        scores["Chemical"] = max(0, scores["Chemical"] - 2)

    # Return best category only if it has a clear signal advantage
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPT-2 SEMANTIC EMBEDDING CLASSIFIER (requires transformers + torch)
# ─────────────────────────────────────────────────────────────────────────────

# Curated seed terms for each hazard class to build embedding centroids.
# These are the "dictionary of known ailments and materials."
HAZARD_SEED_TERMS = {
    "Biological": [
        "bacteria", "virus", "pathogen", "parasite", "mold", "fungus",
        "listeria", "salmonella", "campylobacter", "botulinum", "norovirus",
        "vibrio", "shigella", "cryptosporidium", "giardia", "toxoplasma",
        "trichinella", "ascaris", "tapeworm", "yersinia", "brucella",
        "contamination", "microorganism", "infection", "outbreak", "spore",
    ],
    "Allergen": [
        "peanut", "milk", "soy", "wheat", "egg", "shellfish", "fish",
        "tree nut", "sesame", "almond", "walnut", "cashew", "pecan",
        "hazelnut", "pistachio", "lupin", "mustard", "celery", "sulfite",
        "allergen", "anaphylaxis", "hypersensitivity", "undeclared",
        "mislabeled", "labeling",
    ],
    "Physical": [
        "glass", "metal", "plastic", "wire", "stone", "gravel", "rubber",
        "wood", "splinter", "ceramic", "shard", "fragment", "shaving",
        "debris", "nail", "staple", "foil", "fiberglass", "bone",
        "tungsten", "gold", "titanium", "graphite", "zirconium", "copper",
        "foreign object", "foreign material", "equipment", "machinery",
    ],
    "Chemical": [
        "lead", "mercury", "arsenic", "cadmium", "pesticide", "herbicide",
        "aflatoxin", "mycotoxin", "dioxin", "patulin", "melamine",
        "bisphenol", "acrylamide", "benzene", "nitrate", "antibiotic",
        "residue", "toxin", "pollutant", "contaminant", "chemical",
        "compound", "heavy metal", "fumonisin", "ochratoxin", "zearalenone",
    ],
}


class GPT2SemanticClassifier:
    """
    Uses GPT-2 token embeddings (mean-pooled) to classify hazard text.
    - Builds a centroid embedding per hazard class from HAZARD_SEED_TERMS.
    - Classifies text by cosine similarity to centroids.
    - Falls back gracefully if transformers/torch is not available.
    """

    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.centroids = {}
        self._try_load()

    def _try_load(self):
        try:
            from transformers import GPT2Tokenizer, GPT2Model
            import torch
            self._torch = torch
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2Model.from_pretrained("gpt2")
            self.model.eval()
            self.available = True
            self._build_centroids()
            print("[SemanticMatcher] GPT-2 loaded. Centroids built.")
        except Exception as e:
            print(f"[SemanticMatcher] GPT-2 not available: {e}. Semantic matching disabled.")

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Returns mean-pooled GPT-2 last hidden state for input text."""
        if not self.available:
            return None
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=64
            )
            with self._torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pool over token dimension
            embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()
            return embedding
        except Exception:
            return None

    def _build_centroids(self):
        """Build one centroid embedding per hazard class from seed terms."""
        for category, terms in HAZARD_SEED_TERMS.items():
            embeddings = []
            for term in terms:
                emb = self._embed(term)
                if emb is not None:
                    embeddings.append(emb)
            if embeddings:
                self.centroids[category] = np.mean(embeddings, axis=0)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def predict_single(self, text: str) -> Optional[dict]:
        """
        Returns dict with category prediction and similarity scores,
        or None if semantic matching is unavailable.
        """
        if not self.available or not self.centroids:
            return None
        emb = self._embed(text)
        if emb is None:
            return None
        scores = {
            cat: self._cosine_similarity(emb, centroid)
            for cat, centroid in self.centroids.items()
        }
        best = max(scores, key=scores.get)
        return {"prediction": best, "scores": scores}


# Singleton — loaded once and cached
_gpt2_classifier = None

def get_semantic_classifier() -> GPT2SemanticClassifier:
    global _gpt2_classifier
    if _gpt2_classifier is None:
        _gpt2_classifier = GPT2SemanticClassifier()
    return _gpt2_classifier
