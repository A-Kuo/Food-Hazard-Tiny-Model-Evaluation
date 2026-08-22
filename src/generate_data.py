import os
import pandas as pd
import random

def generate_dataset():
    data = []

    # === BIOLOGICAL ===
    bio_rows = [
        # Classic pathogens
        ("Company X is recalling its frozen spinach due to potential contamination with Listeria monocytogenes.", "Biological", "High"),
        ("Voluntary recall of ground beef that may be contaminated with E. coli O157:H7.", "Biological", "High"),
        ("Routine testing revealed Salmonella enterica in peanut butter batches.", "Biological", "High"),
        ("Smoked salmon recalled due to risk of Clostridium botulinum.", "Biological", "High"),
        ("Cantaloupes linked to an outbreak of Listeria.", "Biological", "High"),
        ("Raw milk recall due to Campylobacter jejuni risk.", "Biological", "Moderate"),
        ("Sprouts recalled after Salmonella detected in routine testing.", "Biological", "High"),
        ("Norovirus traced back to frozen berries.", "Biological", "High"),
        ("Vibrio parahaemolyticus found in raw oysters, prompting recall.", "Biological", "High"),
        ("Frozen chicken products recalled due to Salmonella typhimurium contamination.", "Biological", "High"),
        ("Ready-to-eat deli meats recalled after Listeria monocytogenes detected.", "Biological", "High"),
        ("Shigella sonnei contamination found in bagged salad mix.", "Biological", "High"),
        ("Hepatitis A virus traced to imported green onions.", "Biological", "High"),
        ("Yersinia enterocolitica found in pasteurized pork chitterlings.", "Biological", "High"),
        ("Bacillus cereus toxin detected in pre-cooked rice products.", "Biological", "Moderate"),
        ("Staphylococcus aureus enterotoxin found in deli ham.", "Biological", "Moderate"),
        ("Cyclospora cayetanensis linked to fresh basil imports.", "Biological", "High"),
        ("Cryptosporidium parvum detected in apple cider.", "Biological", "High"),
        ("Giardia lamblia found in contaminated spring water.", "Biological", "Moderate"),
        ("Toxoplasma gondii detected in undercooked lamb.", "Biological", "Moderate"),
        ("Brucella abortus risk from raw goat cheese product.", "Biological", "High"),
        ("Trichinella spiralis parasite risk in wild boar sausage.", "Biological", "High"),
        ("Ascaris lumbricoides eggs found in imported herbs.", "Biological", "Moderate"),
        ("Recall triggered by detection of Cronobacter sakazakii in infant formula.", "Biological", "High"),
        ("Aspergillus flavus mold contamination in dried figs.", "Biological", "Moderate"),
        ("Penicillium roqueforti contamination in aged cheddar (unexpected strain).", "Biological", "Low"),
        ("Recall of raw sprouts due to enterohemorrhagic E. coli (EHEC).", "Biological", "High"),
        ("Salmonella heidelberg detected in smoked trout.", "Biological", "High"),
        ("Legionella pneumophila risk in commercially bottled spring water.", "Biological", "High"),
        ("Taenia saginata tapeworm cysts found in beef tartare.", "Biological", "High"),
        # Tricky biological phrasing
        ("The product may support growth of harmful microorganisms including Staphylococcus.", "Biological", "Moderate"),
        ("Risk of bacterial proliferation of E. coli O104:H4 identified.", "Biological", "High"),
        ("Spore-forming pathogen Bacillus anthracis flagged in protein powder recall.", "Biological", "High"),
    ]

    # === ALLERGEN ===
    allergen_rows = [
        ("Chocolate bars recalled because they contain undeclared peanuts.", "Allergen", "High"),
        ("Undeclared milk allergen found in vegan cheese alternative.", "Allergen", "High"),
        ("Mislabeled packaging failed to disclose presence of soy in teriyaki sauce.", "Allergen", "High"),
        ("Recall due to undeclared wheat in gluten-free oat cookies.", "Allergen", "High"),
        ("Undeclared crustacean shellfish (shrimp) in spring rolls.", "Allergen", "High"),
        ("Tree nut warning: product may contain traces of undeclared almonds.", "Allergen", "Moderate"),
        ("Undeclared egg in pasta product.", "Allergen", "High"),
        ("Sesame seed buns packaged and labeled as plain buns.", "Allergen", "High"),
        ("Allergy alert: undeclared fish (pollock) in fish sticks.", "Allergen", "High"),
        ("Undeclared lupin flour in bread product.", "Allergen", "High"),
        ("Mustard seeds not declared on label of spice blend.", "Allergen", "Moderate"),
        ("Undeclared cashew in mixed granola bar.", "Allergen", "High"),
        ("Pecan pie contains undeclared walnuts.", "Allergen", "High"),
        ("Undeclared hazelnut in chocolate spread.", "Allergen", "High"),
        ("Recall of almond milk that was found to contain undeclared dairy.", "Allergen", "High"),
        ("Undeclared celery in vegetable broth concentrate.", "Allergen", "Moderate"),
        ("Product contains latex-reactive fruit (kiwi) not declared.", "Allergen", "Moderate"),
        ("Undeclared sulfites in wine labelled as sulfite-free.", "Allergen", "Moderate"),
        ("Recall of protein bar due to undeclared milk and soy.", "Allergen", "High"),
        ("Corn starch product contaminated with undeclared wheat gluten.", "Allergen", "High"),
        ("Coconut-based product contains undeclared tree nuts.", "Allergen", "High"),
        ("Undeclared pistachio in premium nut mix.", "Allergen", "High"),
        ("Baby formula contains undeclared soy protein.", "Allergen", "High"),
        ("Macadamia nuts not declared on confectionery label.", "Allergen", "Moderate"),
        ("Canned soup recalled for undeclared barley (gluten).", "Allergen", "High"),
    ]

    # === PHYSICAL ===
    physical_rows = [
        ("Recalling canned baked beans because they may contain small pieces of plastic.", "Physical", "Moderate"),
        ("Glass fragments found in jars of baby food.", "Physical", "High"),
        ("Metal shavings discovered in ground turkey.", "Physical", "High"),
        ("Wood splinters found in bagged salads.", "Physical", "Moderate"),
        ("Rubber pieces reported by consumers in frozen chicken nuggets.", "Physical", "Moderate"),
        ("Bread recalled due to possible presence of foreign material (wire).", "Physical", "High"),
        ("Stones found in bulk lentils.", "Physical", "Low"),
        ("Bone fragments found in boneless chicken breast.", "Physical", "Low"),
        ("Ceramic shards detected in canned tomatoes.", "Physical", "High"),
        ("Nails found in imported dried fish product.", "Physical", "High"),
        ("Foil fragments in chocolate bar.", "Physical", "Moderate"),
        ("Cardboard contamination in protein powder.", "Physical", "Low"),
        ("Silica desiccant packets found inside chocolate packaging.", "Physical", "Moderate"),
        ("Metal staple found in ready-made salad.", "Physical", "High"),
        ("Plastic film strip discovered in frozen lasagna.", "Physical", "Moderate"),
        ("Fiberglass insulation particles detected in dried spice mixture.", "Physical", "High"),
        ("Broken thermometer glass inside beverage bottle.", "Physical", "High"),
        ("Gravel contamination reported in imported quinoa.", "Physical", "Low"),
        ("Jewelry fragment (ring) found in bread loaf.", "Physical", "High"),
        ("Sand and soil contamination in frozen peas.", "Physical", "Low"),
        # Unusual physical — things that don't normally appear
        ("Recall triggered by presence of tungsten filings found in canned tuna.", "Physical", "High"),
        ("Gold flake contamination from manufacturing equipment in candy product.", "Physical", "Moderate"),
        ("Titanium shavings from industrial machinery found in rice flour.", "Physical", "High"),
        ("Graphite particles from conveyor belt in granola bars.", "Physical", "Moderate"),
        ("Carbon fiber fragments from processing equipment in dried pasta.", "Physical", "High"),
        ("Zirconium oxide ceramic debris detected in baby cereal.", "Physical", "High"),
        ("Lead solder fragments from equipment failure found in olive oil.", "Physical", "High"),  # note: lead also chemical
        ("Copper wire strand found inside packaged cheese.", "Physical", "Moderate"),
    ]

    # === CHEMICAL ===
    chemical_rows = [
        ("High levels of lead detected in cinnamon apple sauce.", "Chemical", "High"),
        ("Elevated levels of aflatoxin found in corn meal.", "Chemical", "High"),
        ("Product recalled due to pesticide residue above legal limit.", "Chemical", "Moderate"),
        ("Cleaning agent contamination (sanitizer) found in bottled water.", "Chemical", "Moderate"),
        ("Juice recalled due to elevated levels of patulin.", "Chemical", "Moderate"),
        ("Histamine levels above FDA limits in tuna steaks.", "Chemical", "High"),
        ("Melamine contamination detected in protein powder.", "Chemical", "High"),
        ("Mercury levels exceed safety threshold in imported swordfish.", "Chemical", "High"),
        ("Cadmium contamination found in organic brown rice.", "Chemical", "High"),
        ("Arsenic levels above FDA guidance in apple juice.", "Chemical", "High"),
        ("Dioxin contamination in farmed salmon feed product.", "Chemical", "High"),
        ("Polychlorinated biphenyls (PCBs) found in fish oil capsules.", "Chemical", "High"),
        ("Bisphenol A (BPA) leaching from can lining into tomato product.", "Chemical", "Moderate"),
        ("Acrylamide levels exceed ALARA guidelines in potato chips.", "Chemical", "Moderate"),
        ("Malachite green dye illegally used on farmed shrimp.", "Chemical", "High"),
        ("Sudan red dye found in paprika spice product.", "Chemical", "High"),
        ("Excessive nitrate levels in leafy greens.", "Chemical", "Moderate"),
        ("Ochratoxin A detected in dried grape products.", "Chemical", "Moderate"),
        ("Chloramphenicol antibiotic residue in imported honey.", "Chemical", "High"),
        ("Fumonisins B1 and B2 above threshold in corn-based snack.", "Chemical", "Moderate"),
        ("Trichothecene mycotoxin detected in wheat flour.", "Chemical", "Moderate"),
        ("Deoxynivalenol (vomitoxin/DON) found in oat products.", "Chemical", "Moderate"),
        ("Hydrogen cyanide above safe levels in improperly processed cassava flour.", "Chemical", "High"),
        ("Ethylene oxide residue found in sesame seeds.", "Chemical", "High"),
        ("Benzene detected above EPA limits in carbonated soft drink.", "Chemical", "High"),
        ("Phthalate plasticizers found migrating from packaging into dairy product.", "Chemical", "Moderate"),
        ("Residual veterinary drug (tetracycline) above MRL in chicken.", "Chemical", "Moderate"),
        ("Zearalenone mycotoxin found in corn-based breakfast cereal.", "Chemical", "Moderate"),
    ]

    # === EDGE CASES (intentionally ambiguous / adversarial for model stress-testing) ===
    edge_rows = [
        # Negation traps
        ("The product was tested negative for Salmonella and E. coli, but is recalled due to undeclared milk.", "Allergen", "High"),
        ("Lab results were negative for Listeria. Recall is due to metal wire fragments.", "Physical", "High"),
        ("Despite testing free of all pathogens, the product is recalled due to elevated arsenic.", "Chemical", "High"),
        ("No biological contamination found. Recall due to undeclared peanuts in the chocolate.", "Allergen", "High"),
        # Certification paradox
        ("Certified peanut-free facility, but recalling due to metal shavings from equipment.", "Physical", "High"),
        ("Gluten-free pasta recalled for potential Listeria contamination.", "Biological", "High"),
        ("Dairy-free label recalled because milk protein casein was undeclared.", "Allergen", "High"),
        # Cross-hazard descriptions
        ("Warning: processed in a facility that also handles soy. Actual recall is for glass pieces.", "Physical", "Moderate"),
        ("Product contains added calcium (from milk sources) not listed, and has elevated lead.", "Allergen", "High"),  # borderline allergen/chemical
        # Unusual physical foreign objects
        ("Recall of canned fish due to presence of mercury thermometer glass.", "Physical", "High"),  # physical, not chemical mercury
        ("Consumer found what appeared to be a small piece of gold jewelry in the bread loaf.", "Physical", "High"),
        ("Tungsten carbide grinding media found in spice blend.", "Physical", "High"),
        # Latin root stress tests (unknown pathogen names, model must use morphology)
        ("Recall triggered by detection of Pseudomonella virulens, an emerging pathogen.", "Biological", "High"),  # fake name ending in -ella
        ("Contamination with Clostriferens toxicus detected in canned goods.", "Biological", "High"),  # fake Clostri- + -ferens
        ("Unknown Staphylocryptus strain found in ready-to-eat salad.", "Biological", "High"),  # fake -coccus variant
        # Chemical compound stress tests
        ("Recall due to elevated concentrations of diethylhexylphthalate in packaging migration.", "Chemical", "Moderate"),
        ("Polybrominated diphenyl ethers (PBDEs) above limit in fish product.", "Chemical", "High"),
        ("Presence of acetaldehyde above safety threshold in juice product.", "Chemical", "Moderate"),
        # Allergen-chemical boundary (sulfites)
        ("Undeclared sulfites in dried apricots may cause reactions in sensitive individuals.", "Allergen", "Moderate"),
        # Unclear subject first (predicate-first phrasing)
        ("Found in routine inspection: Vibrio cholerae in oyster harvest batch.", "Biological", "High"),
        ("Detected during export screening: aflatoxin B1 above threshold in groundnuts.", "Chemical", "High"),
        ("Confirmed by consumer report: broken ceramic in jarred baby food.", "Physical", "High"),
        ("Reported by allergist: anaphylaxis linked to undeclared sesame in crackers.", "Allergen", "High"),
        # Mixed low-stakes
        ("Tested positive for elevated sodium, no pathogens found.", "Chemical", "Low"),
        ("Mild spoilage odor reported; no pathogens detected above threshold.", "Biological", "Low"),
    ]

    random.seed(42)
    all_rows = bio_rows + allergen_rows + physical_rows + chemical_rows + edge_rows
    random.shuffle(all_rows)

    for i, (text, category, risk) in enumerate(all_rows):
        data.append({
            "id": f"REC-{1000 + i}",
            "text": text,
            "true_category": category,
            "risk_level": risk
        })

    df = pd.DataFrame(data)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "food_recalls_sample.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows in {out_path}")
    print(df['true_category'].value_counts().to_string())
    return df

if __name__ == "__main__":
    generate_dataset()
