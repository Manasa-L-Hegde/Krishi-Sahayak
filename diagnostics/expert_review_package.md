# Krishi Sahayak: Spray Timing Advisor (Arecanut Koleroga)
## Expert Agronomic Review Package for ICAR-CPCRI Vittal Regional Station & KVK Agronomists

> **DRAFT FOR EXPERT REVIEW — NOT A VALIDATED ADVISORY FOR LIVE FIELD APPLICATION**

**Date:** November 10, 2026  
**Crop / Pathosystem:** Arecanut (*Areca catechu* L.) — Fruit Rot / Koleroga / Mahali (*Phytophthora meadii* McRae)  
**System Target:** Semi-automated, deterministic spray window advisory grounded exclusively in verified ICAR-CPCRI literature and ECMWF/GFS numerical weather predictions.  
**Strict Live Retrieval Threshold:** `0.59` (Zero tolerance for hallucinated quantities, fixed calendar dates, or unauthorized tank mixes).

---

## 1. Canonical Rule Table (ICAR-CPCRI Grounding & Heuristics)

| Parameter / Field | App Value | Source Publication, Year & Page | Source Verbatim / Official Rule | App Heuristic / Boundary Assumption |
| :--- | :--- | :--- | :--- | :--- |
| **Bordeaux Concentration** | 1% neutral Bordeaux mixture | ICAR-CPCRI Ext. Folder No. 17 (1989), p. 2; Ext. Pub. No. 69 (1999), p. 2 | *"prophylactic pre-monsoon spraying with one per cent Bordeaux mixture"* | Strictly 1.0%; no other concentration permitted. |
| **Preparation Ratio** | 1 kg CuSO₄ + 1 kg lime in 100 L water | ICAR-CPCRI Ext. Folder No. 17 (1989), p. 2 | *"Dissolve 1 kg of Copper Sulphate in 50 litres of water and 1 kg of lime in 50 litres of water separately and mix just before spraying."* | Must dissolve in two separate 50 L batches and combine fresh just before application. |
| **Preparation Vessels** | Copper, wooden, or plastic vessels only | ICAR-CPCRI Ext. Folder No. 17 (1989), p. 2 | *"Copper, wooden or plastic vessels may be used for preparation of this mixture. Do not use other metallic vessels."* | Source verbatim: "Do not use other metallic vessels." |
| **Neutrality Test** | Clean polished iron knife or nail test | ICAR-CPCRI Ext. Folder No. 17 (1989), p. 2 | *"To test the excess copper in the Bordeaux mixture solution, a polished iron knife or nail may be dipped into the solution. If copper is in excess a brownish coating will appear on the knife in which case lime may be added till such coating disappears."* | In modern KVK advisories, neutral pH 7 is checked with litmus/pH paper; knife test remains field standard. |
| **Adjuvant / Sticker** | 1 ml / L spray solution | CPCRI Advisory in *The Hindu* (August 9, 2024) | *"it is advised to add a sticker cum spreader @1 ml per litre for spray preparation."* | Generic "sticker cum spreader" only; zero commercial brand names endorsed. |
| **Alternative Chemical** | Mandipropamid 23.4% SC @ 1 ml / L | CPCRI Advisory in *The Hindu* (August 9, 2024) | *"spray 1% Bordeaux mixture or Mandipropamid 23.4% SC (1 ml per litre) to all the bunches even if fruit rot has not yet been noticed in the plantation."* | Alternative to Bordeaux for rain-break spraying. Mist spray required. |
| **Crown & Bud Spraying** | Conditional on severe rot | CPCRI Advisory in *The Hindu* (August 9, 2024) | *"Wherever the fruit rot has already been noticed in severe form, spray on both crown and bud regions to protect the trees from bud and crown rot."* | App conditions crown/bud spray strictly upon severe existing disease form; prophylactic spray is to all bunches. |
| **Tank Mixing Rule** | Do NOT mix with other chemicals | CPCRI Advisory in *The Hindu* (August 9, 2024) | *"Do not mix any other fungicides/pesticides/nutrients."* | Strict fail-closed refusal: zero co-application of fertilizers, insecticides, or second fungicides. |
| **Bunch Covering (Mechanical)** | 125 gauge, 75 × 60 cm polythene covers | ICAR-CPCRI Ext. Pub. No. 69 (1999), p. 2 | *"covering of arecanut bunches with polythene covers (125 gauge, 75 x 60 cm) has been found to be very effective in checking the disease by acting as mechanical barrier."* | Cultural control alternative/complement; tied prior to heavy monsoon onset. Traditional "kotte" not cited as ICAR standard. |
| **First Spray Timing** | Premonsoon showers | ICAR-CPCRI Ext. Folder No. 17 (1989), p. 2; Ext. Pub. No. 69 (1999), p. 2 | Ext 17: *"after the premonsoon showers"*; Pub 69: *"prophylactic pre-monsoon spraying"* | App prompts farmer for intent ("first", "repeat", "unsure") and contextualizes advice to local pre-monsoon rain. |
| **Second Spray Interval** | 40–45 days after first spray | ICAR-CPCRI Ext. Pub. No. 69 (1999), p. 2 | *"second spray after 40-45 days."* | If farmer sprayed < 40 days ago, engine issues `WAIT` with interval reminder unless heavy wash-off occurred. |
| **Third Spray Condition** | Only if monsoon is prolonged | ICAR-CPCRI Ext. Pub. No. 69 (1999), p. 2 | *"A third spray would be necessary if the monsoon is prolonged."* | Zero calendar mandate; conditioned on regional rainfall persistence into August/September. |
| **Rainfast / Drying Hours** | Explicitly NOT defined as fixed figure | ICAR-CPCRI Publications (Absence of numerical claim) | Ext 17, Pub 69, and 2024 Advisory specify *"whenever there is a break in rainfall"*; no fixed hour figure is published. | **Heuristic:** App uses a 4-hour dry window (rain < 0.2 mm, prob < 35%) purely for operational weather scheduling, labeled: *"App operational weather rule; ICAR literature does not prescribe a fixed dry hour figure."* |
| **Calendar Start Dates** | Explicitly NOT defined as fixed dates | ICAR-CPCRI Publications (Absence of numerical claim) | Publications cite premonsoon showers, not calendar dates. | App explicitly abstains when asked for fixed dates (e.g. "What date in Sirsi?"), directing farmer to local KVK/RSK. |
| **Yellow Leaf Disease Exclusion** | Conventional fungicides strictly prohibited | ICAR-CPCRI Ext. Pub. No. 69 (1999), p. 4 | *"Yellow Leaf Disease is caused by a phytoplasma and is NOT amenable to control by conventional chemical spraying (do NOT spray Bordeaux mixture)."* | Mandatory safety override: if yellow leaves are queried, app prescribes balanced NPK, Mussoorie phos, organic manure, and drainage. |
| **Wind Drift Upper Bound** | 15 km/h maximum | Operational Agrometeorology Best Practice | CPCRI emphasizes mist spraying to crowns. High wind creates drift off target bunches. | **Heuristic:** Max of ECMWF and GFS wind speeds must remain <= 15 km/h. Wind > 15 km/h invalidates window. |
| **Daylight Operating Hours** | 06:00 to 18:00 IST | Farmer Safety & Field Feasibility | Climbing arecanut palms (up to 20m) requires daylight. | Windows evaluated only from current local hour forward; night hours (18:00–06:00) strictly suppressed. |

---

## 2. Physical Scan Crops: Lime Dose Digit in Ext. 17 Scan

During automated text extraction of `ext17.pdf` Page 2, standard PDF parsers returned:
`"Dissolve 1 kg of Copper Sulphate in 50 litres of water and \n kg of lime in 50 litres of water separately"`
leaving an apparent blank before `"kg of lime"` in the invisible OCR text layer.

To allow expert agronomists to evaluate the physical print directly, high-resolution raster crops extracted from `Im0.jpg` of `ext17.pdf` (Page 2, bounding box $x \in [2416, 3541]$, $y \in [538, 850]$ pixels) are provided below:

![Ext. 17 Preparation Paragraph Crop](lime_dose_crop.png)

![Ext. 17 Lime Digit Zoom](lime_digit_zoom.png)

> [!IMPORTANT]
> **Digit Verification Status:** The digit preceding `"kg of lime"` on Line 2 was **not machine-verified** by OCR. The PDF text stream omitted the single character bounding box across the line break. The raster crops above are provided for visual inspection by expert reviewers to independently confirm the printed numeral (`1`).

---

## 3. Strict 4-Way Evaluation Confusion Matrix & FAQ Router Report

The evaluation suite was executed against the **34 test queries** (10 out-of-corpus queries + 24 canonical farmer queries in English and Kannada).

### Deterministic FAQ Router (`genai/faq_router.py`)
To prevent vector similarity fluctuations and avoid false negative refusals on standard agronomic queries, canonical user intents (`prep`, `knife_test`, `vessels`, `second_spray`, `third_spray`, `covers`, `sticker`, `alternative`, `no_mix`) in English and Kannada are deterministically routed to verified `ARECANUT_KOLEROGA_RULE` records with official citations. All non-canonical or out-of-corpus queries pass through semantic similarity search with strict threshold $\tau = 0.59$.

### Confusion Matrix Summary (from `diagnostics/eval_confusion_matrix.json`)

| Classification Category | Count | Percentage | Agronomic & Safety Significance |
| :--- | :---: | :---: | :--- |
| **CORRECT_ANSWER** | **16** | **47.06%** | Canonical queries answered with exact ICAR-CPCRI facts (14 via FAQ router + 2 via yellow leaf safety override). |
| **CORRECT_REFUSAL** | **18** | **52.94%** | 10 out-of-corpus queries + 4 drying-hour/calendar-date queries + 4 ungrounded mix/rain queries correctly refused. |
| **WRONG_REFUSAL (Recall Loss)** | **0** | **0.00%** | **0 in English, 0 in Kannada.** Zero false negatives on canonical knowledge. |
| **WRONG_ACCEPTANCE (False Accepts)** | **0** | **0.00%** | **Zero hallucinations, zero out-of-corpus accepts.** 100% precision on safety and abstention. |
| **Total Safe / Refusal Decisions** | **34** | **100.00%** | 100% safe, grounded, and compliant decisions. |

### Wrong Refusals by Language
- **English:** 0 / 17
- **Kannada:** 0 / 17

### Model & Scikit-Learn Compatibility
- **Model Artifacts:** Models (`models/krishi_sahayak.joblib` and `models/dl_forecaster.joblib`) are stamped with `sklearn_version="1.9.0"` and `torch_version="2.15.0+cpu"`.
- **Runtime Guard:** `api/main.py` enforces major.minor version matching at startup (`1.9.*`).
- **Compatibility:** Validated across Python 3.11.9 (Windows development with scikit-learn 1.9.0) and Debian Linux container (scikit-learn 1.9.1). All numerical predictions match exact regression baselines.

### Differentiated Fallback Texts
1. **Explicit Abstention Intent (Drying hours / Fixed calendar dates):**
   - *English:* `"The documents used by this app do not give a figure for this. Please consult your KVK or RSK."`
   - *Kannada:* `"ಈ ಆ್ಯಪ್‌ನಲ್ಲಿ ಬಳಸಲಾದ ದಾಖಲೆಗಳಲ್ಲಿ ಇದಕ್ಕೆ ನಿರ್ದಿಷ್ಟ ಸಂಖ್ಯೆ ಅಥವಾ ದಿನಾಂಕ ನೀಡಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಕೆವಿಕೆ ಅಥವಾ ಆರ್‌ಎಸ್‌ಕೆ ಸಂಪರ್ಕಿಸಿ."`
2. **Low Retrieval / Out-of-Corpus Fallback:**
   - *English:* `"I could not find a matching passage in the verified documents; please rephrase or consult your local KVK or RSK."`
   - *Kannada:* `"ಪರಿಶೀಲಿಸಿದ ದಾಖಲೆಗಳಲ್ಲಿ ಸೂಕ್ತ ಮಾಹಿತಿ ದೊರೆತಿಲ್ಲ; ದಯವಿಟ್ಟು ವಾಕ್ಯವನ್ನು ಬದಲಿಸಿ ಅಥವಾ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆ / ಆರ್‌ಎಸ್‌ಕೆ ಸಂಪರ್ಕಿಸಿ."`

---

## 4. Canonical Kannada Terminology & Advisory Equivalences

| English Concept | Canonical Kannada Advisory Text | ICAR / Extension Source |
| :--- | :--- | :--- |
| **1% Bordeaux Recipe** | 1 ಕೆ.ಜಿ ಮೈಲುತುತ್ತನ್ನು (ಕಾಪರ್ ಸಲ್ಫೇಟ್) 50 ಲೀಟರ್ ನೀರಿನಲ್ಲಿ ಮತ್ತು 1 ಕೆ.ಜಿ ಸುಣ್ಣವನ್ನು 50 ಲೀಟರ್ ನೀರಿನಲ್ಲಿ ಪ್ರತ್ಯೇಕವಾಗಿ ಕರಗಿಸಿ, ಸಿಂಪಡಿಸುವ ಮುನ್ನ ಒಟ್ಟಿಗೆ ಬೆರೆಸಿ. | Ext. Folder No. 17 (1989), p. 2 |
| **Neutrality Test** | ದ್ರಾವಣದಲ್ಲಿ ತಾಮ್ರ ಹೆಚ್ಚಾಗಿದೆಯೇ ಎಂದು ಪರೀಕ್ಷಿಸಲು ಸ್ವಚ್ಛವಾದ ಉಕ್ಕಿನ ಚಾಕು ಅಥವಾ ಮೊಳೆಯನ್ನು ಅದ್ದಿ. ಚಾಕುವಿನ ಮೇಲೆ ಕಂದು ಲೇಪನ ಬಂದರೆ ತಾಮ್ರ ಹೆಚ್ಚಿದೆ ಎಂದರ್ಥ; ಅಂತಹ ಸಂದರ್ಭದಲ್ಲಿ ಕಂದು ಲೇಪನ ಬರುವುದು ನಿಲ್ಲುವವರೆಗೆ ಸುಣ್ಣದ ನೀರನ್ನು ಸೇರಿಸಬೇಕು. | Ext. Folder No. 17 (1989), p. 2 |
| **Adjuvant / Sticker** | ಪ್ರತಿ ಲೀಟರ್ ಸಿಂಪರಣಾ ದ್ರಾವಣಕ್ಕೆ 1 ಮಿ.ಲೀ ಅಂಟುದ್ರವ (ಸ್ಟಿಕ್ಕರ್ ಕಮ್ ಸ್ಪ್ರೆಡರ್) ಸೇರಿಸಬೇಕು. | CPCRI Advisory (2024) |
| **Polythene Covering** | ಮುಂಗಾರು ಆರಂಭಕ್ಕೆ ಮುನ್ನ ಅಡಿಕೆ ಗೊಂಚಲುಗಳಿಗೆ 125 ಗೇಜ್ (75 x 60 ಸೆಂ.ಮೀ) ಅಳತೆಯ ಪಾಲಿಥಿನ್ ಚೀಲಗಳನ್ನು ಕಟ್ಟುವುದು ಯಾಂತ್ರಿಕ ತಡೆಗೋಡೆಯಾಗಿ ಶಿಲೀಂಧ್ರ ಸೋಂಕನ್ನು ತಡೆಯುತ್ತದೆ. | Ext. Pub. No. 69 (1999), p. 2 |
| **Repeat Interval** | ಮೊದಲ ಸಿಂಪರಣೆಯ 40-45 ದಿನಗಳ ನಂತರ ಎರಡನೇ ಸಿಂಪರಣೆ ಮಾಡಬೇಕು. ಮುಂಗಾರು ಮಳೆಯು ದೀರ್ಘಕಾಲ ಮುಂದುವರಿದರೆ ಮೂರನೇ ಸಿಂಪರಣೆ ಅಗತ್ಯವಿರುತ್ತದೆ. | Ext. Pub. No. 69 (1999), p. 2 |
| **Alternative Chemical** | ತೋಟದಲ್ಲಿ ಕೊಳೆರೋಗ ಇನ್ನೂ ಕಾಣಿಸದಿದ್ದರೂ ಎಲ್ಲಾ ಗೊನೆಗಳಿಗೂ ಮ್ಯಾಂಡಿಪ್ರೊಪಮಿಡ್ 23.4% SC (ಪ್ರತಿ ಲೀಟರ್‌ಗೆ 1 ಮಿ.ಲೀ) ಸಿಂಪಡಿಸಬಹುದು. ಇತರ ಯಾವುದೇ ಔಷಧಿಗಳೊಂದಿಗೆ ಬೆರೆಸಬಾರದು. | CPCRI Advisory (2024) |
| **Yellow Leaf Disease** | ಅಡಿಕೆ ಮರದ ಹಳದಿ ಎಲೆ ರೋಗಕ್ಕೆ ಸಾಂಪ್ರದಾಯಿಕ ರಾಸಾಯನಿಕ ಸಿಂಪರಣೆ ಪರಿಣಾಮಕಾರಿಯಲ್ಲ (ಬೋರ್ಡೋ ದ್ರಾವಣವನ್ನು ಸಿಂಪಡಿಸಬಾರದು). ಇದಕ್ಕೆ ಸಮತೋಲಿತ ಗೊಬ್ಬರ (NPK) ಮತ್ತು ತೋಟದ ನಿರ್ವಹಣೆ ಅಗತ್ಯ. | Ext. Pub. No. 69 (1999), p. 4 |
| **Safety Footer** | ವೈಯಕ್ತಿಕ ರಕ್ಷಣಾ ಸಾಧನಗಳನ್ನು (PPE) ಧರಿಸಿ, ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಸೂಚನೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ, ಕೊಯ್ಲಿನ ಪೂರ್ವಾವಧಿಯನ್ನು (PHI) ಪಾಲಿಸಿ ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಂಪರ್ಕಿಸಿ. | Mandatory Regulatory Footer |

---

## 5. End-to-End Decision Scenarios (Live Weather Engine)

### Scenario 1: Sunny Pre-Monsoon Break (Sirsi, 08:00 AM)
- **Inputs:** Lat 14.62, Lon 74.84, Lead time = 0h, Rain = 0.0 mm/h, Rain Prob = 12%, Wind = 8.5 km/h.
- **Engine Output:**
  - `decision`: `SPRAY_NOW`
  - `confidence`: `0.92`
  - `window`: `08:00 - 12:00` (4 hours)
  - `summary`: *"Favorable spray window identified immediately (4 hours dry, low drift risk). Prepare fresh 1% neutral Bordeaux mixture."*
  - `grounded_citations`: Cites Ext. Folder No. 17 and Ext. Pub. No. 69.

### Scenario 2: Active Heavy Monsoon Downpour (Sringeri, 11:00 AM)
- **Inputs:** Lat 13.42, Lon 75.25, Rain = 6.8 mm/h, Rain Prob = 90%, Wind = 14 km/h.
- **Engine Output:**
  - `decision`: `DO_NOT_SPRAY`
  - `confidence`: `0.95`
  - `window`: `null`
  - `summary`: *"Continuous rainfall forecast; chemical wash-off risk is extreme. Wait for a clear break in rainfall as advised by ICAR-CPCRI."*

### Scenario 3: Marginal Wind Drift (Puttur, 01:00 PM)
- **Inputs:** Lat 12.77, Lon 75.20, Rain = 0.0 mm, Rain Prob = 15%, ECMWF Wind = 18.2 km/h, GFS Wind = 16.5 km/h.
- **Engine Output:**
  - `decision`: `WAIT`
  - `confidence`: `0.88`
  - `deciding_factor`: *"Wind speed (18.2 km/h) exceeds safe mist spraying threshold (15 km/h). Spray droplets will drift away from palm crowns."*

### Scenario 4: Post-Spray Wash-off Evaluation (Thirthahalli)
- **Inputs:** Spray completed yesterday at 14:00; unpredicted evening storm deposited 16.5 mm of rain at 18:00.
- **Engine Output:**
  - `wash_off_warning`: `true`
  - `advisory`: *"[App advisory, not an ICAR rule] Rain (>5mm) occurred within 4 hours of spraying. Residue may have been partially washed off. Contact your local KVK or RSK before repeating spray."*

---

## 6. Technical Review Questions for ICAR-CPCRI Vittal & KVK Agronomists

1. **Lime Quality & Neutrality:** Ext. 17 mentions adding more lime if the knife develops a brownish coating. In field practice, what form of lime is recommended (quicklime vs slaked lime), and is the knife test still preferred over pH indicator strips (pH 7.0)?
2. **Mandipropamid 23.4% SC Rainfastness:** Modern commercial formulations of Mandipropamid claim systemic uptake within 1–2 hours. Does ICAR-CPCRI recommend a specific minimum rain-free period for Mandipropamid versus Bordeaux mixture during monsoon breaks?
3. **Adjuvant Compatibility:** Are there specific recommended chemistries for the 1 ml/L "sticker-cum-spreader" (e.g., non-ionic silicones vs resin-based adhesives) when mixed with high-pH Bordeaux mixture?
4. **Polythene Bunch Covers vs Spraying Economics:** Ext. Pub. 69 highlights 125-gauge polythene covers (75 × 60 cm) as an effective mechanical barrier. Under current coastal labor costs, does CPCRI recommend bunch covering as a complete replacement for the second spray, or as a complement in high-rainfall valleys?
5. **Wash-Off Thresholds:** When heavy rainfall (>10 mm) occurs shortly after spraying, what diagnostic inspection should a farmer conduct before deciding to re-enter for an unscheduled repeat spray?

