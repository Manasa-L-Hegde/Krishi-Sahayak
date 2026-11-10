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
| **Bunch Covering (Mechanical)** | 125 gauge, 75 × 60 cm polythene covers | ICAR-CPCRI Ext. Pub. No. 69 (1999), p. 2 | *"covering of arecanut bunches with polythene covers (125 gauge, 75 x 60 cm) has been found to be very effective in checking the disease by acting as mechanical barrier."* | Cultural control alternative/complement; acts as mechanical barrier against rain splash and inoculum. (app assumption, no source on tying dates). |
| **First Spray Timing** | Premonsoon showers | ICAR-CPCRI Ext. Folder No. 17 (1989), p. 2; Ext. Pub. No. 69 (1999), p. 2 | Ext 17: *"after the premonsoon showers"*; Pub 69: *"prophylactic pre-monsoon spraying"* | App prompts farmer for intent ("first", "repeat", "unsure") and contextualizes advice to local pre-monsoon rain. |
| **Second Spray Interval** | 40–45 days after first spray | ICAR-CPCRI Ext. Pub. No. 69 (1999), p. 2 | *"second spray after 40-45 days."* | If farmer sprayed < 40 days ago, engine issues `WAIT` with interval reminder unless heavy wash-off occurred. |
| **Third Spray Condition** | Only if monsoon is prolonged | ICAR-CPCRI Ext. Pub. No. 69 (1999), p. 2 | *"A third spray would be necessary if the monsoon is prolonged."* | Zero calendar mandate; conditioned on regional rainfall persistence into August/September. |
| **Rainfast / Drying Hours** | Explicitly NOT defined as fixed figure | ICAR-CPCRI Publications (Absence of numerical claim) | Ext 17, Pub 69, and 2024 Advisory specify *"whenever there is a break in rainfall"*; no fixed hour figure is published. | **Heuristic:** App uses a 4-hour dry window (rain < 0.5 mm, prob < 35%) purely for operational weather scheduling, labeled: *"App operational weather rule; ICAR literature does not prescribe a fixed dry hour figure."* |
| **Calendar Start Dates** | Explicitly NOT defined as fixed dates | ICAR-CPCRI Publications (Absence of numerical claim) | Publications cite premonsoon showers, not calendar dates. | App explicitly abstains when asked for fixed dates (e.g. "What date in Sirsi?"), directing farmer to local KVK/RSK. |
| **Yellow Leaf Disease Exclusion** | Non-chemical garden management | ICAR-CPCRI Ext. Pub. No. 69 (1999), p. 4 | *"The disease is not amenable to conventional plant protection measures. Garden management is the only method suggested to increase the yield in affected palms."* | Mandatory safety override: if yellow leaves are queried, app prescribes balanced NPK, Mussoorie phos, organic manure, and drainage. Conventional spray (Bordeaux) is not amenable. |
| **Wind Drift Upper Bound** | 15 km/h maximum | None | None | app assumption, no source |
| **Daylight Operating Hours** | 06:00 to 18:00 IST | None | None | app assumption, no source |

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

The evaluation suite was executed against **39 test queries** (10 out-of-corpus queries + 24 farmer queries in English and Kannada + 5 empirical near-miss queries).

### Deterministic FAQ Router & Safety Guards (`genai/faq_router.py`)
To prevent vector similarity fluctuations and avoid false negative refusals on standard agronomic queries, canonical user intents (`prep`, `knife_test`, `vessels`, `second_spray`, `third_spray`, `covers`, `sticker`, `alternative`, `no_mix`) in English and Kannada are deterministically routed to verified `ARECANUT_KOLEROGA_RULE` records with official citations.

To address near-miss hazards revealed during adversarial testing, two additional guards operate upstream:
1. **Quantity Guard:** Rejects scaling requests (e.g., 50 L, 200 L, per-acre, per-palm) and returns a fixed constraint: *"The source recipe makes 100 litres in total (1 kg copper sulphate in 50 L + 1 kg lime in 50 L). For other quantities consult your KVK/RSK."*
2. **Entity Guard:** If an in-domain question names an unverified chemical entity (e.g., mancozeb, copper oxychloride, quicklime, slaked lime, chlorpyrifos, carbendazim), the passage is rejected even if vector similarity exceeds threshold ($\tau = 0.59$), returning a safe fallback refusal.

### Confusion Matrix Summary (from `diagnostics/eval_confusion_matrix.json`)

| Classification Category | Count | Percentage | Agronomic & Safety Significance |
| :--- | :---: | :---: | :--- |
| **CORRECT_ANSWER** | **18** | **46.15%** | Canonical queries answered with exact ICAR-CPCRI facts (14 canonical + 2 yellow leaf overrides + 2 quantity guard / alternative prohibitions). |
| **CORRECT_REFUSAL** | **21** | **53.85%** | 10 out-of-corpus queries + 4 drying-hour/calendar-date queries + 4 ungrounded mix/rain queries + 3 entity-guard near-miss refusals. |
| **WRONG_REFUSAL (Recall Loss)** | **0** | **0.00%** | **0 in English, 0 in Kannada.** Zero false negatives on verified canonical knowledge. |
| **WRONG_ACCEPTANCE (False Accepts)** | **0** | **0.00%** | **Zero hallucinations, zero out-of-corpus accepts.** False similarity matches on near-miss chemicals stopped by Entity Guard. |
| **Total Test Decisions** | **39** | **100.00%** | All 39 benchmark queries categorized safely without false acceptances. |

### Statistical Error Bound (Rule of Three)
> [!NOTE]
> Observing zero errors ($k = 0$) across $n = 39$ empirical queries does **not** prove the system is "100% safe". Under the **Rule of Three**, a zero-error trial gives a 95% confidence upper bound on the true failure rate:
> $$p_{95} \le \frac{3}{n} = \frac{3}{39} \approx 7.7\%$$
> This confirms that the true error rate is bounded below $\sim 7.7\%$ at 95% confidence for this distribution. Human expert review and independent held-out evaluation remain mandatory prior to any farmer pilot.

### Empirical Near-Miss Benchmark Results

| Case ID | Query Text | Top Sim | Guard / Router Action | Classification | Output Excerpt |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **NM_01_EN** | Mandipropamid mixed with a micronutrient spray | 0.4776 | FAQ Router: `alternative` | **CORRECT_ANSWER** | *"Mandipropamid 23.4% SC @ 1 ml per litre can be sprayed as an alternative... Do not mix with other fungicides, pesticides, or nutrients."* |
| **NM_02_EN** | Can I mix Bordeaux mixture with mancozeb or copper oxychloride? | 0.6275 | **Entity Guard Rejection** (Mancozeb / COC unmentioned) | **CORRECT_REFUSAL** | *"I could not find a matching passage in the verified documents; please rephrase or consult your local KVK or RSK."* |
| **NM_03_EN** | Should I use quicklime or slaked lime for Bordeaux mixture? | 0.6002 | **Entity Guard Rejection** (Quicklime / Slaked lime unmentioned) | **CORRECT_REFUSAL** | *"I could not find a matching passage in the verified documents; please rephrase or consult your local KVK or RSK."* |
| **NM_04_EN** | How much copper sulphate and lime for 50 L water? | 0.6778 | **Quantity Guard** (Detects non-100L volume) | **CORRECT_ANSWER** | *"The source recipe makes 100 litres in total (1 kg copper sulphate in 50 L + 1 kg lime in 50 L). For other quantities consult your KVK/RSK."* |
| **NM_05_EN** | What is the treatment for bud rot? | 0.5248 | Similarity Abstention ($< 0.59$) | **CORRECT_REFUSAL** | *"I could not find a matching passage in the verified documents; please rephrase or consult your local KVK or RSK."* (Does NOT output 1% Koleroga recipe). |

### Wrong Refusals by Language
- **English:** 0 / 22
- **Kannada:** 0 / 17

### Model & Scikit-Learn Compatibility
- **Model Artifacts:** Models (`models/krishi_sahayak.joblib` and `models/dl_forecaster.joblib`) are stamped with `sklearn_version="1.9.0"` and `torch_version="2.15.0+cpu"`.
- **Runtime Guard:** `api/main.py` enforces major.minor version matching at startup (`1.9.*`).
- **Compatibility:** Validated across Python 3.11.9 (Windows development with scikit-learn 1.9.0) and Debian Linux container (scikit-learn 1.9.0). All numerical predictions match exact regression baselines.

---

## 4. Canonical Kannada Terminology & Advisory Equivalences

| English Concept | Canonical Kannada Advisory Text (Verbatim from API) | ICAR / Extension Source |
| :--- | :--- | :--- |
| **1% Bordeaux Recipe** | 1 ಕೆಜಿ ಮೈಲುತುತ್ತನ್ನು (ಕಾಪರ್ ಸಲ್ಫೇಟ್) 50 ಲೀಟರ್ ನೀರಿನಲ್ಲಿ ಮತ್ತು 1 ಕೆಜಿ ಸುಣ್ಣವನ್ನು 50 ಲೀಟರ್ ನೀರಿನಲ್ಲಿ ಪ್ರತ್ಯೇಕವಾಗಿ ಕರಗಿಸಿ, ಸಿಂಪಡಿಸುವ ಮುನ್ನವೇ ಎರಡನ್ನೂ ಒಟ್ಟಿಗೆ ಬೆರೆಸಿ. ತಾಮ್ರ, ಮರ ಅಥವಾ ಪ್ಲಾಸ್ಟಿಕ್ ಪಾತ್ರೆಗಳನ್ನು ಬಳಸಿ; ಇತರ ಲೋಹದ ಪಾತ್ರೆಗಳನ್ನು ಬಳಸಬೇಡಿ. | Ext. Folder No. 17 (1989), p. 2 |
| **Neutrality Test** | ಸ್ವಚ್ಛವಾದ ಕಬ್ಬಿಣದ ಚಾಕು ಅಥವಾ ಮೊಳೆಯನ್ನು ದ್ರಾವಣದಲ್ಲಿ ಮುಳುಗಿಸಿ. ಚಾಕುವಿನ ಮೇಲೆ ಕಂದು ಬಣ್ಣದ ತಾಮ್ರದ ಲೇಪನ ಕಂಡುಬಂದರೆ, ತಾಮ್ರದ ಅಂಶವು ಹೆಚ್ಚಾಗಿದೆ ಎಂದು ತಿಳಿಯಿರಿ; ಕಂದು ಬಣ್ಣ ಮಾಯವಾಗುವವರೆಗೆ ಇನ್ನಷ್ಟು ಸುಣ್ಣದ ತಿಳಿನೀರನ್ನು ಸೇರಿಸಿ. | Ext. Folder No. 17 (1989), p. 2 |
| **Preparation Vessels** | ಬೋರ್ಡೋ ಮಿಶ್ರಣ ತಯಾರಿಸಲು ತಾಮ್ರ, ಮರ ಅಥವಾ ಪ್ಲಾಸ್ಟಿಕ್ ಪಾತ್ರೆಗಳನ್ನು ಬಳಸಬಹುದು. ಇತರ ಲೋಹದ ಪಾತ್ರೆಗಳನ್ನು ಬಳಸಬೇಡಿ. | Ext. Folder No. 17 (1989), p. 2 |
| **Adjuvant / Sticker** | ಪ್ರತಿ ಲೀಟರ್ ಸಿಂಪಡಣಾ ದ್ರಾವಣಕ್ಕೆ 1 ಮಿ.ಲೀ ಅಂಟುದ್ರವವನ್ನು (ಸ್ಟಿಕ್ಕರ್/ಸ್ಪ್ರೆಡರ್) ಸೇರಿಸಿ. | CPCRI Advisory in *The Hindu* (Aug 2024) |
| **Polythene Covering** | ಯಾಂತ್ರಿಕ ತಡೆಗೋಡೆಯಾಗಿ 125 ಗೇಜ್ (75 x 60 ಸೆಂ.ಮೀ) ಅಳತೆಯ ಪಾಲಿಥಿನ್ ಚೀಲಗಳಿಂದ ಅಡಿಕೆ ಗೊಂಚಲುಗಳನ್ನು ಮುಚ್ಚಿ. | Ext. Pub. No. 69 (1999), p. 2 |
| **Repeat Interval** | ಎರಡನೇ ಸಿಂಪರಣೆಯನ್ನು ಮೊದಲ ಮುನ್ನೆಚ್ಚರಿಕೆ ಸಿಂಪರಣೆಯ 40 ರಿಂದ 45 ದಿನಗಳ ನಂತರ ಮಾಡಬೇಕು. ಮುಂಗಾರು ಮಳೆಯು ದೀರ್ಘಕಾಲ ಮುಂದುವರಿದರೆ ಮಾತ್ರ ಮೂರನೇ ಸಿಂಪರಣೆ ಅಗತ್ಯವಿರುತ್ತದೆ. | Ext. Pub. No. 69 (1999), p. 2 |
| **Alternative Chemical** | ಬದಲಿ ಶಿಲೀಂಧ್ರನಾಶಕವಾಗಿ ಮ್ಯಾಂಡಿಪ್ರೊಪಮಿಡ್ 23.4% SC ಯನ್ನು ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ 1 ಮಿ.ಲೀ ನಂತೆ ಸಿಂಪಡಿಸಬಹುದು. ಸೂಕ್ಷ್ಮ ಮಂಜಿನಂತೆ ಸಿಂಪಡಿಸಿ. ಇತರ ಯಾವುದೇ ಶಿಲೀಂಧ್ರನಾಶಕ, ಕೀಟನಾಶಕ ಅಥವಾ ಪೋಷಕಾಂಶಗಳೊಂದಿಗೆ ಬೆರೆಸಬೇಡಿ. | CPCRI Advisory in *The Hindu* (Aug 2024) |
| **Yellow Leaf Disease** | ಅಡಿಕೆ ಮರದ ಹಳದಿ ಎಲೆ ರೋಗಕ್ಕೆ ಸಾಂಪ್ರದಾಯಿಕ ಸಸ್ಯ ಸಂರಕ್ಷಣಾ ಕ್ರಮಗಳು ಪರಿಣಾಮಕಾರಿಯಲ್ಲ. ಪೀಡಿತ ಮರಗಳಲ್ಲಿ ಇಳುವರಿಯನ್ನು ಹೆಚ್ಚಿಸಲು ತೋಟದ ನಿರ್ವಹಣೆಯು ಸೂಚಿಸಲಾದ ಏಕೈಕ ವಿಧಾನವಾಗಿದೆ. | Ext. Pub. No. 69 (1999), p. 4 |
| **Safety Footer** | ಸುರಕ್ಷತಾ ಸಲಹೆ: ಸಿಂಪಡಿಸುವಾಗ ಯಾವಾಗಲೂ ವೈಯಕ್ತಿಕ ಸುರಕ್ಷತಾ ಸಾಧನಗಳನ್ನು (ಪಿಪಿಇ: ಕೈಗವಸು, ಮುಖಗವಸು, ಕಣ್ಣಿನ ರಕ್ಷಣೆ) ಧರಿಸಿ, ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಮೇಲಿನ ಸೂಚನೆಗಳನ್ನು ಕಟ್ಟುನಿಟ್ಟಾಗಿ ಪಾಲಿಸಿ, ಕಟಾವಿಗೆ ಮುನ್ನ ಕಾಯುವ ಅವಧಿಯನ್ನು (PHI) ಗಮನಿಸಿ, ಮತ್ತು ಸ್ಥಳೀಯ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (ಕೆವಿಕೆ) ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು (ಆರ್‌ಎಸ್‌ಕೆ) ಸಂಪರ್ಕಿಸಿ. | Mandatory Regulatory Footer |

---

## 5. End-to-End Decision Scenarios (Weather Engine Reference Cases)

> **NOTE: The scenarios below are illustrative reference examples, not live API outputs.**

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
