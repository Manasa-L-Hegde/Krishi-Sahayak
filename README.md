# Krishi Sahayak (ಕೃಷಿ ಸಹಾಯಕ)
### Multi-Layer Intelligent Agriculture Advisory Platform: ML + DL + ICAR-Grounded GenAI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render%20Cloud-00c7b7?style=for-the-badge&logo=render&logoColor=white)](https://krishi-sahayak-bgi0.onrender.com/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Live Application URL:** **[https://krishi-sahayak-bgi0.onrender.com/](https://krishi-sahayak-bgi0.onrender.com/)**  
> * **Interactive Crop Advisor:** [https://krishi-sahayak-bgi0.onrender.com/](https://krishi-sahayak-bgi0.onrender.com/)  
> * **Arecanut Spray Timing Advisor:** [https://krishi-sahayak-bgi0.onrender.com/spray](https://krishi-sahayak-bgi0.onrender.com/spray)  
> * **Interactive OpenAPI Docs:** [https://krishi-sahayak-bgi0.onrender.com/docs](https://krishi-sahayak-bgi0.onrender.com/docs)  

---

## 🌾 Problem Statement

Smallholder and commercial farmers in India face three interconnected challenges:

1. **Suboptimal Crop Selection & Climate Vulnerability:**  
   Farmers frequently select crops based on tradition or regional trends rather than empirical soil composition ($N, P, K$, soil $\text{pH}$) and upcoming seasonal weather (temperature, humidity, rainfall). When micro-climates shift unexpectedly, uncalibrated crop choices result in severe yield reduction or total harvest failure.

2. **Catastrophic Fungal Disease Outbreaks (e.g., Arecanut Koleroga / Mahali):**  
   In coastal Karnataka and the Western Ghats, *Phytophthora meadii* (Koleroga/Fruit Rot) causes up to **75–90% nut-drop and crop destruction** during heavy monsoon rain. Timely prophylactic spraying of 1% Bordeaux mixture before incessant rainfall breaks is critical. However, spraying during rainfall washes fungicide away, wasting expensive chemicals and labor, while spraying too late guarantees orchard infection.

3. **The "AI in Agriculture" Trust Gap (Hallucinations vs. Reality):**  
   Generic Large Language Models (LLMs) hallucinate chemical concentrations, invent toxic fungicide dosages, or leak unscaled recipes when asked for smaller/larger spray volumes. In agriculture, an incorrect chemical recipe or bad fungicide timing can destroy an entire plantation and poison soil ecosystems.

---

## 💡 What We Have Done (Solution & Core Innovations)

**Krishi Sahayak** addresses these challenges by implementing an engineering-first, multi-layered architecture where **mission-critical decisions are deterministic and scientifically grounded**, while Generative AI is strictly constrained by official ICAR agronomic literature:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             KRISHI SAHAYAK PLATFORM                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│     LAYER 1     │           │     LAYER 2     │           │     LAYER 3     │
│   ML CROP &     │           │   DETERMINISTIC │           │   ICAR RAG &    │
│ RISK REGRESSION │           │  SPRAY ADVISOR  │           │  VISION ADVISOR │
└────────┬────────┘           └────────┬────────┘           └────────┬────────┘
         │                             │                             │
         ▼                             ▼                             ▼
• Random Forest (700 trees)   • Zero LLM / Zero Halluc      • Google Gemini API
• 99.32% Macro-F1             • Open-Meteo ECMWF & GFS      • 369-Chunk FAISS Index
• 2-Sigma Climate Risk Metric • Daylight Window Analysis   • Fail-Closed Safety Guard
• SHAP Local Explainability   • ICAR-CPCRI Verified Rules   • Regex Quantity Scaling
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       │
                                       ▼
             ┌───────────────────────────────────────────────────┐
             │            FARMER-FIRST BILINGUAL UI              │
             │   • Responsive Glassmorphic Multi-Tone Palette    │
             │   • Synchronized Sliders & One-Click Presets      │
             │   • Instant English ↔ ಕನ್ನಡ (Kannada) Switcher    │
             │   • Leaflet Interactive Farm Pinpoint Mapping     │
             └───────────────────────────────────────────────────┘
```

---

### 1. Layer 1: Machine Learning Crop Recommendation & Climate Risk
* **Classification Pipeline:** Evaluates 22 major Indian agricultural crops across 7 physical parameters ($N$, $P$, $K$, temperature, relative humidity, soil $\text{pH}$, and seasonal rainfall).
* **Champion Model Selection:** Benchmarked Random Forest, XGBoost, and Logistic Regression on held-out test splits. Selected `RandomForest_700Trees` achieving **99.32% Macro-F1** and superior perturbation robustness.
* **Engineered Climate Mismatch Risk Metric ($0 - 100$):** Standard datasets lack continuous risk targets. We engineered an agronomic risk regressor based on two-sigma distance from optimal historical crop rainfall:
  $$\text{Risk} = \text{clip}\left(\frac{|\text{rainfall} - \mu_{\text{crop}}|}{2 \times \sigma_{\text{crop}}} \times 100,\, 0,\, 100\right)$$
* **SHAP Explainability:** Predictions do not operate as a black box; every inference calculates exact SHAP (SHapley Additive exPlanations) values indicating the top 3 features supporting or pulling away from the prediction.

### 2. Layer 2: Deterministic Koleroga Spray Timing Advisor (`/spray`)
* **Zero-LLM Safety Guarantee:** Fungicide decisions do **NOT** invoke an LLM. Dosages, preparation steps, safety intervals, and neutrality tests are hard-coded verbatim from **ICAR-CPCRI (Central Plantation Crops Research Institute)** literature.
* **Dual-Model Numerical Weather Forecasts:** Integrates live 7-day hourly forecasts from **ECMWF IFS 0.25°** and **GFS Seamless 0.25°** via Open-Meteo.
* **Daylight Rain-Free Window Evaluation:** Computes consecutive daylight hours without rain to render actionable decisions:
  * 🟢 **`SPRAY_NOW`**: $\ge 3$ consecutive dry daylight hours projected before subsequent rainfall.
  * 🟡 **`WAIT`**: Imminent precipitation within the drying window ($< 3$ hours); recommends postponing to protect chemical efficacy.
  * 🔴 **`DO_NOT_SPRAY`**: Sustained rainfall or zero drying windows.
* **Interactive Mapping & Geo-Resolution:** Leaflet GPS pin-drop and reverse-geocoding bounded strictly to Indian coordinates ($6.0^\circ\text{N} - 37.5^\circ\text{N}$, $68.0^\circ\text{E} - 97.5^\circ\text{E}$).

### 3. Layer 3: ICAR-Grounded GenAI & Vision Advisory (`/genai` & `/vision`)
* **Dual-Provider Architecture:** Powered by **Google Gemini API** (`gemini-flash-latest` via Google AI Studio) with automatic fallback to Groq (`llama-3.1-70b-versatile`).
* **Dense Vector RAG (FAISS):** Searches 369 indexed semantic chunks of verified ICAR and university agronomic publications.
* **Fail-Closed Safety Threshold (Cosine $\ge 0.59$):** If an agricultural question cannot be grounded in official ICAR documents, the model automatically abstains from answering rather than guessing.
* **Generalized Quantity Scaling Guard:** Prevents the dangerous 100 L Bordeaux preparation recipe from leaking when a farmer asks for custom amounts (e.g. 25 L, 50 L, 200 L) using bilingual regex guards in English and Kannada.
* **Plant Vision Analysis (`/vision/advise`):** Accepts leaf/crop photographs, identifies visible pathological symptoms, and searches the verified index for matching extension remedies.

### 4. Layer 4: Modern Farmer-First Interactive UI
* **No Monotonous Green or Sterile White:** Styled with a multi-tone ambient mesh background (warm almond, soft celadon, morning mist) and soft pastel input cards.
* **Synchronized Dual Controls:** Real-time synchronized range sliders and precision numeric inputs with live agronomic status badges (e.g., Acidic vs. Neutral soil pH).
* **One-Click Field Presets:** Instant parameters for 🌾 Rice, 🌽 Maize, 🌿 Cotton, ☕ Coffee, and 🧆 Chickpea.
* **Full Bilingual Support:** Instant toggle between **English** and **ಕನ್ನಡ (Kannada)** across all headers, inputs, advisories, and result cards.

---

## 📊 Model Evaluation & Benchmarks

The models were evaluated using rigorous held-out test splits (random state 42) with perturbation robustness tests ($\pm 10\%$ input noise):

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Perturbation Flip Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (700 Trees - Champion)** | **99.32%** | **99.35%** | **99.32%** | **99.32%** | **1.62%** |
| Random Forest (350 Trees - Baseline) | 99.32% | 99.35% | 99.32% | 99.32% | 1.74% |
| XGBoost (Classifier) | 99.09% | 99.15% | 99.09% | 99.09% | 2.15% |
| Logistic Regression | 95.23% | 95.40% | 95.23% | 95.18% | 5.82% |

* **Rainfall Risk Regressor:** $R^2 = 0.835$, $\text{MAE} = 7.20$, $\text{RMSE} = 10.11$.
* **RAG Retrieval Safeguard Benchmark:** Evaluated against 15 out-of-corpus adversarial prompts; achieved **100% fail-closed abstention rate** (zero ungrounded hallucinated recipes generated).

---

## 🚀 Live API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Responsive bilingual crop recommendation & advisory dashboard |
| `GET` | `/spray` | Live Arecanut Koleroga spray timing advisor with Leaflet map |
| `GET` | `/health` | Container deep-probe healthcheck (verifies model & FAISS index availability) |
| `POST` | `/predict` | ML crop classification + risk regression + SHAP factor analysis |
| `POST` | `/spray/advise` | Deterministic weather-evaluated spray window decision (`SPRAY_NOW`, `WAIT`) |
| `GET` | `/weather/forecast`| 7-day dual-model numerical weather forecast (ECMWF & GFS) |
| `GET` | `/geo/search` | Geocoding bounded strictly to Indian agricultural territories |
| `POST` | `/genai/ask` | Bilingual ICAR-grounded conversational RAG advisor (Google Gemini) |
| `POST` | `/vision/advise` | Visual plant symptom analysis and verified extension lookup |
| `GET` | `/docs` | Interactive Swagger UI API documentation |

---

## 🛠️ Local Development & Quickstart

### Prerequisites
* **Python 3.11.x** (pinned: `Python 3.11.9`)
* Git & Virtualenv

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/Manasa-L-Hegde/Krishi-Sahayak.git
cd Krishi-Sahayak

# Create virtual environment with Python 3.11
python -m venv .venv
source .venv/bin/activate        # On Linux/macOS
# .\.venv\Scripts\Activate.ps1   # On Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory (refer to `.env.example`):
```ini
GEMINI_API_KEY=your_google_ai_studio_api_key_here
GROQ_API_KEY=your_groq_api_key_here (optional fallback)
PILOT_MODE=false
PORT=8000
PYTHONPATH=.
```

### 3. Run the Application
```bash
# Start FastAPI ASGI server
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 🐳 Docker Container Deployment

Krishi Sahayak is fully containerized with CPU-optimized PyTorch and dynamic `$PORT` binding:

```bash
# Build the Docker image
docker build -t krishi-sahayak:latest .

# Run container locally
docker run -p 8000:8000 \
  -e GEMINI_API_KEY="your_api_key_here" \
  -e PILOT_MODE="false" \
  krishi-sahayak:latest
```

---

## 📚 Agronomic Citations & References

1. **ICAR-CPCRI Extension Publication No. 69 (1999):** *Management of Fruit Rot (Koleroga) of Arecanut*. Central Plantation Crops Research Institute, Kasaragod.
2. **ICAR-CPCRI Extension Folder No. 17 (1989):** *Preparation of 1% Bordeaux Mixture and Neutrality Testing*.
3. **ICAR-CPCRI Monsoon Advisory (2024):** *Prophylactic Spray Schedules for Phytophthora Outbreaks in Coastal Regions*.
4. **Open-Meteo Weather API:** ECMWF IFS 0.25° & GFS Seamless numerical weather prediction models.
5. **Crop Recommendation Dataset:** Attributed to Atharva Ingle (Kaggle).

---

## 👥 Authors & Acknowledgments

* **Lead Developer & Researcher:** [Manasa L Hegde](https://github.com/Manasa-L-Hegde)
* **Institutional Grounding:** Indian Council of Agricultural Research (ICAR) & CPCRI public extension literature.

---

## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
