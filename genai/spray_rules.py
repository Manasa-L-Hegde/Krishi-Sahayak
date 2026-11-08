"""
Krishi Sahayak - Hand-curated Agronomic Spray Rules for Arecanut Koleroga (Fruit Rot).

STRICT GROUNDING POLICY:
- No values are hallucinated or copied blindly from noisy OCR.
- Every rule parameter maps to a cited document with exact publication year, page number,
  and verbatim passage checked against the original document.
- Parameters not explicitly stated in ICAR/CPCRI publications (e.g. calendar start dates,
  exact minimum dry rainfast hours) are strictly set to None, with fail-closed fallback
  to the local Krishi Vigyan Kendra (KVK) or Raitha Samparka Kendra (RSK).
- Media advisories from The Hindu are explicitly documented as "News report of a CPCRI release".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProvenanceRecord:
    source_id: str
    source_type: str  # e.g. "ICAR-CPCRI Official Publication", "News report of a CPCRI release"
    title: str
    publication_year: int
    page: Optional[int]
    url: str
    verbatim_text: str
    verification_status: str  # "VERIFIED" or "UNVERIFIED_NULL"
    notes: Optional[str] = None


@dataclass
class AgronomicSprayRule:
    crop: str
    target_disease: str
    causal_organism: str

    # Primary chemical practice (1% Neutral Bordeaux Mixture)
    fungicide_name: str
    concentration: str
    preparation: Dict[str, Any]
    neutrality_test: Dict[str, Any]
    adjuvant: Dict[str, Any]

    # Application intervals & conditions
    timing_relative_to_rain: str
    first_spray_timing: str
    repeat_interval_days_min: Optional[int]
    repeat_interval_days_max: Optional[int]
    third_spray_condition: Optional[str]

    # Cultural & mechanical barrier practice
    bunch_covering: Dict[str, Any]

    # Modern alternative fungicide (Mandipropamid)
    alternative_fungicide: Optional[Dict[str, Any]]

    # Fields NOT established in ICAR/CPCRI corpus (fail-closed nulls)
    calendar_start_dates: Optional[str] = None
    min_dry_hours: Optional[float] = None  # None: ICAR corpus does not give a numeric rainfast threshold

    # Provenance mapping
    provenance: Dict[str, ProvenanceRecord] = field(default_factory=dict)
    kvk_fallback_message_en: str = (
        "Exact calendar dates and rainfast drying hours vary by microclimate and are not defined "
        "as fixed thresholds in the ICAR-CPCRI publications. Consult your local Krishi Vigyan Kendra (KVK) "
        "or Raitha Samparka Kendra (RSK) for localized guidance."
    )
    kvk_fallback_message_kn: str = (
        "ಖಚಿತ ಕ್ಯಾಲೆಂಡರ್ ದಿನಾಂಕಗಳು ಮತ್ತು ಮಳೆ ನಿರೋಧಕ ಒಣಗುವ ಸಮಯವು ಸ್ಥಳೀಯ ಹವಾಮಾನಕ್ಕೆ ಅನುಗುಣವಾಗಿ ಬದಲಾಗುತ್ತವೆ "
        "ಮತ್ತು ಐಸಿಎಆರ್-ಸಿಪಿಸಿಆರ್‌ಐ ಪ್ರಕಟಣೆಗಳಲ್ಲಿ ನಿಶ್ಚಿತ ಮಿತಿಯಾಗಿ ನಮೂದಿಸಿಲ್ಲ. ಸ್ಥಳೀಯ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ "
        "ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (ಕೆವಿಕೆ) ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು (ಆರ್‌ಎಸ್‌ಕೆ) ಸಂಪರ್ಕಿಸಿ."
    )


# --- Hand-Entered Citations Checked Against Genuine Sources ---

PROVENANCE_EXT17 = ProvenanceRecord(
    source_id="ext17_pdf",
    source_type="ICAR-CPCRI Official Publication",
    title="Koleroga (Mahali) of Arecanut (Extension Folder No. 17)",
    publication_year=1989,
    page=2,
    url="https://cpcri.gov.in/filemgr/webfs/publication/ext17.pdf",
    verbatim_text=(
        "CONTROL: Prophylactic spraying with 1 % Bordeaux mixture gives satisfactory control of the disease. "
        "Two sprayings are generally required, first after the premonsoon showers and the second 40-45 days thereafter. "
        "If the monsoon prolongs, a third spraying will be required. Infected and fallen fruits must be collected "
        "periodically and destroyed to reduce the inoculum and further spread of the disease. Since Bordeaux mixture "
        "is a protective fungicide, care should be taken to cover the entire surface by giving a very fine spray. "
        "PREPARATION OF BORDEAUX MIXTURE: Dissolve 1 kg of Copper Sulphate in 50 litres of water and 1 kg of lime "
        "in 50 litres of water separately and mix just before spraying. If the quality of lime is inferior, 1 kg of lime "
        "may not be sufficient to neutralise the copper sulphate. To test the excess copper in the Bordeaux mixture solution, "
        "a polished iron knife or nail may be dipped into the solution. If copper is in excess a brownish coating will "
        "appear on the knife in which case lime may be added till such coating disappears. Copper, wooden or plastic vessels "
        "may be used for preparation of this mixture. Do not use other metallic vessels."
    ),
    verification_status="VERIFIED",
    notes="Extension Folder No. 17, published August 1989 by CPCRI Kasaragod.",
)

PROVENANCE_LEAFLET_P2 = ProvenanceRecord(
    source_id="cpcri_vittal_leaflet_pdf",
    source_type="ICAR-CPCRI Official Publication",
    title="Disease Management Strategies in Arecanut and Cocoa (Extension Publication No. 69)",
    publication_year=1999,
    page=2,
    url="https://cpcri.gov.in/filemgr/webfs/publication/disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf",
    verbatim_text=(
        "Fruit rot ('Kole roga/mahali'): The disease is caused by Phytophthora meadii... "
        "Disease can be controlled satisfactorily by prophylactic pre-monsoon spraying with one per cent Bordeaux mixture "
        "and second spray after 40-45 days. A third spray would be necessary if the monsoon is prolonged. "
        "The cultural practice of covering of arecanut bunches with polythene covers (125 gauge, 75 x 60 cm) "
        "has been found to be very effective in checking the disease by acting as mechanical barrier."
    ),
    verification_status="VERIFIED",
    notes="Extension Publication No. 69, published December 1999 by CPCRI Regional Station Vittal. Polythene 125 gauge, 75x60 cm verified on Page 2.",
)

PROVENANCE_THE_HINDU_2024 = ProvenanceRecord(
    source_id="cpcri_advisory_2024_thehindu",
    source_type="News report of a CPCRI release",
    title="CPCRI issues advisory to control koleroga disease in arecanut plantations (The Hindu, Aug 9, 2024)",
    publication_year=2024,
    page=None,
    url="https://www.thehindu.com/news/cities/Mangalore/cpcri-issues-advisory-to-control-koleroga-disease-in-arecanut-plantations/article68506079.ece",
    verbatim_text=(
        "Therefore, it is important to take up control measures timely whenever there is a break in rainfall, a CPCRI release said. "
        "It advised farmers to spray 1% Bordeaux mixture or Mandipropamid 23.4% SC (1 ml per litre) to all the bunches even "
        "if fruit rot has not yet been noticed in the plantation. Wherever the fruit rot has already been noticed in severe form, "
        "spray on both crown and bud regions to protect the trees from bud and crown rot. Do not mix any other fungicides/pesticides/nutrients. "
        "The Bordeaux mixture should be prepared fresh, the pH of the preparation should be neutral (pH 7) and it is advised "
        "to add a sticker cum spreader @1 ml per litre for spray preparation. Ensure proper mist spray of the fungicide."
    ),
    verification_status="VERIFIED",
    notes="Verbatim text in downloaded article body (Line 3). Third-party news coverage of ICAR-CPCRI advisory, August 9, 2024.",
)

PROVENANCE_UNVERIFIED_MIN_DRY_HOURS = ProvenanceRecord(
    source_id="none",
    source_type="Agronomic Absence",
    title="Minimum Rainfast Dry Hours Threshold",
    publication_year=2026,
    page=None,
    url="",
    verbatim_text="None of the cited ICAR-CPCRI publications specify a numeric minimum dry hours threshold for Bordeaux mixture or Mandipropamid.",
    verification_status="UNVERIFIED_NULL",
    notes="Set to None in agronomic rule. An advisory default (e.g. 4.0 h) is configured in engine defaults and explicitly labeled as an advisory default, not an ICAR rule.",
)

PROVENANCE_UNVERIFIED_CALENDAR_DATES = ProvenanceRecord(
    source_id="none",
    source_type="Agronomic Absence",
    title="Fixed Calendar Spray Dates (e.g. late May / early June)",
    publication_year=2026,
    page=None,
    url="",
    verbatim_text="Official publications state 'after the premonsoon showers' or 'pre-monsoon', but do not prescribe fixed calendar dates.",
    verification_status="UNVERIFIED_NULL",
    notes="Set to None in agronomic rule. Microclimatic onset of monsoon determines timing, not calendar dates.",
)


# --- Hand-Entered Canonical Rule Instance ---

ARECANUT_KOLEROGA_RULE = AgronomicSprayRule(
    crop="arecanut",
    target_disease="koleroga",
    causal_organism="Phytophthora meadii",
    fungicide_name="Bordeaux mixture (1%)",
    concentration="1% (1 kg copper sulphate + 1 kg lime per 100 L water)",
    preparation={
        "copper_sulphate_kg": 1.0,
        "copper_sulphate_water_l": 50.0,
        "lime_kg": 1.0,
        "lime_water_l": 50.0,
        "total_water_l": 100.0,
        "instruction_en": (
            "Dissolve 1 kg of Copper Sulphate in 50 litres of water and 1 kg of lime in 50 litres "
            "of water separately and mix just before spraying. Copper, wooden or plastic vessels may be used "
            "for preparation; never use other metallic vessels."
        ),
        "instruction_kn": (
            "1 ಕೆಜಿ ಮೈಲುತುತ್ತವನ್ನು (ಕಾಪರ್ ಸಲ್ಫೇಟ್) 50 ಲೀಟರ್ ನೀರಿನಲ್ಲಿ ಮತ್ತು 1 ಕೆಜಿ ಸುಣ್ಣವನ್ನು 50 ಲೀಟರ್ ನೀರಿನಲ್ಲಿ "
            "ಪ್ರತ್ಯೇಕವಾಗಿ ಕರಗಿಸಿ, ಸಿಂಪಡಿಸುವ ಮುನ್ನವೇ ಎರಡನ್ನೂ ಒಟ್ಟಿಗೆ ಬೆರೆಸಿ. ತಾಮ್ರ, ಮರ ಅಥವಾ ಪ್ಲಾಸ್ಟಿಕ್ ಪಾತ್ರೆಗಳನ್ನು ಬಳಸಿ; "
            "ಇತರ ಲೋಹದ ಪಾತ್ರೆಗಳನ್ನು ಬಳಸಬೇಡಿ."
        ),
        "source_id": "ext17_pdf",
        "source": "ext17_pdf (Page 2, Col 2, Aug 1989)",
        "source_title": "Koleroga (Mahali) of Arecanut (Extension Folder No. 17)",
        "publication_year": 1989,
        "page": 2,
        "url": "https://cpcri.gov.in/filemgr/webfs/publication/ext17.pdf",
    },
    neutrality_test={
        "method": "Polished iron knife or nail test",
        "instruction_en": (
            "Dip a clean polished iron knife or nail into the solution. If a brownish copper coating appears, "
            "copper is in excess; add more lime solution until no brownish deposit forms."
        ),
        "instruction_kn": (
            "ಸ್ವಚ್ಛವಾದ ಕಬ್ಬಿಣದ ಚಾಕು ಅಥವಾ ಮೊಳೆಯನ್ನು ದ್ರಾವಣದಲ್ಲಿ ಮುಳುಗಿಸಿ. ಚಾಕುವಿನ ಮೇಲೆ ಕಂದು ಬಣ್ಣದ ತಾಮ್ರದ ಲೇಪನ ಕಂಡುಬಂದರೆ, "
            "ತಾಮ್ರದ ಅಂಶವು ಹೆಚ್ಚಾಗಿದೆ ಎಂದು ತಿಳಿಯಿರಿ; ಕಂದು ಬಣ್ಣ ಮಾಯವಾಗುವವರೆಗೆ ಇನ್ನಷ್ಟು ಸುಣ್ಣದ ತಿಳಿನೀರನ್ನು ಸೇರಿಸಿ."
        ),
        "required_ph": 7.0,  # Stated in The Hindu / CPCRI 2024 advisory
        "source_id": "ext17_pdf",
        "source": "ext17_pdf (Page 2, 1989) & The Hindu CPCRI advisory (Aug 2024)",
        "source_title": "Koleroga (Mahali) of Arecanut (Extension Folder No. 17)",
        "publication_year": 1989,
        "page": 2,
        "url": "https://cpcri.gov.in/filemgr/webfs/publication/ext17.pdf",
    },
    adjuvant={
        "type": "Sticker cum spreader",
        "dose_per_litre": "1 ml / L",
        "instruction_en": "Add a sticker cum spreader @ 1 ml per litre of spray preparation.",
        "instruction_kn": "ಪ್ರತಿ ಲೀಟರ್ ಸಿಂಪಡಣಾ ದ್ರಾವಣಕ್ಕೆ 1 ಮಿ.ಲೀ ಅಂಟುದ್ರವವನ್ನು (ಸ್ಟಿಕ್ಕರ್/ಸ್ಪ್ರೆಡರ್) ಸೇರಿಸಿ.",
        "source_id": "thehindu_cpcri_2024",
        "source": "The Hindu / CPCRI advisory (Aug 2024)",
        "source_title": "CPCRI issues advisory to control koleroga disease in arecanut plantations (The Hindu, Aug 9, 2024)",
        "publication_year": 2024,
        "page": None,
        "url": "https://www.thehindu.com/news/cities/Mangalore/cpcri-issues-advisory-to-control-koleroga-disease-in-arecanut-plantations/article68506079.ece",
    },
    timing_relative_to_rain="Whenever there is a break in rainfall (The Hindu / CPCRI advisory, Aug 2024)",
    first_spray_timing="Prophylactic pre-monsoon / after premonsoon showers (ext17_pdf, Page 2, 1989)",
    repeat_interval_days_min=40,
    repeat_interval_days_max=45,
    third_spray_condition="Necessary if the monsoon is prolonged (ext17_pdf, Page 2, 1989; Leaflet, Page 2, 1999)",
    bunch_covering={
        "recommended": True,
        "material": "Polythene covers (125 gauge, 75 x 60 cm)",
        "mechanism": "Mechanical barrier against rain splash and inoculum",
        "instruction_en": (
            "Cover arecanut bunches with 125-gauge polythene covers (75 x 60 cm) as a mechanical barrier."
        ),
        "instruction_kn": (
            "ಯಾಂತ್ರಿಕ ತಡೆಗೋಡೆಯಾಗಿ 125 ಗೇಜ್ (75 x 60 ಸೆಂ.ಮೀ) ಅಳತೆಯ ಪಾಲಿಥಿನ್ ಚೀಲಗಳಿಂದ ಅಡಿಕೆ ಗೊಂಚಲುಗಳನ್ನು ಮುಚ್ಚಿ."
        ),
        "source_id": "cpcri_vittal_leaflet_pdf",
        "source": "cpcri_vittal_leaflet_pdf (Page 2, 1999)",
        "source_title": "Disease Management Strategies in Arecanut and Cocoa (Extension Publication No. 69)",
        "publication_year": 1999,
        "page": 2,
        "url": "https://cpcri.gov.in/filemgr/webfs/publication/disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf",
    },
    alternative_fungicide={
        "fungicide_name": "Mandipropamid 23.4% SC",
        "dose_per_litre": "1 ml / L",
        "application_scope": "Spray to all bunches even if fruit rot has not yet been noticed",
        "severe_infestation_scope": "Spray on both crown and bud regions if fruit rot is severe to protect against bud and crown rot",
        "instruction_en": (
            "Mandipropamid 23.4% SC @ 1 ml per litre can be sprayed as an alternative. "
            "Ensure proper mist spray. Do not mix with other fungicides, pesticides, or nutrients."
        ),
        "instruction_kn": (
            "ಬದಲಿ ಶಿಲೀಂಧ್ರನಾಶಕವಾಗಿ ಮ್ಯಾಂಡಿಪ್ರೊಪಮಿಡ್ 23.4% SC ಯನ್ನು ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ 1 ಮಿ.ಲೀ ನಂತೆ ಸಿಂಪಡಿಸಬಹುದು. "
            "ಸೂಕ್ಷ್ಮ ಮಂಜಿನಂತೆ ಸಿಂಪಡಿಸಿ. ಇತರ ಯಾವುದೇ ಶಿಲೀಂಧ್ರನಾಶಕ, ಕೀಟನಾಶಕ ಅಥವಾ ಪೋಷಕಾಂಶಗಳೊಂದಿಗೆ ಬೆರೆಸಬೇಡಿ."
        ),
        "source_id": "thehindu_cpcri_2024",
        "source": "The Hindu / CPCRI advisory (Aug 2024)",
        "source_title": "CPCRI issues advisory to control koleroga disease in arecanut plantations (The Hindu, Aug 9, 2024)",
        "publication_year": 2024,
        "page": None,
        "url": "https://www.thehindu.com/news/cities/Mangalore/cpcri-issues-advisory-to-control-koleroga-disease-in-arecanut-plantations/article68506079.ece",
    },
    calendar_start_dates=None,  # Unverified in ICAR corpus -> Fail closed
    min_dry_hours=None,  # Unverified in ICAR corpus -> Fail closed
    provenance={
        "ext17": PROVENANCE_EXT17,
        "leaflet_p2": PROVENANCE_LEAFLET_P2,
        "the_hindu_2024": PROVENANCE_THE_HINDU_2024,
        "unverified_min_dry_hours": PROVENANCE_UNVERIFIED_MIN_DRY_HOURS,
        "unverified_calendar_dates": PROVENANCE_UNVERIFIED_CALENDAR_DATES,
    },
)


def get_spray_rule(crop: str, disease: str) -> Optional[AgronomicSprayRule]:
    """
    Retrieve hand-curated, ICAR-grounded spray rule.
    Fails closed (returns None) if crop or disease is not verified in corpus.
    """
    c = crop.strip().lower()
    d = disease.strip().lower()
    if c == "arecanut" and any(k in d for k in ["koleroga", "mahali", "fruit rot", "fruit_rot"]):
        return ARECANUT_KOLEROGA_RULE
    return None
