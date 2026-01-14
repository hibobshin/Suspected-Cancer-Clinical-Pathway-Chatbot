"""
Terms Index for NG12 Guideline.

Provides:
- Term definitions from "Terms used in this guideline" section
- Age term expansion (e.g., "young people" → 16-24)
- Symptom normalization (e.g., "coughing up blood" → "haemoptysis")
- Query expansion for better matching
"""

import re
from dataclasses import dataclass, field

from config.logging_config import get_logger
from models.rule_models import ExtractedFacts

logger = get_logger(__name__)


@dataclass
class TermDefinition:
    """Definition of a term from NG12."""
    term: str
    definition: str
    age_min: int | None = None
    age_max: int | None = None
    timeframe_hours: int | None = None  # For urgency terms
    usage_sites: list[str] = field(default_factory=list)  # Rule IDs where used


class TermsIndex:
    """
    Pre-computed index of NG12 terminology.
    
    Handles:
    - Age term expansion (children, young people, etc.)
    - Urgency term definitions (urgent, very urgent, immediate)
    - Symptom normalization and synonyms
    """
    
    # NG12-defined terms from "Terms used in this guideline" section
    DEFINITIONS: dict[str, TermDefinition] = {
        "children": TermDefinition(
            term="children",
            definition="From birth to 15 years",
            age_min=0,
            age_max=15,
        ),
        "young people": TermDefinition(
            term="young people",
            definition="Aged 16 to 24 years",
            age_min=16,
            age_max=24,
        ),
        "young person": TermDefinition(
            term="young person",
            definition="Aged 16 to 24 years (same as young people)",
            age_min=16,
            age_max=24,
        ),
        "children and young people": TermDefinition(
            term="children and young people",
            definition="From birth to 24 years",
            age_min=0,
            age_max=24,
        ),
        "immediate": TermDefinition(
            term="immediate",
            definition="Within 24 hours",
            timeframe_hours=24,
        ),
        "very urgent": TermDefinition(
            term="very urgent",
            definition="Within 48 hours",
            timeframe_hours=48,
        ),
        "urgent": TermDefinition(
            term="urgent",
            definition="Before 2 weeks",
            timeframe_hours=336,  # 14 days * 24 hours
        ),
        "non-urgent": TermDefinition(
            term="non-urgent",
            definition="Routine referral, no specific timeframe",
            timeframe_hours=None,
        ),
        "persistent": TermDefinition(
            term="persistent",
            definition="Symptoms lasting 3 weeks or more",
        ),
        "unexplained": TermDefinition(
            term="unexplained",
            definition="Not attributable to a non-cancer cause after initial assessment",
        ),
        "suspected cancer pathway referral": TermDefinition(
            term="suspected cancer pathway referral",
            definition="Referral for first appointment within 2 weeks (2-week-wait)",
            timeframe_hours=336,
        ),
        "direct access": TermDefinition(
            term="direct access",
            definition="GP can arrange investigation without specialist referral",
        ),
        "safety netting": TermDefinition(
            term="safety netting",
            definition="Advice to patient to return if symptoms persist or worsen",
        ),
    }
    
    # Symptom synonyms for normalization
    # Key: canonical term, Value: list of synonyms
    SYMPTOM_SYNONYMS: dict[str, list[str]] = {
        "haemoptysis": [
            "coughing up blood",
            "blood in sputum",
            "hemoptysis",
            "coughing blood",
            "bloody sputum",
            "blood when coughing",
        ],
        "dysphagia": [
            "difficulty swallowing",
            "trouble swallowing",
            "hard to swallow",
            "swallowing problems",
            "food stuck",
        ],
        "dyspepsia": [
            "indigestion",
            "heartburn",
            "acid reflux",
            "stomach upset",
            "gastric discomfort",
        ],
        "haematuria": [
            "blood in urine",
            "bloody urine",
            "hematuria",
            "visible blood in urine",
        ],
        "haematemesis": [
            "vomiting blood",
            "blood in vomit",
            "hematemesis",
        ],
        "rectal bleeding": [
            "blood in stool",
            "bloody stool",
            "blood when passing stool",
            "bleeding from bottom",
            "pr bleeding",
        ],
        "post-menopausal bleeding": [
            "pmb",
            "vaginal bleeding after menopause",
            "bleeding after menopause",
        ],
        "lymphadenopathy": [
            "swollen lymph nodes",
            "enlarged lymph nodes",
            "swollen glands",
            "lumps in neck",
        ],
        "hepatomegaly": [
            "enlarged liver",
            "big liver",
            "liver enlargement",
        ],
        "splenomegaly": [
            "enlarged spleen",
            "big spleen",
            "spleen enlargement",
        ],
        "hepatosplenomegaly": [
            "enlarged liver and spleen",
        ],
        "abdominal pain": [
            "stomach ache",
            "belly pain",
            "tummy pain",
            "stomach pain",
            "abdominal discomfort",
        ],
        "weight loss": [
            "losing weight",
            "lost weight",
            "unintentional weight loss",
            "unexplained weight loss",
        ],
        "fatigue": [
            "tiredness",
            "exhaustion",
            "feeling tired",
            "low energy",
            "lethargy",
        ],
        "shortness of breath": [
            "breathlessness",
            "difficulty breathing",
            "dyspnea",
            "dyspnoea",
            "sob",
            "out of breath",
        ],
        "hoarseness": [
            "hoarse voice",
            "voice changes",
            "croaky voice",
            "raspy voice",
        ],
        "night sweats": [
            "sweating at night",
            "nocturnal sweating",
        ],
        "vulval bleeding": [
            "vulva bleeding",
            "vulvar bleeding",
            "bleeding from vulva",
            # Common typos
            "vuval bleeding",
            "vuvla bleeding",
            "vulval bleed",
            "vulva bleed",
        ],
        "vaginal bleeding": [
            "vaginal bleed",
            "bleeding vaginally",
        ],
        "vaginal discharge": [
            "discharge from vagina",
        ],
    }
    
    # Terms that are NOT defined in NG12 and require clarification
    AMBIGUOUS_AGE_TERMS = [
        "elderly",
        "older",
        "middle-aged",
        "young adult",
        "adult",
        "senior",
        "geriatric",
        "pediatric",
        "paediatric",
    ]
    
    def __init__(self):
        # Build reverse lookup for symptoms
        self._symptom_lookup: dict[str, str] = {}
        for canonical, synonyms in self.SYMPTOM_SYNONYMS.items():
            self._symptom_lookup[canonical.lower()] = canonical
            for syn in synonyms:
                self._symptom_lookup[syn.lower()] = canonical
    
    def get_definition(self, term: str) -> TermDefinition | None:
        """Get definition for a term."""
        return self.DEFINITIONS.get(term.lower())
    
    def expand_age_term(self, query: str) -> tuple[int | None, int | None, str | None]:
        """
        Expand age terms in query using NG12 definitions.
        
        Returns:
            (age_min, age_max, clarification_message)
            
        If term is NG12-defined: returns age range, no clarification
        If term is ambiguous: returns None, None, clarification message
        If no age term: returns None, None, None
        """
        query_lower = query.lower()
        
        # Check NG12-defined terms (order matters - longer first)
        ordered_terms = sorted(self.DEFINITIONS.keys(), key=len, reverse=True)
        for term in ordered_terms:
            defn = self.DEFINITIONS[term]
            if term in query_lower and defn.age_min is not None:
                logger.debug(
                    "Expanded age term",
                    term=term,
                    age_min=defn.age_min,
                    age_max=defn.age_max,
                )
                return defn.age_min, defn.age_max, None
        
        # Check ambiguous terms
        for term in self.AMBIGUOUS_AGE_TERMS:
            if term in query_lower:
                return None, None, (
                    f"The term '{term}' is not defined in NG12. "
                    "Please specify the patient's age to identify applicable referral criteria."
                )
        
        return None, None, None
    
    def expand_facts_age(self, facts: ExtractedFacts) -> int | None:
        """
        Get age from facts, expanding terms if needed.
        
        If facts.age is set, returns it.
        If facts.age_term or raw_query contains NG12 terms, returns midpoint of range.
        """
        if facts.age is not None:
            return facts.age
        
        # Check for age term
        query = facts.age_term or facts.raw_query
        age_min, age_max, _ = self.expand_age_term(query)
        
        if age_min is not None and age_max is not None:
            # Return midpoint for matching
            return (age_min + age_max) // 2
        elif age_min is not None:
            return age_min
        
        return None
    
    def normalize_symptom(self, symptom: str) -> str:
        """
        Normalize a symptom to its canonical form.
        
        Examples:
            "coughing up blood" → "haemoptysis"
            "difficulty swallowing" → "dysphagia"
        """
        symptom_lower = symptom.lower().strip()
        
        # Direct lookup
        if symptom_lower in self._symptom_lookup:
            return self._symptom_lookup[symptom_lower]
        
        # Partial match (for longer phrases)
        for synonym, canonical in self._symptom_lookup.items():
            if synonym in symptom_lower or symptom_lower in synonym:
                return canonical
        
        # No match - return original
        return symptom
    
    def normalize_symptoms(self, symptoms: list[str]) -> list[str]:
        """Normalize a list of symptoms."""
        normalized = []
        seen = set()
        
        for symptom in symptoms:
            norm = self.normalize_symptom(symptom)
            if norm.lower() not in seen:
                normalized.append(norm)
                seen.add(norm.lower())
        
        return normalized
    
    def get_clarification_for_age(self, query: str) -> str | None:
        """Get clarification message if query contains ambiguous age terms."""
        _, _, clarification = self.expand_age_term(query)
        return clarification
    
    def get_urgency_timeframe(self, urgency_term: str) -> int | None:
        """Get timeframe in hours for an urgency term."""
        defn = self.DEFINITIONS.get(urgency_term.lower())
        if defn:
            return defn.timeframe_hours
        return None
    
    def extract_qualifiers(self, text: str) -> list[str]:
        """Extract NG12 qualifiers from text (unexplained, persistent, etc.)."""
        qualifiers = []
        text_lower = text.lower()
        
        qualifier_terms = ["unexplained", "persistent", "recurrent", "treatment-resistant"]
        for q in qualifier_terms:
            if q in text_lower:
                qualifiers.append(q)
        
        return qualifiers


# Singleton instance
_terms_index: TermsIndex | None = None


def get_terms_index() -> TermsIndex:
    """Get the singleton terms index instance."""
    global _terms_index
    if _terms_index is None:
        _terms_index = TermsIndex()
    return _terms_index
