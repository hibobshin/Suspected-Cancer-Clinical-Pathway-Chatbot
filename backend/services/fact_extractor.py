"""
Fact Extractor for NG12 Rule Engine.

Uses a small LLM to extract structured facts from natural language queries.
Simple, clean, maintainable - no complex regex patterns.
"""

import json
import re
from typing import Optional

from openai import AsyncOpenAI

from config.config import get_settings
from config.logging_config import get_logger
from models.rule_models import ExtractedFacts

logger = get_logger(__name__)

# Lazy import to avoid circular dependencies
_symptom_normalizer = None


def get_normalizer():
    """Lazy load symptom normalizer."""
    global _symptom_normalizer
    if _symptom_normalizer is None:
        from services.symptom_normalizer import get_symptom_normalizer
        _symptom_normalizer = get_symptom_normalizer()
    return _symptom_normalizer


class FactExtractor:
    """
    Extract structured facts from natural language using LLM.
    
    Architecture:
    1. LLM extracts raw facts (handles typos, variations, natural language)
    2. Embedding normalizer maps symptoms to NG12 vocabulary
    3. Minimal fallback for network errors only
    """
    
    EXTRACTION_PROMPT = """Extract patient facts from this clinical query. Return valid JSON only.

{{
  "age": <integer or null if not mentioned>,
  "gender": <"male" or "female" or null>,
  "symptoms": [<list of symptoms mentioned>],
  "findings": [<list of test/exam findings>],
  "history": [<list of history items like smoking, family history>]
}}

RULES:
- Extract age as integer (e.g., "52-year-old" → 52, "patient is 45" → 45)
- Infer gender from anatomy if not explicit:
  - vulval, vaginal, cervical, ovarian, uterine, breast → female
  - prostate, testicular, scrotal, penile → male
- Normalize symptoms to medical terms:
  - "coughing up blood" → "haemoptysis"
  - "blood in urine" → "haematuria"
  - "difficulty swallowing" → "dysphagia"
  - "indigestion" → "dyspepsia"
- Fix obvious typos: "haemoptsis" → "haemoptysis", "vuval" → "vulval"
- Only extract explicitly mentioned facts - do NOT infer or assume
- findings = test results, X-ray findings, scan results
- history = smoking, asbestos, alcohol, family history

Query: {query}

JSON:"""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-chat"
    
    async def extract(self, query: str) -> ExtractedFacts:
        """
        Extract structured facts from a natural language query.
        
        Returns ExtractedFacts with normalized symptoms.
        """
        # LLM extraction
        raw_facts = await self._llm_extract(query)
        
        if raw_facts:
            # Normalize symptoms using embeddings
            normalized_symptoms = self._normalize_symptoms(raw_facts.get("symptoms", []))
            
            facts = ExtractedFacts(
                age=raw_facts.get("age"),
                gender=raw_facts.get("gender"),
                symptoms=[s for s, _ in normalized_symptoms],
                symptoms_raw=raw_facts.get("symptoms", []),
                findings=raw_facts.get("findings", []),
                history=raw_facts.get("history", []),
                raw_query=query,
            )
            
            logger.info(
                "Extracted facts",
                age=facts.age,
                gender=facts.gender,
                symptoms=facts.symptoms,
            )
            
            return facts
        
        # Minimal fallback - only if LLM completely fails
        logger.warning("LLM extraction failed, using minimal fallback", query=query[:50])
        return self._minimal_fallback(query)
    
    async def _llm_extract(self, query: str) -> Optional[dict]:
        """Extract facts using LLM."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical fact extractor. Return only valid JSON, no markdown."
                    },
                    {
                        "role": "user", 
                        "content": self.EXTRACTION_PROMPT.format(query=query)
                    }
                ],
                temperature=0,
                max_tokens=300,
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_json(content)
            
        except Exception as e:
            logger.error("LLM extraction error", error=str(e))
            return None
    
    def _parse_json(self, content: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        # Remove markdown code blocks if present
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON object
            match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            
            logger.warning("JSON parse failed", content=content[:100])
            return None
    
    def _normalize_symptoms(self, raw_symptoms: list[str]) -> list[tuple[str, float]]:
        """Normalize symptoms using embedding similarity."""
        if not raw_symptoms:
            return []
        
        try:
            normalizer = get_normalizer()
            results = []
            seen = set()
            
            for raw in raw_symptoms:
                result = normalizer.normalize(raw)
                if result:
                    canonical, score = result
                    if canonical not in seen:
                        results.append((canonical, score))
                        seen.add(canonical)
                else:
                    # Keep raw symptom if no match
                    if raw.lower() not in seen:
                        results.append((raw.lower(), 0.5))
                        seen.add(raw.lower())
            
            return results
            
        except Exception as e:
            logger.warning("Symptom normalization failed", error=str(e))
            return [(s.lower(), 0.5) for s in raw_symptoms]
    
    def _minimal_fallback(self, query: str) -> ExtractedFacts:
        """
        Minimal fallback when LLM fails completely.
        
        Only extracts obvious patterns - not meant to be comprehensive.
        """
        query_lower = query.lower()
        
        # Simple age extraction
        age = None
        age_match = re.search(r"(\d{1,3})[\s-]*(?:year|yo|y/?o)", query_lower)
        if age_match:
            age = int(age_match.group(1))
            if age < 1 or age > 120:
                age = None
        
        # Simple gender
        gender = None
        if any(w in query_lower for w in ["female", "woman", "women"]):
            gender = "female"
        elif any(w in query_lower for w in ["male", " man ", " men "]):
            gender = "male"
        
        # Use embedding normalizer to find symptoms in query
        symptoms = []
        try:
            normalizer = get_normalizer()
            # Try to match the whole query against symptom vocabulary
            result = normalizer.normalize(query_lower)
            if result and result[1] > 0.6:
                symptoms.append(result[0])
        except Exception:
            pass
        
        return ExtractedFacts(
            age=age,
            gender=gender,
            symptoms=symptoms,
            symptoms_raw=[],
            findings=[],
            history=[],
            raw_query=query,
        )


# Singleton instance
_extractor_instance: FactExtractor | None = None


def get_fact_extractor() -> FactExtractor:
    """Get the singleton fact extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = FactExtractor()
    return _extractor_instance
