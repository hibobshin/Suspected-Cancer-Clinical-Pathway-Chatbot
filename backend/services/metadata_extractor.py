"""
Discrete metadata extraction module for NG12 chunks.

CRITICAL: All metadata extraction is DETERMINISTIC (regex/pattern matching), NOT LLM-based.

This module extracts:
- Rule IDs (from numbered headings)
- Age ranges (from explicit patterns)
- Symptom tags (from controlled vocabulary)
- Action types (from keyword matching)
- Trigger types (from keyword matching)

All extraction uses deterministic pattern matching, never LLM inference.
"""

import re
from typing import Literal

from config.custom_config import CustomPipelineSettings, get_custom_settings
from config.logging_config import get_logger
from models.custom_models import (
    ActionType,
    AuditMetadata,
    InheritedMetadata,
    LocalMetadata,
    MetadataQuality,
    TriggerType,
)

logger = get_logger(__name__)


class MetadataExtractor:
    """
    Deterministic metadata extractor for NG12 chunks.
    
    All extraction is pattern-based, not LLM-based.
    """
    
    def __init__(self, settings: CustomPipelineSettings | None = None):
        """
        Initialize the metadata extractor.
        
        Args:
            settings: Custom pipeline settings. Uses default if not provided.
        """
        self.settings = settings or get_custom_settings()
    
    def extract_local_metadata(
        self,
        chunk_text: str,
        parent_metadata: InheritedMetadata | None = None,
    ) -> tuple[LocalMetadata, AuditMetadata]:
        """
        Extract local metadata from chunk text using deterministic pattern matching.
        
        Args:
            chunk_text: The text of the chunk to extract from.
            parent_metadata: Inherited metadata from parent sections.
            
        Returns:
            Tuple of (LocalMetadata, AuditMetadata).
        """
        logger.debug("Extracting local metadata", text_preview=chunk_text[:100])
        
        # Extract rule ID
        rule_id = self._extract_rule_id(chunk_text)
        
        # Extract age range
        age_min, age_max, age_text = self._extract_age_range(chunk_text)
        
        # Extract symptom tags
        symptom_tags = self._extract_symptom_tags(chunk_text)
        
        # Extract action type
        action_type = self._extract_action_type(chunk_text)
        
        # Extract trigger type
        trigger_type = self._extract_trigger_type(chunk_text)
        
        # Build audit metadata
        source_heading = self._extract_heading(chunk_text)
        extraction_notes = self._build_extraction_notes(
            rule_id=rule_id,
            age_text=age_text,
            action_type=action_type,
        )
        
        local_metadata = LocalMetadata(
            rule_id=rule_id,
            age_min=age_min,
            age_max=age_max,
            age_text=age_text,
            symptom_tags=symptom_tags,
            action_type=action_type,
            trigger_type=trigger_type,
        )
        
        audit_metadata = AuditMetadata(
            source_heading=source_heading,
            source_offsets=None,  # Can be enhanced later
            age_text=age_text,
            extraction_notes=extraction_notes,
        )
        
        logger.debug(
            "Local metadata extracted",
            rule_id=rule_id,
            action_type=action_type.value if action_type else None,
            symptom_count=len(symptom_tags),
        )
        
        return local_metadata, audit_metadata
    
    def _extract_rule_id(self, text: str) -> str | None:
        """
        Extract rule ID using deterministic regex patterns.
        
        Patterns:
        - Numbered headings: "1.2.1", "1.3.4"
        - Explicit phrases: "recommendation 1.2.1", "section 1.3.4"
        
        Precedence: Section heading > earliest occurrence in text.
        
        Args:
            text: Text to extract from.
            
        Returns:
            Rule ID string (e.g., "1.2.1") or None if not found.
        """
        # Pattern for rule IDs: X.Y.Z format
        rule_id_pattern = r'\b(\d+\.\d+\.\d+)\b'
        
        # First, check for rule ID in headings (## or ###)
        heading_pattern = r'^#{2,3}\s*.*?(\d+\.\d+\.\d+)'
        heading_match = re.search(heading_pattern, text, re.MULTILINE)
        if heading_match:
            rule_id = heading_match.group(1)
            logger.debug("Rule ID found in heading", rule_id=rule_id)
            return rule_id
        
        # Check for explicit phrases
        explicit_pattern = r'(?:recommendation|section)\s+(\d+\.\d+\.\d+)'
        explicit_match = re.search(explicit_pattern, text, re.IGNORECASE)
        if explicit_match:
            rule_id = explicit_match.group(1)
            logger.debug("Rule ID found in explicit phrase", rule_id=rule_id)
            return rule_id
        
        # Check for earliest occurrence
        all_matches = re.findall(rule_id_pattern, text)
        if all_matches:
            rule_id = all_matches[0]  # Earliest occurrence
            logger.debug("Rule ID found (earliest occurrence)", rule_id=rule_id)
            return rule_id
        
        logger.debug("No rule ID found")
        return None
    
    def _extract_age_range(
        self,
        text: str,
    ) -> tuple[int | None, int | None, str | None]:
        """
        Extract age range using deterministic regex patterns.
        
        Patterns:
        - "aged X and over" → age_min = X, age_max = None
        - "X+" or "≥X" → age_min = X, age_max = None
        - "under X" or "children under X" → age_min = None, age_max = X-1
        - "X to Y" → age_min = X, age_max = Y
        
        If ambiguous ("older people", "adults"): return None values with verbatim text.
        
        Args:
            text: Text to extract from.
            
        Returns:
            Tuple of (age_min, age_max, age_text).
        """
        age_text = None
        age_min = None
        age_max = None
        
        # Pattern: "aged X and over"
        pattern1 = r'aged\s+(\d+)\s+and\s+over'
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            age_min = int(match.group(1))
            age_max = None
            age_text = match.group(0)
            logger.debug("Age extracted (aged X and over)", age_min=age_min)
            return age_min, age_max, age_text
        
        # Pattern: "X+" or "≥X"
        pattern2 = r'(\d+)\s*\+|≥\s*(\d+)'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            age_min = int(match.group(1) or match.group(2))
            age_max = None
            age_text = match.group(0)
            logger.debug("Age extracted (X+)", age_min=age_min)
            return age_min, age_max, age_text
        
        # Pattern: "under X" or "children under X"
        pattern3 = r'(?:children\s+)?under\s+(\d+)'
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            age_min = None
            age_max = int(match.group(1)) - 1
            age_text = match.group(0)
            logger.debug("Age extracted (under X)", age_max=age_max + 1)
            return age_min, age_max, age_text
        
        # Pattern: "X to Y" or "X-Y"
        pattern4 = r'(\d+)\s*(?:to|-)\s*(\d+)'
        match = re.search(pattern4, text, re.IGNORECASE)
        if match:
            age_min = int(match.group(1))
            age_max = int(match.group(2))
            age_text = match.group(0)
            logger.debug("Age extracted (X to Y)", age_min=age_min, age_max=age_max)
            return age_min, age_max, age_text
        
        # Check for ambiguous patterns
        ambiguous_patterns = [
            r'older\s+people',
            r'adults',
            r'elderly',
            r'children',
        ]
        for pattern in ambiguous_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age_text = match.group(0)
                logger.debug("Age ambiguous", age_text=age_text)
                return None, None, age_text
        
        logger.debug("No age range found")
        return None, None, None
    
    def _extract_symptom_tags(self, text: str) -> list[str]:
        """
        Extract symptom tags using controlled vocabulary matching.
        
        Only tags symptoms that are EXPLICITLY mentioned (case-insensitive).
        Does NOT use NLP or LLM to infer symptoms.
        
        Args:
            text: Text to extract from.
            
        Returns:
            List of symptom tags from controlled vocabulary.
        """
        text_lower = text.lower()
        symptom_tags = []
        
        for symptom in self.settings.symptom_vocabulary:
            # Case-insensitive exact match (word boundaries)
            pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
            if re.search(pattern, text_lower):
                symptom_tags.append(symptom)
                logger.debug("Symptom tag found", symptom=symptom)
        
        return symptom_tags
    
    def _extract_action_type(self, text: str) -> ActionType | None:
        """
        Extract action type using deterministic keyword matching.
        
        Keywords:
        - "refer on suspected cancer pathway" or "2WW" → 2WW_REFERRAL
        - "urgent test" or "urgent endoscopy" → URGENT_TEST
        - "urgent CXR" or "urgent chest X-ray" → URGENT_CXR
        - "consider urgent CXR" → CONSIDER_URGENT_CXR
        - "routine" or "non-urgent" → ROUTINE_WORKUP
        - "safety net" or "follow-up" → SAFETY_NET
        - Informational only → INFO_ONLY
        
        Args:
            text: Text to extract from.
            
        Returns:
            ActionType enum or None if not found.
        """
        text_lower = text.lower()
        
        # 2WW referral
        if re.search(r'refer\s+on\s+suspected\s+cancer\s+pathway|2WW|two\s+week\s+wait', text_lower):
            return ActionType.TWO_WW_REFERRAL
        
        # Urgent CXR
        if re.search(r'consider\s+urgent\s+(?:CXR|chest\s+X-ray)', text_lower):
            return ActionType.CONSIDER_URGENT_CXR
        
        if re.search(r'urgent\s+(?:CXR|chest\s+X-ray)', text_lower):
            return ActionType.URGENT_CXR
        
        # Urgent test
        if re.search(r'urgent\s+(?:test|endoscopy|investigation)', text_lower):
            return ActionType.URGENT_TEST
        
        # Routine workup
        if re.search(r'routine|non-urgent', text_lower):
            return ActionType.ROUTINE_WORKUP
        
        # Safety net
        if re.search(r'safety\s+net|follow-up|follow\s+up', text_lower):
            return ActionType.SAFETY_NET
        
        # If no action language found, return None
        return None
    
    def _extract_trigger_type(self, text: str) -> TriggerType | None:
        """
        Extract trigger type using deterministic keyword matching.
        
        Args:
            text: Text to extract from.
            
        Returns:
            TriggerType enum or None if not found.
        """
        text_lower = text.lower()
        
        if re.search(r'symptom', text_lower):
            return TriggerType.SYMPTOM
        
        if re.search(r'sign|clinical\s+sign', text_lower):
            return TriggerType.SIGN
        
        if re.search(r'test\s+result|investigation\s+result', text_lower):
            return TriggerType.TEST_RESULT
        
        if re.search(r'incidental\s+finding', text_lower):
            return TriggerType.INCIDENTAL_FINDING
        
        return None
    
    def _extract_heading(self, text: str) -> str:
        """Extract heading from text (first markdown heading)."""
        heading_pattern = r'^#{1,3}\s+(.+)$'
        match = re.search(heading_pattern, text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Unknown"
    
    def _build_extraction_notes(
        self,
        rule_id: str | None,
        age_text: str | None,
        action_type: ActionType | None,
    ) -> str | None:
        """Build extraction notes for audit."""
        notes = []
        
        if rule_id is None:
            notes.append("rule_id not found")
        
        if age_text is None:
            notes.append("age not found")
        elif "older" in (age_text or "").lower() or "adult" in (age_text or "").lower():
            notes.append("age ambiguous")
        
        if action_type is None:
            notes.append("action_type not found")
        
        return "; ".join(notes) if notes else None
    
    def assign_metadata_quality(
        self,
        local_metadata: LocalMetadata,
    ) -> MetadataQuality:
        """
        Assign metadata quality level using deterministic rules.
        
        Rules:
        - HIGH: rule_id present + explicit action_type + explicit condition(s) (age/symptoms)
        - MEDIUM: action_type present but conditions partially missing
        - LOW: Informational or ambiguous (no action, no rule_id, ambiguous conditions)
        
        Args:
            local_metadata: Local metadata to evaluate.
            
        Returns:
            MetadataQuality enum.
        """
        has_rule_id = local_metadata.rule_id is not None
        has_action = local_metadata.action_type is not None
        has_age = local_metadata.age_min is not None or local_metadata.age_max is not None
        has_symptoms = len(local_metadata.symptom_tags) > 0
        has_conditions = has_age or has_symptoms
        
        if has_rule_id and has_action and has_conditions:
            return MetadataQuality.HIGH
        
        if has_action and not has_conditions:
            return MetadataQuality.MEDIUM
        
        return MetadataQuality.LOW


def get_metadata_extractor() -> MetadataExtractor:
    """Get the metadata extractor singleton."""
    return MetadataExtractor()
