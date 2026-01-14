"""
Rule Parser for NG12 Guideline.

Parses the NG12 markdown document into structured NG12Rule objects
with properly parsed condition trees (AND/OR/COUNT logic).
"""

import hashlib
import json
import re
from pathlib import Path

from config.logging_config import get_logger
from models.rule_models import (
    ActionType,
    AgeConstraint,
    AtomicCondition,
    CompositeCondition,
    CountCondition,
    NG12Rule,
)

logger = get_logger(__name__)

# Paths
GUIDELINE_PATH = Path(__file__).parent.parent.parent / "data" / "final.md"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_RULES = CACHE_DIR / "parsed_rules.json"
CACHE_HASH = CACHE_DIR / "parsed_rules_hash.txt"


class RuleParser:
    """Parse NG12 markdown into structured rules with condition trees."""
    
    # Action patterns - order matters (most specific first)
    ACTION_PATTERNS = [
        (r"refer.*immediate", ActionType.REFER_IMMEDIATE),
        (r"very urgent referral", ActionType.OFFER_VERY_URGENT),
        (r"refer.*suspected cancer pathway", ActionType.REFER_SUSPECTED_CANCER),
        (r"suspected cancer pathway referral", ActionType.REFER_SUSPECTED_CANCER),
        (r"offer.*urgent", ActionType.OFFER_URGENT),
        (r"consider.*urgent", ActionType.CONSIDER_URGENT),
        (r"consider.*non-urgent", ActionType.CONSIDER_NON_URGENT),
        (r"consider.*referral", ActionType.CONSIDER_REFERRAL),
    ]
    
    # Age extraction patterns
    AGE_PATTERNS = [
        (r"aged?\s*(\d+)\s*(?:and|or)?\s*(?:over|above)", "min"),  # "aged 40 and over"
        (r"aged?\s*(\d+)\s*\+", "min"),  # "aged 40+"
        (r"aged?\s*(?:under|below)\s*(\d+)", "max"),  # "aged under 50"
        (r"aged?\s*(\d+)\s*(?:and|or)?\s*(?:older|more)", "min"),  # "aged 50 or older"
    ]
    
    # Qualifiers to extract
    QUALIFIERS = ["unexplained", "persistent", "recurrent", "treatment-resistant", "visible"]
    
    # History indicators
    HISTORY_INDICATORS = [
        "ever smoked", "smoking", "smoker", "smoked",
        "asbestos", "exposure", "exposed",
        "history of", "previous", "family history",
    ]
    
    # Finding indicators
    FINDING_INDICATORS = [
        "x-ray", "xray", "scan", "ultrasound", "ct ", "mri",
        "findings", "test", "result", "blood", "fit result",
        "ca125", "serum", "haemoglobin", "platelet",
    ]
    
    def __init__(self):
        self._rules: list[NG12Rule] = []
        self._loaded = False
    
    def get_rules(self) -> list[NG12Rule]:
        """Get all parsed rules, loading from cache if available."""
        if not self._loaded:
            self._load_rules()
        return self._rules
    
    def _load_rules(self) -> None:
        """Load rules from cache or parse from document."""
        if self._loaded:
            return
        
        if not GUIDELINE_PATH.exists():
            logger.warning("Guideline file not found", path=str(GUIDELINE_PATH))
            self._loaded = True
            return
        
        # Check cache validity
        current_hash = self._compute_document_hash()
        cached_hash = self._load_cached_hash()
        
        if current_hash == cached_hash and CACHE_RULES.exists():
            try:
                self._rules = self._load_from_cache()
                logger.info("Loaded rules from cache", count=len(self._rules))
                self._loaded = True
                return
            except Exception as e:
                logger.warning("Failed to load cache, reparsing", error=str(e))
        
        # Parse document
        logger.info("Parsing NG12 document...")
        with open(GUIDELINE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        
        self._rules = self._parse_document(content)
        
        # Save cache
        self._save_to_cache(self._rules)
        self._save_document_hash(current_hash)
        
        logger.info("Parsed rules", count=len(self._rules))
        self._loaded = True
    
    def _compute_document_hash(self) -> str:
        """Compute MD5 hash of document for cache invalidation."""
        with open(GUIDELINE_PATH, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_cached_hash(self) -> str | None:
        """Load cached document hash."""
        if CACHE_HASH.exists():
            return CACHE_HASH.read_text().strip()
        return None
    
    def _save_document_hash(self, hash_value: str) -> None:
        """Save document hash."""
        CACHE_HASH.write_text(hash_value)
    
    def _load_from_cache(self) -> list[NG12Rule]:
        """Load rules from JSON cache."""
        with open(CACHE_RULES, "r") as f:
            data = json.load(f)
        return [NG12Rule.model_validate(r) for r in data]
    
    def _save_to_cache(self, rules: list[NG12Rule]) -> None:
        """Save rules to JSON cache."""
        data = [r.model_dump() for r in rules]
        with open(CACHE_RULES, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _parse_document(self, content: str) -> list[NG12Rule]:
        """Parse the entire document into rules."""
        rules = []
        
        # Track current section context
        current_section = ""
        current_subsection = ""
        
        # Split into lines for processing
        lines = content.split("\n")
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Track section headers
            if line.startswith("## 1."):
                # Main section like "## 1.1 Lung and pleural cancers"
                match = re.match(r"## (1\.\d+)\s+(.+)", line)
                if match:
                    current_section = match.group(2).strip()
                    current_subsection = ""
            
            elif line.startswith("### "):
                # Subsection like "### Lung cancer"
                current_subsection = line[4:].strip()
            
            # Look for rule patterns (e.g., "1.1.1 Refer...")
            rule_match = re.match(r"^(\d+\.\d+(?:\.\d+)?)\s+(.+)", line)
            if rule_match:
                rule_id = rule_match.group(1)
                
                # Collect full rule text (may span multiple lines)
                rule_text, end_idx = self._collect_rule_text(lines, i)
                char_start = content.find(line)
                
                # Parse the rule
                try:
                    rule = self._parse_rule(
                        rule_id=rule_id,
                        rule_text=rule_text,
                        cancer_site=current_section,
                        cancer_type=current_subsection,
                        char_start=char_start,
                        char_end=char_start + len(rule_text),
                    )
                    if rule:
                        rules.append(rule)
                except Exception as e:
                    logger.warning(
                        "Failed to parse rule",
                        rule_id=rule_id,
                        error=str(e),
                    )
                
                i = end_idx
                continue
            
            i += 1
        
        return rules
    
    def _collect_rule_text(self, lines: list[str], start_idx: int) -> tuple[str, int]:
        """Collect full rule text spanning multiple lines."""
        collected = [lines[start_idx]]
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i]
            
            # Stop conditions: new rule, new section, or empty line followed by non-continuation
            if re.match(r"^\d+\.\d+(?:\.\d+)?\s+", line):
                break
            if line.startswith("## ") or line.startswith("### "):
                break
            if line.startswith("> ") and not collected[-1].strip().endswith(":"):
                break
            
            # Include bullet points and indented content
            if line.strip().startswith("*") or line.strip().startswith("-"):
                collected.append(line)
            elif line.strip() and not line.strip().startswith("#"):
                collected.append(line)
            elif not line.strip():
                # Empty line - check if next line is continuation
                if i + 1 < len(lines) and (
                    lines[i + 1].strip().startswith("*") or
                    lines[i + 1].strip().startswith("-") or
                    lines[i + 1].strip().startswith("•")
                ):
                    collected.append(line)
                else:
                    break
            else:
                break
            
            i += 1
        
        return "\n".join(collected), i
    
    def _parse_rule(
        self,
        rule_id: str,
        rule_text: str,
        cancer_site: str,
        cancer_type: str,
        char_start: int,
        char_end: int,
    ) -> NG12Rule | None:
        """Parse a single rule into structured format."""
        
        # Extract action type
        action = self._extract_action(rule_text)
        if not action:
            return None
        
        # Extract action text (first sentence/clause)
        action_text = self._extract_action_text(rule_text)
        
        # Extract age constraint
        age_constraint = self._extract_age_constraint(rule_text)
        
        # Extract conditions
        conditions = self._extract_conditions(rule_text)
        
        # Extract source year
        source_year = self._extract_year(rule_text)
        
        # Build section path
        section_path = f"NG12 > {cancer_site}"
        if cancer_type:
            section_path += f" > {cancer_type}"
        
        return NG12Rule(
            rule_id=rule_id,
            cancer_site=cancer_site,
            cancer_type=cancer_type if cancer_type else None,
            section_path=section_path,
            action=action,
            action_text=action_text,
            age_constraint=age_constraint,
            conditions=conditions,
            verbatim_text=rule_text.strip(),
            source_year=source_year,
            char_start=char_start,
            char_end=char_end,
        )
    
    def _extract_action(self, text: str) -> ActionType | None:
        """Extract action type from rule text."""
        text_lower = text.lower()
        for pattern, action in self.ACTION_PATTERNS:
            if re.search(pattern, text_lower):
                return action
        return None
    
    def _extract_action_text(self, text: str) -> str:
        """Extract the action portion of the rule."""
        # Find the first sentence or up to "if they" / "in people"
        match = re.match(
            r"^(\d+\.\d+(?:\.\d+)?)\s+(.+?)(?:\s+if\s+|\s+in\s+people|\s+in\s+adults|\s+in\s+women|\s+in\s+men|\s+for\s+people)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(2).strip()
        
        # Fallback: first line
        first_line = text.split("\n")[0]
        return re.sub(r"^\d+\.\d+(?:\.\d+)?\s+", "", first_line).strip()
    
    def _extract_age_constraint(self, text: str) -> AgeConstraint | None:
        """Extract age constraint from rule text."""
        text_lower = text.lower()
        
        for pattern, age_type in self.AGE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                age_text = match.group(0)
                
                if age_type == "min":
                    return AgeConstraint(min_age=age, text=age_text)
                else:
                    return AgeConstraint(max_age=age, text=age_text)
        
        return None
    
    def _extract_year(self, text: str) -> str:
        """Extract source year from rule text."""
        # Look for [2015] or [2015, amended 2025] patterns
        match = re.search(r"\[(\d{4}(?:,\s*amended\s*\d{4})?)\]", text)
        if match:
            return match.group(1)
        return "unknown"
    
    def _extract_conditions(self, text: str) -> CompositeCondition | AtomicCondition | None:
        """Extract conditions from rule text into a condition tree."""
        
        # Find the condition portion (after "if they", "in people with", etc.)
        condition_portion = self._get_condition_portion(text)
        if not condition_portion:
            return None
        
        # Check for bullet list (OR structure)
        bullet_items = self._extract_bullet_items(condition_portion)
        
        if bullet_items:
            # Parse each bullet as a branch
            children = []
            for item in bullet_items:
                child = self._parse_condition_branch(item)
                if child:
                    children.append(child)
            
            if len(children) == 1:
                return children[0]
            elif len(children) > 1:
                return CompositeCondition(type="or", children=children)
        
        # No bullets - parse as single condition or AND group
        return self._parse_condition_branch(condition_portion)
    
    def _get_condition_portion(self, text: str) -> str:
        """Extract the condition portion of rule text."""
        # Try different patterns
        patterns = [
            r"if they[:\s]+(.+)",
            r"if (?:a )?(?:person|people|patient)[:\s]+(.+)",
            r"in (?:people|adults|women|men|children)(?: aged \d+[^:]+)?[:\s]+(.+)",
            r"with[:\s]+(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return text
    
    def _extract_bullet_items(self, text: str) -> list[str]:
        """Extract bullet list items from text."""
        items = []
        
        # Split by bullet markers
        lines = text.split("\n")
        current_item = ""
        
        for line in lines:
            stripped = line.strip()
            
            # Check for bullet marker
            if stripped.startswith("*") or stripped.startswith("-") or stripped.startswith("•"):
                if current_item:
                    items.append(current_item.strip())
                # Remove bullet and leading/trailing whitespace
                current_item = re.sub(r"^[\*\-•]\s*", "", stripped)
            elif stripped.startswith("- "):
                # Sub-bullet (indented)
                current_item += " " + stripped[2:]
            elif stripped and current_item:
                # Continuation of previous item
                current_item += " " + stripped
        
        if current_item:
            items.append(current_item.strip())
        
        return items
    
    def _parse_condition_branch(self, text: str) -> AtomicCondition | CompositeCondition | CountCondition | None:
        """Parse a single condition branch (may contain AND or COUNT)."""
        text = text.strip()
        if not text:
            return None
        
        # Remove trailing year markers and "or" suffixes
        text = re.sub(r"\s*\[\d{4}.*?\]\s*$", "", text)
        text = re.sub(r"\s+(?:\*\*)?or(?:\*\*)?\s*\.?\s*$", "", text, flags=re.IGNORECASE)
        
        # Check for "N or more of the following" pattern
        count_match = re.search(
            r"(\d+)\s+or more of the following",
            text,
            re.IGNORECASE,
        )
        if count_match:
            threshold = int(count_match.group(1))
            # Get the items that follow
            items_text = text[count_match.end():]
            items = self._extract_inline_items(items_text)
            if items:
                options = [self._parse_atomic(item) for item in items]
                return CountCondition(type="count_gte", threshold=threshold, options=options)
        
        # Check for "any of the following" pattern
        if re.search(r"any of the following", text, re.IGNORECASE):
            items_text = re.split(r"any of the following[:\s]*", text, flags=re.IGNORECASE)[-1]
            items = self._extract_inline_items(items_text)
            if items:
                options = [self._parse_atomic(item) for item in items]
                return CountCondition(type="count_any", threshold=1, options=options)
        
        # Check for AND patterns within the text
        and_parts = self._split_by_and(text)
        if len(and_parts) > 1:
            children = [self._parse_atomic(part) for part in and_parts]
            return CompositeCondition(type="and", children=children)
        
        # Single atomic condition
        return self._parse_atomic(text)
    
    def _split_by_and(self, text: str) -> list[str]:
        """Split text by AND conjunctions, being careful with 'and over' age patterns."""
        # Don't split on "and over", "and above", etc.
        # Split on " and " when not followed by age-related words
        parts = re.split(r"\s+and\s+(?!over|above|older|more)", text, flags=re.IGNORECASE)
        
        # Also split on " with " as it often indicates AND
        result = []
        for part in parts:
            subparts = re.split(r"\s+with\s+", part, flags=re.IGNORECASE)
            result.extend(subparts)
        
        return [p.strip() for p in result if p.strip()]
    
    def _extract_inline_items(self, text: str) -> list[str]:
        """Extract items from inline lists (colon-separated or bullet-separated)."""
        items = []
        
        # Try bullet format first
        bullet_items = self._extract_bullet_items(text)
        if bullet_items:
            return bullet_items
        
        # Try comma/newline separated
        parts = re.split(r"[,\n]", text)
        for part in parts:
            cleaned = part.strip().strip(".-•*")
            if cleaned and len(cleaned) > 2:
                items.append(cleaned)
        
        return items
    
    # Age indicators - conditions that are really age constraints
    AGE_INDICATORS = [
        "aged ", "age ", "years old", "year old",
        "and over", "or over", "and above", "or older",
        "under ", "below ",
    ]
    
    def _parse_atomic(self, text: str) -> AtomicCondition:
        """Parse a single atomic condition (symptom, finding, or history)."""
        text = text.strip().lower()
        
        # Remove common suffixes
        text = re.sub(r"\s*\[\d{4}.*?\]\s*$", "", text)
        text = re.sub(r"\s*\*\*\s*$", "", text)
        text = re.sub(r"^\*\*\s*", "", text)
        text = re.sub(r"\s+or\s*$", "", text)
        
        # Detect condition type
        condition_type = "symptom"  # default
        
        # Check for age-type condition (should be skipped in matching)
        for indicator in self.AGE_INDICATORS:
            if indicator in text:
                # This is an age condition embedded in the text
                # Mark it as "age" type so matcher can handle it
                return AtomicCondition(type="age", value=text.strip(), qualifier=None)
        
        # Check for history
        for indicator in self.HISTORY_INDICATORS:
            if indicator in text:
                condition_type = "history"
                break
        
        # Check for finding
        if condition_type == "symptom":
            for indicator in self.FINDING_INDICATORS:
                if indicator in text:
                    condition_type = "finding"
                    break
        
        # Extract qualifier
        qualifier = None
        for q in self.QUALIFIERS:
            if q in text:
                qualifier = q
                break
        
        return AtomicCondition(type=condition_type, value=text.strip(), qualifier=qualifier)


# Singleton instance
_parser_instance: RuleParser | None = None


def get_rule_parser() -> RuleParser:
    """Get the singleton rule parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = RuleParser()
    return _parser_instance
