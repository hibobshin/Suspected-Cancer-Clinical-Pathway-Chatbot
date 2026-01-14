"""
Rule Matcher for NG12 Guideline.

Provides deterministic matching of extracted facts against parsed rules.
NO LLM is used in this module - all matching is based on explicit criteria.
"""

from collections import defaultdict

from config.logging_config import get_logger
from models.rule_models import (
    AtomicCondition,
    CompositeCondition,
    CountCondition,
    ExtractedFacts,
    MatchResult,
    NG12Rule,
)
from services.rule_parser import get_rule_parser
from services.terms_index import get_terms_index, TermsIndex

logger = get_logger(__name__)


class RuleMatcher:
    """
    Deterministic rule matching engine.
    
    Matches extracted facts against NG12 rules using explicit criteria.
    No LLM is involved - all matching is based on:
    - Age constraints
    - Symptom presence
    - Finding presence
    - Patient history
    - Logical operators (AND, OR, COUNT)
    """
    
    def __init__(
        self,
        rules: list[NG12Rule] | None = None,
        terms_index: TermsIndex | None = None,
    ):
        """
        Initialize the matcher.
        
        Args:
            rules: List of parsed rules. If None, loads from parser.
            terms_index: Terms index for normalization. If None, uses singleton.
        """
        self.terms = terms_index or get_terms_index()
        
        if rules is None:
            parser = get_rule_parser()
            self._rules = parser.get_rules()
        else:
            self._rules = rules
        
        # Build indexes for fast lookup
        self._build_indexes()
        
        logger.info("Rule matcher initialized", rule_count=len(self._rules))
    
    # Key symptom words that should be indexed individually
    SYMPTOM_KEYWORDS = [
        "bleeding", "lump", "mass", "pain", "ulcer", "ulceration", "discharge",
        "swelling", "cough", "hoarseness", "fatigue", "weight", "loss",
        "haemoptysis", "dysphagia", "dyspepsia", "haematuria", "anaemia",
        "vulval", "vaginal", "breast", "rectal", "abdominal", "chest",
        "lymph", "bone", "headache", "night", "sweats",
    ]
    
    def _build_indexes(self) -> None:
        """Build lookup indexes for fast candidate selection."""
        self.symptom_to_rules: dict[str, list[NG12Rule]] = defaultdict(list)
        self.cancer_site_to_rules: dict[str, list[NG12Rule]] = defaultdict(list)
        self.finding_to_rules: dict[str, list[NG12Rule]] = defaultdict(list)
        
        for rule in self._rules:
            # Index by cancer site
            self.cancer_site_to_rules[rule.cancer_site.lower()].append(rule)
            
            # Index by conditions
            if rule.conditions:
                for condition in self._flatten_conditions(rule.conditions):
                    if isinstance(condition, AtomicCondition):
                        value_lower = condition.value.lower()
                        if condition.type == "symptom":
                            # Index by full value
                            self.symptom_to_rules[value_lower].append(rule)
                            
                            # Also index by individual symptom keywords
                            for keyword in self.SYMPTOM_KEYWORDS:
                                if keyword in value_lower:
                                    self.symptom_to_rules[keyword].append(rule)
                        elif condition.type == "finding":
                            self.finding_to_rules[value_lower].append(rule)
    
    def _flatten_conditions(self, condition) -> list[AtomicCondition]:
        """Recursively flatten a condition tree to get all atomic conditions."""
        if isinstance(condition, AtomicCondition):
            return [condition]
        elif isinstance(condition, CountCondition):
            return condition.options
        elif isinstance(condition, CompositeCondition):
            result = []
            for child in condition.children:
                result.extend(self._flatten_conditions(child))
            return result
        return []
    
    def match(self, facts: ExtractedFacts) -> list[MatchResult]:
        """
        Match extracted facts against all rules.
        
        Args:
            facts: Extracted facts from user query
            
        Returns:
            List of MatchResult objects, sorted by confidence (highest first)
        """
        results = []
        
        # Expand age from terms if needed
        expanded_age = self.terms.expand_facts_age(facts)
        
        # Normalize symptoms
        normalized_symptoms = self.terms.normalize_symptoms(facts.symptoms)
        
        # Get candidate rules based on symptoms/findings
        candidates = self._get_candidate_rules(normalized_symptoms, facts.findings)
        
        logger.debug(
            "Matching facts",
            age=expanded_age,
            symptoms=normalized_symptoms,
            candidate_count=len(candidates),
        )
        
        # Evaluate each candidate
        for rule in candidates:
            result = self._evaluate_rule(rule, facts, expanded_age, normalized_symptoms)
            if result.confidence > 0:
                results.append(result)
        
        # Sort by confidence (highest first), then by match type
        results.sort(key=lambda r: (-r.confidence, r.match_type != "full"))
        
        logger.info(
            "Matching complete",
            total_candidates=len(candidates),
            matches=len(results),
            full_matches=sum(1 for r in results if r.match_type == "full"),
        )
        
        return results
    
    def _get_candidate_rules(
        self,
        symptoms: list[str],
        findings: list[str],
    ) -> set[NG12Rule]:
        """Get candidate rules that might match based on symptoms/findings."""
        candidates = set()
        
        # Add rules matching symptoms
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            candidates.update(self.symptom_to_rules.get(symptom_lower, []))
            
            # Also check partial matches
            for key, rules in self.symptom_to_rules.items():
                if symptom_lower in key or key in symptom_lower:
                    candidates.update(rules)
        
        # Add rules matching findings
        for finding in findings:
            finding_lower = finding.lower()
            candidates.update(self.finding_to_rules.get(finding_lower, []))
            
            # Partial matches
            for key, rules in self.finding_to_rules.items():
                if finding_lower in key or key in finding_lower:
                    candidates.update(rules)
        
        # If no candidates from symptoms/findings, consider all rules
        # (useful for age-only queries like "60 year old patient")
        if not candidates:
            candidates = set(self._rules)
        
        return candidates
    
    # Age population constraints based on rule title/section
    CHILDREN_YOUNG_PEOPLE_MAX_AGE = 24
    ADULT_MIN_AGE = 18
    
    def _check_population_constraint(self, rule: NG12Rule, age: int | None) -> tuple[bool, str | None]:
        """
        Check if patient age matches the rule's population constraint.
        
        Rules for "children and young people" should only match age ≤ 24.
        Rules for "adults" should only match age ≥ 18.
        
        Returns (matches, reason) where reason is None if matches.
        """
        if age is None:
            return True, None  # Can't check without age
        
        title_lower = (rule.cancer_type or rule.cancer_site or "").lower()
        
        # Check for children/young people constraint
        if "children and young people" in title_lower or "children or young people" in title_lower:
            if age > self.CHILDREN_YOUNG_PEOPLE_MAX_AGE:
                return False, f"Rule applies to children/young people (age ≤ {self.CHILDREN_YOUNG_PEOPLE_MAX_AGE}), patient is {age}"
        
        # Check for adults-only constraint
        if " in adults" in title_lower and "children" not in title_lower:
            if age < self.ADULT_MIN_AGE:
                return False, f"Rule applies to adults (age ≥ {self.ADULT_MIN_AGE}), patient is {age}"
        
        return True, None
    
    def _evaluate_rule(
        self,
        rule: NG12Rule,
        facts: ExtractedFacts,
        age: int | None,
        normalized_symptoms: list[str],
    ) -> MatchResult:
        """
        Evaluate if a rule matches the given facts.
        
        Returns a MatchResult with match type and confidence.
        """
        matched_conditions: list[str] = []
        unmatched_conditions: list[str] = []
        
        # Check population constraint first (children vs adults)
        population_match, population_reason = self._check_population_constraint(rule, age)
        if not population_match:
            return MatchResult(
                rule=rule,
                match_type="no_match",
                confidence=0.0,
                matched_conditions=[],
                unmatched_conditions=[population_reason] if population_reason else [],
            )
        
        # Check age constraint (at rule level)
        age_matched = self._check_age_constraint(rule, age, matched_conditions, unmatched_conditions)
        
        # Check conditions (may include embedded age conditions)
        conditions_result = self._check_conditions(
            rule.conditions,
            facts,
            normalized_symptoms,
            matched_conditions,
            unmatched_conditions,
            age,
        )
        
        # Calculate confidence
        total = len(matched_conditions) + len(unmatched_conditions)
        if total == 0:
            confidence = 0.0
        else:
            confidence = len(matched_conditions) / total
        
        # Determine match type
        if not unmatched_conditions and matched_conditions:
            match_type = "full"
        elif matched_conditions and unmatched_conditions:
            if age_matched and not conditions_result:
                match_type = "age_only"
            elif conditions_result and not age_matched:
                match_type = "symptom_only"
            else:
                match_type = "partial"
        else:
            match_type = "partial"
        
        return MatchResult(
            rule=rule,
            match_type=match_type,
            matched_conditions=matched_conditions,
            unmatched_conditions=unmatched_conditions,
            confidence=confidence,
        )
    
    def _check_age_constraint(
        self,
        rule: NG12Rule,
        age: int | None,
        matched: list[str],
        unmatched: list[str],
    ) -> bool:
        """Check if age constraint is satisfied."""
        if not rule.age_constraint:
            return True  # No constraint
        
        if age is None:
            unmatched.append(f"Age required: {rule.age_constraint.text}")
            return False
        
        # Check min age
        if rule.age_constraint.min_age is not None:
            if age < rule.age_constraint.min_age:
                unmatched.append(f"Age {age} < {rule.age_constraint.min_age}")
                return False
            else:
                matched.append(f"Age {age} meets '{rule.age_constraint.text}'")
        
        # Check max age
        if rule.age_constraint.max_age is not None:
            if age > rule.age_constraint.max_age:
                unmatched.append(f"Age {age} > {rule.age_constraint.max_age}")
                return False
            else:
                matched.append(f"Age {age} meets '{rule.age_constraint.text}'")
        
        return True
    
    def _check_conditions(
        self,
        condition,
        facts: ExtractedFacts,
        normalized_symptoms: list[str],
        matched: list[str],
        unmatched: list[str],
        age: int | None = None,
    ) -> bool:
        """
        Recursively check if conditions are satisfied.
        
        Returns True if conditions are fully satisfied.
        """
        if condition is None:
            return True
        
        if isinstance(condition, AtomicCondition):
            return self._check_atomic(condition, facts, normalized_symptoms, matched, unmatched, age)
        
        elif isinstance(condition, CountCondition):
            return self._check_count(condition, facts, normalized_symptoms, matched, unmatched, age)
        
        elif isinstance(condition, CompositeCondition):
            return self._check_composite(condition, facts, normalized_symptoms, matched, unmatched, age)
        
        return True
    
    # Anatomical site keywords for precise matching
    ANATOMICAL_SITES = {
        "vulval", "vulva", "vaginal", "vagina", "breast", "rectal", "rectum",
        "abdominal", "abdomen", "chest", "lung", "throat", "oral", "mouth",
        "skin", "bone", "liver", "kidney", "bladder", "prostate", "testicular",
        "thyroid", "brain", "head", "neck", "axillary", "groin", "scrotal",
        "pleural", "peritoneal", "hepat", "spleno", "lymph",
    }
    
    # Generic symptom descriptors that should NOT match site-specific symptoms
    GENERIC_QUALIFIERS = {"unexplained", "persistent", "recurrent", "new"}
    
    def _is_symptom_match(self, patient_symptom: str, rule_condition: str) -> bool:
        """
        Check if a patient symptom matches a rule condition with stricter logic.
        
        Key principle: Site-specific symptoms should NOT match generic conditions.
        - "vulval bleeding" should NOT match "unexplained bleeding"
        - "vulval bleeding" SHOULD match "vulval lump, ulceration or bleeding"
        - "haemoptysis" SHOULD match "haemoptysis" (exact medical term)
        """
        # Exact match
        if patient_symptom == rule_condition:
            return True
        
        # Direct containment (e.g., "haemoptysis" in "unexplained haemoptysis")
        if patient_symptom in rule_condition or rule_condition in patient_symptom:
            # But check for site-specificity mismatch
            patient_sites = self._extract_sites(patient_symptom)
            condition_sites = self._extract_sites(rule_condition)
            
            # If patient symptom has a site, condition must have same site or no site
            if patient_sites and condition_sites:
                if not patient_sites.intersection(condition_sites):
                    return False  # Site mismatch: "vulval bleeding" vs "rectal bleeding"
            elif patient_sites and not condition_sites:
                # Patient has site, condition is generic - check if it's just qualifiers
                condition_words = set(rule_condition.replace(',', '').replace('.', '').split())
                non_qualifier_words = condition_words - self.GENERIC_QUALIFIERS
                if len(non_qualifier_words) <= 1:
                    # Condition is just "unexplained bleeding" - too generic for site-specific
                    return False
            
            return True
        
        # Word-level match for complex conditions like "vulval lump, ulceration or bleeding"
        patient_words = set(patient_symptom.split())
        condition_words = set(rule_condition.replace(',', '').replace('.', '').split())
        
        # Get meaningful words (exclude common qualifiers)
        patient_meaningful = patient_words - self.GENERIC_QUALIFIERS - {"or", "and", "with"}
        condition_meaningful = condition_words - self.GENERIC_QUALIFIERS - {"or", "and", "with"}
        
        overlap = patient_meaningful & condition_meaningful
        
        if len(overlap) >= 2:
            # Two+ meaningful words overlap
            # Check site compatibility
            patient_sites = patient_meaningful & self.ANATOMICAL_SITES
            condition_sites = condition_meaningful & self.ANATOMICAL_SITES
            
            if patient_sites and condition_sites:
                # Both have sites - they must match
                if patient_sites.intersection(condition_sites):
                    return True
            elif patient_sites and not condition_sites:
                # Patient has site, condition doesn't - be more careful
                # "vulval bleeding" should NOT match just because "bleeding" overlaps
                return False
            else:
                # Neither has site, or only condition has site - OK
                return True
        
        return False
    
    def _extract_sites(self, text: str) -> set[str]:
        """Extract anatomical sites from text."""
        words = set(text.lower().replace(',', '').replace('.', '').split())
        return words.intersection(self.ANATOMICAL_SITES)
    
    def _check_atomic(
        self,
        condition: AtomicCondition,
        facts: ExtractedFacts,
        normalized_symptoms: list[str],
        matched: list[str],
        unmatched: list[str],
        age: int | None = None,
    ) -> bool:
        """Check a single atomic condition."""
        value_lower = condition.value.lower().rstrip('.')
        
        if condition.type == "symptom":
            # Check if symptom is present with stricter matching
            symptoms_lower = [s.lower() for s in normalized_symptoms]
            
            for symptom in symptoms_lower:
                if self._is_symptom_match(symptom, value_lower):
                    matched.append(f"Symptom: {symptom} (matches '{condition.value}')")
                    return True
                
            unmatched.append(f"Symptom: {condition.value}")
            return False
        
        elif condition.type == "finding":
            # Check if finding is present
            findings_lower = [f.lower() for f in facts.findings]
            for finding in findings_lower:
                if value_lower in finding or finding in value_lower:
                    matched.append(f"Finding: {condition.value}")
                    return True
            unmatched.append(f"Finding: {condition.value}")
            return False
        
        elif condition.type == "history":
            # Check if history item is present
            history_lower = [h.lower() for h in facts.history]
            for history in history_lower:
                if value_lower in history or history in value_lower:
                    matched.append(f"History: {condition.value}")
                    return True
            unmatched.append(f"History: {condition.value}")
            return False
        
        elif condition.type == "age":
            # Age conditions embedded in text - extract and check age
            import re
            age_match = re.search(r"(\d+)", condition.value)
            if age_match and age is not None:
                required_age = int(age_match.group(1))
                if "under" in condition.value or "below" in condition.value:
                    if age < required_age:
                        matched.append(f"Age {age} < {required_age}")
                        return True
                    unmatched.append(f"Age {age} >= {required_age}")
                    return False
                else:  # "and over", "or over", etc.
                    if age >= required_age:
                        matched.append(f"Age {age} >= {required_age}")
                        return True
                    unmatched.append(f"Age {age} < {required_age}")
                    return False
            elif age is None:
                unmatched.append(f"Age required: {condition.value}")
                return False
        
        return True
    
    def _check_count(
        self,
        condition: CountCondition,
        facts: ExtractedFacts,
        normalized_symptoms: list[str],
        matched: list[str],
        unmatched: list[str],
        age: int | None = None,
    ) -> bool:
        """Check a count condition (N or more of the following)."""
        count = 0
        matched_options = []
        
        for option in condition.options:
            # Create temporary lists for this option
            temp_matched = []
            temp_unmatched = []
            
            if self._check_atomic(option, facts, normalized_symptoms, temp_matched, temp_unmatched, age):
                count += 1
                matched_options.append(option.value)
        
        threshold = condition.threshold
        if count >= threshold:
            matched.append(f"Count {count}/{threshold} met: {', '.join(matched_options)}")
            return True
        else:
            unmatched.append(f"Count {count}/{threshold} not met (need {threshold} of: {', '.join(o.value for o in condition.options[:3])}...)")
            return False
    
    def _check_composite(
        self,
        condition: CompositeCondition,
        facts: ExtractedFacts,
        normalized_symptoms: list[str],
        matched: list[str],
        unmatched: list[str],
        age: int | None = None,
    ) -> bool:
        """Check a composite condition (AND/OR)."""
        if condition.type == "or":
            # OR: at least one child must match
            for child in condition.children:
                temp_matched = []
                temp_unmatched = []
                if self._check_conditions(child, facts, normalized_symptoms, temp_matched, temp_unmatched, age):
                    matched.extend(temp_matched)
                    return True
            
            # None matched - add first unmatched as example
            if condition.children:
                first_child = condition.children[0]
                if isinstance(first_child, AtomicCondition):
                    unmatched.append(f"None of OR conditions met (e.g., {first_child.value})")
            return False
        
        elif condition.type == "and":
            # AND: all children must match
            all_matched = True
            for child in condition.children:
                if not self._check_conditions(child, facts, normalized_symptoms, matched, unmatched, age):
                    all_matched = False
            return all_matched
        
        return True
    
    def get_near_matches(
        self,
        facts: ExtractedFacts,
        max_results: int = 5,
    ) -> list[MatchResult]:
        """
        Get rules that nearly match (partial matches).
        
        Useful for suggesting what additional info would trigger a referral.
        """
        all_results = self.match(facts)
        
        # Filter to partial matches only
        partial = [r for r in all_results if r.match_type == "partial" and r.confidence > 0.3]
        
        return partial[:max_results]
    
    def get_rules_by_symptom(self, symptom: str) -> list[NG12Rule]:
        """Get all rules that reference a specific symptom."""
        symptom_lower = symptom.lower()
        normalized = self.terms.normalize_symptom(symptom_lower)
        
        rules = set()
        rules.update(self.symptom_to_rules.get(normalized, []))
        rules.update(self.symptom_to_rules.get(symptom_lower, []))
        
        return list(rules)
    
    def get_rules_by_cancer_site(self, cancer_site: str) -> list[NG12Rule]:
        """Get all rules for a specific cancer site."""
        return self.cancer_site_to_rules.get(cancer_site.lower(), [])


# Singleton instance
_matcher_instance: RuleMatcher | None = None


def get_rule_matcher() -> RuleMatcher:
    """Get the singleton rule matcher instance."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = RuleMatcher()
    return _matcher_instance
