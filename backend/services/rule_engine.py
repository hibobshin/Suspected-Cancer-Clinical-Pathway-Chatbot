"""
Rule Engine for NG12 Guideline.

Main orchestrator that replaces the LangGraph pipeline.
Provides deterministic rule matching with LLM only for:
1. Fact extraction (from natural language)
2. Response formatting (presenting matched rules)

All rule matching decisions are deterministic - NO LLM involved.
"""

from datetime import datetime, timedelta
from typing import Literal

from openai import AsyncOpenAI

from config.config import get_settings
from config.logging_config import get_logger
from models.rule_models import (
    Artifact,
    ConversationState,
    ConversationTurn,
    ExtractedFacts,
    IntakeField,
    IntakeRequest,
    MatchResult,
    RuleEngineResponse,
)
from services.fact_extractor import get_fact_extractor, FactExtractor
from services.rule_matcher import get_rule_matcher, RuleMatcher
from services.terms_index import get_terms_index, TermsIndex

logger = get_logger(__name__)


class SafetyGate:
    """Deterministic fail-closed safety checks - runs BEFORE any processing."""
    
    BLOCKED_INTENTS = {
        "diagnosis": [
            "does patient have cancer",
            "does this patient have cancer",
            "is it cancer",
            "diagnose",
            "confirm cancer",
            "rule out cancer",
            "do they have cancer",
            "cancer diagnosis",
        ],
        "prognosis": [
            "going to be okay",
            "will they survive",
            "survival rate",
            "how long",
            "prognosis",
            "life expectancy",
            "will they die",
        ],
        "treatment": [
            "prescribe",
            "treatment",
            "what medication",
            "chemotherapy",
            "radiotherapy",
            "surgery",
            "how to treat",
        ],
    }
    
    FAIL_CLOSED_RESPONSES = {
        "diagnosis": (
            "This tool assists with recognition and referral pathways based on NICE NG12. "
            "It cannot make clinical diagnoses. Please provide patient symptoms and demographics "
            "to identify appropriate referral criteria."
        ),
        "prognosis": (
            "This tool provides referral guidance only and cannot assess clinical outcomes. "
            "Prognosis discussions should occur with the treating clinical team."
        ),
        "treatment": (
            "NICE NG12 covers recognition and referral for suspected cancer. "
            "Treatment decisions require consultation of appropriate treatment guidelines "
            "and specialist input."
        ),
        "scope": (
            "This query falls outside the scope of NICE NG12 suspected cancer "
            "recognition and referral guidance."
        ),
    }
    
    def check(self, query: str) -> tuple[bool, str | None]:
        """
        Check if query is safe to process.
        
        Returns:
            (is_safe, fail_response) - If not safe, returns fail response text
        """
        query_lower = query.lower()
        
        for intent, patterns in self.BLOCKED_INTENTS.items():
            if any(p in query_lower for p in patterns):
                logger.warning("Safety gate blocked query", intent=intent, query=query[:100])
                return False, self.FAIL_CLOSED_RESPONSES[intent]
        
        return True, None


class ConversationMemory:
    """Manage conversation state and fact accumulation."""
    
    SESSION_TIMEOUT = timedelta(hours=1)
    
    def __init__(self):
        self._sessions: dict[str, ConversationState] = {}
    
    def get_or_create(self, conversation_id: str) -> ConversationState:
        """Get existing session or create new one."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        return self._sessions[conversation_id]
    
    def merge_facts(
        self,
        state: ConversationState,
        new_facts: ExtractedFacts,
    ) -> ExtractedFacts:
        """
        Merge new facts into accumulated state.

        Rules:
        - If new query appears to be about a DIFFERENT patient, reset state
        - New explicit values override old values
        - Lists (symptoms, findings) are unioned only for same patient
        - None values don't override existing values
        """
        accumulated = state.accumulated_facts
        
        # Detect if this is a new patient (different from accumulated state)
        if self._is_new_patient(accumulated, new_facts):
            logger.info(
                "Detected new patient, resetting conversation state",
                old_age=accumulated.age,
                new_age=new_facts.age,
                old_symptoms=accumulated.symptoms[:2],
                new_symptoms=new_facts.symptoms[:2],
            )
            # Reset accumulated facts to new facts
            state.accumulated_facts = ExtractedFacts(
                age=new_facts.age,
                age_term=new_facts.age_term,
                gender=new_facts.gender,
                symptoms=new_facts.symptoms,
                symptoms_raw=new_facts.symptoms_raw,
                findings=new_facts.findings,
                history=new_facts.history,
                raw_query=new_facts.raw_query,
            )
            return state.accumulated_facts

        # Same patient - merge facts
        # Age: new overrides if provided
        if new_facts.age is not None:
            accumulated.age = new_facts.age

        # Age term: new overrides if provided
        if new_facts.age_term is not None:
            accumulated.age_term = new_facts.age_term

        # Gender: new overrides if provided
        if new_facts.gender is not None:
            accumulated.gender = new_facts.gender

        # Symptoms: union (deduplicated)
        accumulated.symptoms = list(set(accumulated.symptoms + new_facts.symptoms))

        # Symptoms raw: union
        accumulated.symptoms_raw = list(set(accumulated.symptoms_raw + new_facts.symptoms_raw))
        
        # Findings: union
        accumulated.findings = list(set(accumulated.findings + new_facts.findings))
        
        # History: union
        accumulated.history = list(set(accumulated.history + new_facts.history))
        
        # Update raw_query
        if accumulated.raw_query:
            accumulated.raw_query = f"{accumulated.raw_query} | {new_facts.raw_query}"
        else:
            accumulated.raw_query = new_facts.raw_query
        
        state.updated_at = datetime.now()
        return accumulated
    
    def _is_new_patient(
        self,
        accumulated: ExtractedFacts,
        new_facts: ExtractedFacts,
    ) -> bool:
        """
        Detect if the new query is about a different patient.
        
        Indicators of a new patient:
        1. Different age (and both are specified)
        2. Completely different symptoms with no overlap
        3. Query contains "new patient", "another patient", etc.
        """
        # If accumulated has no facts yet, not a new patient scenario
        if not accumulated.age and not accumulated.symptoms:
            return False
        
        # If new facts have no age or symptoms, can't determine - assume same patient
        if not new_facts.age and not new_facts.symptoms:
            return False
        
        # Different age is a strong signal of new patient
        if accumulated.age and new_facts.age:
            if abs(accumulated.age - new_facts.age) >= 3:  # Allow small discrepancies
                return True
        
        # Different symptoms with no overlap is a signal
        if accumulated.symptoms and new_facts.symptoms:
            old_symptoms = set(s.lower() for s in accumulated.symptoms)
            new_symptoms = set(s.lower() for s in new_facts.symptoms)
            
            # If no symptom overlap, likely a new patient
            if not old_symptoms.intersection(new_symptoms):
                return True
        
        # Check for explicit "new patient" language in raw query
        if new_facts.raw_query:
            query_lower = new_facts.raw_query.lower()
            new_patient_phrases = [
                "new patient", "another patient", "different patient",
                "next patient", "second patient", "other patient",
            ]
            if any(phrase in query_lower for phrase in new_patient_phrases):
                return True
        
        return False
    
    def add_turn(
        self,
        state: ConversationState,
        user_message: str,
        extracted_facts: ExtractedFacts,
        matches: list[MatchResult],
        response: str,
        response_type: Literal["answer", "clarification", "intake_form", "fail_closed"],
    ) -> None:
        """Record a conversation turn for audit trail."""
        turn = ConversationTurn(
            turn_id=len(state.turns) + 1,
            timestamp=datetime.now(),
            user_message=user_message,
            extracted_facts=extracted_facts,
            matches=matches,
            response=response,
            response_type=response_type,
        )
        state.turns.append(turn)
        state.previous_matches = matches
    
    def clear_session(self, conversation_id: str) -> None:
        """Clear a conversation session."""
        if conversation_id in self._sessions:
            del self._sessions[conversation_id]
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        now = datetime.now()
        expired = [
            cid for cid, state in self._sessions.items()
            if now - state.updated_at > self.SESSION_TIMEOUT
        ]
        for cid in expired:
            del self._sessions[cid]
        return len(expired)


class ResponseGenerator:
    """Format matched rules into human-readable response using LLM."""
    
    RESPONSE_PROMPT = """You are presenting NICE NG12 guideline matches to a healthcare professional.

MATCHED RULES (these are the ONLY rules that apply):
{matched_rules}

PATIENT FACTS (TRUST THESE - do NOT infer or guess different facts):
{facts}

FORMAT your response as plain text (no markdown headers):
**Outcome:** [1-2 sentences stating the recommendation with citation in format [rule_id]]
**Why this applies:** [explain why the patient matches the rule criteria]
**Evidence:** [verbatim quote from rule]
**Next steps:** [action to take based on the recommendation]

CRITICAL RULES:
- TRUST the patient facts provided above - do NOT guess or infer different demographics
- You CANNOT add rules that weren't matched
- You CANNOT remove or ignore matched rules  
- Every claim must cite [rule_id] (e.g., [1.1.1])
- Use clinical language, not conversational
- No diagnosis or prognosis statements
- End with: *This tool supports recognition and referral based on NICE NG12. Clinical decisions rest with the treating clinician.*

Query: {query}"""

    NO_MATCH_TEMPLATE = """Based on the information provided ({facts_summary}), no urgent suspected cancer referral pathway is currently indicated under NICE NG12.

{conditional_pathways}

**Safety Netting (per NG12 section 1.15):**
- Advise the patient to return if symptoms persist or worsen
- Provide a specific timeframe for review
- Document the safety-netting discussion

*This tool supports recognition and referral based on NICE NG12. Clinical decisions rest with the treating clinician.*"""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
    
    async def generate(
        self,
        query: str,
        facts: ExtractedFacts,
        matches: list[MatchResult],
    ) -> str:
        """Generate response for matched rules."""
        if not matches or all(m.match_type != "full" for m in matches):
            return self._no_match_response(facts, matches)
        
        # Filter to full matches only for recommendation
        full_matches = [m for m in matches if m.match_type == "full"]
        
        rules_text = self._format_matched_rules(full_matches)
        facts_text = self._format_facts(facts)
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                    "role": "user",
                    "content": self.RESPONSE_PROMPT.format(
                        matched_rules=rules_text,
                        facts=facts_text,
                        query=query,
                    )
                }],
                temperature=0.3,
                max_tokens=600,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error("Failed to generate response", error=str(e))
            # Fallback: structured response without LLM
            return self._fallback_response(full_matches, facts)
    
    def _format_matched_rules(self, matches: list[MatchResult]) -> str:
        """Format matched rules for prompt."""
        lines = []
        for m in matches[:5]:  # Limit to top 5
            lines.append(f"Rule {m.rule.rule_id} ({m.rule.cancer_site}):")
            lines.append(f"  Action: {m.rule.action_text}")
            lines.append(f"  Matched conditions: {', '.join(m.matched_conditions)}")
            lines.append(f"  Verbatim: {m.rule.verbatim_text[:300]}...")
            lines.append("")
        return "\n".join(lines)
    
    def _format_facts(self, facts: ExtractedFacts) -> str:
        """Format facts for prompt."""
        parts = []
        if facts.age:
            parts.append(f"Age: {facts.age}")
        if facts.gender:
            parts.append(f"Gender: {facts.gender}")
        if facts.symptoms:
            parts.append(f"Symptoms: {', '.join(facts.symptoms)}")
        if facts.findings:
            parts.append(f"Findings: {', '.join(facts.findings)}")
        if facts.history:
            parts.append(f"History: {', '.join(facts.history)}")
        return "\n".join(parts) if parts else "No structured facts extracted"
    
    def _no_match_response(self, facts: ExtractedFacts, partial_matches: list[MatchResult]) -> str:
        """Generate response when no full matches found."""
        # Summarize facts
        fact_parts = []
        if facts.age:
            fact_parts.append(f"{facts.age}-year-old")
        if facts.symptoms:
            fact_parts.append(f"with {', '.join(facts.symptoms)}")
        facts_summary = " ".join(fact_parts) if fact_parts else "the provided information"
        
        # Generate conditional pathways from partial matches
        conditional = ""
        if partial_matches:
            conditional = "**Conditional Pathways:**\n"
            for m in partial_matches[:3]:
                if m.unmatched_conditions:
                    missing = m.unmatched_conditions[0]
                    conditional += f"- If {missing} → {m.rule.action_text} [{m.rule.rule_id}]\n"
        
        return self.NO_MATCH_TEMPLATE.format(
            facts_summary=facts_summary,
            conditional_pathways=conditional,
        )
    
    def _fallback_response(self, matches: list[MatchResult], facts: ExtractedFacts) -> str:
        """Fallback structured response without LLM."""
        lines = ["Based on the provided information and NICE NG12 criteria:\n"]
        
        for m in matches[:3]:
            lines.append(f"**{m.rule.action.value.replace('_', ' ').title()}** [{m.rule.rule_id}]")
            lines.append(f"Cancer site: {m.rule.cancer_site}")
            lines.append(f"Matched: {', '.join(m.matched_conditions)}")
            lines.append("")
        
        lines.append("*This tool supports recognition and referral based on NICE NG12. Clinical decisions rest with the treating clinician.*")
        
        return "\n".join(lines)


class RuleEngine:
    """
    Main orchestrator for the Document-Native Rule Engine.
    
    Replaces the LangGraph pipeline with a simpler, more deterministic flow:
    1. Safety gate (deterministic)
    2. Fact extraction (LLM)
    3. Rule matching (deterministic)
    4. Response generation (LLM)
    """
    
    def __init__(self):
        self.safety_gate = SafetyGate()
        self.memory = ConversationMemory()
        self.extractor = get_fact_extractor()
        self.matcher = get_rule_matcher()
        self.terms = get_terms_index()
        self.generator = ResponseGenerator()
        
        logger.info("Rule engine initialized")
    
    async def process(
        self,
        query: str,
        conversation_id: str | None = None,
    ) -> RuleEngineResponse:
        """
        Process a query through the rule engine.
        
        Args:
            query: User's natural language query
            conversation_id: Optional conversation ID for memory
            
        Returns:
            RuleEngineResponse with response, facts, matches, and artifacts
        """
        # Get or create conversation state
        state = None
        if conversation_id:
            state = self.memory.get_or_create(conversation_id)
        
        # Phase 0: Safety gate (ALWAYS first)
        is_safe, fail_response = self.safety_gate.check(query)
        if not is_safe:
            response = RuleEngineResponse(
                response=fail_response,
                response_type="fail_closed",
                facts=ExtractedFacts(raw_query=query),
                matches=[],
                artifacts=[],
                conversation_id=conversation_id,
            )
            if state:
                self.memory.add_turn(state, query, ExtractedFacts(raw_query=query), [], fail_response, "fail_closed")
            return response
        
        # Phase 1: Extract facts (LLM)
        current_facts = await self.extractor.extract(query)
        
        # Phase 1.5: Merge with accumulated facts
        if state:
            merged_facts = self.memory.merge_facts(state, current_facts)
        else:
            merged_facts = current_facts
        
        # Check for age clarification needed
        age_clarification = self.terms.get_clarification_for_age(query)
        if age_clarification and merged_facts.age is None:
            response = RuleEngineResponse(
                response=age_clarification,
                response_type="clarification",
                facts=merged_facts,
                matches=[],
                artifacts=[],
                conversation_id=conversation_id,
            )
            if state:
                self.memory.add_turn(state, query, current_facts, [], age_clarification, "clarification")
            return response
        
        # Phase 2: Match rules (deterministic - NO LLM)
        matches = self.matcher.match(merged_facts)
        
        logger.info(
            "Rule matching complete",
            query=query[:50],
            facts_age=merged_facts.age,
            facts_symptoms=merged_facts.symptoms,
            match_count=len(matches),
            full_matches=sum(1 for m in matches if m.match_type == "full"),
        )
        
        # Phase 3: Determine response type and generate response
        full_matches = [m for m in matches if m.match_type == "full"]
        
        if full_matches:
            # Full matches - provide recommendation
            response_text = await self.generator.generate(query, merged_facts, matches)
            response_type = "answer"
        elif self._needs_intake(merged_facts, matches):
            # Need more info
            intake = self._generate_intake_request(merged_facts, matches, state)
            if isinstance(intake, IntakeRequest):
                response_text = intake
                response_type = "intake_form"
            else:
                response_text = intake
                response_type = "clarification"
        else:
            # Partial matches or no match - provide guidance
            response_text = await self.generator.generate(query, merged_facts, matches)
            response_type = "answer"
        
        # Build artifacts for citation
        artifacts = self._build_artifacts(matches)
        
        # Record turn
        if state:
            response_str = response_text if isinstance(response_text, str) else str(response_text)
            self.memory.add_turn(state, query, current_facts, matches, response_str, response_type)
        
        return RuleEngineResponse(
            response=response_text,
            response_type=response_type,
            facts=merged_facts,
            matches=matches,
            artifacts=artifacts,
            conversation_id=conversation_id,
        )
    
    def _needs_intake(self, facts: ExtractedFacts, matches: list[MatchResult]) -> bool:
        """Determine if structured intake is needed."""
        # Need intake if no symptoms/findings at all
        if not facts.symptoms and not facts.findings:
            return True
        
        # Need intake if age is missing and rules require it
        if facts.age is None:
            for m in matches:
                if m.rule.age_constraint is not None:
                    return True
        
        return False
    
    def _generate_intake_request(
        self,
        facts: ExtractedFacts,
        matches: list[MatchResult],
        state: ConversationState | None,
    ) -> IntakeRequest | str:
        """Generate intake request for missing information."""
        fields = []
        
        # Check if age is needed
        if facts.age is None:
            fields.append(IntakeField(
                id="age",
                label="Patient Age",
                field_type="number",
                required=True,
                context="Age thresholds apply to multiple referral pathways (40+, 55+, 60+)",
            ))
        
        # Check if symptoms are needed
        if not facts.symptoms:
            fields.append(IntakeField(
                id="symptoms",
                label="Presenting Symptoms",
                field_type="multiselect",
                required=True,
                options=[
                    "Unexplained weight loss",
                    "Fatigue",
                    "Cough",
                    "Haemoptysis (coughing up blood)",
                    "Dysphagia (difficulty swallowing)",
                    "Abdominal pain",
                    "Rectal bleeding",
                    "Visible haematuria (blood in urine)",
                    "Lymphadenopathy (swollen lymph nodes)",
                    "Breast lump",
                ],
                context="Select all symptoms the patient is presenting with",
            ))
        
        # Filter out already-asked fields
        if state:
            fields = [f for f in fields if f.id not in state.asked_fields]
            state.asked_fields.update(f.id for f in fields)
        
        if not fields:
            return (
                "I've gathered all available information but cannot identify a matching "
                "referral pathway. Please consult NG12 directly or seek specialist advice."
            )
        
        if len(fields) == 1:
            # Single field - inline clarification
            return f"To complete the assessment, please provide: **{fields[0].label}** ({fields[0].context})"
        
        # Multiple fields - structured form
        return IntakeRequest(
            type="intake_required",
            reason="Additional information needed to identify applicable referral criteria",
            fields=fields,
            partial_assessment=(
                f"Based on the current information ({self._summarize_facts(facts)}), "
                "additional details are needed to determine if referral criteria are met."
            ),
        )
    
    def _summarize_facts(self, facts: ExtractedFacts) -> str:
        """Create a brief summary of facts."""
        parts = []
        if facts.age:
            parts.append(f"{facts.age}-year-old")
        if facts.gender:
            parts.append(facts.gender)
        if facts.symptoms:
            parts.append(f"with {', '.join(facts.symptoms[:3])}")
        return " ".join(parts) if parts else "limited information"
    
    def _build_artifacts(self, matches: list[MatchResult]) -> list[Artifact]:
        """Build artifacts for citation and highlighting."""
        artifacts = []
        
        for m in matches[:5]:  # Top 5 matches
            artifacts.append(Artifact(
                section=m.rule.section_path,
                text=m.rule.verbatim_text[:500],
                rule_id=m.rule.rule_id,
                relevance_score=m.confidence,
                char_start=m.rule.char_start,
                char_end=m.rule.char_end,
            ))
        
        return artifacts


# Singleton instance
_engine_instance: RuleEngine | None = None


def get_rule_engine() -> RuleEngine:
    """Get the singleton rule engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RuleEngine()
    return _engine_instance
