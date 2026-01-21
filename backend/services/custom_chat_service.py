"""
Custom chat service for the NG12 Assistant Pipeline.

This service uses document-centric section retrieval:
1. Safety Gate (deterministic keyword matching)
2. Section Retrieval (hybrid BM25 + semantic search)
3. Response Formatting (LLM with verbatim citations)
4. Deterministic pathway button logic (based on has_criteria)

The LLM's only role is formatting responses - all retrieval and
pathway decisions are deterministic.
"""

import json
import re
import time
from collections.abc import AsyncGenerator
from typing import Any, Optional
from uuid import UUID, uuid4

from openai import AsyncOpenAI

from config.config import Settings, get_settings
from config.custom_config import CustomPipelineSettings, get_custom_settings
from config.logging_config import get_logger
from models.custom_models import (
    CaseFields,
    Citation as CustomCitation,
    ConfidenceFactors,
    ConfidenceScore,
    IntentClassification,
    IntentType,
    IntakeOption,
    IntakeQuestion,
    IntakeResult,
    MetadataQuality,
    RetrievalResult as CustomRetrievalResult,
    SafetyGateResult,
    SafetyCheck,
    StructuredResponse,
    VerbatimEvidence,
)
from services.section_retriever import get_section_retriever, SectionRetriever, RetrievalResult
from models.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationContext,
    MessageRole,
    PathwayRouteType,
    ResponseType,
    Citation,
    Artifact,
    PathwaySpec,
)

logger = get_logger(__name__)


class CustomChatService:
    """
    Service for processing chat messages with document-centric section retrieval.
    
    Uses hybrid BM25 + semantic search to find relevant NG12 sections,
    then LLM for response formatting with verbatim citations.
    Pathway button is deterministically shown when sections have criteria.
    """
    
    # System prompt for LLM response formatting
    SYSTEM_PROMPT = """You are NG12, the NICE guideline for suspected cancer recognition and referral.

YOUR ROLE:
- Provide authoritative, direct guidance based ONLY on the provided context
- Your audience is healthcare professionals who need quick, accurate answers

FIRST: IDENTIFY THE QUERY TYPE

**Definitional/Conceptual Questions** (e.g., "What is X?", "What does Y mean?", "Explain Z"):
- Provide a clear, concise definition or explanation
- Use plain language, not the clinical criteria format
- Keep it brief (2-4 sentences)
- Example: "A suspected cancer pathway referral is an urgent referral route..."

**Clinical Patient Queries** (e.g., "50 year old with thrombocytosis", specific symptoms + age):
- Use the structured format below with Criteria and Action for each recommendation

STRUCTURED FORMAT (for clinical queries only):
1. Opening: "NG12 recommends the following for [symptom/topic]:"

2. FOR EACH NUMBERED RECOMMENDATION (X.X.X) IN THE CONTEXT:
   ONLY include it if the patient's symptom is EXPLICITLY listed in that recommendation's criteria.
   
   MANDATORY FORMAT for each recommendation (use this EXACT structure):
   
   For assessment of [cancer type] NG12 X.X.X
   
   * Criteria: [age and symptom requirements - copy EXACTLY from the recommendation]
   * Action: [what to do]
   
   CRITICAL FORMATTING RULES:
   - Put "For assessment of [cancer type] NG12 X.X.X" on ONE line
   - The cancer type MUST come before the NG12 reference
   - Use descriptive cancer types: "suspected colorectal cancer", "oesophageal cancer", "stomach cancer", "pancreatic cancer", "lung cancer", "mesothelioma", etc.
   
   CORRECT Examples:
   - "For assessment of suspected colorectal cancer NG12 1.3.1"
   - "For assessment of oesophageal cancer NG12 1.2.1"
   - "For assessment of stomach cancer NG12 1.2.7"
   - "For assessment of pancreatic cancer NG12 1.2.5"
   
   WRONG Examples (do NOT use):
   - "1.3.1 For assessment of..." (ID comes first)
   - "NG12 1.3.1 For assessment of..." (NG12 comes before cancer type)
   
   If the symptom is NOT in the criteria list, skip that recommendation entirely.

3. Summary: List all applicable pathways

4. PATHWAY CRITERIA (for interactive pathway tool):
   After the footer, add a structured section with this exact format:
   
   ---PATHWAY_CRITERIA_START---
   Recommendation IDs: [comma-separated list of ALL recommendation IDs you included in your response above, e.g., "1.1.2, 1.1.5"]
   Extracted Symptoms: [comma-separated list of all symptoms/conditions from the query, e.g., "chest pain"]
   ---PATHWAY_CRITERIA_END---
   
   CRITICAL: Include EVERY recommendation ID (X.X.X format) that you mentioned in your response above.
   If you mentioned 1.1.2 and 1.1.5, list BOTH: "1.1.2, 1.1.5"
   This section is used to build the interactive pathway checker and must be included for clinical queries.

RULES FOR CLINICAL QUERIES - CRITICAL VERIFICATION:

Before including ANY recommendation, you MUST verify:
1. Read the recommendation's criteria list carefully
2. Check if the patient's symptom appears EXACTLY in that list
3. If the symptom is NOT in the list, DO NOT include that recommendation

EXAMPLES:
- Patient has "finger clubbing"
  - 1.1.3 lists: "chest infection, finger clubbing, lymphadenopathy, chest signs, thrombocytosis" → INCLUDE (finger clubbing is in list)
  - 1.1.2 lists: "cough, fatigue, shortness of breath, chest pain, weight loss, appetite loss" → DO NOT INCLUDE (finger clubbing NOT in list)
  
- Patient has "chest pain"
  - 1.1.2 lists: "cough, fatigue, shortness of breath, chest pain, weight loss, appetite loss" → INCLUDE (chest pain is in list)
  - 1.1.3 lists: "chest infection, finger clubbing, lymphadenopathy, chest signs, thrombocytosis" → DO NOT INCLUDE (chest pain NOT in list)

- Patient has "vomiting" and "weight loss"
  - 1.2.3 says "nausea or vomiting with any of the following: weight loss, reflux, dyspepsia, upper abdominal pain" → INCLUDE
    (Patient has "vomiting" which matches "nausea or vomiting", AND has "weight loss" which is in "the following" list)

- Patient has "weight loss"
  - 1.1.1 lists: "chest X-ray findings that suggest lung cancer" OR "unexplained haemoptysis" → DO NOT INCLUDE
    (Weight loss is NOT in the criteria list - 1.1.1 is for lung cancer with haemoptysis or chest X-ray findings only)
  - 1.1.2 lists: "cough, fatigue, shortness of breath, chest pain, weight loss, appetite loss" → INCLUDE
    (Weight loss IS explicitly in the criteria list)
  - 1.2.1 lists: "upper abdominal pain, reflux, dyspepsia" (with weight loss) → INCLUDE
    (Weight loss is mentioned as a required symptom alongside the others)
  
CRITICAL: If the patient's symptom is "weight loss", you MUST check if "weight loss" appears in the recommendation's criteria list.
- If YES → Include it
- If NO → Do NOT include it, even if it's for the same cancer type
  
COMPOUND CRITERIA:
- If a recommendation says "X or Y with any of the following: A, B, C", the patient needs:
  * Either X or Y (one of them), AND
  * At least one of A, B, or C
- Example: "nausea or vomiting with any of the following: weight loss, reflux"
  * Patient with "vomiting" + "weight loss" → MEETS criteria
  * Patient with "nausea" + "reflux" → MEETS criteria
  * Patient with only "vomiting" → DOES NOT MEET (missing the "with any of" part)

VERIFICATION CHECKLIST:
For each recommendation you include, ask yourself:
- Is the patient's symptom explicitly listed in that recommendation's criteria?
- If NO → Do NOT include it
- If YES → Include it

NEVER:
- Include recommendations where the symptom is NOT in the criteria list
- Assume a symptom "counts" if it's not explicitly listed
- Use clinical criteria format for definitional questions
- Say "not a standalone symptom" for items in "any of" lists

FOOTER: *Disclaimer: NICE NG12 guidance. Clinical decisions remain with the treating clinician.*

CONTEXT:
{context}

USER QUERY: {query}"""
    
    def __init__(
        self,
        settings: Settings | None = None,
        custom_settings: CustomPipelineSettings | None = None,
    ):
        """
        Initialize the custom chat service.
        
        Args:
            settings: Application settings. Uses default if not provided.
            custom_settings: Custom pipeline settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self.custom_settings = custom_settings or get_custom_settings()
        self._retriever: SectionRetriever | None = None
        self._openai_client: AsyncOpenAI | None = None
        
        logger.info(
            "CustomChatService initialized with section retrieval",
            config_version=self.custom_settings.config_version,
        )
    
    @property
    def retriever(self) -> SectionRetriever:
        """Get or create the section retriever."""
        if self._retriever is None:
            self._retriever = get_section_retriever()
        return self._retriever
    
    @property
    def openai_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client."""
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                # Uses OpenAI's default base URL
            )
        return self._openai_client
    
    async def process_message(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Process a chat message through document-centric section retrieval.
        
        Flow:
        1. Safety gate (deterministic keyword matching)
        2. Section retrieval (hybrid BM25 + semantic search)
        3. LLM response formatting (with verbatim citations)
        4. Deterministic pathway button (if sections have criteria)
        
        Args:
            request: Chat request with message and context.
            
        Returns:
            ChatResponse with the assistant's response and pathway info.
        """
        start_time = time.perf_counter()
        conversation_id = request.conversation_id or uuid4()
        
        logger.info(
            "Processing custom chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
        )
        
        try:
            query = request.message.strip()
            
            # Input validation
            if not query or len(query) < self.custom_settings.min_query_length:
                processing_time = int((time.perf_counter() - start_time) * 1000)
                return ChatResponse(
                    conversation_id=conversation_id,
                    message=f"Query must be at least {self.custom_settings.min_query_length} characters long.",
                    response_type=ResponseType.CLARIFICATION,
                    citations=[],
                    artifacts=[],
                    follow_up_questions=[],
                    processing_time_ms=processing_time,
                )
            
            # 1. Safety gate (deterministic)
            safety_result = self._check_safety_gate(query)
            if not safety_result["passed"]:
                processing_time = int((time.perf_counter() - start_time) * 1000)
                return ChatResponse(
                    conversation_id=conversation_id,
                    message=safety_result["message"],
                    response_type=ResponseType.REFUSAL,
                    citations=[],
                    artifacts=[],
                    follow_up_questions=[],
                    processing_time_ms=processing_time,
                )
            
            # 2. Section retrieval - multi-pass with score-based ranking
            sections = self._retrieve_with_ranking(query)
            
            if not sections:
                processing_time = int((time.perf_counter() - start_time) * 1000)
                return ChatResponse(
                    conversation_id=conversation_id,
                    message="NG12 does not appear to contain information specifically addressing your query. Please try rephrasing or ask about a different topic.",
                    response_type=ResponseType.CLARIFICATION,
                    citations=[],
                    artifacts=[],
                    follow_up_questions=[],
                    processing_time_ms=processing_time,
                )
            
            # 3. LLM response formatting
            response_text = await self._format_response(query, sections)
            
            # 4. Parse pathway criteria from LLM response (before removing it)
            rec_ids_from_response, symptoms_from_response = self._parse_pathway_criteria_from_response(response_text)
            
            # Remove the PATHWAY_CRITERIA section from response (it's metadata, not for display)
            response_text = re.sub(r'---PATHWAY_CRITERIA_START---.*?---PATHWAY_CRITERIA_END---', '', response_text, flags=re.DOTALL).strip()
            
            # 5. Build artifacts from sections
            artifacts = self._build_artifacts(sections)
            
            # 6. Build citations from sections
            citations = self._build_citations(sections)
            
            # 7. Build pathway spec using recommendation IDs from LLM response
            pathway_available, pathway_spec = self._build_pathway_spec_from_ids(sections, rec_ids_from_response)
            
            processing_time = int((time.perf_counter() - start_time) * 1000)
            
            logger.info(
                "Section retrieval processed message",
                conversation_id=str(conversation_id),
                sections_found=len(sections),
                artifacts_count=len(artifacts),
                pathway_available=pathway_available,
                processing_time_ms=processing_time,
            )
            
            return ChatResponse(
                conversation_id=conversation_id,
                message=response_text,
                response_type=ResponseType.ANSWER,
                citations=citations,
                artifacts=artifacts,
                follow_up_questions=[],
                processing_time_ms=processing_time,
                pathway_available=pathway_available,
                pathway_spec=pathway_spec,
            )
            
        except Exception as e:
            logger.exception("Unexpected error in custom chat", error=str(e), conversation_id=str(conversation_id))
            processing_time = int((time.perf_counter() - start_time) * 1000)
            return ChatResponse(
                conversation_id=conversation_id,
                message="I encountered an unexpected error. Please try again.",
                response_type=ResponseType.ERROR,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
    
    def _check_safety_gate(self, query: str) -> dict:
        """
        Deterministic safety gate using keyword matching.
        
        Args:
            query: The user query to check.
            
        Returns:
            Dict with 'passed' bool and 'message' if blocked.
        """
        query_lower = query.lower()
        
        # Check for emergency keywords
        emergency_keywords = self.custom_settings.emergency_keywords
        for keyword in emergency_keywords:
            if keyword.lower() in query_lower:
                return {
                    "passed": False,
                    "message": (
                        "I cannot provide emergency medical advice. "
                        "If this is a medical emergency, please call emergency services immediately."
                    )
                }
        
        # Check for diagnostic requests
        diagnostic_patterns = [
            "diagnose me",
            "what is wrong with me",
            "what disease do i have",
            "what condition do i have",
            "tell me what i have",
        ]
        for pattern in diagnostic_patterns:
            if pattern in query_lower:
                return {
                    "passed": False,
                    "message": (
                        "I cannot provide diagnoses. NG12 provides guidance on recognising symptoms "
                        "that may warrant investigation or referral, but diagnosis requires clinical assessment. "
                        "Please consult a healthcare professional."
                    )
                }
        
        # Check for treatment/prescription requests
        treatment_patterns = [
            "prescribe",
            "what medication should",
            "what treatment should",
            "how to treat",
            "what drug",
        ]
        for pattern in treatment_patterns:
            if pattern in query_lower:
                return {
                    "passed": False,
                    "message": (
                        "I cannot recommend treatments or medications. "
                        "NG12 focuses on cancer recognition and referral pathways, not treatment. "
                        "Treatment decisions should be made by your healthcare team."
                    )
                }
        
        return {"passed": True, "message": None}
    
    def _is_definitional_query(self, query: str) -> bool:
        """
        Check if a query is asking for a definition or explanation rather than clinical guidance.
        
        Definitional queries get fewer retrieved sections to keep responses concise.
        """
        query_lower = query.lower().strip()
        
        # Patterns that indicate definitional/conceptual questions
        definitional_patterns = [
            "what is",
            "what are", 
            "what does",
            "what do",
            "explain",
            "define",
            "meaning of",
            "tell me about",
            "how does",
            "can you explain",
            "what's a",
            "what's the",
        ]
        
        for pattern in definitional_patterns:
            if query_lower.startswith(pattern):
                return True
        
        # Also check for short questions without age/patient info
        # Clinical queries usually mention age or patient
        has_age = bool(re.search(r'\b\d{1,3}\s*(year|yr|yo|y\.?o\.?)\b', query_lower))
        has_patient = "patient" in query_lower
        
        # If it's a question (ends with ?) but no clinical context, likely definitional
        if query.strip().endswith("?") and not has_age and not has_patient:
            if len(query.split()) < 12:  # Short questions
                return True
        
        return False
    
    def _retrieve_with_ranking(self, query: str) -> list[RetrievalResult]:
        """
        Multi-pass retrieval with score-based ranking and reference extraction.
        
        Gets top 5 from each section type, extracts referenced recommendations
        from symptom tables, and merges with score-based ranking.
        """
        # Pass 1: General context sections (symptom tables, overviews)
        context_sections = self.retriever.search(query, top_k=5)
        
        # Pass 2: Sections with actionable criteria (numbered recommendations)
        criteria_sections = self.retriever.search(query, top_k=5, require_criteria=True)
        
        # Pass 3: Extract referenced recommendations from symptom tables
        # E.g., if symptom table says "[1.1.2] [1.1.5]", get both recommendations
        referenced_sections = []
        referenced_ids = set()
        for s in context_sections:
            if s.section_type == "symptom_table":
                # Extract recommendation IDs like [1.1.2] from content
                refs = re.findall(r'\[(\d+\.\d+\.\d+)\]', s.content)
                for ref_id in refs:
                    if ref_id not in referenced_ids:
                        referenced_ids.add(ref_id)
                        # Look up this recommendation
                        for section in self.retriever.sections:
                            if section["id"] == ref_id:
                                referenced_sections.append(
                                    RetrievalResult.from_section(section, score=s.score * 0.95)
                                )
                                break
        
        # Pass 4: Find related recommendations from same cancer site
        related_sections = []
        found_prefixes = set()
        for s in criteria_sections:
            if s.section_id and '.' in s.section_id:
                parts = s.section_id.split('.')
                if len(parts) >= 2:
                    prefix = f"{parts[0]}.{parts[1]}"
                    if prefix not in found_prefixes:
                        found_prefixes.add(prefix)
                        for section in self.retriever.sections:
                            if (section.get("has_criteria") and 
                                section["id"].startswith(prefix) and
                                section["id"] != s.section_id):
                                related_sections.append(
                                    RetrievalResult.from_section(section, score=s.score * 0.9)
                                )
        
        # Merge all sections, deduplicating by ID
        # Priority: criteria > referenced > related > context
        all_sections = {}
        for s in criteria_sections + referenced_sections + related_sections + context_sections:
            if s.section_id not in all_sections:
                all_sections[s.section_id] = s
            else:
                # Keep the higher score
                if s.score > all_sections[s.section_id].score:
                    all_sections[s.section_id] = s
        
        # Sort by score descending
        ranked = sorted(all_sections.values(), key=lambda x: x.score, reverse=True)
        
        if not ranked:
            return []
        
        # Determine cutoff: take top results with reasonable score gap
        MIN_RESULTS = 5
        MAX_RESULTS = 10  # Increased to accommodate related recommendations
        SCORE_TIE_THRESHOLD = 0.15  # Slightly wider threshold
        
        if len(ranked) <= MIN_RESULTS:
            return ranked
        
        cutoff_score = ranked[MIN_RESULTS - 1].score
        
        # Include all results within tie threshold of the cutoff
        results = []
        for s in ranked:
            if len(results) < MIN_RESULTS:
                results.append(s)
            elif s.score >= cutoff_score - SCORE_TIE_THRESHOLD:
                results.append(s)
                if len(results) >= MAX_RESULTS:
                    break
            else:
                break
        
        logger.info(
            "Ranked retrieval",
            context_found=len(context_sections),
            criteria_found=len(criteria_sections),
            referenced_found=len(referenced_sections),
            related_found=len(related_sections),
            unique_sections=len(all_sections),
            final_count=len(results),
            top_score=ranked[0].score if ranked else 0,
            cutoff_score=cutoff_score if ranked else 0
        )
        
        return results
    
    async def _format_response(self, query: str, sections: list[RetrievalResult]) -> str:
        """
        Format a response using LLM with retrieved sections as context.
        
        Args:
            query: The user query.
            sections: Retrieved sections to use as context.
            
        Returns:
            Formatted response string with citations.
        """
        # Build context from sections - prioritize numbered recommendations
        # First, add numbered recommendations (X.X.X format) as they're most actionable
        numbered_sections = []
        other_sections = []
        
        for section in sections:
            section_ref = section.section_id
            if section.section_id[0].isdigit():
                # This is a numbered recommendation - prioritize it
                section_ref = f"NG12 {section.section_id}"
                numbered_sections.append(f"### {section_ref}: {section.header}\n{section.content}")
            else:
                # Other context (definitions, tables, etc.)
                other_sections.append(f"### {section_ref}: {section.header}\n{section.content[:300]}")
        
        # Build context with numbered recommendations first, prominently marked
        context_parts = []
        if numbered_sections:
            context_parts.append("=== NUMBERED RECOMMENDATIONS (YOU MUST ADDRESS EACH ONE) ===")
            context_parts.extend(numbered_sections)
            context_parts.append(f"=== END OF {len(numbered_sections)} NUMBERED RECOMMENDATIONS ===")
        
        if other_sections:
            context_parts.append("\n=== SUPPORTING CONTEXT ===")
            context_parts.extend(other_sections[:3])  # Limit supporting context
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = self.SYSTEM_PROMPT.format(context=context, query=query)
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,  # Lower temperature for factual responses
                max_tokens=1000,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM response formatting failed: {e}")
            # Fallback: return verbatim section content
            fallback_parts = []
            for section in sections[:2]:  # Top 2 sections
                if section.section_id[0].isdigit():
                    fallback_parts.append(f"**NG12 {section.section_id}:** {section.content}")
                else:
                    fallback_parts.append(f"**{section.header}:** {section.content}")
            
            return "\n\n".join(fallback_parts) + "\n\n*Source: NICE NG12. Clinical decisions remain with the treating clinician.*"
    
    def _build_artifacts(self, sections: list[RetrievalResult]) -> list[Artifact]:
        """
        Build artifact list from retrieved sections.
        
        Args:
            sections: Retrieved sections.
            
        Returns:
            List of Artifact objects for the response.
        """
        artifacts = []
        for section in sections:
            artifact = Artifact(
                section=" > ".join(section.header_path) if section.header_path else section.header,
                text=section.content,
                source="NICE NG12",
                source_url="https://www.nice.org.uk/guidance/ng12",
                relevance_score=section.score,
                rule_id=section.section_id if section.section_id[0].isdigit() else None,
                start_line=section.start_line,
                end_line=section.end_line,
            )
            artifacts.append(artifact)
        return artifacts
    
    def _build_citations(self, sections: list[RetrievalResult]) -> list[Citation]:
        """
        Build citation list from retrieved sections.
        
        Args:
            sections: Retrieved sections.
            
        Returns:
            List of Citation objects.
        """
        citations = []
        for section in sections:
            if section.section_id[0].isdigit():  # Only numbered recommendations
                citation = Citation(
                    statement_id=section.section_id,
                    section=" > ".join(section.header_path) if section.header_path else section.header,
                    text=section.content[:200] if section.content else None,
                )
                citations.append(citation)
        return citations
    
    def _parse_pathway_criteria_from_response(self, response_text: str) -> tuple[list[str], set[str]]:
        """
        Parse the PATHWAY_CRITERIA section from LLM response.
        
        Returns:
            Tuple of (recommendation_ids, symptoms)
        """
        
        pattern = r'---PATHWAY_CRITERIA_START---(.*?)---PATHWAY_CRITERIA_END---'
        match = re.search(pattern, response_text, re.DOTALL)
        
        if not match:
            logger.warning("No PATHWAY_CRITERIA section found in LLM response, using fallback extraction")
            # Fallback: extract all recommendation IDs from the response text
            # Look for patterns like "NG12 1.1.2", "1.1.2", or "Recommendation 1.1.2"
            fallback_ids = re.findall(r'(?:NG12|Recommendation|For assessment of[^\n]*)\s*(\d+\.\d+\.\d+)', response_text, re.IGNORECASE)
            if not fallback_ids:
                # Try simpler pattern
                fallback_ids = re.findall(r'\b(\d+\.\d+\.\d+)\b', response_text)
            if fallback_ids:
                unique_ids = list(set(fallback_ids))
                logger.info("Using fallback ID extraction from response text", rec_ids=unique_ids)
                return unique_ids, set()
            return [], set()
        
        criteria_text = match.group(1).strip()
        logger.info("Found PATHWAY_CRITERIA section", content=criteria_text[:200])
        
        rec_ids = []
        symptoms = set()
        
        # Parse recommendation IDs - be flexible with format
        rec_match = re.search(r'Recommendation IDs?:\s*\[?([^\n\]]+)\]?', criteria_text, re.IGNORECASE)
        if rec_match:
            ids_text = rec_match.group(1).strip()
            rec_ids = [r.strip() for r in re.split(r'[,;]', ids_text) if r.strip() and re.match(r'\d+\.\d+\.\d+', r.strip())]
        
        # Parse symptoms
        symptom_match = re.search(r'Extracted Symptoms?:\s*\[?([^\n\]]+)\]?', criteria_text, re.IGNORECASE)
        if symptom_match:
            symptoms = {s.strip().lower() for s in symptom_match.group(1).split(',') if s.strip()}
        
        # Also extract IDs from response text to ensure we get all recommendations
        # (LLM might mention more in the response than in the structured section)
        response_ids = re.findall(r'(?:NG12|Recommendation|For assessment of[^\n]*)\s*(\d+\.\d+\.\d+)', response_text, re.IGNORECASE)
        if not response_ids:
            response_ids = re.findall(r'\b(\d+\.\d+\.\d+)\b', response_text)
        
        # Merge: use IDs from structured section if present, otherwise use all from response
        if rec_ids:
            # Combine both sources and deduplicate
            all_ids = list(set(rec_ids + response_ids))
            logger.info("Merged IDs from PATHWAY_CRITERIA and response text", 
                       structured_ids=rec_ids, 
                       response_ids=response_ids,
                       merged_ids=all_ids)
            rec_ids = all_ids
        elif response_ids:
            rec_ids = list(set(response_ids))
            logger.info("Using IDs extracted from response text", rec_ids=rec_ids)
        
        logger.info("Final parsed pathway criteria", rec_ids=rec_ids, symptoms=list(symptoms))
        return rec_ids, symptoms
    
    async def _extract_symptoms_from_query(self, query: str) -> set[str]:
        """
        Use LLM to extract medical symptoms/conditions from the query.
        
        Returns a set of symptom terms (normalized to lowercase) that can be used for matching.
        """
        try:
            prompt = f"""Extract all medical symptoms, signs, and clinical findings mentioned in this patient query.
Return ONLY a comma-separated list of symptom terms, nothing else.

Query: {query}

Examples:
- "45 years old with chest pain and was a smoker" → chest pain
- "50 year old with thrombocytosis" → thrombocytosis
- "patient has unexplained weight loss and fatigue" → weight loss, fatigue
- "haematuria in a 60 year old woman" → haematuria

Symptoms:"""
            
            response = await self.openai_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            
            symptoms_text = response.choices[0].message.content.strip()
            # Parse comma-separated list
            symptoms = {s.strip().lower() for s in symptoms_text.split(',') if s.strip()}
            
            logger.info("Extracted symptoms from query", query=query[:50], symptoms=list(symptoms))
            return symptoms
            
        except Exception as e:
            logger.warning(f"Failed to extract symptoms with LLM: {e}, falling back to keyword matching")
            # Fallback: simple keyword extraction
            query_lower = query.lower()
            # Common medical terms pattern
            medical_terms = re.findall(r'\b(?:chest pain|weight loss|shortness of breath|haematuria|haemoptysis|thrombocytosis|fatigue|cough|hoarseness|dysphagia|jaundice|vaginal discharge|lymphadenopathy|clubbing|chest infection)\b', query_lower)
            return {term.lower() for term in medical_terms}
    
    def _build_pathway_spec_from_ids(self, sections: list[RetrievalResult], rec_ids: list[str]) -> tuple[bool, Optional[PathwaySpec]]:
        """
        Build pathway spec from sections matching the recommendation IDs provided by LLM.
        
        Args:
            sections: Retrieved sections.
            rec_ids: Recommendation IDs extracted from LLM response (e.g., ["1.1.2", "1.1.5"]).
            
        Returns:
            Tuple of (pathway_available, pathway_spec).
        """
        if not rec_ids:
            return False, None
        
        # Filter sections to only those matching the recommendation IDs
        matching_sections = []
        rec_ids_set = {rid.strip() for rid in rec_ids}
        
        for section in sections:
            if (section.has_criteria and 
                section.criteria_spec and 
                section.section_id in rec_ids_set):
                matching_sections.append(section)
        
        logger.info(
            "Building pathway spec from LLM IDs",
            requested_ids=rec_ids,
            matched_sections=[s.section_id for s in matching_sections],
        )
        
        if not matching_sections:
            return False, None
        
        # Sort by score
        matching_sections.sort(key=lambda s: s.score, reverse=True)
        
        # Aggregate all criteria into a single PathwaySpec
        if len(matching_sections) == 1:
            section = matching_sections[0]
            spec = section.criteria_spec
            pathway_spec = PathwaySpec(
                recommendation_id=spec.get("recommendation_id", section.section_id),
                title=f"NG12 {section.section_id} - Check Patient Criteria",
                verbatim_text=section.content,
                criteria_groups=spec.get("criteria_groups", []),
                action_if_met=spec.get("action", ""),
            )
        else:
            # Multiple recommendations - aggregate criteria and collect ALL symptoms
            all_rec_ids = [s.section_id for s in matching_sections]
            all_criteria_groups = []
            all_verbatim_parts = []
            all_symptoms = set()  # Collect all unique symptoms across all recommendations
            
            for section in matching_sections:
                spec = section.criteria_spec
                if spec:
                    all_criteria_groups.extend(spec.get("criteria_groups", []))
                    all_verbatim_parts.append(f"NG12 {section.section_id}: {section.content[:200]}...")
                    
                    # Extract all symptoms from this section's criteria
                    for group in spec.get("criteria_groups", []):
                        for criterion in group.get("criteria", []):
                            if criterion.get("field") == "symptoms":
                                symptoms = criterion.get("value", [])
                                if isinstance(symptoms, list):
                                    all_symptoms.update(symptoms)
            
            # Create a unified criteria group with all symptoms if we have multiple recommendations
            if all_symptoms:
                # Add a unified symptom criterion that includes all symptoms from all recommendations
                unified_symptom_group = {
                    "operator": "OR",
                    "criteria": [{
                        "field": "symptoms",
                        "operator": "has_any",
                        "value": sorted(list(all_symptoms)),  # Sort for consistency
                        "label": "Any of the following symptoms"
                    }]
                }
                # Prepend the unified group so it appears first
                all_criteria_groups.insert(0, unified_symptom_group)
            
            pathway_spec = PathwaySpec(
                recommendation_id=",".join(all_rec_ids),
                title=f"NG12 {', '.join(all_rec_ids)} - Check Patient Criteria",
                verbatim_text="\n\n".join(all_verbatim_parts),
                criteria_groups=all_criteria_groups,
                action_if_met="See individual recommendations above.",
            )
        
        logger.info(
            "Pathway available",
            rec_ids=pathway_spec.recommendation_id.split(','),
            recommendation_count=len(matching_sections),
            criteria_groups=len(pathway_spec.criteria_groups),
        )
        
        return True, pathway_spec
    
    async def _build_pathway_spec(self, sections: list[RetrievalResult], query: str = "") -> tuple[bool, Optional[PathwaySpec]]:
        """
        Build pathway spec for sections that match the queried symptoms.
        
        Uses LLM to extract symptoms from query, then matches against section criteria.
        Only includes sections whose criteria contain symptoms that appear in the query.
        
        Args:
            sections: Retrieved sections.
            query: The user's query (used to match symptoms).
            
        Returns:
            Tuple of (pathway_available, pathway_spec).
        """
        import re
        import json
        
        # Extract symptoms from query using LLM (with fallback)
        query_symptoms = await self._extract_symptoms_from_query(query)
        
        # If no symptoms extracted, try simple fallback
        if not query_symptoms:
            query_lower = query.lower()
            # Extract common medical phrases
            medical_patterns = [
                r'chest pain', r'weight loss', r'shortness of breath', r'chest infection',
                r'haematuria', r'haemoptysis', r'thrombocytosis', r'fatigue', r'cough',
                r'hoarseness', r'dysphagia', r'jaundice', r'vaginal discharge',
                r'lymphadenopathy', r'finger clubbing', r'appetite loss'
            ]
            for pattern in medical_patterns:
                if re.search(pattern, query_lower):
                    query_symptoms.add(re.search(pattern, query_lower).group().lower())
        
        logger.info(
            "Building pathway spec",
            query_symptoms=list(query_symptoms),
            sections_count=len(sections),
            sections_with_criteria=[s.section_id for s in sections if s.has_criteria],
        )
        
        # Simple approach: include sections with criteria where the symptom is in the content
        matching_sections = []
        for section in sections:
            if not section.has_criteria or not section.criteria_spec:
                continue
            
            # Check if ANY query symptom appears in the section content or criteria spec
            content_lower = section.content.lower()
            criteria_text = ""
            if section.criteria_spec:
                # Extract text from criteria spec for matching
                criteria_text = json.dumps(section.criteria_spec).lower()
            
            has_matching_symptom = False
            for qs in query_symptoms:
                if qs in content_lower or qs in criteria_text:
                    has_matching_symptom = True
                    logger.info(f"Section {section.section_id} matches symptom '{qs}'")
                    break
            
            if has_matching_symptom:
                matching_sections.append(section)
        
        logger.info(
            "Matched sections",
            matched_ids=[s.section_id for s in matching_sections],
        )
        
        # Sort by score and limit to top matching sections
        matching_sections.sort(key=lambda s: s.score, reverse=True)
        # Allow up to 5 recommendations if they all match the symptoms
        matching_sections = matching_sections[:5]
        
        # Note: We no longer add recommendations from symptom tables
        # since we've already done proper symptom-based filtering above
        
        if not matching_sections:
            return False, None
        
        # Aggregate all criteria into a single PathwaySpec
        if len(matching_sections) == 1:
            # Single recommendation - simple case
            section = matching_sections[0]
            spec = section.criteria_spec
            pathway_spec = PathwaySpec(
                recommendation_id=spec.get("recommendation_id", section.section_id),
                title=f"NG12 {section.section_id} - Check Patient Criteria",
                verbatim_text=section.content,
                criteria_groups=spec.get("criteria_groups", []),
                action_if_met=spec.get("action", ""),
            )
        else:
            # Multiple recommendations - aggregate criteria
            all_rec_ids = [s.section_id for s in matching_sections]
            all_criteria_groups = []
            all_verbatim_parts = []
            all_actions = []
            
            # Collect unique symptoms across all criteria groups
            all_symptoms = set()
            has_age = False
            has_smoking = False
            
            for section in matching_sections:
                spec = section.criteria_spec
                all_verbatim_parts.append(f"**NG12 {section.section_id}:**\n{section.content}")
                
                if spec.get("action"):
                    all_actions.append(f"NG12 {section.section_id}: {spec['action']}")
                
                for group in spec.get("criteria_groups", []):
                    for criterion in group.get("criteria", []):
                        field = criterion.get("field")
                        if field == "age":
                            has_age = True
                        elif field == "smoking":
                            has_smoking = True
                        elif field == "symptoms":
                            values = criterion.get("value", [])
                            if isinstance(values, list):
                                all_symptoms.update(values)
                            elif isinstance(values, str):
                                all_symptoms.add(values)
            
            # Build aggregated criteria groups
            aggregated_criteria = []
            if has_age:
                aggregated_criteria.append({
                    "field": "age",
                    "operator": ">=",
                    "value": 40,  # Most common threshold
                    "label": "Patient age"
                })
            if has_smoking:
                aggregated_criteria.append({
                    "field": "smoking",
                    "operator": "==",
                    "value": True,
                    "label": "Smoking history"
                })
            if all_symptoms:
                aggregated_criteria.append({
                    "field": "symptoms",
                    "operator": "any_of",
                    "value": sorted(list(all_symptoms)),
                    "label": "Symptoms present"
                })
            
            if aggregated_criteria:
                all_criteria_groups.append({
                    "operator": "AND",
                    "criteria": aggregated_criteria
                })
            
            pathway_spec = PathwaySpec(
                recommendation_id=",".join(all_rec_ids),
                title=f"Check Criteria for {len(matching_sections)} NG12 Recommendations",
                verbatim_text="\n\n---\n\n".join(all_verbatim_parts),
                criteria_groups=all_criteria_groups,
                action_if_met="\n".join(all_actions) if all_actions else "",
            )
        
        logger.info(
            "Pathway available",
            recommendation_count=len(matching_sections),
            rec_ids=[s.section_id for s in matching_sections],
            criteria_groups=len(pathway_spec.criteria_groups),
        )
        return True, pathway_spec
    
    async def compile_recommendation(
        self,
        recommendation_id: str,
        patient_criteria: dict,
    ) -> dict:
        """
        Compile recommendation(s) with patient criteria.
        
        Supports multiple recommendation IDs (comma-separated) for cases where
        multiple recommendations apply (e.g., lung cancer AND endometrial cancer).
        
        Args:
            recommendation_id: The NG12 recommendation ID(s) (e.g., "1.1.3" or "1.1.3,1.5.12")
            patient_criteria: Dict of patient criteria from the pathway UI
            
        Returns:
            Dict with response, meets_criteria, and artifacts.
        """
        # Handle multiple recommendation IDs
        rec_ids = [r.strip() for r in recommendation_id.split(",")]
        
        if len(rec_ids) == 1:
            # Single recommendation - original behavior
            return await self._compile_single_recommendation(rec_ids[0], patient_criteria)
        
        # Multiple recommendations - check each and aggregate results
        all_responses = []
        all_artifacts = []
        any_meets = False
        all_matched = []
        
        for rec_id in rec_ids:
            result = await self._compile_single_recommendation(rec_id, patient_criteria)
            all_responses.append(result["response"])
            all_artifacts.extend(result.get("artifacts", []))
            if result["meets_criteria"]:
                any_meets = True
            all_matched.append(result["matched_recommendation"])
        
        # Combine responses with clear separation
        combined_response = "\n\n---\n\n".join(all_responses)
        combined_response += "\n\n---\n*Source: NICE NG12. Clinical decisions remain with the treating clinician.*"
        
        return {
            "response": combined_response,
            "meets_criteria": any_meets,
            "matched_recommendation": ", ".join(all_matched),
            "artifacts": all_artifacts,
        }
    
    async def _compile_single_recommendation(
        self,
        recommendation_id: str,
        patient_criteria: dict,
    ) -> dict:
        """
        Compile a single recommendation with patient criteria.
        
        Args:
            recommendation_id: The NG12 recommendation ID (e.g., "1.1.2")
            patient_criteria: Dict of patient criteria from the pathway UI
            
        Returns:
            Dict with response, meets_criteria, and artifacts.
        """
        # Look up the section
        section = self.retriever.get_by_id(recommendation_id)
        
        if not section:
            return {
                "response": f"Recommendation {recommendation_id} not found in NG12.",
                "meets_criteria": False,
                "matched_recommendation": recommendation_id,
                "artifacts": [],
            }
        
        # Check if patient meets criteria
        meets_criteria = self._evaluate_criteria(section.criteria_spec, patient_criteria, section.content)
        
        # Format the compiled recommendation
        response = self._format_compiled_recommendation(
            section=section,
            patient_criteria=patient_criteria,
            meets_criteria=meets_criteria,
        )
        
        # Build artifact
        artifact = Artifact(
            section=" > ".join(section.header_path) if section.header_path else section.header,
            text=section.content,
            source="NICE NG12",
            source_url="https://www.nice.org.uk/guidance/ng12",
            relevance_score=1.0,
            rule_id=section.section_id,
            start_line=section.start_line,
            end_line=section.end_line,
        )
        
        return {
            "response": response,
            "meets_criteria": meets_criteria,
            "matched_recommendation": recommendation_id,
            "artifacts": [artifact],
        }
    
    def _evaluate_criteria(self, criteria_spec: Optional[dict], patient_criteria: dict, section_content: str = "") -> bool:
        """
        Evaluate if patient criteria meet the recommendation criteria.
        
        Note: The NG12 recommendations often have complex OR conditions like:
        "2+ symptoms OR (ever smoked AND 1+ symptom)"
        
        The parser simplifies these, so we use pragmatic evaluation:
        - If patient has age + smoking + any symptom from list = meets criteria
        - If patient has age + 2+ symptoms from list = meets criteria
        
        Args:
            criteria_spec: The pre-parsed criteria from the section.
            patient_criteria: The patient data from the pathway UI.
            section_content: The full section content text (for checking compound patterns).
            
        Returns:
            True if criteria are met, False otherwise.
        """
        if not criteria_spec or not criteria_spec.get("criteria_groups"):
            return True  # No criteria means it applies
        
        for group in criteria_spec.get("criteria_groups", []):
            criteria_list = group.get("criteria", [])
            
            # Extract values
            age_criterion = None
            smoking_criterion = None
            symptom_criterion = None
            
            for criterion in criteria_list:
                if criterion.get("field") == "age":
                    age_criterion = criterion
                elif criterion.get("field") == "smoking":
                    smoking_criterion = criterion
                elif criterion.get("field") == "symptoms":
                    symptom_criterion = criterion
            
            # Get patient data
            patient_age = patient_criteria.get("age")
            patient_smoking = patient_criteria.get("smoking", False)
            patient_symptoms = patient_criteria.get("symptoms", [])
            if isinstance(patient_symptoms, str):
                patient_symptoms = [patient_symptoms]
            
            # Check age requirement
            age_met = True
            if age_criterion:
                age_threshold = age_criterion.get("value", 0)
                age_met = patient_age is not None and patient_age >= age_threshold
            
            if not age_met:
                continue  # Age not met, try next group
            
            # Check symptom requirements
            if symptom_criterion:
                expected_symptoms = symptom_criterion.get("value", [])
                operator = symptom_criterion.get("operator", "has_any")
                
                # Check for compound criteria patterns in the section content
                # Some recommendations require symptoms in combination (e.g., "nausea or vomiting with weight loss")
                content_lower = section_content.lower() if section_content else ""
                has_compound_patterns = any(
                    phrase in content_lower for phrase in [
                        "with any of the following",
                        "with low haemoglobin",
                        "with raised platelet",
                        "nausea or vomiting with"
                    ]
                )
                
                # Count matching symptoms
                matching_symptoms = [s for s in patient_symptoms if s.lower() in [e.lower() for e in expected_symptoms]]
                
                # If compound patterns exist, validate compound criteria
                if has_compound_patterns:
                    # Check for specific compound patterns that must be satisfied
                    # Pattern 1: "nausea or vomiting with any of the following: weight loss, reflux, dyspepsia, upper abdominal pain"
                    has_nausea_or_vomiting = any(s.lower() in ["nausea", "vomiting"] for s in patient_symptoms)
                    has_compound_secondary = any(s.lower() in ["weight loss", "reflux", "dyspepsia", "upper abdominal pain"] for s in patient_symptoms)
                    
                    if has_nausea_or_vomiting and has_compound_secondary:
                        return True
                    
                    # Pattern 2: "raised platelet count with any of the following: nausea, vomiting, weight loss, reflux, dyspepsia, upper abdominal pain"
                    # Note: We don't have "raised platelet count" as a patient symptom field, so skip this for now
                    # (This would require additional patient data fields)
                    
                    # Pattern 3: "upper abdominal pain with low haemoglobin levels"
                    # Note: We don't have "low haemoglobin levels" as a patient symptom field, so skip this
                    
                    # Pattern 4: Standalone "treatment-resistant dyspepsia"
                    if any("treatment-resistant dyspepsia" in s.lower() for s in patient_symptoms):
                        return True
                    
                    # If we have compound patterns but none of the compound conditions are met,
                    # and the patient only has symptoms that appear in compound contexts,
                    # then criteria are NOT met
                    # Symptoms that ONLY appear in compound contexts (not standalone):
                    compound_only_symptoms = ["weight loss", "reflux", "dyspepsia", "upper abdominal pain"]
                    # Check if patient only has symptoms that require compounds
                    if matching_symptoms:
                        # Check if ALL matching symptoms are compound-only (not standalone)
                        all_compound_only = all(
                            s.lower() in compound_only_symptoms 
                            for s in matching_symptoms
                        )
                        if all_compound_only:
                            # These symptoms only appear in compound contexts for this recommendation
                            # Need to check if compound condition is met
                            if not (has_nausea_or_vomiting and has_compound_secondary):
                                # Compound condition not met - criteria NOT satisfied
                                continue  # Try next group (will eventually return False)
                
                # Standard logic for non-compound or when compound conditions are met
                # Pragmatic logic based on NG12 patterns:
                # If ever smoked, only need 1 symptom
                # If not smoked, need 2+ symptoms
                if patient_smoking:
                    if len(matching_symptoms) >= 1:
                        return True
                else:
                    # Check operator for threshold
                    if "2_or_more" in operator:
                        if len(matching_symptoms) >= 2:
                            return True
                    elif len(matching_symptoms) >= 1:
                        return True
            else:
                # No symptom criteria, age alone sufficient
                return True
        
        return False
    
    def _format_compiled_recommendation(
        self,
        section: RetrievalResult,
        patient_criteria: dict,
        meets_criteria: bool,
    ) -> str:
        """
        Format the compiled recommendation with bold labels.
        
        Args:
            section: The matched section.
            patient_criteria: The patient criteria.
            meets_criteria: Whether criteria are met.
            
        Returns:
            Formatted markdown string with bold labels.
        """
        # Format patient criteria as readable string
        criteria_parts = []
        if patient_criteria.get("age"):
            criteria_parts.append(f"Age {patient_criteria['age']}")
        if patient_criteria.get("sex"):
            criteria_parts.append(patient_criteria["sex"])
        if patient_criteria.get("smoking"):
            criteria_parts.append("ever smoked")
        if patient_criteria.get("symptoms"):
            symptoms = patient_criteria["symptoms"]
            if isinstance(symptoms, list):
                criteria_parts.append(f"symptoms: {', '.join(symptoms)}")
            else:
                criteria_parts.append(f"symptom: {symptoms}")
        
        criteria_str = ", ".join(criteria_parts) if criteria_parts else "provided criteria"
        
        # Extract cancer type - prioritize specific cancer names from content (more accurate than broad cancer_site)
        cancer_type = "cancer"
        header_lower = section.header.lower()
        content_lower = section.content.lower()[:400]  # Check first 400 chars for specificity
        
        # Specific cancer type patterns - check content FIRST (most accurate)
        cancer_patterns = [
            (r'pancreatic\s+cancer', 'pancreatic cancer'),
            (r'oesophageal\s+cancer', 'oesophageal cancer'),
            (r'stomach\s+cancer', 'stomach cancer'),
            (r'colorectal\s+cancer', 'colorectal cancer'),
            (r'lung\s+cancer', 'lung cancer'),
            (r'mesothelioma', 'mesothelioma'),
            (r'endometrial\s+cancer', 'endometrial cancer'),
            (r'cervical\s+cancer', 'cervical cancer'),
            (r'ovarian\s+cancer', 'ovarian cancer'),
            (r'prostate\s+cancer', 'prostate cancer'),
            (r'bladder\s+cancer', 'bladder cancer'),
            (r'renal\s+cancer', 'renal cancer'),
            (r'laryngeal\s+cancer', 'laryngeal cancer'),
            (r'oral\s+cancer', 'oral cancer'),
            (r'thyroid\s+cancer', 'thyroid cancer'),
            (r'breast\s+cancer', 'breast cancer'),
        ]
        
        # First, try to extract from content (most specific)
        for pattern, cancer_name in cancer_patterns:
            if re.search(pattern, content_lower):
                cancer_type = cancer_name
                break
        
        # If not found in content, try header
        if cancer_type == "cancer":
            for pattern, cancer_name in cancer_patterns:
                if re.search(pattern, header_lower):
                    cancer_type = cancer_name
                    break
        
        # Last resort: use cancer_site field (but this is too broad, so only as fallback)
        if cancer_type == "cancer" and section.cancer_site:
            cancer_site_map = {
                'lung': 'lung cancer',
                'lower_gi': 'colorectal cancer',
                'breast': 'breast cancer',
                'gynaecological': 'gynaecological cancer',
                'urological': 'urological cancer',
                'skin': 'skin cancer',
                'head_neck': 'head and neck cancer',
                'cns': 'brain and central nervous system cancer',
                'haematological': 'haematological cancer',
                # Note: upper_gi is too broad - don't use it, extract from content instead
            }
            if section.cancer_site in cancer_site_map:
                cancer_type = cancer_site_map[section.cancer_site]
        
        # Get action from criteria spec or header
        action = "See recommendation for appropriate action."
        if section.criteria_spec and section.criteria_spec.get("action"):
            action = section.criteria_spec["action"]
        
        if meets_criteria:
            response = f"""✅ **NG12 Recommendation for {cancer_type}**

**Based on:** {criteria_str}

**Action:** {action}

**Rationale:** Patient meets the criteria specified in NG12 {section.section_id} for assessment of {cancer_type}.

**Source:** NG12 {section.section_id} (Lines {section.start_line}-{section.end_line})

---
*Source: NICE NG12. Clinical decisions remain with the treating clinician.*"""
        else:
            response = f"""⚠️ **NG12 Criteria Not Met for {cancer_type}**

**Based on:** {criteria_str}

**Assessment:** The provided patient criteria do not fully meet the threshold for NG12 {section.section_id} (assessment of {cancer_type}).

**Recommendation:** Review the full criteria in NG12 {section.section_id} and consider whether additional patient information may be relevant.

**Source:** NG12 {section.section_id} (Lines {section.start_line}-{section.end_line})

---
*Source: NICE NG12. Clinical decisions remain with the treating clinician.*"""
        
        return response
    
    # ============================================================
    # LEGACY METHODS BELOW (kept for compatibility)
    # ============================================================
    
    async def _process_with_rule_engine_legacy(
        self,
        query: str,
        conversation_id: UUID,
        start_time: float,
    ) -> ChatResponse:
        """Process query using the old Rule Engine (DEPRECATED)."""
        
        # This method is kept for reference but no longer used
        from services.rule_engine import get_rule_engine
        from models.rule_models import IntakeRequest as RuleIntakeRequest
        
        rule_engine = get_rule_engine()
        result = await rule_engine.process(query, str(conversation_id))
        
        processing_time = int((time.perf_counter() - start_time) * 1000)
        
        # Convert response type
        response_type_map = {
            "answer": ResponseType.ANSWER,
            "clarification": ResponseType.CLARIFICATION,
            "intake_form": ResponseType.CLARIFICATION,
            "fail_closed": ResponseType.CLARIFICATION,
        }
        response_type = response_type_map.get(result.response_type, ResponseType.ANSWER)
        
        # Extract response text
        if isinstance(result.response, RuleIntakeRequest):
            # Format intake request as text
            response_text = result.response.partial_assessment or result.response.reason
            if result.response.fields:
                response_text += "\n\n**Please provide:**\n"
                for field in result.response.fields:
                    response_text += f"- {field.label}: {field.context}\n"
        else:
            response_text = result.response
        
        # Convert artifacts
        artifacts = [
            Artifact(
                section=a.section,
                text=a.text,
                source="NICE NG12",
                source_url="https://www.nice.org.uk/guidance/ng12",
                relevance_score=a.relevance_score,
                chunk_id=a.rule_id,
                rule_id=a.rule_id,
            )
            for a in result.artifacts
        ]
        
        # Parse citations from matched rules
        citations = []
        for match in result.matches[:5]:
            citations.append(Citation(
                statement_id=match.rule.rule_id,
                section=match.rule.section_path,
                text=match.rule.verbatim_text[:200],
            ))
        
        logger.info(
            "Rule engine processed message",
            conversation_id=str(conversation_id),
            response_type=response_type.value,
            query_type=result.query_type,
            matches_count=len(result.matches),
            full_matches=sum(1 for m in result.matches if m.match_type == "full"),
            artifacts_count=len(artifacts),
            processing_time_ms=processing_time,
        )
        
        return ChatResponse(
            conversation_id=conversation_id,
            message=response_text,
            response_type=response_type,
            citations=citations,
            artifacts=artifacts,
            follow_up_questions=[],
            processing_time_ms=processing_time,
            query_type=result.query_type,
        )
    
    async def _process_with_legacy_pipeline(
        self,
        query: str,
        conversation_id: UUID,
        start_time: float,
    ) -> ChatResponse:
        """Process query using the legacy LangGraph pipeline (deprecated)."""
        
        # Validate preprocessor is initialized
        preprocessor = get_document_preprocessor()
        sections = preprocessor.get_sections()
        if not sections:
            processing_time = int((time.perf_counter() - start_time) * 1000)
            logger.error("Preprocessor not initialized", sections_count=len(sections))
            return ChatResponse(
                conversation_id=conversation_id,
                message="The guideline document is not available. Please contact support.",
                response_type=ResponseType.ERROR,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
        
        # Run LangGraph pipeline
        pipeline_state = await self.pipeline.run(query, str(conversation_id))
        
        processing_time = int((time.perf_counter() - start_time) * 1000)
        
        # Check for errors
        if pipeline_state.get("error"):
            error_message = pipeline_state["error"]
            logger.error("Pipeline error", error=error_message, conversation_id=str(conversation_id))
            return ChatResponse(
                conversation_id=conversation_id,
                message=f"I encountered an error: {error_message}. Please try again.",
                response_type=ResponseType.ERROR,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
        
        # Extract response
        response_text = pipeline_state.get("response", "")
        if not response_text:
            logger.warning("Empty response from pipeline", conversation_id=str(conversation_id))
            response_text = "I was unable to generate a response. Please try rephrasing your query."
        
        # Parse citations from response
        citations = self._parse_citations_from_response(response_text, pipeline_state)
        
        # Build artifacts from ranked chunks
        artifacts = self._build_artifacts_from_chunks(pipeline_state)
        
        # Determine response type
        response_type = ResponseType.ANSWER
        if "cannot provide" in response_text.lower() or "do not contain" in response_text.lower():
            response_type = ResponseType.CLARIFICATION
        
        logger.info(
            "Legacy pipeline processed message",
            conversation_id=str(conversation_id),
            response_type=response_type.value,
            citations_count=len(citations),
            artifacts_count=len(artifacts),
            processing_time_ms=processing_time,
        )
        
        return ChatResponse(
            conversation_id=conversation_id,
            message=response_text,
            response_type=response_type,
            citations=citations,
            artifacts=artifacts,
            follow_up_questions=[],
            processing_time_ms=processing_time,
        )
    
    def _parse_citations_from_response(
        self,
        response_text: str,
        pipeline_state: dict[str, Any],
    ) -> list[Citation]:
        """
        Parse citations from response text.
        
        Args:
            response_text: Response text that may contain citations.
            pipeline_state: Pipeline state with ranked chunks.
            
        Returns:
            List of Citation objects.
        """
        citations = []
        
        # Pattern: [subsection_path] or [rule_id: subsection_path]
        citation_pattern = r'\[([^\]]+):?\s*([^\]]*)\]'
        matches = re.finditer(citation_pattern, response_text)
        
        # Get subsection map from preprocessor
        preprocessor = get_document_preprocessor()
        all_subsections = preprocessor.get_all_subsections()
        subsection_map = {sub.subsection_path: sub for sub in all_subsections}
        
        for match in matches:
            if match.lastindex >= 2 and match.group(2):
                # Format: [rule_id: subsection_path]
                rule_id_or_path = match.group(1).strip()
                subsection_path = match.group(2).strip()
                
                if not subsection_path:
                    subsection_path = rule_id_or_path
                    rule_id = None
                else:
                    rule_id = rule_id_or_path
            else:
                # Format: [subsection_path]
                subsection_path = match.group(1).strip()
                rule_id = None
            
            # Find subsection
            subsection = subsection_map.get(subsection_path)
            if subsection:
                citation_text = subsection.content[:200]  # First 200 chars
                citations.append(Citation(
                    statement_id=rule_id or subsection_path,
                    section=subsection_path,
                    text=citation_text,
                ))
        
        logger.info("Citations parsed", count=len(citations))
        return citations
    
    def _build_artifacts_from_chunks(
        self,
        pipeline_state: dict[str, Any],
    ) -> list[Artifact]:
        """
        Build artifacts from ranked chunks.
        
        Args:
            pipeline_state: Pipeline state with ranked chunks.
            
        Returns:
            List of Artifact objects.
        """
        artifacts = []
        
        ranked_chunks = pipeline_state.get("ranked_chunks", [])
        if not ranked_chunks:
            return artifacts
        
        # Get subsection map from preprocessor
        preprocessor = get_document_preprocessor()
        all_subsections = preprocessor.get_all_subsections()
        subsection_map = {sub.subsection_id: sub for sub in all_subsections}
        
        for subsection_id, score in ranked_chunks:
            subsection = subsection_map.get(subsection_id)
            if not subsection:
                continue
            
            artifacts.append(Artifact(
                section=subsection.subsection_path,
                text=subsection.content[:500],  # Truncate for display
                source="NICE NG12",
                source_url="https://www.nice.org.uk/guidance/ng12",
                relevance_score=float(score),
                chunk_id=subsection.subsection_id,
                char_count=len(subsection.content),
                rule_id=None,  # Could extract rule IDs if available
            ))
        
        logger.info("Artifacts built", count=len(artifacts))
        return artifacts
    
    async def process_message_stream(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat message and stream the response using section retrieval.
        
        Args:
            request: Chat request with message and context.
            
        Yields:
            SSE-formatted event strings.
        """
        conversation_id = request.conversation_id or uuid4()
        
        logger.info(
            "Processing custom chat message stream",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
        )
        
        try:
            import json as json_module
            
            # Send start event
            yield f"data: {json_module.dumps({'type': 'start', 'conversation_id': str(conversation_id)})}\n\n"
            
            query = request.message.strip()
            
            # 1. Safety gate
            safety_result = self._check_safety_gate(query)
            if not safety_result["passed"]:
                yield f"data: {json_module.dumps({'type': 'chunk', 'content': safety_result['message']})}\n\n"
                yield f"data: {json_module.dumps({'type': 'done', 'response_type': 'refusal', 'citations': [], 'artifacts': []})}\n\n"
                return
            
            # 2. Section retrieval - multi-pass with score-based ranking
            sections = self._retrieve_with_ranking(query)
            logger.info("Section retrieval", sections_count=len(sections))
            
            if not sections:
                no_results_msg = "NG12 does not appear to contain information specifically addressing your query."
                yield f"data: {json_module.dumps({'type': 'chunk', 'content': no_results_msg})}\n\n"
                yield f"data: {json_module.dumps({'type': 'done', 'response_type': 'clarification', 'citations': [], 'artifacts': []})}\n\n"
                return
            
            # 3. Format response with LLM (streaming)
            response_text = await self._format_response(query, sections)
            
            # 4. Parse pathway criteria from LLM response (before removing it)
            rec_ids_from_response, symptoms_from_response = self._parse_pathway_criteria_from_response(response_text)
            
            # Remove the PATHWAY_CRITERIA section from response (it's metadata, not for display)
            response_text_clean = re.sub(r'---PATHWAY_CRITERIA_START---.*?---PATHWAY_CRITERIA_END---', '', response_text, flags=re.DOTALL).strip()
            
            # Stream response in chunks
            chunk_size = 50
            for i in range(0, len(response_text_clean), chunk_size):
                chunk = response_text_clean[i:i + chunk_size]
                yield f"data: {json_module.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # 5. Build artifacts from sections
            artifacts = self._build_artifacts(sections)
            
            # 6. Build citations from sections
            citations = self._build_citations(sections)
            
            # 7. Build pathway spec using recommendation IDs from LLM response
            pathway_available, pathway_spec = self._build_pathway_spec_from_ids(sections, rec_ids_from_response)
            
            # Send done event
            done_data = {
                'type': 'done',
                'response_type': 'answer',
                'artifacts': [artifact.model_dump(mode='json') for artifact in artifacts],
                'citations': [
                    {
                        'statement_id': c.statement_id,
                        'section': c.section,
                        'text': c.text,
                    }
                    for c in citations
                ],
                'pathway_available': pathway_available,
                'pathway_spec': pathway_spec.model_dump(mode='json') if pathway_spec else None,
                }
            yield f"data: {json_module.dumps(done_data)}\n\n"
            
        except Exception as e:
            logger.exception("Error in custom chat stream", error=str(e))
            import json as json_err
            yield f"data: {json_err.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    # Old methods kept for compatibility but not used in new pipeline
    # These methods are deprecated - the new LangGraph pipeline handles everything
    async def classify_intent(
        self,
        request: ChatRequest,
    ) -> IntentClassification:
        """
        Classify the intent of the user's message.
        
        Uses LLM with structured output to extract intent classification.
        LLM extracts facts; deterministic logic uses them as signals.
        
        Args:
            request: Chat request with message and context.
            
        Returns: 
            IntentClassification with intent type, confidence, and reasoning.
        """
        logger.info("Classifying intent", message_preview=request.message[:100])
        
        # Build prompt for intent classification
        system_prompt = """You are an NG12 Suspected Cancer Recognition & Referral Assistant.
Your purpose is to support clinicians by mapping structured inputs to NICE NG12 guidance, with verbatim citations, traceability, and fail-closed behavior.

Core Operating Principles (Non-negotiable):
- Grounding first: Every recommendation must be explicitly grounded in retrieved NG12 rule chunks. No citation → no recommendation.
- Fail closed by default: If required information is missing, ambiguous, or confidence is low, you must ask targeted follow-up questions or refuse. A refusal with clarity is always safer than a confident guess.
- Rule-level specificity: Apply guidance at the individual recommendation (rule) level, not high-level subsections. Each action must map to a single NG12 rule with clear conditions.
- Deterministic safety controls: LLMs may extract structured facts, but confidence scoring, thresholds, and gating are deterministic and auditable.

Allowed Scope:
You may: Navigate and apply NICE NG12 guidance, identify whether suspected cancer referral or urgent investigation criteria apply, generate structured outputs (recommendations, follow-ups, referral drafts, safety-netting), cite exact NG12 sections verbatim.
You must not: Diagnose cancer, guess missing inputs (age, symptoms, duration, risk factors), provide uncited clinical advice, recommend actions during intake when required fields are missing.

YOUR TASK: Classify the user's query into one of three intent types based on their question pattern. This classification determines the conversation flow.

INTENT TYPES (choose exactly one):

1. guideline_lookup
   - Pattern: General questions about guidelines, symptoms, referral pathways
   - Examples:
     * "What are symptoms of lung cancer?"
     * "What is the referral pathway for suspected colon cancer?"
     * "When should I refer a patient with dysphagia?"
   - Key indicators: General guideline information, symptom lists, pathway descriptions
   - Flow: Proceeds directly to retrieval without structured intake

2. case_triage
   - Pattern: Questions about specific patient cases that require clinical triage
   - Examples:
     * "58-year-old patient with weight loss and heartburn"
     * "45-year-old woman with persistent abdominal pain and change in bowel habit"
     * "Patient aged 60 with haemoptysis and smoking history"
     * "Patient with weight loss and fatigue — what should I do?"
     * "What should I do for a patient with weight loss?"
   - Key indicators: Specific patient demographics (age, sex), presenting symptoms, case details, "what should I do", "patient with X"
   - CRITICAL: If query mentions "patient" + symptoms, classify as case_triage (even if age not mentioned)
   - Flow: REQUIRES structured intake (age, symptoms) before retrieval. Fail-closed if required fields missing.

3. documentation
   - Pattern: Questions about how to document, format, or structure referrals
   - Examples:
     * "How do I document a referral?"
     * "What information should I include in a 2WW referral letter?"
     * "What format is required for urgent cancer pathway referrals?"
   - Key indicators: Documentation processes, referral formatting, administrative procedures
   - Flow: Proceeds directly to retrieval

OUTPUT FORMAT (JSON object):
{
  "intent": "guideline_lookup" | "case_triage" | "documentation",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief 1-2 sentence explanation>"
}

CONFIDENCE SCORING GUIDELINES:
- 0.9-1.0: Clear, unambiguous match with strong indicators
- 0.7-0.89: Good match with most indicators present
- 0.5-0.69: Partial match, some ambiguity
- Below 0.5: Significant ambiguity, prefer lower confidence
- Be conservative: If uncertain, use lower confidence and note ambiguity in reasoning

CRITICAL RULES:
- Classify based on question pattern, not content domain
- If query contains patient-specific details (age, symptoms), prefer "case_triage"
- If query mentions "patient" + symptoms (even without age), prefer "case_triage"
- If query asks "what should I do" or "what action", prefer "case_triage"
- If query asks "how to" document/format, prefer "documentation"
- If query is general information-seeking (no specific patient), prefer "guideline_lookup"
- If truly ambiguous between two types, choose the more specific one and lower confidence
- Example: "Patient with weight loss and fatigue" → case_triage (not guideline_lookup)

This tool supports recognition and referral based on NICE NG12 and does not replace clinical judgement or local pathways."""

        # Build user prompt with context
        if request.context and request.context.messages:
            history = "\n".join([
                f"{msg.role.value}: {msg.content}"
                for msg in request.context.messages[-3:]  # Last 3 messages for context
            ])
            user_prompt = f"""PREVIOUS CONVERSATION:
{history}

CURRENT QUERY:
{request.message}

TASK: Classify the current query into one of the three intent types. Consider the conversation context when determining intent."""
        else:
            user_prompt = f"""QUERY TO CLASSIFY:
{request.message}

TASK: Classify this query into one of the three intent types based on the question pattern and content."""
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.custom_settings.llm_temperature_intent,
                max_tokens=300,  # Increased for more detailed reasoning
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            
            result = json.loads(content)
            
            # Validate and create IntentClassification
            intent_str = result.get("intent", "guideline_lookup")
            try:
                intent = IntentType(intent_str)
            except ValueError:
                logger.warning(
                    "Invalid intent from LLM, defaulting to guideline_lookup",
                    received_intent=intent_str,
                )
                intent = IntentType.GUIDELINE_LOOKUP
            
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            
            reasoning = result.get("reasoning", "No reasoning provided")
            
            classification = IntentClassification(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
            )
            
            logger.info(
                "Intent classified",
                intent=intent.value,
                confidence=confidence,
                reasoning=reasoning,
            )
            
            return classification
            
        except Exception as e:
            logger.exception("Error classifying intent", error=str(e))
            # Fail closed: default to guideline_lookup with low confidence
            return IntentClassification(
                intent=IntentType.GUIDELINE_LOOKUP,
                confidence=0.3,
                reasoning=f"Error during classification: {str(e)}",
            )
    
    async def check_safety_gate(
        self,
        request: ChatRequest,
    ) -> SafetyGateResult:
        """
        Check safety gate with deterministic heuristic rules.
        
        This is DETERMINISTIC logic, not LLM-based. Uses keyword matching
        and pattern matching to detect red flags.
        
        Args:
            request: Chat request with message and context.
            
        Returns:
            SafetyGateResult indicating whether gate passed and intent classification.
        """
        logger.info("Checking safety gate", message_preview=request.message[:100])
        
        # Deterministic safety checks (code-based, not LLM)
        message_lower = request.message.lower()
        red_flags = []
        
        # Check for emergency keywords
        for keyword in self.custom_settings.emergency_keywords:
            if keyword.lower() in message_lower:
                red_flags.append(f"Contains emergency keyword: {keyword}")
        
        # Check for diagnostic requests (pattern matching)
        diagnostic_patterns = [
            "diagnose",
            "what is wrong with",
            "what disease",
            "what condition",
        ]
        for pattern in diagnostic_patterns:
            if pattern in message_lower:
                red_flags.append(f"Diagnostic request detected: {pattern}")
        
        # Check for treatment requests
        treatment_patterns = [
            "prescribe",
            "what medication",
            "what treatment",
            "how to treat",
            "give me",
        ]
        for pattern in treatment_patterns:
            if pattern in message_lower:
                red_flags.append(f"Treatment request detected: {pattern}")
        
        # If red flags found, fail closed
        if red_flags:
            escalation_message = (
                "I cannot provide diagnostic or treatment recommendations. "
                "This is outside the scope of the NICE NG12 guideline assistant. "
                "Please consult with a qualified healthcare professional for clinical decisions."
            )
            
            logger.warning(
                "Safety gate failed",
                red_flags=red_flags,
                message_preview=request.message[:100],
            )
            
            return SafetyGateResult(
                passed=False,
                escalation_message=escalation_message,
                intent_classification=None,
            )
        
        # Safety gate passed - classify intent
        intent_classification = await self.classify_intent(request)
        
        logger.info(
            "Safety gate passed",
            intent=intent_classification.intent.value,
            confidence=intent_classification.confidence,
        )
        
        return SafetyGateResult(
            passed=True,
            escalation_message=None,
            intent_classification=intent_classification,
        )
    
    def _build_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        """
        Build messages list for LLM from request.
        
        Args:
            request: Chat request.
            
        Returns:
            List of message dictionaries.
        """
        messages = []
        
        # System prompt (fallback - used in streaming, should match final response prompt style)
        system_prompt = (
            "You are an NG12 Suspected Cancer Recognition & Referral Assistant. "
            "Your purpose is to support clinicians by mapping structured inputs to NICE NG12 guidance, with verbatim citations, traceability, and fail-closed behavior. "
            "You do not diagnose, do not replace clinical judgement, and do not infer missing information.\n\n"
            "Core Operating Principles: Grounding first (every recommendation must be explicitly grounded in retrieved NG12 rule chunks; no citation → no recommendation), "
            "fail closed by default (if required information is missing, ambiguous, or confidence is low, ask targeted follow-up questions or refuse), "
            "rule-level specificity (apply guidance at the individual recommendation level, not high-level subsections), "
            "deterministic safety controls (LLMs extract structured facts; confidence scoring, thresholds, and gating are deterministic and auditable).\n\n"
            "Provide clear, concise, and STRICTLY evidence-grounded responses based ONLY on provided NICE NG12 guideline evidence. "
            "Always cite sources when referencing guidelines using [rule_id: section_path] format. "
            "If you don't know something or evidence is insufficient, state that clearly rather than guessing. "
            "Do not provide diagnostic or treatment advice—focus on recognition criteria and referral pathways. "
            "This tool supports recognition and referral based on NICE NG12 and does not replace clinical judgement or local pathways."
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if available
        if request.context and request.context.messages:
            for msg in request.context.messages:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        return messages
    
    async def extract_case_fields(
        self,
        request: ChatRequest,
    ) -> CaseFields:
        """
        Extract case fields from conversation using LLM structured extraction.
        
        LLM extracts facts; deterministic validation uses them.
        
        Args:
            request: Chat request with message and context.
            
        Returns:
            CaseFields with extracted fields and ambiguity flags.
        """
        logger.info("Extracting case fields", message_preview=request.message[:100])
        
        # Build prompt for field extraction
        system_prompt = """You are an NG12 Suspected Cancer Recognition & Referral Assistant.
Your purpose is to support clinicians by mapping structured inputs to NICE NG12 guidance, with verbatim citations, traceability, and fail-closed behavior.

Core Operating Principles (Non-negotiable):
- Grounding first: Every recommendation must be explicitly grounded in retrieved NG12 rule chunks. No citation → no recommendation.
- Fail closed by default: If required information is missing, ambiguous, or confidence is low, you must ask targeted follow-up questions or refuse. A refusal with clarity is always safer than a confident guess.
- Rule-level specificity: Apply guidance at the individual recommendation (rule) level, not high-level subsections. Each action must map to a single NG12 rule with clear conditions.
- Deterministic safety controls: LLMs may extract structured facts, but confidence scoring, thresholds, and gating are deterministic and auditable.

Allowed Scope:
You may: Navigate and apply NICE NG12 guidance, identify whether suspected cancer referral or urgent investigation criteria apply, generate structured outputs (recommendations, follow-ups, referral drafts, safety-netting), cite exact NG12 sections verbatim.
You must not: Diagnose cancer, guess missing inputs (age, symptoms, duration, risk factors), provide uncited clinical advice, recommend actions during intake when required fields are missing.

YOUR TASK: Extract specific case fields from the conversation during Structured Intake (Stage 2). Extract ONLY explicitly stated information. Do NOT infer, guess, or assume. This extraction occurs BEFORE retrieval and is required for case_triage intent.

Structured Intake Rules:
- Before retrieving or applying rules: Determine whether required fields are present (e.g. age, relevant symptoms, duration).
- If any required field is missing: Ask concise, targeted follow-up questions. Explain why the information is needed. Do not reference pathways, tests, or rule numbers at this stage.
- Retrieval only happens after intake is complete.

FIELD DEFINITIONS:

1. age (integer | null)
   - Extract only if numeric age is explicitly stated (e.g., "58-year-old", "age 45", "patient is 60")
   - Patterns to extract: "X-year-old", "aged X", "age X", "X years old"
   - Set to null if: ambiguous ("older person", "elderly", "adult", "child"), not mentioned, or range given without specific value
   - Validation: Must be between 0-150 if provided
   - Required for case_triage: YES

2. sex (string | null)
   - Extract only if explicitly stated: "male", "female", "man", "woman", "M", "F"
   - Set to null if not mentioned or ambiguous
   - Required for case_triage: NO (optional)

3. symptoms (array of strings)
   - Extract ONLY symptoms explicitly mentioned in the text
   - Use controlled vocabulary when exact match: dysphagia, weight loss, rectal bleeding, abdominal pain, haemoptysis, cough, fatigue, dyspepsia, heartburn, reflux, change in bowel habit, constipation, diarrhoea, bloating, nausea, vomiting, chest pain, shortness of breath, hoarseness, lump, mass, bleeding, pain
   - If symptom is paraphrased (e.g., "trouble swallowing" for dysphagia), include the original phrase as-is
   - Return empty array [] if no symptoms mentioned
   - DO NOT infer symptoms from context
   - Required for case_triage: YES

4. symptom_duration (string | null)
   - Extract only if explicitly stated (e.g., "3 months", "6 weeks", "persistent for 2 years")
   - Set to null if not mentioned or unclear
   - Required for case_triage: NO (optional, but helpful)

5. key_triggers (array of strings)
   - Extract explicit trigger factors mentioned (e.g., "smoking history", "family history", "positive FIT test")
   - Return empty array [] if none mentioned
   - Required for case_triage: NO (optional)

6. missing_fields (array of strings)
   - List field names that are required but missing or ambiguous
   - Required fields for case triage: "age", "symptoms"
   - Add to this list if: field is not mentioned, ambiguous, or cannot be determined
   - CRITICAL: If required field is missing, system will fail-closed and ask follow-up questions

OUTPUT FORMAT (JSON object):
{
  "age": <integer | null>,
  "sex": <string | null>,
  "symptoms": [<array of strings>],
  "symptom_duration": <string | null>,
  "key_triggers": [<array of strings>],
  "missing_fields": [<array of field names>]
}

CRITICAL EXTRACTION RULES (NON-NEGOTIABLE):
1. EXTRACT ONLY: Extract only what is explicitly stated in the conversation. Do NOT infer, assume, or guess.
2. NO INFERENCE: If information is missing, ambiguous, or cannot be determined, mark it as missing. Do not fill gaps with assumptions.
3. CONSERVATIVE APPROACH: When in doubt, mark as missing rather than guessing. A refusal with clarity is always safer than a confident guess.
4. AMBIGUITY HANDLING:
   - "Older person", "elderly", "adult", "child" → age = null, add "age" to missing_fields
   - "Some symptoms", "various symptoms", "symptoms" without specifics → symptoms = [], add "symptoms" to missing_fields
   - Age ranges without specific value (e.g., "50s", "40-50") → age = null, add "age" to missing_fields
5. SYMPTOM EXTRACTION:
   - Match against controlled vocabulary when possible
   - Preserve original phrasing if not exact match
   - Do not combine or summarize multiple symptoms into one
   - Each symptom must be explicitly mentioned, not inferred from context
6. VALIDATION:
   - age must be 0-150 if provided
   - symptoms must be explicit mentions, not inferred from context
   - missing_fields must include ALL required fields ("age", "symptoms") that are absent or ambiguous
7. FAIL-CLOSED BEHAVIOR:
   - If required field (age or symptoms) is missing or ambiguous, it MUST be added to missing_fields
   - The system will not proceed to retrieval until required fields are present
   - Do NOT guess age, symptoms, or any other field

EXAMPLES:
Input: "58-year-old man with weight loss and persistent heartburn"
Output: {"age": 58, "sex": "male", "symptoms": ["weight loss", "heartburn"], "symptom_duration": null, "key_triggers": [], "missing_fields": []}

Input: "Older patient with abdominal symptoms"
Output: {"age": null, "sex": null, "symptoms": [], "symptom_duration": null, "key_triggers": [], "missing_fields": ["age", "symptoms"]}

Input: "Patient with unexplained weight loss"
Output: {"age": null, "sex": null, "symptoms": ["weight loss"], "symptom_duration": null, "key_triggers": [], "missing_fields": ["age"]}

This tool supports recognition and referral based on NICE NG12 and does not replace clinical judgement or local pathways."""

        # Build conversation history
        conversation_text = request.message
        if request.context and request.context.messages:
            history = "\n".join([
                f"{msg.role.value}: {msg.content}"
                for msg in request.context.messages[-5:]  # Last 5 messages
            ])
            conversation_text = f"{history}\n\nCurrent message: {request.message}"
        
        user_prompt = f"""CONVERSATION:
{conversation_text}

TASK: Extract the structured case fields from this conversation. Extract ONLY explicitly stated information. Do not infer or guess. Mark ambiguous or missing information appropriately in the missing_fields array."""
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.custom_settings.llm_temperature_extraction,
                max_tokens=400,  # Increased for detailed field extraction
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            
            result = json.loads(content)
            
            # Extract fields with validation
            age = result.get("age")
            if age is not None:
                try:
                    age = int(age)
                    if age < 0 or age > 150:
                        age = None
                except (ValueError, TypeError):
                    age = None
            
            sex = result.get("sex")
            if sex:
                sex = str(sex).strip()
                if not sex:
                    sex = None
            
            symptoms = result.get("symptoms", [])
            if not isinstance(symptoms, list):
                symptoms = []
            symptoms = [str(s).strip().lower() for s in symptoms if s]
            
            symptom_duration = result.get("symptom_duration")
            if symptom_duration:
                symptom_duration = str(symptom_duration).strip()
                if not symptom_duration:
                    symptom_duration = None
            
            key_triggers = result.get("key_triggers", [])
            if not isinstance(key_triggers, list):
                key_triggers = []
            key_triggers = [str(t).strip() for t in key_triggers if t]
            
            missing_fields = result.get("missing_fields", [])
            if not isinstance(missing_fields, list):
                missing_fields = []
            missing_fields = [str(f).strip() for f in missing_fields if f]
            
            case_fields = CaseFields(
                age=age,
                sex=sex,
                symptoms=symptoms,
                symptom_duration=symptom_duration,
                key_triggers=key_triggers,
                missing_fields=missing_fields,
            )
            
            logger.info(
                "Case fields extracted",
                age=age,
                symptoms_count=len(symptoms),
                missing_fields=missing_fields,
            )
            
            return case_fields
            
        except Exception as e:
            logger.exception("Error extracting case fields", error=str(e))
            # Fail closed: return empty fields with all marked as missing
            return CaseFields(
                age=None,
                sex=None,
                symptoms=[],
                symptom_duration=None,
                key_triggers=[],
                missing_fields=["age", "symptoms"],
            )
    
    def validate_intake(
        self,
        case_fields: CaseFields,
    ) -> IntakeResult:
        """
        Validate intake fields using deterministic logic.
        
        This is DETERMINISTIC validation, not LLM-based.
        
        Args:
            case_fields: Extracted case fields.
            
        Returns:
            IntakeResult with validation status and current question if incomplete.
        """
        logger.info("Validating intake", missing_fields=case_fields.missing_fields)
        
        # Deterministic validation: check required fields
        required_fields = ["age", "symptoms"]
        missing_required = []
        
        if case_fields.age is None or "age" in case_fields.missing_fields:
            missing_required.append("age")
        
        if not case_fields.symptoms or "symptoms" in case_fields.missing_fields:
            missing_required.append("symptoms")
        
        is_complete = len(missing_required) == 0
        
        if is_complete:
            logger.info("Intake validation passed - all required fields present")
            return IntakeResult(
                fields_collected=case_fields,
                is_complete=True,
                current_question=None,
                follow_up_questions=[],
            )
        
        # Generate next question deterministically
        current_question = self.generate_intake_questions(missing_required)[0]
        
        logger.info(
            "Intake validation incomplete",
            missing_required=missing_required,
            current_question_id=current_question.question_id,
        )
        
        return IntakeResult(
            fields_collected=case_fields,
            is_complete=False,
            current_question=current_question,
            follow_up_questions=[],
        )
    
    def generate_intake_questions(
        self,
        missing_fields: list[str],
    ) -> list[IntakeQuestion]:
        """
        Generate intake questions deterministically based on missing fields.
        
        This is DETERMINISTIC question generation, not LLM-based.
        Questions are generated with pathway-correlated options.
        
        Args:
            missing_fields: List of missing field names.
            
        Returns:
            List of IntakeQuestion objects, one per missing field.
        """
        questions = []
        
        for field in missing_fields:
            if field == "age":
                # Age question with pathway-correlated options
                options = []
                for threshold in self.custom_settings.age_thresholds:
                    options.append(IntakeOption(
                        option_id=f"age_{threshold['min']}_{threshold['max']}",
                        label=threshold["label"],
                        value=f"{threshold['min']}-{threshold['max']}",
                        pathway_correlation=f"NG12 age threshold: {threshold['label']}",
                    ))
                
                questions.append(IntakeQuestion(
                    question_id="age_question",
                    question_text="What is the patient's age?",
                    question_type="single_choice",
                    options=options,
                    required=True,
                ))
            
            elif field == "symptoms":
                # Symptom question with controlled vocabulary options
                options = []
                for symptom in self.custom_settings.symptom_vocabulary[:10]:  # Top 10 symptoms
                    options.append(IntakeOption(
                        option_id=f"symptom_{symptom.replace(' ', '_')}",
                        label=symptom.title(),
                        value=symptom,
                        pathway_correlation=f"NG12 symptom: {symptom}",
                    ))
                
                questions.append(IntakeQuestion(
                    question_id="symptoms_question",
                    question_text="What symptoms is the patient experiencing? (Select all that apply)",
                    question_type="multi_choice",
                    options=options,
                    required=True,
                ))
        
        return questions
    
    def _generate_intake_fail_closed_message(
        self,
        case_fields: CaseFields,
        intake_result: IntakeResult,
        original_query: str,
    ) -> str:
        """
        Generate fail-closed message matching gold standard format.
        
        This message explicitly states what information is missing and why
        no recommendation can be made. Matches the exact format from gold standard.
        
        Args:
            case_fields: Extracted case fields (may be incomplete).
            intake_result: Validation result showing what's missing.
            original_query: Original user query for context.
            
        Returns:
            Fail-closed message in gold standard format.
        """
        missing_fields = case_fields.missing_fields or []
        if not missing_fields:
            # Determine missing fields from validation
            required_missing = []
            if not case_fields.age or "age" in missing_fields:
                required_missing.append("age")
            if not case_fields.symptoms or len(case_fields.symptoms) == 0 or "symptoms" in missing_fields:
                required_missing.append("symptoms")
            missing_fields = required_missing
        
        # Check query for context (weight loss, fatigue, etc.)
        query_lower = original_query.lower()
        has_weight_loss = "weight loss" in query_lower
        has_fatigue = "fatigue" in query_lower
        
        # Build message matching gold standard format
        message_parts = [
            "Based on the information provided, I cannot yet apply NG12 suspected cancer referral criteria.",
            "",
            "Reason:",
        ]
        
        if has_weight_loss or has_fatigue:
            message_parts.append(
                "NG12 recommendations for unexplained weight loss and fatigue depend on patient age and the presence of specific associated symptoms. These details are required to determine whether any suspected cancer pathway applies."
            )
        else:
            message_parts.append(
                "NG12 recommendations depend on patient age and the presence of specific symptoms. These details are required to determine whether any suspected cancer pathway applies."
            )
        
        message_parts.extend([
            "",
            "Information needed to proceed:",
            "",
            "Please provide:",
        ])
        
        if "age" in missing_fields or not case_fields.age:
            message_parts.append("• Patient age")
        
        if has_weight_loss:
            message_parts.append("• Duration and magnitude of weight loss")
        
        if "symptoms" in missing_fields or not case_fields.symptoms or len(case_fields.symptoms) == 0:
            if has_weight_loss or has_fatigue:
                message_parts.append(
                    "• Presence or absence of associated symptoms (e.g. upper abdominal pain, reflux, dyspepsia, cough, bleeding)"
                )
            else:
                message_parts.append("• Specific symptoms the patient is experiencing")
        elif has_weight_loss:
            # Check if we have basic symptoms but need GI-specific ones
            has_gi_symptoms = any(
                s.lower() in [symptom.lower() for symptom in case_fields.symptoms]
                for s in ["upper abdominal pain", "reflux", "dyspepsia", "heartburn", "abdominal"]
            )
            if not has_gi_symptoms:
                message_parts.append(
                    "• Presence or absence of associated symptoms (e.g. upper abdominal pain, reflux, dyspepsia, cough, bleeding)"
                )
        
        message_parts.extend([
            "",
            "Until this information is available, no NG12 referral recommendation can be made with confidence.",
            "",
            "Safety note:",
            "Unexplained weight loss is recognised in NG12 as a potential red-flag symptom. This tool supports recognition and referral based on NICE guidance and does not replace clinical judgement or local pathways.",
        ])
        
        return "\n".join(message_parts)
    
    def _extract_required_fields_questions(
        self,
        case_fields: CaseFields,
    ) -> list[str]:
        """
        Extract question text for required fields that are missing.
        
        Args:
            case_fields: Case fields (may be incomplete).
            
        Returns:
            List of question texts for missing required fields.
        """
        questions = []
        missing_fields = case_fields.missing_fields or []
        
        if "age" in missing_fields or not case_fields.age:
            questions.append("What is the patient's age?")
        
        if "symptoms" in missing_fields or not case_fields.symptoms or len(case_fields.symptoms) == 0:
            questions.append(
                "What symptoms is the patient experiencing? Please specify duration and any associated symptoms (e.g. upper abdominal pain, reflux, dyspepsia)."
            )
        
        return questions
    
    def extract_evidence(
        self,
        retrieval_result: RetrievalResult,
    ) -> list[VerbatimEvidence]:
        """
        Extract verbatim evidence from retrieved chunks.
        
        No LLM summarization - direct mapping to VerbatimEvidence.
        
        Args:
            retrieval_result: Result from structured retrieval.
            
        Returns:
            List of VerbatimEvidence objects.
        """
        logger.info("Extracting evidence", chunks_count=len(retrieval_result.rule_chunks))
        
        evidence_list = []
        
        for chunk in retrieval_result.rule_chunks:
            evidence = VerbatimEvidence(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                section_path=chunk.metadata.inherited.section_path,
                rule_id=chunk.metadata.local.rule_id,
                relevance_score=retrieval_result.retrieval_scores.get("combined_avg", 0.0),
                metadata_quality=chunk.metadata.metadata_quality,
            )
            evidence_list.append(evidence)
        
        logger.info("Evidence extracted", count=len(evidence_list))
        return evidence_list
    
    def compute_confidence(
        self,
        evidence: list[VerbatimEvidence],
        case_fields: CaseFields | None = None,
    ) -> ConfidenceScore:
        """
        Compute confidence score using deterministic calculations.
        
        CRITICAL: All confidence computation is DETERMINISTIC (code-based), NOT LLM-based.
        
        Args:
            evidence: List of verbatim evidence chunks.
            case_fields: Optional case fields for constraint matching.
            
        Returns:
            ConfidenceScore with overall score and factors.
        """
        logger.info("Computing confidence", evidence_count=len(evidence))
        
        if not evidence:
            return ConfidenceScore(
                overall=0.0,
                factors=ConfidenceFactors(
                    retrieval_strength=0.0,
                    constraint_match=0.0,
                    evidence_specificity=0.0,
                    coverage=0.0,
                    metadata_quality_score=0.0,
                    action_consensus=0.0,
                ),
                threshold_met=False,
            )
        
        # Factor 1: Retrieval strength (average relevance score)
        retrieval_strength = sum(e.relevance_score for e in evidence) / len(evidence)
        retrieval_strength = max(0.0, min(1.0, retrieval_strength))
        
        # Factor 2: Constraint match (if case_fields provided)
        constraint_match = 1.0
        if case_fields:
            matches = 0
            total = 0
            
            # Age match
            if case_fields.age is not None:
                total += 1
                # Check if any evidence has matching age constraints
                # (simplified - would need to check chunk metadata)
                matches += 1  # Placeholder
            
            # Symptom match
            if case_fields.symptoms:
                total += 1
                # Check if any evidence has matching symptoms
                # (simplified - would need to check chunk metadata)
                matches += 1  # Placeholder
            
            if total > 0:
                constraint_match = matches / total
        
        # Factor 3: Evidence specificity (based on rule_id presence)
        rule_ids_present = sum(1 for e in evidence if e.rule_id is not None)
        evidence_specificity = rule_ids_present / len(evidence) if evidence else 0.0
        
        # Factor 4: Coverage (simplified - all evidence covers query)
        coverage = 1.0  # Placeholder
        
        # Factor 5: Metadata quality score (deterministic mapping)
        quality_scores = {
            MetadataQuality.HIGH: 1.0,
            MetadataQuality.MEDIUM: 0.6,
            MetadataQuality.LOW: 0.3,
        }
        avg_quality = sum(quality_scores.get(e.metadata_quality, 0.3) for e in evidence) / len(evidence)
        metadata_quality_score = avg_quality
        
        # Factor 6: Action consensus (would need action_type from chunks)
        action_consensus = 1.0  # Placeholder
        
        # Overall calculation (deterministic formula)
        overall = (
            self.custom_settings.confidence_weight_retrieval * retrieval_strength +
            self.custom_settings.confidence_weight_constraint * constraint_match +
            self.custom_settings.confidence_weight_specificity * evidence_specificity +
            self.custom_settings.confidence_weight_coverage * coverage +
            self.custom_settings.confidence_weight_metadata * metadata_quality_score +
            self.custom_settings.confidence_weight_consensus * action_consensus
        )
        
        # Cap overall if metadata quality is low
        if metadata_quality_score < 0.6:
            overall = min(overall, 0.6)
        
        overall = max(0.0, min(1.0, overall))
        threshold_met = overall >= self.custom_settings.confidence_threshold
        
        factors = ConfidenceFactors(
            retrieval_strength=retrieval_strength,
            constraint_match=constraint_match,
            evidence_specificity=evidence_specificity,
            coverage=coverage,
            metadata_quality_score=metadata_quality_score,
            action_consensus=action_consensus,
        )
        
        confidence = ConfidenceScore(
            overall=overall,
            factors=factors,
            threshold_met=threshold_met,
        )
        
        logger.info(
            "Confidence computed",
            overall=overall,
            threshold_met=threshold_met,
        )
        
        return confidence
    
    async def generate_final_response(
        self,
        evidence: list[VerbatimEvidence],
        confidence: ConfidenceScore,
        request: ChatRequest,
        case_fields: CaseFields | None = None,
        intent: IntentType | None = None,
    ) -> StructuredResponse:
        """
        Generate final response with fail-closed logic and citation validation.
        
        Args:
            evidence: List of verbatim evidence.
            confidence: Confidence score.
            request: Original chat request.
            case_fields: Optional case fields.
            
        Returns:
            StructuredResponse with answer, citations, and evidence.
        """
        logger.info("Generating final response", evidence_count=len(evidence))
        
        # Fail-closed check: If threshold not met, return clarification
        if not confidence.threshold_met:
            logger.warning("Confidence threshold not met, returning clarification")
            return StructuredResponse(
                answer="I need more information to provide a reliable recommendation. Please provide additional details about the patient's age, symptoms, or other relevant factors.",
                citations=[],
                evidence=evidence,
                confidence=confidence,
                referral_note_draft=None,
            )
        
        # Check for HIGH quality chunks
        high_quality_count = sum(
            1 for e in evidence if e.metadata_quality == MetadataQuality.HIGH
        )
        if high_quality_count == 0:
            logger.warning("No HIGH quality chunks, returning clarification")
            return StructuredResponse(
                answer="I found some information, but it's not specific enough to make a reliable recommendation. Please provide more details or consult the full NICE NG12 guideline.",
                citations=[],
                evidence=evidence,
                confidence=confidence,
                referral_note_draft=None,
            )
        
        # Generate answer using LLM (LangStruct extraction)
        # Format evidence with clear boundaries
        evidence_text = "\n\n".join([
            f"--- Evidence Chunk {idx + 1} ---\n"
            f"Rule ID: {e.rule_id or 'N/A'}\n"
            f"Section: {e.section_path}\n"
            f"Text: {e.text}\n"
            for idx, e in enumerate(evidence)
        ])
        
        system_prompt = """You are an NG12 Suspected Cancer Recognition & Referral Assistant.
Your purpose is to support clinicians by mapping structured inputs to NICE NG12 guidance, with verbatim citations, traceability, and fail-closed behavior.
You do not diagnose, do not replace clinical judgement, and do not infer missing information.

Core Operating Principles (Non-negotiable):

1. Grounding first:
   - Every recommendation must be explicitly grounded in retrieved NG12 rule chunks.
   - No citation → no recommendation.
   - Use ONLY information that is EXPLICITLY stated in the provided evidence chunks.
   - Do NOT use external medical knowledge, general guidelines, or information not in the evidence.
   - Do NOT infer, extrapolate, or add information that is not in the evidence.
   - Every statement must be traceable to a specific evidence chunk with verbatim citation.

2. Fail closed by default:
   - If required information is missing, ambiguous, or confidence is low, you must ask targeted follow-up questions or refuse.
   - A refusal with clarity is always safer than a confident guess.
   - If the evidence does NOT contain information needed to answer the query, you MUST state:
     "The provided NICE NG12 evidence does not contain specific guidance for this scenario. I cannot provide a recommendation based on the available evidence."
   - Do NOT make up recommendations, pathways, or guidance.
   - Do NOT use general medical knowledge to fill gaps.
   - Do NOT provide "common sense" or "standard practice" recommendations.
   - If only partial information is available, state what IS in the evidence and what IS NOT.

3. Rule-level specificity:
   - Apply guidance at the individual recommendation (rule) level, not high-level subsections.
   - Each action must map to a single NG12 rule with clear conditions.
   - Quote or closely paraphrase the EXACT text from the evidence chunks.
   - Do NOT add details, examples, or explanations not in the evidence.
   - Preserve IF-THEN conditional structure: "IF age ≥ X AND symptom Y is present, THEN consider Z"
   - Do NOT collapse conditional logic to "patients with X should do Y" when criteria are conditional.

4. Deterministic safety controls:
   - LLMs may extract structured facts, but confidence scoring, thresholds, and gating are deterministic and auditable.
   - If confidence < threshold → fail closed.
   - If confidence ≥ threshold → proceed with recommendation.

Allowed Scope:
You may: Navigate and apply NICE NG12 guidance, identify whether suspected cancer referral or urgent investigation criteria apply, generate structured outputs (recommendations, follow-ups, referral drafts, safety-netting), cite exact NG12 sections verbatim.

You must not: Diagnose cancer, guess missing inputs (age, symptoms, duration, risk factors), provide uncited clinical advice, recommend actions during intake when required fields are missing, provide recommendations not explicitly stated in the evidence, add test lists, examination procedures, or clinical actions not in the evidence, provide general medical advice or best practices.

CITATION REQUIREMENT (MANDATORY):
   - Every recommendation, pathway, or guideline reference MUST be cited using: [rule_id: section_path]
   - Example: [1.1.1: NG12 > Lung cancer]
   - If rule_id is "N/A", use: [N/A: section_path]
   - If you cannot cite a specific rule_id and section_path, the information is NOT in the evidence and MUST NOT be included.

OUTPUT STRUCTURE (Final Output - Stage 4):
When permitted (confidence ≥ threshold), respond with:
1. Clear triage outcome (e.g. 2WW referral, urgent test, insufficient info)
2. Recommended next steps (based ONLY on evidence)
3. Safety-netting advice (if applicable)
4. Verbatim citations (rule ID + section path) for every claim
5. Confidence score with brief rationale (if applicable)

Always include at the end:
"This tool supports recognition and referral based on NICE NG12 and does not replace clinical judgement or local pathways."

OUTPUT STRUCTURE DETAILS:
1. Direct answer: Address ONLY what is in the evidence (2-4 sentences). State triage outcome clearly if applicable.
2. Evidence-based recommendations: ONLY recommendations explicitly stated in evidence, with verbatim citations.
3. Missing information: Explicitly state what information is NOT in the provided evidence.
4. Citations: Include [rule_id: section_path] citations for all guideline references. Every claim must be cited.
5. Fail-closed statement: If evidence is insufficient, state this clearly and explain why.

CITATION FORMAT RULES (MANDATORY):
- Every recommendation must have a citation: [rule_id: section_path]
- Multiple citations: [1.1.1: NG12 > Lung cancer] [1.1.2: NG12 > Lung cancer]
- If rule_id unavailable: [N/A: NG12 > Section name]
- Place citations immediately after the referenced recommendation
- If you cannot cite it, do NOT include it in the response

TONE AND STYLE:
- Professional and clinical
- Clear and concise (avoid unnecessary verbosity)
- Specific rather than generic
- Use guideline terminology accurately
- Acknowledge uncertainty when evidence is incomplete
- Use conditional language (IF-THEN) to preserve NG12 logic structure

FORMATTING REQUIREMENTS:
- Use plain text format - NO markdown structure (no # headers, no ``` code blocks, no markdown links)
- Use **bold** only for emphasis of key terms (e.g., **urgent referral**, **2WW pathway**)
- Write in natural paragraphs with proper sentence structure
- Use simple line breaks between paragraphs (double line break)
- Avoid markdown lists - use natural prose instead (e.g., "First, ... Second, ..." instead of "- First")
- Keep output clean and readable without markdown syntax

SAFETY BOUNDARIES:
- Do not provide diagnostic advice or treatment recommendations
- Do not interpret test results or clinical findings
- Do not make patient-specific treatment decisions
- Focus on referral pathways, recognition criteria, and guideline recommendations
- If asked about treatment or diagnosis, redirect to guideline scope

PRESERVE LOGICAL STRUCTURE (CRITICAL):
- NG12 uses conditional logic: "IF age ≥ X AND symptom Y is present, THEN consider Z"
- Do NOT collapse this to: "weight loss + fatigue → do X"
- Always frame as conditional: "If the patient is aged X and has symptom Y, then [rule_id] recommends..."
- Do NOT assume criteria are met - state them as conditions
- Do NOT answer as if the user already meets criteria when information is missing

DO NOT ASSUME MISSING INPUTS (CRITICAL):
- If age is NOT PROVIDED in available case information, do NOT introduce age brackets (18+, 55+)
- If symptoms are NOT PROVIDED in available case information, do NOT introduce symptom sets
- If information is missing, state what is needed, do NOT proceed as if it exists
- Check the AVAILABLE CASE INFORMATION section in the user prompt - only use what is explicitly listed there

EXAMPLE OUTPUT FORMAT (clean text, no markdown structure, preserve IF-THEN logic):
"Based on the provided NICE NG12 evidence [1.2.1: NG12 > Upper gastrointestinal tract cancers], **if** a patient is aged 55 and over **and** has unexplained **weight loss** **and** dyspepsia, then the guideline recommends offering **urgent direct access upper GI endoscopy**. The evidence states that the endoscopy should be performed and results returned within **2 weeks** [1.2.1: NG12 > Upper gastrointestinal tract cancers].

The provided evidence does not specify additional tests, examination procedures, or follow-up actions beyond what is stated above.

This tool supports recognition and referral based on NICE NG12 and does not replace clinical judgement or local pathways."

NOTE: Always use conditional language (IF-THEN) - do NOT collapse to "patients with X should do Y" when criteria are conditional.

EXAMPLE FOR INSUFFICIENT EVIDENCE:
"The provided NICE NG12 evidence does not contain specific guidance for this scenario. I cannot provide a recommendation based on the available evidence chunks. Please consult the full NICE NG12 guideline or provide more specific details about the patient's presentation.

This tool supports recognition and referral based on NICE NG12 and does not replace clinical judgement or local pathways."

CRITICAL REMINDERS:
- If you cannot find the information in the provided evidence chunks, you MUST say so explicitly
- Do NOT fill gaps with general medical knowledge
- Every claim must be traceable to a specific evidence chunk with verbatim citation
- Use **bold** sparingly for key terms only. Write in natural prose, not markdown lists or headers.
- Always include the safety disclaimer at the end: "This tool supports recognition and referral based on NICE NG12 and does not replace clinical judgement or local pathways."
"""

        # Build context about what information we have (for case_triage)
        # This helps the LLM know what NOT to assume
        context_info = ""
        if intent == IntentType.CASE_TRIAGE and case_fields:
            context_parts = []
            if case_fields.age:
                context_parts.append(f"✓ Patient age: {case_fields.age}")
            else:
                context_parts.append("✗ Patient age: NOT PROVIDED - DO NOT assume age")
            if case_fields.symptoms and len(case_fields.symptoms) > 0:
                context_parts.append(f"✓ Symptoms: {', '.join(case_fields.symptoms)}")
            else:
                context_parts.append("✗ Symptoms: NOT PROVIDED - DO NOT assume symptoms")
            context_info = (
                f"\n\nAVAILABLE CASE INFORMATION (ONLY USE WHAT IS MARKED WITH ✓):\n"
                + "\n".join(context_parts)
                + "\n\nCRITICAL: Do NOT introduce age brackets, symptom sets, or assume any information marked with ✗.\n"
            )
        
        user_prompt = f"""USER QUERY:
{request.message}
{context_info}
EVIDENCE (NICE NG12 Guidelines - USE ONLY THIS INFORMATION):
{evidence_text}

CRITICAL INSTRUCTIONS (FOLLOW STRICTLY):
1. Answer the query using ONLY the information in the evidence chunks above - nothing else
2. If the evidence does NOT contain information needed to answer the query, you MUST explicitly state: "The provided NICE NG12 evidence does not contain specific guidance for this scenario. I cannot provide a recommendation based on the available evidence."
3. Do NOT add information, tests, procedures, or recommendations not explicitly stated in the evidence
4. Do NOT use markdown structure (no headers, no lists with bullets, no code blocks)
5. Cite every claim using [rule_id: section_path] format - if you cannot cite it, do not include it
6. If you cannot find the answer in the evidence, say so explicitly - do NOT guess, infer, or use general medical knowledge
7. Write in plain prose paragraphs - use **bold** only for key terms, not for structure

PRESERVE LOGICAL STRUCTURE (CRITICAL):
- NG12 uses conditional logic: "IF age ≥ X AND symptom Y is present, THEN consider Z"
- Do NOT collapse this to: "weight loss + fatigue → do X"
- Always frame as conditional: "If the patient is aged X and has symptom Y, then [rule_id] recommends..."
- Do NOT assume criteria are met - state them as conditions
- Do NOT answer as if the user already meets criteria when information is missing

DO NOT ASSUME MISSING INPUTS (CRITICAL):
- If age is NOT PROVIDED in available case information, do NOT introduce age brackets (18+, 55+)
- If symptoms are NOT PROVIDED in available case information, do NOT introduce symptom sets
- If information is missing, state what is needed, do NOT proceed as if it exists
- Check the AVAILABLE CASE INFORMATION section above - only use what is explicitly listed there

TASK:
Generate a concise answer using ONLY the evidence provided. Write in plain text prose (no markdown). If information is missing, explicitly state what is NOT in the evidence. Every recommendation must have a citation. Preserve IF-THEN conditional structure."""

        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.custom_settings.llm_temperature_response,  # Lower temp for strict grounding
                max_tokens=600,  # Reduced to encourage concise, grounded responses
            )
            
            answer_text = response.choices[0].message.content or "I cannot provide a recommendation based on the available evidence."
            
            # Validate grounding: Check if response is actually grounded in evidence
            answer_text = self._validate_grounding(answer_text, evidence)
            
            # Clean markdown structure while preserving bold formatting
            answer_text = self._clean_markdown(answer_text)
            
            # Parse and validate citations (deterministic)
            citations = self._parse_citations(answer_text, evidence)
            
            # Generate referral note draft if applicable
            referral_note = None
            if case_fields and case_fields.age and case_fields.symptoms:
                referral_note = self._generate_referral_note_draft(case_fields, evidence)
            
            return StructuredResponse(
                answer=answer_text,
                citations=citations,
                evidence=evidence,
                confidence=confidence,
                referral_note_draft=referral_note,
                follow_up_questions=[],
            )
            
        except Exception as e:
            logger.exception("Error generating final response", error=str(e))
            return StructuredResponse(
                answer="I encountered an error generating the response. Please try again.",
                citations=[],
                evidence=evidence,
                confidence=confidence,
                referral_note_draft=None,
            )
    
    def _parse_citations(
        self,
        answer_text: str,
        evidence: list[VerbatimEvidence],
    ) -> list[Citation]:
        """
        Parse citations from answer text using deterministic regex.
        
        Args:
            answer_text: Answer text that may contain citations.
            evidence: List of evidence chunks to validate against.
            
        Returns:
            List of validated Citation objects.
        """
        citations = []
        
        # Pattern: [rule_id: section_path] or [1.2.1: NG12 > Section]
        citation_pattern = r'\[([^\]]+):\s*([^\]]+)\]'
        matches = re.finditer(citation_pattern, answer_text)
        
        evidence_rule_ids = {e.rule_id for e in evidence if e.rule_id}
        evidence_sections = {e.section_path for e in evidence}
        
        for match in matches:
            rule_id_str = match.group(1).strip()
            section_path = match.group(2).strip()
            
            # Validate rule_id exists in evidence
            if rule_id_str not in evidence_rule_ids:
                logger.warning("Invalid citation rule_id", rule_id=rule_id_str)
                continue
            
            # Validate section_path matches evidence
            if section_path not in evidence_sections:
                logger.warning("Invalid citation section_path", section_path=section_path)
                continue
            
            # Find evidence text
            evidence_text = ""
            for e in evidence:
                if e.rule_id == rule_id_str and e.section_path == section_path:
                    evidence_text = e.text[:200]  # First 200 chars
                    break
            
            citations.append(Citation(
                rule_id=rule_id_str if rule_id_str != "N/A" else None,
                section_path=section_path,
                evidence_text=evidence_text,
            ))
        
        logger.info("Citations parsed and validated", count=len(citations))
        return citations
    
    def _generate_referral_note_draft(
        self,
        case_fields: CaseFields,
        evidence: list[VerbatimEvidence],
    ) -> str:
        """
        Generate referral note draft using deterministic template.
        
        This is template-based, not LLM-based.
        
        Args:
            case_fields: Case fields.
            evidence: Evidence chunks.
            
        Returns:
            Draft referral note text.
        """
        note_parts = [
            f"Patient: {case_fields.age}-year-old",
            f"Symptoms: {', '.join(case_fields.symptoms)}",
            "",
            "Based on NICE NG12 guidelines:",
        ]
        
        for e in evidence[:3]:  # First 3 evidence chunks
            if e.rule_id:
                note_parts.append(f"- [{e.rule_id}]: {e.section_path}")
            else:
                note_parts.append(f"- {e.section_path}")
        
        return "\n".join(note_parts)
    
    def _validate_grounding(
        self,
        answer_text: str,
        evidence: list[VerbatimEvidence],
    ) -> str:
        """
        Validate that the answer is grounded in evidence and enforce fail-closed behavior.
        
        This is a deterministic check to ensure the LLM didn't hallucinate.
        If citations are missing or evidence doesn't support the claims, add fail-closed statements.
        
        Args:
            answer_text: LLM-generated answer text.
            evidence: List of evidence chunks that should ground the answer.
            
        Returns:
            Validated answer text with fail-closed statements if needed.
        """
        # Check if answer has citations
        citation_pattern = r'\[([^\]]+):\s*([^\]]+)\]'
        citations_found = re.findall(citation_pattern, answer_text)
        
        # If no citations and answer is substantial, add warning
        if not citations_found and len(answer_text) > 100:
            logger.warning(
                "Answer generated without citations - may not be grounded",
                answer_preview=answer_text[:100],
            )
            # Add fail-closed statement
            answer_text = (
                "⚠️ WARNING: The following response may not be fully grounded in the provided evidence. "
                "Please verify all recommendations against the NICE NG12 guideline.\n\n"
                + answer_text
            )
        
        # Check for common hallucination patterns (deterministic detection)
        hallucination_patterns = [
            # Test lists not in evidence
            r'\b(should|must|recommended to)\s+(perform|conduct|order|take)\s+.*?(?:FBC|blood test|examination|investigation|test)',
            r'\b(typically include|usually include|commonly include|often include)\s+',
            r'\b(standard practice|common practice|best practice|standard approach)\b',
            r'\b(generally|usually|typically|commonly|often)\s+',
            # Examination procedures
            r'\b(perform|conduct|carry out)\s+(?:a|an)\s+(?:full|complete|comprehensive)\s+(?:examination|assessment|history)',
            # Generic recommendations
            r'\b(important to|essential to|crucial to|advisable to)\s+(?:note|remember|consider)',
            # Lists of actions not cited
            r'^\s*[-*•]\s+(?:Take|Perform|Conduct|Order|Consider)',
            # Assuming missing inputs - age brackets
            r'\b(?:aged|age)\s+(?:18|55|60)\s+(?:and over|\+)',
            r'\b(?:For|In)\s+(?:adults|patients)\s+(?:aged|age)\s+(?:18|55|60)',
            # Collapsed conditional logic (weight loss + fatigue → do X) - BAD
            r'(?:weight loss|fatigue).*?(?:→|should|must|recommend)',
            r'\b(?:For|In)\s+(?:patients|adults)\s+with\s+(?:weight loss|fatigue).*?(?:recommend|should|must)',
            # Answering as if criteria are met when they're not
            r'\b(?:For|In)\s+(?:adults|patients)\s+(?:aged|age)\s+(?:18|55|60)\s+(?:and over|\+).*?(?:with|having).*?(?:should|must|recommend)',
            # Assuming missing age - more patterns
            r'\b(?:aged|age)\s+(?:18|55|60)\s+(?:and over|\+)',
            r'\b(?:For|In)\s+(?:adults|patients)\s+(?:aged|age)\s+(?:18|55|60)',
            r'\b(?:adults|patients)\s+(?:aged|age)\s+(?:18|55|60)\s+(?:and over|\+)',
            r'\b(?:aged|age)\s+(?:≥|>=)\s*(?:18|40|55|60)',
            # Assuming missing symptoms
            r'\b(?:with|having)\s+(?:upper abdominal pain|reflux|dyspepsia|GI symptoms)',
            # Direct recommendations without conditional structure
            r'\b(?:arrange|order|perform|conduct)\s+(?:a|an)\s+(?:direct access|CT scan|urgent)',
            r'\b(?:refer|referral)\s+(?:via|for|to)\s+(?:an|a)\s+(?:urgent|suspected cancer)',
            # Summary/action lists that assume criteria
            r'Summary of action:',
            r'Your next step is',
            r'You should',
            # Age-specific recommendations when age not provided
            r'\b(?:For|In)\s+(?:adults|children|young people)',
        ]
        
        hallucination_detected = False
        detected_patterns = []
        for pattern in hallucination_patterns:
            if re.search(pattern, answer_text, re.IGNORECASE):
                detected_patterns.append(pattern)
                hallucination_detected = True
        
        if hallucination_detected:
            logger.warning(
                "Potential hallucination detected - enforcing fail-closed",
                patterns=detected_patterns,
                answer_preview=answer_text[:200],
            )
            # CRITICAL: If we detect assumptions, we MUST fail-closed
            # Don't just add a warning - replace with explicit fail-closed message
            answer_text = (
                "⚠️ FAIL-CLOSED: The system cannot provide a recommendation because the response contains assumptions or information not explicitly stated in the provided evidence.\n\n"
                "The NICE NG12 guideline requires specific patient information (age, symptoms) to determine appropriate pathways. "
                "Without this information explicitly provided, the system cannot make recommendations.\n\n"
                "Please provide:\n"
                "- Patient age (specific number, not age range)\n"
                "- Specific symptoms with duration\n"
                "- Any relevant test results or findings\n\n"
                "The system will then be able to apply NG12 criteria based on the evidence chunks provided."
            )
        
        # If evidence is empty or very limited, enforce fail-closed
        if not evidence or len(evidence) == 0:
            return (
                "The provided NICE NG12 evidence does not contain information relevant to this query. "
                "I cannot provide a recommendation based on the available evidence. "
                "Please consult the full NICE NG12 guideline or provide more specific details."
            )
        
        # If only LOW quality evidence, add warning
        low_quality_count = sum(
            1 for e in evidence if e.metadata_quality == MetadataQuality.LOW
        )
        if len(evidence) > 0 and low_quality_count == len(evidence):
            answer_text = (
                "⚠️ WARNING: The available evidence has low specificity. "
                "The following information is based on limited evidence:\n\n"
                + answer_text
            )
        
        return answer_text
    
    def _clean_markdown(self, text: str) -> str:
        """
        Clean markdown structure from text while preserving bold formatting and citations.
        
        Removes:
        - Markdown headers (# ## ### ####)
        - Code blocks (```...```)
        - Inline code (`code`)
        - Markdown links [text](url) -> text
        - Horizontal rules (---, ***)
        - Blockquotes (>)
        - Excessive whitespace
        
        Preserves:
        - **bold** formatting for emphasis (frontend supports this)
        - Simple bullet lists (- or *) - frontend supports these
        - Citations [rule_id: section_path] format
        - Natural paragraph structure
        
        Args:
            text: Text that may contain markdown.
            
        Returns:
            Cleaned text with complex markdown structure removed, simple formatting preserved.
        """
        cleaned = text
        
        # Remove markdown headers (# ## ### ####) - keep the text content
        cleaned = re.sub(r'^#{1,6}\s+(.+)$', r'\1', cleaned, flags=re.MULTILINE)
        
        # Remove code blocks (```...```) - remove entirely
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
        
        # Remove inline code (`code`) - keep the text
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
        
        # Remove markdown links but keep the text: [text](url) -> text
        # BUT preserve citations [rule_id: section_path] - they don't have (url)
        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
        
        # Remove horizontal rules (--- or ***)
        cleaned = re.sub(r'^[-*]{3,}\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove blockquotes (>) - keep the text
        cleaned = re.sub(r'^>\s+', '', cleaned, flags=re.MULTILINE)
        
        # Clean up excessive whitespace (3+ newlines -> 2 newlines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove leading/trailing whitespace from each line
        lines = cleaned.split('\n')
        cleaned = '\n'.join(line.strip() for line in lines)
        
        # Remove leading/trailing whitespace overall
        cleaned = cleaned.strip()
        
        # Preserve **bold** formatting (frontend converts this to <strong>)
        # Preserve simple bullet lists (- or *) - frontend supports these
        # Preserve citations [rule_id: section_path] - these are handled separately
        
        logger.debug(
            "Markdown cleaned",
            original_length=len(text),
            cleaned_length=len(cleaned),
        )
        
        return cleaned


# Singleton instance
_custom_chat_service: CustomChatService | None = None


def get_custom_chat_service() -> CustomChatService:
    """Get the custom chat service singleton."""
    global _custom_chat_service
    if _custom_chat_service is None:
        _custom_chat_service = CustomChatService()
    return _custom_chat_service
