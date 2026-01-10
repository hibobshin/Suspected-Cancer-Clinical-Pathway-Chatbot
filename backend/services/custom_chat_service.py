"""
Custom chat service for the NG12 Assistant Pipeline.

This service implements a 4-stage pipeline:
1. Intent + Safety Gate
2. Structured Intake (conditional)
3. Structured Retrieval
4. Evidence Extraction + Confidence + Final Output
"""

import json
import re
import time
from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

from openai import AsyncOpenAI, OpenAIError

from config.config import Settings, get_settings
from config.custom_config import CustomPipelineSettings, get_custom_settings
from config.logging_config import get_logger
from models.custom_models import (
    CaseFields,
    Citation,
    ConfidenceFactors,
    ConfidenceScore,
    IntentClassification,
    IntentType,
    IntakeOption,
    IntakeQuestion,
    IntakeResult,
    MetadataQuality,
    SafetyGateResult,
    SafetyCheck,
    StructuredResponse,
    VerbatimEvidence,
)
from services.custom_guideline_service import RetrievalResult, get_custom_guideline_service
from models.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationContext,
    MessageRole,
    PathwayRouteType,
    ResponseType,
)

logger = get_logger(__name__)


class CustomChatService:
    """
    Service for processing chat messages with the Custom NG12 Assistant Pipeline.
    
    Implements a 4-stage pipeline with strict validation and fail-closed behavior.
    """
    
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
        self._client: AsyncOpenAI | None = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client (for DeepSeek API)."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        return self._client
    
    async def process_message(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Process a chat message through the 4-stage pipeline.
        
        Args:
            request: Chat request with message and context.
            
        Returns:
            ChatResponse with the assistant's response.
        """
        start_time = time.perf_counter()
        conversation_id = request.conversation_id or uuid4()
        
        logger.info(
            "Processing custom chat message",
            conversation_id=str(conversation_id),
            message_length=len(request.message),
            route_type=request.route_type.value if request.route_type else None,
        )
        
        try:
            # Stage 1: Intent + Safety Gate
            safety_result = await self.check_safety_gate(request)
            if not safety_result.passed:
                processing_time = int((time.perf_counter() - start_time) * 1000)
                return ChatResponse(
                    conversation_id=conversation_id,
                    message=safety_result.escalation_message or "I cannot assist with this request.",
                    response_type=ResponseType.REFUSAL,
                    citations=[],
                    artifacts=[],
                    follow_up_questions=[],
                    processing_time_ms=processing_time,
                )
            
            intent = safety_result.intent_classification
            if not intent:
                processing_time = int((time.perf_counter() - start_time) * 1000)
                return ChatResponse(
                    conversation_id=conversation_id,
                    message="I encountered an error classifying your intent. Please try again.",
                    response_type=ResponseType.ERROR,
                    citations=[],
                    artifacts=[],
                    follow_up_questions=[],
                    processing_time_ms=processing_time,
                )
            
            logger.info(
                "Intent classified",
                intent=intent.intent.value,
                confidence=intent.confidence,
            )
            
            # Stage 2: Structured Intake (conditional on intent)
            case_fields: CaseFields | None = None
            intake_complete = False
            
            if intent.intent == IntentType.CASE_TRIAGE:
                # For case triage, intake is REQUIRED before retrieval
                case_fields = await self.extract_case_fields(request)
                intake_result = self.validate_intake(case_fields)
                intake_complete = intake_result.is_complete
                
                if not intake_complete:
                    # FAIL-CLOSED: Do NOT proceed to retrieval
                    # Generate proper fail-closed message matching gold standard
                    processing_time = int((time.perf_counter() - start_time) * 1000)
                    fail_closed_message = self._generate_intake_fail_closed_message(
                        case_fields,
                        intake_result,
                        request.message,
                    )
                    
                    return ChatResponse(
                        conversation_id=conversation_id,
                        message=fail_closed_message,
                        response_type=ResponseType.CLARIFICATION,
                        citations=[],
                        artifacts=[],
                        follow_up_questions=self._extract_required_fields_questions(
                            case_fields,
                        ),
                        processing_time_ms=processing_time,
                    )
            elif intent.intent == IntentType.GUIDELINE_LOOKUP:
                # For guideline lookup, intake not required - can proceed directly
                intake_complete = True
            else:
                # Documentation intent - proceed
                intake_complete = True
            
            # Stage 3: Structured Retrieval (ONLY if intake complete or guideline_lookup)
            # For case_triage, we should ONLY retrieve if we have required fields
            if intent.intent == IntentType.CASE_TRIAGE and not intake_complete:
                # This should not be reached due to early return above, but defensive check
                processing_time = int((time.perf_counter() - start_time) * 1000)
                return ChatResponse(
                    conversation_id=conversation_id,
                    message="I cannot provide a recommendation without required case information. Please provide the missing details.",
                    response_type=ResponseType.CLARIFICATION,
                    citations=[],
                    artifacts=[],
                    follow_up_questions=[],
                    processing_time_ms=processing_time,
                )
            
            guideline_service = get_custom_guideline_service()
            
            # Build query from request and case fields
            query = request.message
            cancer_site = None
            age = None
            symptoms = None
            
            # Only pass filters if we have complete case_fields (for case_triage)
            if intent.intent == IntentType.CASE_TRIAGE and case_fields:
                if case_fields.age:
                    age = case_fields.age
                if case_fields.symptoms:
                    symptoms = case_fields.symptoms
            
            # For guideline_lookup, don't filter by age/symptoms - just search
            retrieval_result = guideline_service.retrieve(
                query=query,
                cancer_site=cancer_site,
                age=age if intent.intent == IntentType.CASE_TRIAGE else None,  # Only filter by age for case_triage
                symptoms=symptoms if intent.intent == IntentType.CASE_TRIAGE else None,  # Only filter by symptoms for case_triage
                max_chunks=self.custom_settings.max_retrieved_chunks,
            )
            
            # Stage 4: Evidence Extraction + Confidence + Final Output
            evidence = self.extract_evidence(retrieval_result)
            confidence = self.compute_confidence(evidence, case_fields)
            
            structured_response = await self.generate_final_response(
                evidence=evidence,
                confidence=confidence,
                request=request,
                case_fields=case_fields,
                intent=intent.intent,
            )
            
            # Map to ChatResponse
            processing_time = int((time.perf_counter() - start_time) * 1000)
            
            # Map citations
            citations = [
                {
                    "statement_id": c.rule_id or "N/A",
                    "section": c.section_path,
                    "text": c.evidence_text,
                }
                for c in structured_response.citations
            ]
            
            # Map artifacts with location information for highlighting
            artifacts = [
                {
                    "section": e.section_path,
                    "text": e.text[:500],  # Truncate for display
                    "relevance_score": e.relevance_score,
                    "source": "NICE NG12",
                    "source_url": "https://www.nice.org.uk/guidance/ng12",
                    "chunk_id": e.chunk_id,
                    "rule_id": e.rule_id,  # For section highlighting
                }
                for e in evidence
            ]
            
            response_type = ResponseType.ANSWER
            if not confidence.threshold_met:
                response_type = ResponseType.CLARIFICATION
            
            return ChatResponse(
                conversation_id=conversation_id,
                message=structured_response.answer,
                response_type=response_type,
                citations=citations,
                artifacts=artifacts,
                follow_up_questions=structured_response.follow_up_questions if hasattr(structured_response, 'follow_up_questions') else [],
                processing_time_ms=processing_time,
            )
            
        except OpenAIError as e:
            logger.error("OpenAI API error in custom chat", error=str(e))
            processing_time = int((time.perf_counter() - start_time) * 1000)
            return ChatResponse(
                conversation_id=conversation_id,
                message=f"I encountered an error processing your request: {str(e)}",
                response_type=ResponseType.ERROR,
                citations=[],
                artifacts=[],
                follow_up_questions=[],
                processing_time_ms=processing_time,
            )
        except Exception as e:
            logger.exception("Unexpected error in custom chat", error=str(e))
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
    
    async def process_message_stream(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat message and stream the response.
        
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
            import json
            
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': str(conversation_id)})}\n\n"
            
            # Build messages
            messages = self._build_messages(request)
            
            # Stream from LLM
            stream = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True,
            )
            
            # Stream chunks
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
            
            # Send done event
            yield f"data: {json.dumps({'type': 'done', 'response_type': 'answer'})}\n\n"
            
        except Exception as e:
            logger.exception("Error in custom chat stream", error=str(e))
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
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
