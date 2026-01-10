---
name: Custom NG12 Assistant Pipeline
overview: Build a 4-stage custom NG12 assistant pipeline with intent classification, structured intake, hierarchical retrieval, and confidence-gated fail-closed outputs. Each stage will be implemented and verified before proceeding.
todos:
  - id: stage1_intent
    content: "Stage 1: Create Pydantic schemas for IntentClassification and SafetyGateResult"
    status: pending
  - id: stage1_classify
    content: "Stage 1: Implement classify_intent() with LLM structured output"
    status: pending
  - id: stage1_safety
    content: "Stage 1: Implement check_safety_gate() with red flag heuristics"
    status: pending
  - id: stage1_verify
    content: "Stage 1: Test and verify intent classification and safety gate work correctly"
    status: pending
  - id: stage2_schemas
    content: "Stage 2: Create CaseFields and IntakeResult Pydantic schemas"
    status: pending
  - id: stage2_extract
    content: "Stage 2: Implement extract_case_fields() with structured LLM extraction"
    status: pending
  - id: stage2_validate
    content: "Stage 2: Implement validate_intake() with required field checks"
    status: pending
  - id: stage2_verify
    content: "Stage 2: Test and verify structured intake with complete and incomplete cases"
    status: pending
  - id: stage3_schemas
    content: "Stage 3: Create NG12Chunk and NG12ChunkMetadata Pydantic schemas"
    status: pending
  - id: stage3_parse
    content: "Stage 3: Parse final.md into hierarchical chunks with metadata extraction"
    status: pending
  - id: stage3_embeddings
    content: "Stage 3: Generate and cache embeddings (dense) and BM25 index (sparse) for hybrid retrieval"
    status: pending
  - id: stage3_retrieval
    content: "Stage 3: Implement hybrid retrieval (BM25 + embeddings) with coarse routing, fine retrieval, and metadata filtering"
    status: pending
  - id: stage3_verify
    content: "Stage 3: Test and verify hierarchical retrieval with metadata matching"
    status: pending
  - id: stage4_schemas
    content: "Stage 4: Create VerbatimEvidence, ConfidenceScore, and StructuredResponse schemas"
    status: pending
  - id: stage4_extract
    content: "Stage 4: Implement extract_evidence() with verbatim text extraction"
    status: pending
  - id: stage4_confidence
    content: "Stage 4: Implement compute_confidence() with weighted factors"
    status: pending
  - id: stage4_generate
    content: "Stage 4: Implement generate_final_response() with fail-closed logic"
    status: pending
  - id: stage4_verify
    content: "Stage 4: Test confidence gating, evidence extraction, and final response generation"
    status: pending
  - id: integration
    content: "Integration: Wire all stages together in process_message() pipeline"
    status: pending
  - id: integration_verify
    content: "Integration: End-to-end testing with demo prompts and all failure paths"
    status: pending
---

# Custom NG12 Suspected Cancer Recognition Assistant

## Architecture Overview

**IMPORTANT: Complete Independence from RAG Pipeline**

This custom pipeline is entirely separate from the existing RAG pipeline:

- No imports from `guideline_service.py` or `rag_chat_service.py`
- Separate cache files (custom_ *vs guideline_*)
- Different chunking strategy (hierarchical with metadata vs half-page BoW)
- Different retrieval method (embeddings vs BoW + cosine similarity)
- Different data structures (NG12Chunk with metadata vs simple chunk dict)
- Fresh parsing of `final.md` - no reuse of existing chunks

4-stage pipeline with verification checkpoints:

1. **Intent + Safety Gate** → Classify intent and check red flags
2. **Structured Intake** → Collect required fields (conditional on intent)
3. **Structured Retrieval** → Hierarchical NG12 chunk retrieval with metadata
4. **Evidence Extraction + Confidence + Final Output** → Extract verbatim evidence, compute confidence, generate typed response

All outputs use Pydantic schemas. Fail-closed behavior enforced at each stage.

---

## Stage 1: Intent + Safety Gate

**Files to create/modify:**

- `backend/services/custom_guideline_service.py` (new, basic structure)
- `backend/models/custom_models.py` (new, Pydantic schemas)
- `backend/services/custom_chat_service.py` (modify to call Stage 1)

**Implementation:**

1. Create Pydantic schemas:

   - `IntentType`: `guideline_lookup | case_triage | documentation`
   - `IntentClassification`: intent, confidence, reasoning
   - `SafetyCheck`: has_red_flags, red_flags_list, escalation_message
   - `SafetyGateResult`: passed, escalation_message, intent_classification

2. Implement `classify_intent()` in `CustomChatService`:

   - LLM call with structured output (JSON mode) to classify intent
   - Return `IntentClassification` schema
   - Temperature 0.1 for determinism

3. Implement `check_safety_gate()`:

   - Heuristic rules: emergency keywords, diagnostic requests, treatment requests
   - Return `SafetyGateResult` with escalation if red flags found
   - Enforce "not diagnostic" boundary messaging

**Verification:**

- Test with queries: "What are symptoms of lung cancer?" (guideline_lookup)
- Test with: "58-year-old with weight loss and heartburn" (case_triage)
- Test with: "How do I document a referral?" (documentation)
- Test red flags: "Diagnose this patient", "What treatment should I give?"
- Verify all outputs conform to Pydantic schemas
- **Checkpoint**: Stage 1 verified before proceeding

---

## Stage 2: Structured Intake (Conditional)

**Files to modify:**

- `backend/models/custom_models.py` (add intake schemas)
- `backend/services/custom_chat_service.py` (add intake logic)

**Implementation:**

1. Create Pydantic schemas:

   - `CaseFields`: age, sex, symptoms (list), symptom_duration, key_triggers (list), missing_fields (list)
   - `IntakeResult`: fields_collected (CaseFields), is_complete, follow_up_questions (list[str])
   - `RequiredField`: field_name, is_missing, follow_up_question

2. Implement `extract_case_fields()`:

   - Only run if intent == `case_triage`
   - Extract from conversation history + current message
   - Use LLM with structured output to extract fields
   - Validate: age required, symptoms required

3. Implement `validate_intake()`:

   - Check if all required fields present
   - If missing: generate targeted follow-up questions
   - Return `IntakeResult` with `is_complete=False` if validation fails

**Verification:**

- Test complete case: "58-year-old man, weight loss 3 months, heartburn 6 months"
- Test incomplete: "Patient has weight loss" → should ask for age, sex, duration
- Test with conversation history: previous message had age, current has symptoms
- Verify follow-up questions are specific and actionable
- **Checkpoint**: Stage 2 verified before proceeding

---

## Stage 3: Structured Retrieval

**Files to create:**

- `backend/services/custom_guideline_service.py` (new, completely independent from guideline_service.py - separate parsing, chunking, and caching)

**Implementation:**

1. Create Pydantic schemas:

   - `NG12ChunkMetadata`: section_path (str), cancer_site (str | None), symptom_tags (list[str]), age_min (int | None), age_max (int | None), action_type (str), rule_id (str | None), guideline_version (str)
   - `NG12Chunk`: chunk_id (str), text (str), metadata (NG12ChunkMetadata), section_level (int)
   - `RetrievalResult`: candidate_sections (list[str]), rule_chunks (list[NG12Chunk]), retrieval_scores (dict)

2. Parse `data/final.md` into hierarchical chunks (completely independent from guideline_service.py):

   - Parse markdown structure: `#`, `##`, `###` determine hierarchy
   - Extract section_path: "1.2 Upper gastrointestinal tract cancers > 1.2.1 Urgent referral"
   - Extract metadata from section headers and content:
     - `cancer_site`: from section title (e.g., "Lung and pleural cancers")
     - `symptom_tags`: keywords extracted from content
     - `age_min/age_max`: from text (e.g., "≥55", "adults 18+")
     - `action_type`: "urgent_referral", "routine_referral", "investigation", "safety_netting"
     - `rule_id`: extract NG12 recommendation numbers (e.g., "1.2.1")

3. Generate embeddings and BM25 index (hybrid retrieval):

   - **Dense retrieval (embeddings)**:
     - Use sentence-transformers model (e.g., `all-MiniLM-L6-v2`) or OpenAI embeddings
     - Embed each chunk text
     - Store embeddings in cache (pickle) with chunk metadata
   - **Sparse retrieval (BM25)**:
     - Build BM25 index using `rank-bm25` library or scikit-learn's TfidfVectorizer
     - Index all chunk texts with tokenization (handle clinical terms, abbreviations)
     - Store BM25 index in cache (pickle)

4. Implement hybrid retrieval pipeline:

   - `coarse_routing()`: 
     - Embed query → find top 5-10 candidate sections (section-level similarity)
     - BM25 search → find top sections by keyword match
     - Combine and deduplicate section candidates
   - `fine_retrieval()`: Within candidate sections, retrieve top rule chunks using:
     - **Dense**: Query embedding similarity with chunk embeddings
     - **Sparse**: BM25 score for exact term matches (handles abbreviations, synonyms)
     - **Metadata filters**: age range match, symptom tag match, cancer_site match
     - **Combine scores**: Weighted combination of BM25 score + embedding similarity
   - `rerank_chunks()`: Final reranking using:
     - Combined BM25 + embedding score
     - Metadata match score (age, symptoms, cancer_site)
     - Section depth/specificity (deeper sections = more specific)

5. Cache structure:

   - `data/.cache/custom_chunks.json`: List of NG12Chunk objects (separate from guideline_chunks.json)
   - `data/.cache/custom_embeddings.pkl`: NumPy array of embeddings (separate from guideline_vectors.pkl)
   - `data/.cache/custom_bm25_index.pkl`: BM25 index object (separate from any RAG cache)
   - `data/.cache/custom_metadata.json`: Index mapping chunk_id to metadata
   - **Important**: No sharing with `guideline_service.py` - completely independent parsing, chunking, and caching

**Verification:**

- Test parsing: Verify chunks have correct section_path, metadata extraction works
- Test BM25 index: Query "FIT test" → finds chunks with "FIT" (exact term match)
- Test embeddings: Query "faecal immunochemical testing" → finds same chunks (semantic match)
- Test hybrid: Query "FIT" → BM25 finds exact match, embeddings find semantic variants
- Test coarse routing: Query "oesophageal cancer symptoms" → finds section 1.2 via both methods
- Test fine retrieval: Within section 1.2, combines BM25 + embedding scores for ranking
- Test metadata filtering: Query with age=58 → filters chunks with age_min≤58, age_max≥58
- Test cache: Verify chunks, embeddings, and BM25 index persist and load correctly (independent from RAG cache)
- Verify chunk text is verbatim from final.md (no summarization)
- **Verify independence**: Confirm custom cache files are separate, no imports from guideline_service.py
- **Checkpoint**: Stage 3 verified before proceeding

---

## Stage 4: Evidence Extraction + Confidence + Final Output

**Files to modify:**

- `backend/models/custom_models.py` (add evidence and confidence schemas)
- `backend/services/custom_chat_service.py` (implement Stage 4)

**Implementation:**

1. Create Pydantic schemas:

   - `VerbatimEvidence`: chunk_id (str), text (str), section_path (str), rule_id (str | None), relevance_score (float)
   - `ConfidenceFactors`: retrieval_strength (float), constraint_match (float), evidence_specificity (float), coverage (float)
   - `ConfidenceScore`: overall (float), factors (ConfidenceFactors), threshold_met (bool)
   - `StructuredResponse`: answer (str), citations (list[Citation]), evidence (list[VerbatimEvidence]), confidence (ConfidenceScore), referral_note_draft (str | None)

2. Implement `extract_evidence()`:

   - Take top retrieved chunks from Stage 3
   - Extract verbatim text (no LLM summarization)
   - Map to `VerbatimEvidence` schema
   - Associate with section_path and rule_id from metadata

3. Implement `compute_confidence()`:

   - `retrieval_strength`: Average combined score (BM25 + embedding) of top chunks (0-1)
   - `constraint_match`: Percentage of metadata filters matched (age, symptoms, etc.)
   - `evidence_specificity`: How specific the evidence is (based on section depth, rule_id presence)
   - `coverage`: Whether evidence covers all aspects of query
   - `overall`: Weighted average (e.g., 0.4*retrieval + 0.3*constraint + 0.2*specificity + 0.1*coverage)
   - Threshold: 0.7 (configurable)

4. Implement fail-closed logic:

   - If `confidence.overall < threshold` OR `confidence.threshold_met == False`:
     - Generate follow-up questions based on missing constraints
     - Return `ResponseType.CLARIFICATION` with structured follow-ups
     - Do NOT generate answer
   - If threshold met: proceed to final output generation

5. Implement `generate_final_response()`:

   - Only if confidence threshold met
   - LLM call with:
     - System prompt: "Generate answer grounded ONLY in provided evidence. Cite using [rule_id: section] format."
     - Evidence chunks as context (verbatim)
     - User query + case fields (if case_triage)
   - Parse citations from LLM output (regex: `\[(\d+\.\d+\.\d+): ([^\]]+)\]`)
   - Generate referral note draft if intent == `case_triage` and action_type indicates referral
   - Return `StructuredResponse`

6. Map to `ChatResponse`:

   - `message`: `StructuredResponse.answer`
   - `response_type`: `ANSWER` if threshold met, else `CLARIFICATION` or `REFUSAL`
   - `citations`: Map from `StructuredResponse.citations`
   - `artifacts`: Map from `StructuredResponse.evidence` (with text, section, relevance_score)
   - `follow_up_questions`: From `StructuredResponse` if clarification needed

**Verification:**

- Test high confidence: Complete case with strong evidence → generates answer with citations
- Test low confidence: Incomplete case or weak evidence → returns clarification questions
- Test verbatim evidence: Verify evidence.text is exact substring from final.md
- Test citation format: Verify citations follow `[rule_id: section]` format
- Test fail-closed: Query with insufficient info → no answer, only follow-ups
- Test referral note: Case requiring referral → draft note generated
- Test out-of-scope: Treatment query → `REFUSAL` response with boundary message
- **Checkpoint**: Stage 4 verified before proceeding

---

## Integration & Testing

**Files to modify:**

- `backend/services/custom_chat_service.py` (wire all stages together)
- `backend/main.py` (ensure endpoint works)

**Implementation:**

1. Wire pipeline in `process_message()`:
   ```
   Stage 1: intent_result = classify_intent() + safety_check()
   → If safety gate failed: return escalation response
   
   Stage 2: If intent == case_triage:
     intake_result = extract_case_fields() + validate_intake()
     → If not complete: return clarification with follow-ups
   
   Stage 3: retrieval_result = retrieve_structured_evidence(intent, case_fields)
   
   Stage 4: evidence = extract_evidence(retrieval_result)
            confidence = compute_confidence(evidence, case_fields)
            → If threshold not met: return clarification
            → Else: response = generate_final_response(evidence, confidence)
   
   Return ChatResponse mapped from response
   ```

2. Update streaming to handle staged responses:

   - Stream clarification questions
   - Stream final answer with citations
   - Handle fail-closed cases gracefully

**Verification:**

- End-to-end test: Complete workflow from query to response
- Test all intent types: guideline_lookup, case_triage, documentation
- Test fail-closed paths: safety gate, incomplete intake, low confidence
- Test streaming: Verify SSE events stream correctly
- Performance: Measure latency for each stage
- **Final checkpoint**: Full pipeline verified and tested

---

## Dependencies

**New Python packages needed:**

- `sentence-transformers` (for embeddings) OR `openai` (if using OpenAI embeddings)
- `rank-bm25` (for BM25 sparse retrieval) OR use `scikit-learn`'s TfidfVectorizer
- `numpy` (already in requirements.txt)
- `scikit-learn` (for BM25 alternative or advanced reranking)

**Configuration:**

- Add `CONFIDENCE_THRESHOLD=0.7` to config
- Add `EMBEDDING_MODEL` setting (e.g., "all-MiniLM-L6-v2" or "text-embedding-ada-002")

---

## Testing Strategy

Each stage will have:

1. Unit tests for core functions
2. Integration tests with sample queries
3. Manual verification with demo prompts from `data/demo_prompts.md`
4. Logging for traceability (inputs, outputs, confidence scores at each stage)

**Test cases from demo_prompts.md:**

- Question A: Complete case → should pass all stages, return answer
- Question B: Incomplete case → should fail at Stage 2, return follow-ups
- Question C: Out-of-scope → should fail at Stage 1 safety gate, return refusal

---

## Notes

- **Complete Independence**: This pipeline is entirely separate from the RAG pipeline:
  - No imports from `guideline_service.py` or `rag_chat_service.py`
  - Separate cache files (custom_ *vs guideline_*)
  - Different chunking strategy (hierarchical with metadata vs half-page BoW)
  - Different retrieval method (embeddings vs BoW + cosine similarity)
  - Different data structures (NG12Chunk with metadata vs simple chunk dict)
- All Pydantic models enforce strict validation (fail-closed)
- All evidence must be verbatim (no LLM summarization of chunks)
- Confidence computation is deterministic (no randomness)
- Caching prevents re-parsing final.md on every request
- Logging at each stage for auditability