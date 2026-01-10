# Custom Chatbot Flow Documentation

## Complete Flow: User Message → Response

### 1. **Frontend: User Input** (`ChatWindow.tsx` → `chatStore.ts`)
```
User types message → Clicks send/Enter
↓
chatStore.sendMessage() called
↓
Determines route: solutionMode='custom' → endpoint='custom'
↓
Calls sendChatMessageStream() from api.ts
```

**Key files:**
- `frontend/src/components/ChatWindow.tsx` - UI input handler
- `frontend/src/stores/chatStore.ts` - State management, route determination
- `frontend/src/lib/api.ts` - API client, sends POST to `/api/v1/chat/custom/stream`

**Data sent:**
```typescript
{
  message: "user's question",
  route_type: "cancer_recognition" | "symptom_triage" | "referral_guidance",
  conversation_id: "uuid",
  context: { conversation_id, messages: [...] }  // Optional conversation history
}
```

---

### 2. **Backend: API Route** (`main.py`)
```
POST /api/v1/chat/custom/stream
↓
chat_custom_stream() endpoint handler
↓
Gets CustomChatService instance
↓
Returns StreamingResponse from process_message_stream()
```

**Key file:** `backend/main.py` (lines 271-286)

---

### 3. **Backend: Custom Chat Service** (`custom_chat_service.py`)

#### Step 3.1: Initialize & Retrieve Guidelines
```
process_message_stream() called
↓
Create conversation_id (use existing or generate new UUID)
↓
Call guideline_service.search() to find relevant sections
```

**Guideline Search Process** (`guideline_service.py`):
1. **Load guideline file** (`data/final.md` - NICE NG12)
   - Parses markdown into sections by headings (##, ###)
   - Caches content in memory
   
2. **Search algorithm** (simple keyword matching):
   ```
   - Extract query terms (words) from user message
   - For each section in guideline:
     * Score = count of matching terms
     * +5 points if exact phrase match
     * +3 points if section name matches
   - Sort sections by score (descending)
   - Take top max_chunks * 2 sections (default: 6)
   - Split sections into chunks (~500 chars each)
   - Deduplicate chunks
   - Return top max_chunks (default: 3)
   ```

3. **Format artifacts for LLM**:
   - Creates Artifact objects with:
     - `section`: Section name (e.g., "Upper GI")
     - `text`: Chunk content
     - `relevance_score`: Calculated score
     - `source`: "NICE NG12"
     - `source_url`: "https://www.nice.org.uk/guidance/ng12"
   - Formats as context string for LLM prompt

**Key files:**
- `backend/custom_chat_service.py` (lines 214-235)
- `backend/guideline_service.py` (lines 91-171)

---

#### Step 3.2: Build LLM Messages
```
_build_messages() called
↓
Gets route-specific system prompt from pathway_routes.py
↓
Constructs message array:
  1. System message (route-specific prompt + guideline context)
  2. Conversation history (if provided)
  3. User's current message
```

**Route-Specific Prompts** (`pathway_routes.py`):
- **CANCER_RECOGNITION**: Focus on identifying symptoms by cancer site
- **SYMPTOM_TRIAGE**: Help evaluate symptoms and determine investigations
- **REFERRAL_GUIDANCE**: Determine correct referral pathway (2WW, urgent, routine)

**System prompt format:**
```
{route_specific_prompt}

## Guidelines Context:
{formatted_guideline_artifacts}

Rules:
- Be VERY concise. Use bullet points.
- Cite specific NG12 recommendations: [NG12 1.3.1]
- If info is missing, ask ONE clarifying question
- Refuse treatment/dosing/diagnosis queries
```

**Key files:**
- `backend/custom_chat_service.py` (lines 238, 300-350)
- `backend/pathway_routes.py` - System prompts

---

#### Step 3.3: Call LLM (DeepSeek API)
```
AsyncOpenAI client calls DeepSeek API
↓
POST to https://api.deepseek.com/v1/chat/completions
↓
Streaming response: chunks arrive one by one
↓
Yield each chunk as SSE event
```

**Configuration:**
- Model: `deepseek-chat` (from config)
- Max tokens: 1000 (default)
- Temperature: 0.7 (default)
- Stream: `True`

**Key files:**
- `backend/custom_chat_service.py` (lines 251-258)
- `backend/config.py` - DeepSeek API settings

---

#### Step 3.4: Stream Response to Frontend
```
For each chunk from LLM:
  yield f"data: {json.dumps({'type': 'chunk', 'content': char})}\n\n"
↓
After streaming completes:
  - Extract citations from response text
  - Classify response type (answer/clarification/refusal/error)
  - Generate follow-up questions
  - Yield final 'done' event with metadata
```

**SSE Event Types:**
1. **`start`**: Conversation ID
2. **`chunk`**: Each character/token from LLM
3. **`done`**: Final metadata (citations, artifacts, response_type, processing_time_ms)
4. **`error`**: Error message (if failed)

**Response Classification** (`_classify_response()`):
- Looks for keywords to determine:
  - `answer`: Grounded response with citations
  - `clarification`: Asking for more information
  - `refusal`: Out of scope query
  - `error`: Technical error

**Citation Extraction** (`_extract_citations()`):
- Parses response for citation patterns like `[NG12 1.3.1]` or `[QS124-S2]`
- Creates Citation objects

**Key files:**
- `backend/custom_chat_service.py` (lines 258-305)
- `backend/custom_chat_service.py` (lines 320-380) - Helper methods

---

### 4. **Frontend: Receive & Display** (`api.ts` → `chatStore.ts` → `ChatWindow.tsx`)

```
SSE stream received
↓
Parse each SSE event
↓
onChunk callback: Update message content in real-time (character by character)
↓
onComplete callback: Finalize message with metadata
  - Update response_type
  - Set citations
  - Set artifacts (for traceability)
↓
chatStore updates state
↓
ChatWindow.tsx re-renders with:
  - Message content (formatted markdown)
  - Citations (badges)
  - Artifacts (expandable sections, if showArtifacts=true)
  - Response type indicator (badge)
  - Follow-up questions (if any)
```

**Artifacts Display:**
- Only shown when `solutionMode === 'custom'` and `showArtifacts === true`
- Expandable/collapsible sections showing:
  - Section name
  - Source text used
  - Relevance score
  - Source URL link
- Smooth Framer Motion animations when toggling

**Key files:**
- `frontend/src/lib/api.ts` - SSE parser
- `frontend/src/stores/chatStore.ts` - State updates
- `frontend/src/components/ChatWindow.tsx` - UI rendering

---

## Data Flow Summary

```
User Input
  ↓
[Frontend] chatStore.sendMessage()
  ↓
[Frontend] api.sendChatMessageStream() → POST /api/v1/chat/custom/stream
  ↓
[Backend] main.py → chat_custom_stream()
  ↓
[Backend] CustomChatService.process_message_stream()
  ├─→ GuidelineService.search() → Find relevant sections from data/final.md
  ├─→ Format artifacts for LLM context
  ├─→ Build messages with route-specific system prompt
  ├─→ Call DeepSeek API (streaming)
  └─→ Yield SSE events (chunk, done)
  ↓
[Frontend] Parse SSE stream
  ├─→ onChunk: Update message content (streaming)
  └─→ onComplete: Set metadata (citations, artifacts, response_type)
  ↓
[Frontend] ChatWindow.tsx renders
  ├─→ Message content (markdown formatted)
  ├─→ Citations (badges)
  ├─→ Artifacts (if toggled on) ← Traceability!
  └─→ Response type indicator
```

---

## Key Components

### Backend Services:
1. **CustomChatService** (`custom_chat_service.py`)
   - Orchestrates the flow
   - Handles LLM interaction
   - Formats responses

2. **GuidelineService** (`guideline_service.py`)
   - Loads `data/final.md` (NICE NG12)
   - Simple keyword-based search
   - Returns traceable artifacts

3. **PathwayRoutes** (`pathway_routes.py`)
   - Defines route types and system prompts
   - Provides welcome messages and examples

### Frontend Components:
1. **ChatWindow** (`ChatWindow.tsx`)
   - Main UI component
   - Handles input, displays messages
   - Shows artifacts toggle (custom mode only)

2. **chatStore** (`chatStore.ts`)
   - Zustand state management
   - Handles conversation state
   - Determines which endpoint to call

3. **api.ts**
   - SSE client
   - Handles streaming responses

---

## Current Limitations & Potential Improvements

1. **Search Algorithm**: Simple keyword matching → Could use embeddings/semantic search
2. **Chunking**: Fixed 500 char chunks → Could use smarter chunking
3. **Scoring**: Basic term counting → Could use TF-IDF or similarity scoring
4. **No caching**: Guideline loaded every time → Could cache parsed sections
5. **No conversation memory**: Only last 10 messages → Could use vector store
