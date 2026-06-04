# Agent Memory

Durable, high-signal facts and recurring user preferences for this workspace. Maintained by the continual-learning skill.

## User preferences

- Project is intentionally framed as a **personal learning project**, not a product. README must avoid product/marketing tone; clinical-product language ("clinical-grade", official affiliations, sales framing) is corrected on sight.
- User typically asks for **diagnosis before code** — when a user message describes a symptom (e.g. "why am I getting this error"), explain root cause first and only scaffold code after that's confirmed or explicitly requested.
- Prefers **structured side-by-side comparisons** (tables, columned diffs) when explaining alternatives — used repeatedly for RAG vs Custom, .env layout, code-path tracing.
- Treat secrets exposed in chat as **compromised even if Cursor doesn't persist them**. When agent reads a file containing a real API key (e.g. `.env`), warn the user and recommend rotation. Do not echo full or partial keys back into chat output.

## Workspace facts

- **Repo**: `Suspected-Cancer-Clinical-Pathway-Chatbot`. Monorepo layout: `backend/` (FastAPI + Python), `frontend/` (React 19 + Vite + Tailwind), `data/` (parsed NICE NG12 source markdown and `sections_index.json`).
- **`.env` lives at project root**, one level above `backend/`. Backend loads it via `pydantic-settings` with `env_file="../.env"` in `backend/config/config.py` — relative to the process CWD, so uvicorn must be launched from `backend/`.
- **Config caching**: settings are loaded once via `@lru_cache`. Uvicorn `--reload` does **not** pick up `.env` changes; the process must be fully restarted (Ctrl-C, re-run) after edits.
- **API keys are never hard-coded.** `deepseek_api_key` and `openai_api_key` default to `""` in `backend/config/config.py` and load from env vars. The same file redacts both keys in any logged config (`***REDACTED***`). No `.env` file has ever been committed; `.gitignore` excludes `.env*`.
- **Frontend Vite/TS alias**: `@` → `frontend/src` (configured in both `vite.config.ts` and `tsconfig.json`).
- **Vite dev proxy**: `/api` → `http://localhost:8000` (FastAPI).
- **Backend SSE wire format**: `data: {json}\n\n` per event. Envelope `type` is one of `start | chunk | done | error`. The frontend SSE client lives in `frontend/src/lib/api.ts` (`sendChatMessageStream`).
- **Chat endpoints**: `/api/v1/chat/{rag|graphrag|custom}/stream` and `/api/v1/chat/custom/compile`. All three streaming endpoints still exist on the backend; only `custom` is reachable from the UI.
- **Custom Assistant is the only enabled chatbot mode in the UI.** GraphRAG and RAG buttons in `frontend/src/components/ChatSidebar.tsx` are rendered `disabled` with an "Unavailable" chip. Persisted conversations created in those modes still render — the gate is only on new-conversation creation.
- **Retrieval is local**, not API-driven. `backend/services/section_retriever.py` uses `sentence-transformers` for dense embeddings plus `rank-bm25` for lexical, both on-device. OpenAI is only required for:
  - chat completions in `custom_chat_service.py` (the active path), and
  - embeddings in `document_preprocessor.py` / `langgraph_pipeline.py` (part of the disabled RAG mode).
- **RAG mode uses DeepSeek**, not OpenAI, via the OpenAI SDK pointed at `api.deepseek.com`. Files: `rag_chat_service.py`, `chat_service.py`, `rule_engine.py`, `fact_extractor.py`, `symptom_normalizer.py`, `langgraph_pipeline.py:84`.
- **Custom Assistant fails silently** when the LLM call errors. The `except` block in `custom_chat_service.py` (~lines 619–629) returns the verbatim top-2 NG12 sections as the response, so a missing/invalid `OPENAI_API_KEY` looks like a working answer in the UI. The `PathwayTool` is still populated via regex on those section IDs.
- **`PathwayTool` is deterministic**, not LLM-driven at runtime. The LLM emits a `---PATHWAY_CRITERIA_START---` block of recommendation IDs (e.g. `1.2.7`); the criteria checker then evaluates patient input against `criteria_groups` parsed from `sections_index.json` in pure Python.
- **`frontend/src/lib/{utils,api}.ts`** are the canonical home for shared frontend helpers. `utils.ts` exports `cn`, `generateId`, `formatDate`, `truncate`, `parseCitations`. `api.ts` exports `sendChatMessageStream`, `compileRecommendation`, `StreamEvent`, `ApiClientError`. These are consumed by `chatStore.ts`, `ChatWindow.tsx`, `ChatSidebar.tsx`, `DocumentViewer.tsx`, `PathwayTool.tsx`, `LandingPage.tsx`.
- **Frontend deps required by `lib/utils.ts`**: `clsx`, `tailwind-merge`, `date-fns`, `uuid` — all already pinned in `frontend/package.json`.
- **`parseCitations` recognizes only `NG12 <dotted-id>`** patterns today (e.g. `NG12 1.2.7`, `[NG12 1.2.7]`). Other citation shapes pass through unstyled.
- **LLM defaults** in `backend/config/config.py`: `llm_model="gpt-4o-mini"`, `llm_max_tokens=2048`, `llm_temperature=1.3`.

## Conventions

- **Run backend**: `cd backend && python -m uvicorn main:app --reload --port 8000`. Restart fully (not just rely on `--reload`) after editing `.env` or `config.py`.
- **Run frontend**: `cd frontend && npm run dev`. Vite handles the `/api` proxy.
- **`.env` formatting**: one `KEY=value` per line, **no surrounding quotes**, no trailing whitespace, no shell-escape characters. `pydantic-settings`/`python-dotenv` parses values as-typed; quotes and trailing chars get included in the string and silently corrupt API keys.
- **External environment hazard**: a stray `/Users/ary/package-lock.json` exists *outside* this repo and trips up `npm install` from anywhere under `~`. Recommend deleting if `npm` complains about a lockfile path outside the workspace.

## Security notes

- User has previously pasted real OpenAI API keys into chat (via reading `.env` contents). Treat any key that appears in transcript output as compromised — recommend rotation at `https://platform.openai.com/api-keys` and never echo the key (full or partial) back in responses.
