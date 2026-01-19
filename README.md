# Qualified Health

AI-powered clinical decision support for suspected cancer recognition and referral pathways, based on NICE NG12 guideline.

![NICE NG12](https://img.shields.io/badge/NICE-NG12-0ea5e9)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![React](https://img.shields.io/badge/react-19-61dafb)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

Qualified Health is a clinical decision support chatbot for healthcare professionals. It provides instant, evidence-based guidance on suspected cancer recognition and referral based on NICE NG12.

### Cancer Types Covered

- 🫁 **Lung & Pleural** - Chest X-ray criteria, symptoms
- 🍽️ **Upper GI** - Oesophageal, stomach, pancreatic, liver
- 🔴 **Lower GI** - Colorectal (FIT testing), anal
- 🎀 **Breast** - Lump assessment, age criteria
- 👩 **Gynaecological** - Ovarian, endometrial, cervical
- 🚹 **Urological** - Prostate (PSA), bladder, renal, testicular
- 🔆 **Skin** - Melanoma (7-point checklist), SCC, BCC
- 🗣️ **Head & Neck** - Laryngeal, oral, thyroid
- 🧠 **Brain & CNS** - Neurological symptoms
- 🩸 **Haematological** - Lymphoma, leukaemia, myeloma
- 🦴 **Sarcomas** - Bone and soft tissue
- 👶 **Childhood** - Paediatric presentations

### Key Features

- ⚡ **Streaming responses** with stop button
- 📋 **NG12 citations** in every answer with clickable badges that scroll to document sections
- 🎯 **Multi-pass retrieval** - Combines context sections and actionable recommendations with score-based ranking
- 🔍 **Hybrid search** - BM25 + semantic search for accurate section retrieval
- ✅ **Interactive pathway checker** - Validate patient criteria against NG12 recommendations with visual UI
- 🧠 **LLM-powered symptom extraction** - Automatically identifies symptoms from queries (no hardcoded lists)
- 📊 **Reference extraction** - Automatically follows recommendation references from symptom tables
- 🎨 **Beautiful UI** - Modern, clinical-grade interface with smooth animations
- 🚫 **Fail-closed** for treatment/diagnosis queries
- 🔒 **Stateless queries** - Each query is independent (no conversation history)

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- OpenAI API key (for GPT-4o-mini)

### 1. Clone & Configure

```bash
git clone https://github.com/yourusername/qualified-health.git
cd qualified-health

# Create .env in project root
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run with auto-reload
python -m uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:3000**

### 4. Generate Sections Index (First Time)

The system requires a parsed sections index. Generate it from the NG12 markdown:

```bash
cd backend
python scripts/parse_sections.py
```

This creates `data/sections_index.json` with structured sections, criteria, and metadata.

## Project Structure

```
qualified-health/
├── backend/
│   ├── main.py                    # FastAPI app & routes
│   ├── services/
│   │   ├── custom_chat_service.py # Main chat service with multi-pass retrieval
│   │   ├── section_retriever.py   # Hybrid BM25 + semantic search
│   │   ├── section_parser.py      # Parses NG12 markdown into structured sections
│   │   └── ...
│   ├── models/
│   │   └── models.py              # Pydantic schemas
│   ├── config/
│   │   └── config.py              # Environment settings
│   ├── scripts/
│   │   └── parse_sections.py      # CLI to regenerate sections_index.json
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── pages/                 # Landing, Chat pages
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx     # Main chat interface
│   │   │   ├── PathwayTool.tsx    # Interactive criteria checker
│   │   │   └── DocumentViewer.tsx # NG12 document viewer with scroll-to-section
│   │   ├── stores/
│   │   │   └── chatStore.ts       # Zustand state management
│   │   └── lib/                   # API client, utils
│   └── package.json
│
├── data/
│   ├── final.md                   # NICE NG12 guideline source
│   └── sections_index.json        # Parsed sections with criteria (generated)
│
└── .env                           # API keys (not committed)
```

## API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/chat/custom/stream` | POST | Custom chat with section retrieval (SSE streaming) |
| `/api/v1/chat/custom` | POST | Custom chat (non-streaming) |
| `/api/v1/pathway/compile` | POST | Compile recommendation with patient criteria |
| `/api/v1/document/final` | GET | Get NG12 document source |

### Streaming Example

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/custom/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "45yo with visible haematuria, what pathway?"}'
```

### Compile Pathway Example

```bash
curl -X POST http://localhost:8000/api/v1/pathway/compile \
  -H "Content-Type: application/json" \
  -d '{
    "recommendation_id": "1.1.2",
    "patient_criteria": {
      "age": 50,
      "sex": "male",
      "smoking": true,
      "symptoms": ["chest pain"]
    }
  }'
```

## Tech Stack

**Backend:**
- FastAPI + Uvicorn
- OpenAI SDK (GPT-4o-mini)
- Hybrid search: BM25 (rank-bm25) + Semantic (SentenceTransformers)
- Pydantic + Structlog
- Section-based retrieval with structured criteria parsing

**Frontend:**
- React 19 + TypeScript
- Tailwind CSS + Framer Motion
- Zustand for state management
- Vite for builds
- React Markdown for document rendering

## Architecture

### Retrieval System

The system uses a **multi-pass retrieval approach** with score-based ranking:

1. **Pass 1: Context Sections** - Retrieves top 5 general sections (symptom tables, overviews)
2. **Pass 2: Criteria Sections** - Retrieves top 5 sections with actionable criteria (numbered recommendations)
3. **Pass 3: Reference Extraction** - Extracts recommendation IDs (e.g., `[1.1.2] [1.1.5]`) from symptom tables and includes those specific recommendations
4. **Pass 4: Related Recommendations** - Finds related recommendations from same cancer site section (e.g., if 1.1.2 found, also get 1.1.5 for mesothelioma)
5. **Score-based Ranking** - Merges all results, ranks by similarity score, includes ties within 0.15 threshold (max 10 results)

**Hybrid Search:**
- **BM25** (lexical) - 50% weight, handles exact term matches
- **Semantic** (SentenceTransformers) - 50% weight, handles conceptual similarity
- **Criteria boosting** - Sections with symptoms matching query get additional score boost

### Response Generation

- **LLM-powered symptom extraction** - Automatically identifies symptoms from queries (no hardcoded lists)
- **Structured output** - LLM includes `---PATHWAY_CRITERIA_START---` section with:
  - All recommendation IDs it included
  - Extracted symptoms from the query
- **Pathway tool** - Built from LLM's identified recommendations for interactive criteria checking
- **Clickable badges** - All NG12 references (e.g., "NG12 1.1.2") are clickable and scroll to document sections
- **Cancer type detection** - Automatically identifies cancer type from section content for compiled recommendations

### Interactive Pathway Checker

The PathwayTool component allows clinicians to:
- **Input patient criteria**: Age, biological sex, smoking history, presenting symptoms
- **See all relevant symptoms**: Aggregates symptoms from ALL matching recommendations
- **Validate against NG12**: Real-time feedback on whether criteria are met
- **View specific actions**: Shows cancer type and recommended action for each pathway
- **Clean symptom display**: Automatically removes qualifiers and duplicates

### Data Processing

- **Section Parser** (`section_parser.py`) - Parses NG12 markdown into structured sections with:
  - Header hierarchy and breadcrumbs
  - Extracted criteria (age, symptoms, smoking)
  - Cancer site classification
  - Section types (recommendation, symptom_table, definition, etc.)
- **Sections Index** (`sections_index.json`) - Pre-computed index with:
  - 291 total sections
  - 28 sections with actionable criteria
  - BM25 and semantic embeddings pre-computed for fast retrieval

## Safety

⚠️ **Important:**

- This is a **decision-support tool**, not a replacement for clinical judgment
- **Out of scope:** Treatment, medication dosing, diagnostic interpretation
- Always consult full NICE guidelines for complex cases
- **No patient data is stored or transmitted** - queries are stateless (no conversation history)
- Each query is processed independently based solely on the current question and retrieved sections

## License

MIT License

## Acknowledgments

- [NICE](https://www.nice.org.uk/) for NG12 guideline
- Built with OpenAI (GPT-4o-mini), FastAPI, React, and Tailwind CSS
- Uses SentenceTransformers for semantic search and rank-bm25 for lexical search
