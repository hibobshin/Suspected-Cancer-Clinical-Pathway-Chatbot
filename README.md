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
- 📋 **NG12 citations** in every answer
- ❓ **Smart probing** for missing info (age, symptoms)
- 🚫 **Fail-closed** for treatment/diagnosis queries
- 💾 **Conversation history** persisted locally

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- DeepSeek API key

### 1. Clone & Configure

```bash
git clone https://github.com/yourusername/qualified-health.git
cd qualified-health

# Create .env in project root
echo "DEEPSEEK_API_KEY=your-api-key-here" > .env
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

## Project Structure

```
qualified-health/
├── backend/
│   ├── main.py           # FastAPI app & routes
│   ├── chat_service.py   # LLM streaming integration
│   ├── models.py         # Pydantic schemas
│   ├── config.py         # Environment settings
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── pages/        # Landing, Chat pages
│   │   ├── components/   # ChatWindow, Sidebar
│   │   ├── stores/       # Zustand state
│   │   └── lib/          # API client, utils
│   └── package.json
│
├── data/
│   └── final.md          # NICE NG12 guideline source
│
└── .env                  # API keys (not committed)
```

## API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/chat` | POST | Send message (non-streaming) |
| `/api/v1/chat/stream` | POST | Send message (SSE streaming) |

### Streaming Example

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "45yo with visible haematuria, what pathway?"}'
```

## Tech Stack

**Backend:**
- FastAPI + Uvicorn
- OpenAI SDK (DeepSeek compatible)
- Pydantic + Structlog

**Frontend:**
- React 19 + TypeScript
- Tailwind CSS + Framer Motion
- Zustand for state
- Vite for builds

## Safety

⚠️ **Important:**

- This is a **decision-support tool**, not a replacement for clinical judgment
- **Out of scope:** Treatment, medication dosing, diagnostic interpretation
- Always consult full NICE guidelines for complex cases
- No patient data is stored or transmitted

## License

MIT License

## Acknowledgments

- [NICE](https://www.nice.org.uk/) for NG12 guideline
- Built with DeepSeek, FastAPI, React, and Tailwind CSS
