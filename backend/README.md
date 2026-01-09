# Qualified Health Backend

FastAPI backend for the Qualified Health clinical decision support system.

## Features

- **RESTful API** with OpenAPI documentation
- **Structured logging** for observability and audit trails
- **LLM integration** with OpenAI (GPT-4o)
- **Fail-closed design** for out-of-scope queries
- **Citation extraction** from responses
- **CORS support** for frontend integration

## Setup

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or with uv (faster)
uv pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your settings:

```env
# Application
APP_NAME=Qualified Health
ENVIRONMENT=development
DEBUG=true

# OpenAI
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=console  # or "json" for production
```

### Running

```bash
# Development (with auto-reload)
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --port 8000
```

## API Documentation

When running in debug mode, OpenAPI docs are available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
backend/
├── main.py           # FastAPI app and routes
├── models.py         # Pydantic request/response models
├── chat_service.py   # LLM integration and chat logic
├── config.py         # Settings and configuration
├── logging_config.py # Structured logging setup
├── requirements.txt  # Python dependencies
└── __init__.py
```

## Logging

All operations are logged with structured context:

```python
logger.info(
    "Chat response generated",
    conversation_id=str(conversation_id),
    response_type=response_type.value,
    citations_count=len(citations),
    processing_time_ms=processing_time,
)
```

In production (`LOG_FORMAT=json`), logs are JSON-formatted for aggregation systems.

## Response Types

| Type | Description | Example Trigger |
|------|-------------|-----------------|
| `answer` | Grounded response with citations | "What are the 2WW criteria?" |
| `clarification` | Probing for missing info | "Should I order a FIT test?" (no age given) |
| `refusal` | Out-of-scope refusal | "What treatment should I prescribe?" |
| `error` | System/API error | Network issues, rate limits |

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test
pytest tests/test_chat_service.py -v
```

## Code Quality

```bash
# Format
black .

# Lint
ruff check .

# Type check
mypy .
```
