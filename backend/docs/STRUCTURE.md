# Backend Folder Structure

This document describes the organized structure of the backend folder.

## Directory Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── __init__.py            # Package marker
│
├── config/                # Configuration and logging
│   ├── __init__.py
│   ├── config.py         # Settings and environment variables
│   └── logging_config.py  # Structured logging setup
│
├── models/                # Pydantic data models
│   ├── __init__.py
│   └── models.py         # Request/response models, enums
│
├── services/              # Business logic services
│   ├── __init__.py
│   ├── custom_chat_service.py    # Custom RAG chat service
│   ├── graphrag_chat_service.py  # GraphRAG chat service
│   ├── chat_service.py           # Legacy chat service
│   ├── guideline_service.py       # NICE NG12 guideline retrieval (RAG)
│   ├── graphrag_service.py        # ArangoDB GraphRAG integration
│   └── pathway_routes.py          # Route configuration and prompts
│
├── database/             # Database integration
│   ├── __init__.py
│   └── database.py       # ArangoDB connection and helpers
│
├── docs/                 # Documentation
│   ├── __init__.py
│   ├── README.md
│   ├── CUSTOM_CHAT_FLOW.md
│   ├── TEST_PROXY.md
│   └── STRUCTURE.md      # This file
│
└── scripts/              # Utility and testing scripts
    ├── __init__.py
    └── test_graphrag_notebook.py
```

## Import Patterns

### From main.py
```python
from services.custom_chat_service import get_custom_chat_service
from services.graphrag_chat_service import get_graphrag_chat_service
from config.config import Settings, get_settings
from config.logging_config import configure_logging, get_logger
from models.models import ChatRequest, ChatResponse, ...
from services.pathway_routes import get_all_routes
```

### From services/
```python
from config.config import Settings, get_settings
from config.logging_config import get_logger
from models.models import ChatRequest, ChatResponse, ...
from services.guideline_service import get_guideline_service
from services.graphrag_service import get_graphrag_service
```

### From config/
```python
from config.config import get_settings  # No circular import
```

### From database/
```python
from config.config import get_settings
from config.logging_config import get_logger
```

## Key Principles

1. **Separation of Concerns**: Each folder has a single responsibility
2. **Clear Dependencies**: Services depend on models/config, not vice versa
3. **No Circular Imports**: Config modules don't import each other
4. **Lazy Imports**: Database imports in pathway_routes are lazy to avoid circular deps

## Migration Notes

All imports have been updated to use the new structure:
- `config` → `config.config`
- `logging_config` → `config.logging_config`
- `models` → `models.models`
- Service imports → `services.*`
- Database imports → `database.database`
