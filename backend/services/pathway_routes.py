"""
Pathway routes configuration and management.

Defines different chatbot modes/routes with their specific prompts,
behaviors, and data sources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)


class PathwayRouteType(str, Enum):
    """Available pathway route types."""
    
    # General suspected cancer recognition based on NG12
    CANCER_RECOGNITION = "cancer_recognition"
    
    # Symptom-based triage and investigation
    SYMPTOM_TRIAGE = "symptom_triage"
    
    # Referral pathway guidance (2WW, urgent, routine)
    REFERRAL_GUIDANCE = "referral_guidance"
    
    # GraphRAG-powered knowledge graph queries
    GRAPH_RAG = "graph_rag"
    
    # Custom route (for custom chat service)
    CUSTOM = "custom"


@dataclass
class PathwayRoute:
    """Configuration for a pathway route."""
    
    route_type: PathwayRouteType
    name: str
    description: str
    system_prompt: str
    welcome_message: str
    example_prompts: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API."""
        return {
            "route_type": self.route_type.value,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "welcome_message": self.welcome_message,
            "example_prompts": self.example_prompts,
        }


# ============================================================================
# Route Definitions
# ============================================================================

CANCER_RECOGNITION_PROMPT = """You are a clinical decision support assistant for suspected cancer recognition based on NICE NG12.

## Your Role
Help healthcare professionals identify symptoms and presentations that may indicate cancer, organized by cancer site.

## Rules
- Be VERY concise. Use bullet points.
- Cite specific NG12 recommendations: [NG12 1.3.1]
- If info is missing (age, symptoms), ask ONE clarifying question
- Refuse treatment/dosing/diagnosis interpretation queries

## Cancer Sites Covered
- Lung & Pleural (1.1)
- Upper GI: oesophageal, stomach, pancreatic, liver, gall bladder (1.2)
- Lower GI: colorectal, anal (1.3)
- Breast (1.4)
- Gynaecological: ovarian, endometrial, cervical, vulval (1.5)
- Urological: prostate, bladder, renal, testicular, penile (1.6)
- Skin: melanoma, SCC, BCC (1.7)
- Head & neck: laryngeal, oral, thyroid (1.8)
- Brain & CNS (1.9)
- Haematological (1.10)
- Sarcomas (1.11)
- Childhood cancers (1.12)

Keep responses short and actionable."""

SYMPTOM_TRIAGE_PROMPT = """You are a symptom triage assistant for suspected cancer based on NICE NG12.

## Your Role
Help healthcare professionals evaluate symptoms and determine appropriate investigations or referrals.

## Rules
- Be VERY concise. Use bullet points.
- Cite NG12 recommendations: [NG12 1.3.1]
- Always ask for: patient age, symptom duration, associated symptoms
- Provide specific investigation recommendations

## Symptom Categories
- Abdominal symptoms
- Bleeding (haematuria, haematemesis, rectal, vaginal)
- Lumps or masses
- Neurological symptoms
- Pain patterns
- Respiratory symptoms
- Skin changes
- Urological symptoms
- Weight loss / non-specific features

## Key Age Thresholds
- 40+: Lung symptoms, colorectal with weight loss
- 45+: Haematuria → bladder/renal
- 50+: Rectal bleeding, breast nipple changes
- 55+: Upper GI with weight loss
- 60+: Anaemia, persistent UTI

Keep responses focused on symptom evaluation."""

REFERRAL_GUIDANCE_PROMPT = """You are a referral pathway advisor based on NICE NG12.

## Your Role
Guide healthcare professionals on the correct referral pathway for suspected cancer cases.

## Rules
- Be VERY concise. Use bullet points.
- Cite NG12 recommendations
- Specify exact pathway: 2WW, urgent, routine, or direct access
- Include timing requirements

## Referral Types
1. **Suspected Cancer Pathway (2WW)**: First seen within 2 weeks
   - For high suspicion based on symptoms meeting NG12 criteria
   
2. **Urgent Investigation**: Test within 2 weeks
   - Direct access: GP orders, retains responsibility
   - Examples: Chest X-ray, FIT test, ultrasound
   
3. **Urgent Referral**: Appointment within 2 weeks (not cancer pathway)
   - Lower suspicion but needs specialist review
   
4. **Routine Referral**: Standard waiting times
   - For further evaluation, not meeting urgent criteria

## Key Direct Access Tests
- Chest X-ray: Lung symptoms
- FIT (≥10 µg Hb/g): Colorectal symptoms
- Upper GI endoscopy: Dysphagia, upper GI symptoms
- Ultrasound: Abdominal masses, ovarian
- PSA: Prostate symptoms (with age-specific thresholds)

Keep responses focused on pathway selection."""

GRAPH_RAG_PROMPT = """You are a clinical decision support assistant with access to a knowledge graph.

## Your Role
Answer questions using the retrieved context from the knowledge graph. The context includes:
- **Entities**: Key medical concepts, symptoms, conditions, and their relationships
- **Communities**: High-level topic summaries from the graph
- **Chunks**: Source text from clinical guidelines

## Rules
- Base your answers on the provided context
- Cite sources when available
- If context is insufficient, say so clearly
- Be concise and actionable
- Do not make up information not in the context

## Response Format
- Use bullet points for clarity
- Reference specific entities or sources: [Source: ...]
- If multiple sources agree, synthesize them
- Flag any contradictions in sources"""


# Default routes
DEFAULT_ROUTES: list[PathwayRoute] = [
    PathwayRoute(
        route_type=PathwayRouteType.CANCER_RECOGNITION,
        name="Cancer Recognition",
        description="Identify symptoms that may indicate cancer by site",
        system_prompt=CANCER_RECOGNITION_PROMPT,
        welcome_message="I can help identify symptoms that may indicate cancer based on NICE NG12. Which cancer site or symptom would you like to explore?",
        example_prompts=[
            "What symptoms suggest lung cancer?",
            "Red flags for upper GI malignancy?",
            "Melanoma warning signs?",
        ],
    ),
    PathwayRoute(
        route_type=PathwayRouteType.SYMPTOM_TRIAGE,
        name="Symptom Triage",
        description="Evaluate symptoms and determine investigations",
        system_prompt=SYMPTOM_TRIAGE_PROMPT,
        welcome_message="Describe the patient's symptoms and I'll help determine appropriate investigations based on NG12.",
        example_prompts=[
            "55yo with dyspepsia and weight loss",
            "Visible haematuria in a 50yo male",
            "Post-menopausal bleeding",
        ],
    ),
    PathwayRoute(
        route_type=PathwayRouteType.REFERRAL_GUIDANCE,
        name="Referral Pathway",
        description="Determine correct referral pathway and timing",
        system_prompt=REFERRAL_GUIDANCE_PROMPT,
        welcome_message="I can advise on the appropriate referral pathway (2WW, urgent, routine) based on NG12 criteria. What's the clinical scenario?",
        example_prompts=[
            "When to use 2WW vs urgent referral?",
            "FIT positive at 15 µg - next steps?",
            "Direct access chest X-ray criteria?",
        ],
    ),
    PathwayRoute(
        route_type=PathwayRouteType.GRAPH_RAG,
        name="Knowledge Graph",
        description="Query the clinical knowledge graph for context-rich answers",
        system_prompt=GRAPH_RAG_PROMPT,
        welcome_message="I can search the clinical knowledge graph for relevant information. What would you like to know?",
        example_prompts=[
            "What entities relate to colorectal cancer?",
            "Show connections between FIT testing and referral",
            "What does the graph say about 2WW pathways?",
        ],
    ),
]


# ============================================================================
# Route Management Functions
# ============================================================================

def get_route_by_type(route_type: PathwayRouteType) -> PathwayRoute | None:
    """
    Get a pathway route by type.
    
    Args:
        route_type: The route type to retrieve.
        
    Returns:
        PathwayRoute or None if not found.
    """
    for route in DEFAULT_ROUTES:
        if route.route_type == route_type:
            return route
    return None


def get_all_routes() -> list[PathwayRoute]:
    """
    Get all available pathway routes.
    
    Returns:
        List of all PathwayRoute configurations.
    """
    return DEFAULT_ROUTES


def get_route_system_prompt(route_type: PathwayRouteType) -> str:
    """
    Get the system prompt for a specific route.
    
    Args:
        route_type: The route type.
        
    Returns:
        System prompt string.
    """
    route = get_route_by_type(route_type)
    if route:
        return route.system_prompt
    # Fallback to cancer recognition
    return CANCER_RECOGNITION_PROMPT


def save_route_to_db(route: PathwayRoute) -> dict:
    """
    Save a pathway route to the database.
    
    Args:
        route: The route to save.
        
    Returns:
        Saved document.
    """
    try:
        from database.database import insert_document
        return insert_document("pathway_routes", route.to_dict())
    except ImportError:
        logger.warning("Database module not available")
        return route.to_dict()


def get_routes_from_db() -> list[dict]:
    """
    Get all routes from the database.
    
    Returns:
        List of route documents.
    """
    try:
        from database.database import query_documents
        return query_documents("FOR r IN pathway_routes RETURN r")
    except ImportError:
        logger.warning("Database module not available")
        return []


def init_default_routes() -> None:
    """Initialize default routes in the database if empty."""
    try:
        existing = get_routes_from_db()
        if not existing:
            for route in DEFAULT_ROUTES:
                save_route_to_db(route)
            logger.info("Initialized default pathway routes", count=len(DEFAULT_ROUTES))
    except Exception as e:
        logger.warning("Could not initialize routes in DB", error=str(e))
