"""
GraphRAG service for ArangoDB integration.

Uses ArangoDB's GraphRAG retriever for context-augmented responses.
"""

import base64
import httpx
from typing import Any

from config.config import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class GraphRAGService:
    """
    Service for querying ArangoDB's GraphRAG retriever.
    
    Retrieves relevant context from the knowledge graph and uses it
    to augment LLM responses.
    """
    
    def __init__(self):
        self.settings = get_settings()
        # Exact notebook code: SERVER_URL = os.environ['ARANGO_DEPLOYMENT_ENDPOINT']
        self.server_url = self.settings.arango_deployment_endpoint.rstrip('/')
        
        # Service ID from settings
        self.service_id = self.settings.graphrag_service_id
        
        # Build endpoint exactly like notebook: f"/graphrag/retriever/{retriever_service_id}/v1/graphrag-query"
        # Notebook uses: send_request(f"/graphrag/retriever/{retriever_service_id}/v1/graphrag-query", myBody, "POST")
        # And send_request constructs: url = f"{SERVER_URL}{suffix}"
        self.retriever_endpoint = f"{self.server_url}/graphrag/retriever/{self.service_id}/v1/graphrag-query"
        
        self._client: httpx.AsyncClient | None = None
        self._jwt_token: str | None = None
    
    async def _get_jwt_token(self) -> str | None:
        """Get JWT token from ArangoDB authentication endpoint.
        
        Exact code from notebook: ArangoGraphRAG_Advanced.ipynb
        """
        if self._jwt_token:
            return self._jwt_token
        
        # Exact notebook code: auth_url = f"{SERVER_URL}/_open/auth"
        auth_url = f"{self.server_url}/_open/auth"
        
        # Validate credentials are set
        if not self.settings.arango_username:
            raise ValueError("ArangoDB username is not set. Please set ARANGODB_USERNAME in environment variables.")
        if not self.settings.arango_password:
            raise ValueError("ArangoDB password is not set. Please set ARANGODB_PASSWORD in environment variables.")
        
        payload = {
            "username": self.settings.arango_username,
            "password": self.settings.arango_password
        }
        
        try:
            # Log what we're actually using (without password value)
            logger.info(
                "Authenticating with ArangoDB",
                auth_url=auth_url,
                username=self.settings.arango_username,
                password_length=len(self.settings.arango_password) if self.settings.arango_password else 0,
                password_set=bool(self.settings.arango_password),
            )
            
            # Exact notebook code: response = requests.post(auth_url, json=payload, verify=False)
            # Note: httpx sets verify=False on the client, not on the request
            auth_client = httpx.AsyncClient(timeout=10.0, verify=False)
            auth_response = await auth_client.post(auth_url, json=payload)
            
            # Check response before raising
            if auth_response.status_code == 401:
                error_detail = auth_response.json() if auth_response.headers.get("content-type", "").startswith("application/json") else auth_response.text
                logger.error(
                    "Authentication failed: Wrong credentials",
                    auth_url=auth_url,
                    username=self.settings.arango_username,
                    error_detail=error_detail,
                    note="Please verify ARANGODB_USERNAME and ARANGODB_PASSWORD are correct in your environment variables",
                )
            
            auth_response.raise_for_status()
            
            # Exact notebook code: jwt_token = response.json().get("jwt")
            result = auth_response.json()
            token = result.get("jwt")
            
            if not token:
                raise ValueError("Authentication response does not contain a token.")
            
            self._jwt_token = token
            await auth_client.aclose()
            logger.info("Authentication successful. JWT token retrieved.")
            return token
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Error during authentication: {e}")
            raise
        except ValueError as ve:
            logger.error(f"Error processing authentication response: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            raise
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (notebook pattern: verify=False)."""
        if self._client is None:
            # Notebook uses verify=False for all requests
            self._client = httpx.AsyncClient(
                timeout=30.0,
                verify=False,
            )
        return self._client
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_type: str | int = "UNIFIED",
        level: int = 1,
        provider: int = 0,
        use_llm_planner: bool = False,
    ) -> dict[str, Any]:
        """
        Retrieve relevant context from the GraphRAG retriever.
        
        According to ArangoDB AI Suite Retriever API docs:
        https://docs.arango.ai/ai-suite/reference/retriever/
        
        Args:
            query: The natural language question to ask.
            top_k: Number of results to retrieve (legacy param, may be ignored).
            query_type: Query type - "UNIFIED" (3, Instant Search), "LOCAL" (2, Deep Search), 
                       or "GLOBAL" (1, Global Search). Can be string or integer.
                       Default: "UNIFIED" (3)
            level: Hierarchy level for community grouping (for GLOBAL search, recommended: 1).
            provider: LLM provider - 0 (public LLMs like OpenAI) or 1 (private LLMs).
            use_llm_planner: Whether to use LLM planner (set to True for Deep Search).
            
        Returns:
            Retrieved context with entities, communities, and chunks.
        """
        # Convert query_type to integer for logging
        query_type_for_log = query_type
        if isinstance(query_type, str):
            query_type_map = {"GLOBAL": 1, "LOCAL": 2, "UNIFIED": 3}
            query_type_for_log = f"{query_type} ({query_type_map.get(query_type.upper(), 3)})"
        
        logger.info(
            "GraphRAG retrieval request",
            query=query[:100],
            query_type=query_type_for_log,
            level=level,
            database=self.settings.arango_database,
            username=self.settings.arango_username,
            endpoint=self.retriever_endpoint,
        )
        
        try:
            # GraphRAG Retriever API format according to documentation:
            # https://docs.arango.ai/ai-suite/reference/retriever/
            # API accepts both integer and string, but examples show integers
            # Mapping: 1=GLOBAL, 2=LOCAL, 3=UNIFIED
            query_type_int = 3  # Default: UNIFIED (Instant Search) = 3
            if isinstance(query_type, str):
                # Convert string to integer if needed
                query_type_map = {
                    "GLOBAL": 1,   # Global Search - aggregates across entire graph
                    "LOCAL": 2,    # Deep Search - specific entities/relationships
                    "UNIFIED": 3,  # Instant Search - fast, streamed response
                }
                query_type_int = query_type_map.get(query_type.upper(), 3)
            elif isinstance(query_type, int):
                query_type_int = query_type
            
            # Build payload matching notebook pattern exactly
            # Notebook always includes: query, query_type, level, provider
            payload = {
                "query": query,
                "query_type": query_type_int,  # Integer: 1=GLOBAL, 2=LOCAL, 3=UNIFIED
                "level": level,  # Always include (notebook pattern)
                "provider": provider,  # Always include (0=public LLMs, 1=private LLMs)
            }
            
            # Add optional parameters based on query type
            if query_type_int == 2:  # LOCAL - Deep Search
                payload["use_llm_planner"] = use_llm_planner
            
            # Note: Project name is not in the API payload according to docs
            # The service_id in the URL path identifies the retriever service
            
            # Exact notebook code: headers with JWT token
            jwt_token = await self._get_jwt_token()
            if not jwt_token:
                raise ValueError("JWT token is not set. Please authenticate first.")
            
            headers = {
                "Authorization": f"Bearer {jwt_token}",
                "Content-Type": "application/json"
            }
            
            # Exact notebook code: send_request(f"/graphrag/retriever/{retriever_service_id}/v1/graphrag-query", myBody, "POST")
            request_url = self.retriever_endpoint
            
            logger.debug(
                "GraphRAG request (exact notebook pattern)",
                url=request_url,
                service_id=self.service_id,
                payload=payload,
            )
            
            # Make request (notebook uses verify=False)
            response = await self.client.post(
                request_url,
                json=payload,
                headers=headers,
            )
            
            response.raise_for_status()
            
            result = response.json()
            logger.info(
                "GraphRAG retrieval success",
                response_keys=list(result.keys()),
                response_preview=str(result)[:200],
            )
            return result
            
        except httpx.HTTPStatusError as e:
            error_detail = None
            if e.response:
                try:
                    error_json = e.response.json()
                    error_detail = error_json
                except:
                    error_detail = e.response.text[:1000]
            else:
                error_detail = str(e)
            
            logger.error(
                "GraphRAG retrieval HTTP error",
                status_code=e.response.status_code if e.response else None,
                url=self.retriever_endpoint,
                detail=error_detail,
                username=self.settings.arango_username,
                database=self.settings.arango_database,
                headers_sent=list(headers.keys()) if 'headers' in locals() else None,
            )
            return {"entities": [], "communities": [], "chunks": [], "error": f"HTTP {e.response.status_code if e.response else 'unknown'}: {error_detail}"}
        except Exception as e:
            logger.error("GraphRAG retrieval error", error=str(e))
            return {"entities": [], "communities": [], "chunks": [], "error": str(e)}
    
    
    def format_context(self, retrieval_result: dict[str, Any]) -> str:
        """
        Format retrieved context for LLM consumption.
        
        Handles different response formats from GraphRAG API.
        
        Args:
            retrieval_result: The raw retrieval result.
            
        Returns:
            Formatted context string.
        """
        # Check if there's an error
        if "error" in retrieval_result:
            return f"Error retrieving context: {retrieval_result['error']}"
        
        sections = []
        
        # GraphRAG query endpoint may return different formats
        # Try common response fields
        if "answer" in retrieval_result:
            # Direct answer format
            return retrieval_result.get("answer", "")
        
        if "result" in retrieval_result:
            # Nested result format
            result_data = retrieval_result["result"]
            if isinstance(result_data, str):
                return result_data
            retrieval_result = result_data
        
        # Format entities (if present)
        entities = retrieval_result.get("entities", [])
        if entities:
            entity_lines = []
            for entity in entities[:10]:  # Limit to top 10
                if isinstance(entity, dict):
                    name = entity.get("name", entity.get("_key", "Unknown"))
                    etype = entity.get("type", entity.get("label", "Entity"))
                    desc = entity.get("description", "")
                    if desc:
                        entity_lines.append(f"- **{name}** ({etype}): {desc}")
                    else:
                        entity_lines.append(f"- **{name}** ({etype})")
            if entity_lines:
                sections.append("### Relevant Entities\n" + "\n".join(entity_lines))
        
        # Format communities (high-level summaries)
        communities = retrieval_result.get("communities", [])
        if communities:
            community_lines = []
            for comm in communities[:5]:  # Limit to top 5
                if isinstance(comm, dict):
                    title = comm.get("title", comm.get("name", "Topic"))
                    summary = comm.get("summary", comm.get("description", ""))
                    if summary:
                        community_lines.append(f"- **{title}**: {summary}")
            if community_lines:
                sections.append("### Topic Summaries\n" + "\n".join(community_lines))
        
        # Format chunks (source text)
        chunks = retrieval_result.get("chunks", [])
        if chunks:
            chunk_lines = []
            for chunk in chunks[:5]:  # Limit to top 5
                if isinstance(chunk, dict):
                    text = chunk.get("text", chunk.get("content", ""))
                    source = chunk.get("source", chunk.get("document", ""))
                    if text:
                        preview = text[:300] + "..." if len(text) > 300 else text
                        if source:
                            chunk_lines.append(f"[{source}]: {preview}")
                        else:
                            chunk_lines.append(preview)
            if chunk_lines:
                sections.append("### Source Context\n" + "\n".join(chunk_lines))
        
        # If no structured data, try to return the whole response as text
        if not sections:
            # Return the response as-is if it's a string, or convert to string
            if isinstance(retrieval_result, str):
                return retrieval_result
            # Try to extract any text content
            if "content" in retrieval_result:
                return str(retrieval_result["content"])
            if "text" in retrieval_result:
                return str(retrieval_result["text"])
            # Last resort: return JSON string
            import json
            return json.dumps(retrieval_result, indent=2)
        
        return "\n\n".join(sections)
    
    async def query_with_context(
        self,
        query: str,
        top_k: int = 5,
        query_type: str = "UNIFIED",
    ) -> tuple[str, dict[str, Any]]:
        """
        Retrieve context and format it for LLM augmentation.
        
        Args:
            query_type: "UNIFIED" (Instant Search), "LOCAL" (Deep Search), or "GLOBAL" (Global Search)
            query: User query.
            top_k: Number of results (legacy param, may be ignored).
            query_type: Query type - 1 (GLOBAL), 2 (LOCAL), or 3 (UNIFIED).
            
        Returns:
            Tuple of (formatted_context, raw_result).
        """
        result = await self.retrieve(query, top_k=top_k, query_type=query_type)
        context = self.format_context(result)
        return context, result
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_graphrag_service: GraphRAGService | None = None


def get_graphrag_service() -> GraphRAGService:
    """Get the GraphRAG service singleton."""
    global _graphrag_service
    if _graphrag_service is None:
        _graphrag_service = GraphRAGService()
    return _graphrag_service
