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
        # GraphRAG retriever endpoint
        self.retriever_url = self.settings.graphrag_retriever_url
        self._client: httpx.AsyncClient | None = None
        self._jwt_token: str | None = None
    
    async def _get_jwt_token(self) -> str | None:
        """Get JWT token from ArangoDB authentication endpoint.
        
        According to ArangoDB AI Suite docs, authentication should use:
        1. POST to /_open/auth with username/password
        2. Get JWT token from response
        3. Use JWT as Bearer token in Authorization header
        """
        if self._jwt_token:
            return self._jwt_token
        
        try:
            # Extract base URL from retriever endpoint
            # Handle both internal .svc and external URLs
            if "://" in self.retriever_url:
                base_url = self.retriever_url.split("/graphrag")[0]
                if not base_url:
                    base_url = self.retriever_url.split("/ai")[0]
            else:
                base_url = self.settings.arango_host
            
            # ArangoDB AI Suite uses /_open/auth endpoint
            auth_url = f"{base_url}/_open/auth"
            
            auth_client = httpx.AsyncClient(timeout=10.0, verify=True)
            
            # Disable SSL for internal .svc endpoints
            if ".svc" in auth_url or "deployment.arangodb-platform" in auth_url:
                auth_client = httpx.AsyncClient(timeout=10.0, verify=False)
            
            logger.debug(
                "Requesting JWT token from ArangoDB",
                auth_url=auth_url,
                username=self.settings.arango_username,
            )
            
            # POST to /_open/auth with username/password in JSON body
            auth_response = await auth_client.post(
                auth_url,
                json={
                    "username": self.settings.arango_username,
                    "password": self.settings.arango_password,
                },
                headers={"Content-Type": "application/json"},
            )
            
            if auth_response.status_code == 200:
                result = auth_response.json()
                token = result.get("jwt")
                if token:
                    self._jwt_token = token
                    # Log token info (first/last chars for security)
                    token_preview = f"{token[:20]}...{token[-10:]}" if len(token) > 30 else "***"
                    logger.info(
                        "Successfully obtained JWT token from ArangoDB",
                        token_length=len(token),
                        token_preview=token_preview,
                    )
                    await auth_client.aclose()
                    return token
                else:
                    logger.warning("JWT token not found in auth response", response=result)
            else:
                logger.warning(
                    "Failed to get JWT token from /_open/auth",
                    status=auth_response.status_code,
                    response=auth_response.text[:500],
                    auth_url=auth_url,
                )
            
            await auth_client.aclose()
            return None
            
        except Exception as e:
            logger.warning("JWT token retrieval failed", error=str(e))
            return None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            # Start with BasicAuth, will try JWT if needed
            username = self.settings.arango_username
            
            auth = httpx.BasicAuth(
                username=username,
                password=self.settings.arango_password,
            )
            
            # Internal Kubernetes .svc endpoints typically don't have valid SSL certs
            # External endpoints should verify SSL
            is_internal = ".svc" in self.retriever_url or "deployment.arangodb-platform" in self.retriever_url
            verify_ssl = not is_internal
            
            logger.debug(
                "Creating GraphRAG HTTP client",
                username=username,
                database=self.settings.arango_database,
                endpoint=self.retriever_url,
                is_internal=is_internal,
                verify_ssl=verify_ssl,
            )
            
            self._client = httpx.AsyncClient(
                timeout=30.0,
                verify=verify_ssl,
                auth=auth,
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
            endpoint=self.retriever_url,
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
            
            payload = {
                "query": query,
                "query_type": query_type_int,  # Integer: 1=GLOBAL, 2=LOCAL, 3=UNIFIED
            }
            
            # Add optional parameters based on query type
            if query_type_int == 1:  # GLOBAL - Global Search
                payload["level"] = level
            elif query_type_int == 2:  # LOCAL - Deep Search
                payload["use_llm_planner"] = use_llm_planner
            # UNIFIED (3) doesn't need extra params
            
            # Provider (0=public LLMs, 1=private LLMs)
            payload["provider"] = provider
            
            # Note: Project name is not in the API payload according to docs
            # The service_id in the URL path identifies the retriever service
            
            headers = {
                "Content-Type": "application/json",
            }
            
            # Try JWT token authentication first (ArangoDB AI Suite standard)
            # According to docs: https://docs.arango.ai/ai-suite/reference/ai-orchestrator/
            jwt_token = await self._get_jwt_token()
            if jwt_token:
                headers["Authorization"] = f"Bearer {jwt_token}"
                logger.info(
                    "Using JWT Bearer token for authentication",
                    token_length=len(jwt_token),
                    endpoint=self.retriever_url,
                )
            else:
                logger.warning("JWT token not available, will use Basic Auth as fallback")
            
            # Database headers (may be required for some ArangoDB endpoints)
            # Note: According to Retriever API docs, authentication is via Bearer token
            # Database context might be in the service_id or handled automatically
            if self.settings.arango_database:
                headers["X-Database"] = self.settings.arango_database
                headers["X-Arango-Database"] = self.settings.arango_database
            
            # Determine the actual request URL
            # Internal .svc endpoints might not need /v1/graphrag-query suffix
            request_url = self.retriever_url
            is_internal = ".svc" in request_url or "deployment.arangodb-platform" in request_url
            
            # If internal endpoint doesn't have /v1/graphrag-query, prepare alternative
            request_url_with_suffix = None
            if is_internal and "/v1/graphrag-query" not in request_url:
                request_url_with_suffix = f"{request_url.rstrip('/')}/v1/graphrag-query"
            
            # Log the exact request being made for debugging
            logger.debug(
                "GraphRAG request (attempt 1)",
                url=request_url,
                is_internal=is_internal,
                payload=payload,  # Log actual payload to see integer query_type
                payload_keys=list(payload.keys()),
                headers_keys=list(headers.keys()),
                username=self.settings.arango_username,
                database=self.settings.arango_database,
                project=self.settings.graphrag_project_name,
                using_jwt=bool(jwt_token),
            )
            
            # Try the URL as-is first
            response = await self.client.post(
                request_url,
                json=payload,
                headers=headers,
            )
            
            # If 404 and internal endpoint, try with /v1/graphrag-query suffix
            if response.status_code == 404 and is_internal and request_url_with_suffix:
                logger.debug("404 on internal endpoint, trying with /v1/graphrag-query suffix")
                response = await self.client.post(
                    request_url_with_suffix,
                    json=payload,
                    headers=headers,
                )
            
            logger.debug(
                "GraphRAG request details (attempt 1)",
                url=self.retriever_url,
                status_code=response.status_code,
                response_text=response.text[:500] if response.status_code != 200 else None,
            )
            
            # If 401, log detailed error information
            if response.status_code == 401:
                error_detail = None
                try:
                    error_json = response.json()
                    error_detail = error_json
                except:
                    error_detail = response.text[:1000]
                
                logger.error(
                    "401 Unauthorized - JWT token may be invalid or insufficient permissions",
                    endpoint=request_url,
                    has_jwt_token=bool(jwt_token),
                    jwt_token_length=len(jwt_token) if jwt_token else 0,
                    error_detail=error_detail,
                    headers_sent=list(headers.keys()),
                    payload_keys=list(payload.keys()),
                    username=self.settings.arango_username,
                    database=self.settings.arango_database,
                    service_id="dcajr",
                    note="If JWT token is valid, user may lack permissions for GraphRAG retriever service",
                )
                
                # If JWT was used but failed, try getting a fresh token
                if jwt_token:
                    logger.info("JWT token failed, invalidating and trying fresh token")
                    self._jwt_token = None  # Invalidate cached token
                    fresh_jwt = await self._get_jwt_token()
                    if fresh_jwt and fresh_jwt != jwt_token:
                        logger.info("Got fresh JWT token, retrying request")
                        headers["Authorization"] = f"Bearer {fresh_jwt}"
                        response = await self.client.post(
                            request_url,
                            json=payload,
                            headers=headers,
                        )
                        if response.status_code == 200:
                            logger.info("Fresh JWT token worked!")
                        else:
                            logger.warning("Fresh JWT token also failed", status=response.status_code)
                
                logger.info("401 received, trying alternative authentication methods")
                
                # Attempt 2: username@database format
                username_with_db = f"{self.settings.arango_username}@{self.settings.arango_database}"
                retry_client = httpx.AsyncClient(
                    timeout=30.0,
                    verify=True,
                    auth=httpx.BasicAuth(
                        username=username_with_db,
                        password=self.settings.arango_password,
                    ),
                )
                
                try:
                    retry_headers = headers.copy()
                    if "Authorization" in retry_headers:
                        del retry_headers["Authorization"]
                    
                    logger.debug(
                        "GraphRAG request (attempt 2 - username@database)",
                        url=self.retriever_url,
                        username=username_with_db,
                    )
                    
                    response = await retry_client.post(
                        self.retriever_url,
                        json=payload,
                        headers=retry_headers,
                    )
                    
                    logger.debug(
                        "GraphRAG request details (attempt 2)",
                        status_code=response.status_code,
                        response_text=response.text[:500] if response.status_code != 200 else None,
                    )
                    
                    # If still 401, try with database in URL path
                    if response.status_code == 401 and self.settings.arango_database:
                        logger.info("Still 401, trying database in URL path")
                        # Try: /graphrag/retriever/{service}/{database}/v1/graphrag-query
                        url_parts = self.retriever_url.split("/graphrag/retriever/")
                        if len(url_parts) == 2:
                            service_part = url_parts[1]
                            # Insert database before /v1/
                            if "/v1/" in service_part:
                                service_id = service_part.split("/v1/")[0]
                                new_url = f"{url_parts[0]}/graphrag/retriever/{service_id}/{self.settings.arango_database}/v1/graphrag-query"
                                
                                logger.debug(
                                    "GraphRAG request (attempt 3 - database in URL)",
                                    url=new_url,
                                    username=username_with_db,
                                )
                                
                                response = await retry_client.post(
                                    new_url,
                                    json=payload,
                                    headers=retry_headers,
                                )
                                
                                logger.debug(
                                    "GraphRAG request details (attempt 3)",
                                    status_code=response.status_code,
                                    response_text=response.text[:500] if response.status_code != 200 else None,
                                )
                finally:
                    await retry_client.aclose()
            
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
                url=self.retriever_url,
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
