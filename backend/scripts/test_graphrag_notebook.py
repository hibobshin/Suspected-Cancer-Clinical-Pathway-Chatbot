"""
Jupyter Notebook script to test ArangoDB GraphRAG endpoint from inside the cluster.

This script can be run in ArangoDB's internal Jupyter notebook to:
1. Test the internal GraphRAG endpoint (deployment.arangodb-platform-rnd-fkd0akd3.svc)
2. Debug authentication issues
3. Inspect responses and logs
4. Test different authentication methods

Usage in Jupyter:
1. Open ArangoDB AI Suite → Notebook servers
2. Create/open a notebook
3. Copy this code into a cell
4. Update credentials if needed
5. Run the cell
"""

import requests
import json
from typing import Dict, Any

# Configuration - update these if needed
ARANGO_USERNAME = "root"
ARANGO_PASSWORD = "r=0:v<i-Mm(Y&3h3"  # Update with your password
ARANGO_DATABASE = "ary_db"
GRAPHRAG_PROJECT = "test"

# Internal endpoint (only works from inside the cluster)
INTERNAL_ENDPOINT = "https://deployment.arangodb-platform-rnd-fkd0akd3.svc:8529/graphrag/retriever/dcajr/v1/graphrag-query"
# Alternative without /v1/graphrag-query suffix
INTERNAL_ENDPOINT_ALT = "https://deployment.arangodb-platform-rnd-fkd0akd3.svc:8529/graphrag/retriever/dcajr/"

def get_jwt_token(base_url: str, username: str, password: str) -> str | None:
    """Get JWT token from ArangoDB /_open/auth endpoint."""
    import requests
    
    auth_url = f"{base_url}/_open/auth"
    try:
        response = requests.post(
            auth_url,
            json={"username": username, "password": password},
            headers={"Content-Type": "application/json"},
            verify=False,  # Internal endpoints may use self-signed certs
            timeout=10,
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("jwt")
        else:
            print(f"Failed to get JWT: {response.status_code} - {response.text[:200]}")
            return None
    except Exception as e:
        print(f"Exception getting JWT: {e}")
        return None

def test_graphrag_endpoint(
    endpoint: str,
    query: str = "What are the symptoms of liver cancer?",
    query_type: str = "UNIFIED",  # "UNIFIED", "LOCAL", or "GLOBAL"
    auth_method: str = "jwt",  # Changed default to JWT
    username: str = None,
    password: str = None,
    database: str = None,
    project: str = None,
) -> Dict[str, Any]:
    """
    Test the GraphRAG endpoint with different authentication methods.
    
    Args:
        endpoint: The GraphRAG endpoint URL
        query: The query to test
        auth_method: 'basic', 'basic_with_db', or 'headers'
        username: ArangoDB username
        password: ArangoDB password
        database: Database name
        project: Project name
        
    Returns:
        Dictionary with response details
    """
    username = username or ARANGO_USERNAME
    password = password or ARANGO_PASSWORD
    database = database or ARANGO_DATABASE
    project = project or GRAPHRAG_PROJECT
    
    # Prepare payload
    # Prepare payload according to ArangoDB Retriever API docs:
    # https://docs.arango.ai/ai-suite/reference/retriever/
    payload = {
        "query": query,
        "query_type": query_type,  # "UNIFIED" (Instant), "LOCAL" (Deep), or "GLOBAL"
    }
    
    # Add optional parameters based on query type
    if query_type == "LOCAL":
        payload["use_llm_planner"] = True  # For Deep Search
    elif query_type == "GLOBAL":
        payload["level"] = 1  # Hierarchy level for Global Search
    
    payload["provider"] = 0  # 0=public LLMs, 1=private LLMs
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
    }
    
    # Add database headers
    if database:
        headers["X-Database"] = database
        headers["X-Arango-Database"] = database
        headers["X-DB-Name"] = database
    
    # Add project headers
    if project:
        headers["X-Project"] = project
        headers["X-Project-Name"] = project
    
    # Prepare authentication
    auth = None
    jwt_token = None
    
    if auth_method == "jwt":
        # Get JWT token from /_open/auth
        base_url = endpoint.split("/graphrag")[0] if "/graphrag" in endpoint else endpoint.split("/ai")[0]
        jwt_token = get_jwt_token(base_url, username, password)
        if jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"
            print(f"✅ Obtained JWT token")
        else:
            print(f"⚠️  Failed to get JWT token, falling back to Basic Auth")
            auth = (username, password)
    elif auth_method == "basic":
        auth = (username, password)
    elif auth_method == "basic_with_db":
        auth = (f"{username}@{database}", password)
    
    # Disable SSL verification for internal endpoints (they often use self-signed certs)
    verify_ssl = False
    
    print(f"\n{'='*80}")
    print(f"Testing GraphRAG Endpoint")
    print(f"{'='*80}")
    print(f"Endpoint: {endpoint}")
    print(f"Auth Method: {auth_method}")
    print(f"Username: {username if auth_method == 'basic' else f'{username}@{database}'}")
    print(f"Database: {database}")
    print(f"Project: {project}")
    print(f"Query: {query}")
    print(f"Query Type: {query_type}")
    print(f"\nPayload: {json.dumps(payload, indent=2)}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print(f"\n{'='*80}\n")
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            auth=auth,
            verify=verify_ssl,
            timeout=30,
        )
        
        result = {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "headers": dict(response.headers),
            "response_text": response.text[:1000] if response.text else None,
        }
        
        # Try to parse JSON response
        try:
            result["response_json"] = response.json()
        except:
            result["response_json"] = None
        
        # Print results
        print(f"Status Code: {response.status_code}")
        print(f"Success: {result['success']}")
        
        if response.status_code == 200:
            print(f"\n✅ SUCCESS!")
            if result["response_json"]:
                print(f"\nResponse Keys: {list(result['response_json'].keys())}")
                print(f"\nResponse Preview:")
                print(json.dumps(result["response_json"], indent=2)[:500])
        elif response.status_code == 401:
            print(f"\n❌ UNAUTHORIZED (401)")
            print(f"Response: {response.text[:500]}")
            print(f"\n💡 Try different auth methods:")
            print(f"  - test_graphrag_endpoint(endpoint, auth_method='basic')")
            print(f"  - test_graphrag_endpoint(endpoint, auth_method='basic_with_db')")
        elif response.status_code == 404:
            print(f"\n❌ NOT FOUND (404)")
            print(f"Response: {response.text[:500]}")
            print(f"\n💡 Try the alternative endpoint without /v1/graphrag-query")
        else:
            print(f"\n❌ ERROR ({response.status_code})")
            print(f"Response: {response.text[:500]}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)}")
        return {
            "status_code": None,
            "success": False,
            "error": str(e),
        }


# Example usage - run these in Jupyter cells:

# Test 1: JWT auth with UNIFIED query (Instant Search)
print("Test 1: JWT Auth with UNIFIED Query (Instant Search)")
result1 = test_graphrag_endpoint(
    INTERNAL_ENDPOINT,
    query="What are the symptoms of liver cancer?",
    query_type="UNIFIED",
    auth_method="jwt",
)

# Test 2: JWT auth with LOCAL query (Deep Search)
print("\n\nTest 2: JWT Auth with LOCAL Query (Deep Search)")
result2 = test_graphrag_endpoint(
    INTERNAL_ENDPOINT,
    query="What are the symptoms of liver cancer?",
    query_type="LOCAL",
    auth_method="jwt",
)

# Test 3: JWT auth with GLOBAL query (Global Search)
print("\n\nTest 3: JWT Auth with GLOBAL Query (Global Search)")
result3 = test_graphrag_endpoint(
    INTERNAL_ENDPOINT,
    query="What are the symptoms of liver cancer?",
    query_type="GLOBAL",
    auth_method="jwt",
)

# Test 4: Fallback to Basic Auth if JWT fails
print("\n\nTest 4: Basic Auth with username@database (Fallback)")
result4 = test_graphrag_endpoint(
    INTERNAL_ENDPOINT,
    query="What are the symptoms of liver cancer?",
    query_type="UNIFIED",
    auth_method="basic_with_db",
)

# You can also test with different queries:
# test_graphrag_endpoint(INTERNAL_ENDPOINT, query="Your query here", auth_method="basic_with_db")
