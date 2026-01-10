# Testing the Notebook Proxy

## Quick Test

Run this in a NEW notebook cell to test if the proxy is accessible:

```python
import requests

# Test 1: Health check from inside notebook (should work)
try:
    health = requests.get("http://127.0.0.1:8889/health", timeout=5)
    print(f"✅ Internal health check: {health.status_code}")
    print(f"Response: {health.json()}")
except Exception as e:
    print(f"❌ Internal health check failed: {e}")

# Test 2: GraphRAG query from inside notebook
try:
    response = requests.post(
        "http://127.0.0.1:8889/graphrag/query",
        json={
            "query": "What are the symptoms of liver cancer?",
            "query_type": 3,
            "provider": 0
        },
        timeout=30
    )
    print(f"\n✅ Query test: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response keys: {list(result.keys())}")
        print(f"Result preview: {str(result.get('result', ''))[:200]}")
    else:
        print(f"Error: {response.text[:500]}")
except Exception as e:
    print(f"❌ Query test failed: {e}")
```

## Finding the External URL

The notebook server URL is: `https://fkd0akd3.rnd.pilot.arango.ai/ui/ai-tools/notebooks/qgwsj`

Try these URLs to access the Flask server on port 8889:

1. **Jupyter Proxy Pattern:**
   ```
   https://fkd0akd3.rnd.pilot.arango.ai/notebooks/qgwsj/proxy/8889/health
   https://fkd0akd3.rnd.pilot.arango.ai/ui/ai-tools/notebooks/qgwsj/proxy/8889/health
   ```

2. **Direct Port (if exposed):**
   ```
   https://fkd0akd3.rnd.pilot.arango.ai:8889/health
   ```

3. **Check notebook server configuration** - Some Jupyter servers require you to explicitly expose ports.

## Using Port Forwarding (Recommended)

If the above don't work, use kubectl port forwarding:

```bash
# First, find the notebook pod
kubectl get pods -n arangodb-platform-rnd-fkd0akd3 | grep notebook

# Then port forward (replace <pod-name> with actual pod name)
kubectl port-forward -n arangodb-platform-rnd-fkd0akd3 <pod-name> 8889:8889

# Now access via localhost
# In .env: GRAPHRAG_RETRIEVER_URL=http://localhost:8889/graphrag/query
```

## Update Backend Config

Once you find the working URL, update `.env`:

```bash
# Option 1: Jupyter proxy URL
GRAPHRAG_RETRIEVER_URL=https://fkd0akd3.rnd.pilot.arango.ai/notebooks/qgwsj/proxy/8889/graphrag/query

# Option 2: Port forwarding
GRAPHRAG_RETRIEVER_URL=http://localhost:8889/graphrag/query

# Option 3: Direct port (if exposed)
GRAPHRAG_RETRIEVER_URL=https://fkd0akd3.rnd.pilot.arango.ai:8889/graphrag/query
```
