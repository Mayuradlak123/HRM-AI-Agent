from typing import Dict, Any

def _sanitize_pinecone_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        if v is None:
            safe[k] = "unknown"
        elif isinstance(v, (str, bool, int, float)):
            safe[k] = v
        elif isinstance(v, list):
            # Pinecone allows list of strings only; coerce other types to strings and drop None
            safe[k] = [str(x) for x in v if x is not None]
        else:
            # Fallback to string representation
            safe[k] = str(v)
    return safe


