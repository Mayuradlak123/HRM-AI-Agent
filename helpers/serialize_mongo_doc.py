from typing import Dict, Any
from bson import ObjectId

def serialize_mongo_doc(doc):
    if isinstance(doc, list):
        return [serialize_mongo_doc(d) for d in doc]
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                new_doc[k] = str(v)
            elif isinstance(v, (dict, list)):
                new_doc[k] = serialize_mongo_doc(v)
            else:
                new_doc[k] = v
        return new_doc
    return doc
