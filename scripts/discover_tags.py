import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db import QdrantStorage

def discover_tags():
    storage = QdrantStorage()
    client = storage.client
    collection_name = storage.collection
    
    # Scroll through first 100 points
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    tags = set()
    for rec in records:
        tag = rec.payload.get("source_tag", "None")
        tags.add(tag)
        
    print(f"Các tag tìm thấy trong DB: {tags}")

if __name__ == "__main__":
    discover_tags()
