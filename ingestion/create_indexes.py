from __future__ import annotations

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "medisense_chunks"


def create_payload_indexes() -> None:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    fields = ["section", "specialty", "parent_doc_id"]

    for field in fields:
        print(f"Creating index on '{field}'...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"  ✅ Done")

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection now has {info.points_count} points")


if __name__ == "__main__":
    create_payload_indexes()
