import time
import requests
from db.mongo_client import get_db
from embeddings.nim_client import get_embeddings_batch
from pymongo import UpdateOne

BATCH_SIZE = 50
SECONDS_BETWEEN_CALLS = 1.6  # keeps us under ~40 RPM with margin
MAX_RETRIES = 5


def build_description(product: dict) -> str:
    category = product.get("category_code") or "unknown category"
    brand = product.get("brand") or "unknown brand"
    category_text = category.replace(".", " ").replace("_", " ")
    return f"{brand} {category_text}".strip()


def get_embeddings_with_retry(descriptions):
    for attempt in range(MAX_RETRIES):
        try:
            return get_embeddings_batch(descriptions, input_type="passage")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = e.response.text if e.response is not None else str(e)
            if status in (429, 500, 502, 503):
                wait = 2 ** attempt
                print(f"Error {status}, retrying in {wait}s... Response: {body[:200]}")
                time.sleep(wait)
            else:
                print(f"Non-retryable error {status}: {body[:200]}")
                raise
    raise RuntimeError("Max retries exceeded on NIM API.")


def embed_all_products():
    db = get_db()
    products = list(db.products.find(
        {"embedding": {"$exists": False}},
        {"_id": 1, "product_id": 1, "category_code": 1, "brand": 1}
    ))
    print(f"Found {len(products)} products left to embed.")

    total_updated = 0

    for i in range(0, len(products), BATCH_SIZE):
        batch = products[i:i + BATCH_SIZE]
        descriptions = [build_description(p) for p in batch]

        embeddings = get_embeddings_with_retry(descriptions)

        operations = [
            UpdateOne(
                {"_id": product["_id"]},
                {"$set": {"embedding": embedding, "description": build_description(product)}}
            )
            for product, embedding in zip(batch, embeddings)
        ]

        if operations:
            result = db.products.bulk_write(operations, ordered=False)
            total_updated += result.modified_count

        print(f"Processed {min(i + BATCH_SIZE, len(products))}/{len(products)} products...")
        time.sleep(SECONDS_BETWEEN_CALLS)

    print(f"Done. Total products updated: {total_updated}")


if __name__ == "__main__":
    embed_all_products()