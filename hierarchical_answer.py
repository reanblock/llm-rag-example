from pathlib import Path
from openai import OpenAI
from chromadb import PersistentClient
from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt
from embeddings import embedding_model
from answer import Result, rewrite_query, rerank, make_rag_messages

MODEL = "openai/gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent / "preprocessed_db")
PARENT_COLLECTION = "docs_parent"
CHILD_COLLECTION = "docs_child"

RETRIEVAL_K = 20  # number of child chunks to retrieve per query
FINAL_K = 10      # number of parent chunks to pass to the LLM

wait = wait_exponential(multiplier=1, min=10, max=240)

openai = OpenAI()
chroma = PersistentClient(path=DB_NAME)
child_collection = chroma.get_or_create_collection(CHILD_COLLECTION)
parent_collection = chroma.get_or_create_collection(PARENT_COLLECTION)


def fetch_context_hierarchical_unranked(question: str) -> list[Result]:
    """
    Small-to-big retrieval:
    1. Find the most relevant child chunks via vector similarity search.
    2. Collect the unique parent IDs from those children.
    3. Fetch the full parent sections by ID (direct key lookup, not vector search).
    4. Return parent sections as Result objects for downstream reranking.
    """
    # Step 1: embed question and search child collection
    query_vector = openai.embeddings.create(
        model=embedding_model, input=[question]
    ).data[0].embedding

    child_results = child_collection.query(
        query_embeddings=[query_vector], n_results=RETRIEVAL_K
    )

    # Step 2: extract unique parent_ids preserving relevance order (first hit wins)
    seen: dict[str, dict] = {}
    for meta in child_results["metadatas"][0]:
        pid = meta["parent_id"]
        if pid not in seen:
            seen[pid] = meta

    if not seen:
        return []

    # Step 3: fetch parent documents by ID — O(1) lookup, no vector comparison
    parent_ids = list(seen.keys())
    fetched = parent_collection.get(ids=parent_ids, include=["documents", "metadatas"])

    # Step 4: build Result objects (same type as the flat pipeline — compatible with rerank)
    return [
        Result(page_content=doc, metadata=meta)
        for doc, meta in zip(fetched["documents"], fetched["metadatas"])
    ]


def fetch_context_hierarchical(original_question: str) -> list[Result]:
    """
    Full hierarchical retrieval pipeline:
    - Dual retrieval (original + rewritten question) for broader coverage.
    - Deduplication by parent_id (original query takes precedence on ties).
    - LLM reranking of the merged parent sections.
    - Returns top FINAL_K parent chunks.
    """
    rewritten_question = rewrite_query(original_question)

    parents1 = fetch_context_hierarchical_unranked(original_question)
    parents2 = fetch_context_hierarchical_unranked(rewritten_question)

    # Deduplicate by parent_id; reversed so parents1 overwrites parents2 on conflict
    merged = {r.metadata["parent_id"]: r for r in reversed(parents1 + parents2)}
    merged_list = list(merged.values())

    reranked = rerank(original_question, merged_list)
    return reranked[:FINAL_K]


@retry(wait=wait, stop=stop_after_attempt(1))
def answer_question_hierarchical(
    question: str, history: list[dict] = []
) -> tuple[str, list[Result]]:
    """
    Drop-in replacement for answer_question() using hierarchical retrieval.
    Finds precise child chunks, promotes them to full parent sections, then answers.
    Returns (answer_text, retrieved_parent_chunks).
    """
    chunks = fetch_context_hierarchical(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
