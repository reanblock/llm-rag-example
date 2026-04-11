from pathlib import Path
from uuid import uuid4
from openai import OpenAI
from pydantic import BaseModel
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
from embeddings import embedding_model
from ingest import fetch_documents

MODEL = "openai/gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent / "preprocessed_db")
PARENT_COLLECTION = "docs_parent"
CHILD_COLLECTION = "docs_child"
WORKERS = 3

wait = wait_exponential(multiplier=1, min=10, max=240)
openai = OpenAI()


# --- LLM-facing models (no parent_id — that is assigned programmatically) ---

class ParentChunkLLM(BaseModel):
    section_headline: str
    section_text: str


class ParentChunksLLM(BaseModel):
    chunks: list[ParentChunkLLM]


class ChildChunkLLM(BaseModel):
    headline: str
    summary: str
    child_text: str


class ChildChunksLLM(BaseModel):
    chunks: list[ChildChunkLLM]


# --- Internal data models (include parent_id for storage and linking) ---

class ParentChunk(BaseModel):
    section_headline: str
    section_text: str
    parent_id: str


class ChildChunk(BaseModel):
    headline: str
    summary: str
    child_text: str
    parent_id: str


# --- Prompt builders ---

def make_parent_prompt(document: dict) -> str:
    target_sections = max(3, min(8, len(document["text"]) // 2000))
    return f"""
You split company documents into large, coherent sections for a Knowledge Base.

The document is from the company Insurellm.
Document type: {document["type"]}
Document source: {document["source"]}

Your task: divide the document into {target_sections} sections (you may use between 3 and 8).
Each section should cover a logically cohesive topic (e.g. Pricing, Features, Eligibility, Overview).
Target approximately 500-800 tokens per section.
Allow up to 20% overlap at section boundaries so context is not lost at transitions.
The union of all section_text fields must cover the ENTIRE document — do not omit any text.

For each section provide:
- section_headline: a short title (a few words)
- section_text: the full original text of that section, verbatim

Here is the document:

{document["text"]}

Respond with the sections.
"""


def make_child_prompt(parent_chunk: ParentChunk) -> str:
    target_children = max(2, min(5, len(parent_chunk.section_text) // 500))
    return f"""
You split a section of a company document into small, precise sub-chunks for retrieval.

The section is titled: {parent_chunk.section_headline}

Your task: divide this section into {target_children} child chunks (between 2 and 5).
Each child chunk should be about 100-150 tokens.
Each child should independently answer a specific user question.
Allow about 25% overlap between adjacent children so context is not lost.
Together the children must cover ALL of the section text.

For each child provide:
- headline: a very short, query-oriented heading
- summary: 2-3 sentences capturing the key facts
- child_text: the verbatim text from the section for this chunk

Here is the section text:

{parent_chunk.section_text}

Respond with the child chunks.
"""


# --- Processing functions ---

@retry(wait=wait)
def process_parent_chunks(document: dict) -> list[ParentChunk]:
    messages = [{"role": "user", "content": make_parent_prompt(document)}]
    response = completion(model=MODEL, messages=messages, response_format=ParentChunksLLM)
    raw = ParentChunksLLM.model_validate_json(response.choices[0].message.content)
    return [
        ParentChunk(
            section_headline=c.section_headline,
            section_text=c.section_text,
            parent_id=str(uuid4()),
        )
        for c in raw.chunks
    ]


@retry(wait=wait)
def process_child_chunks(parent_chunk: ParentChunk) -> list[ChildChunk]:
    messages = [{"role": "user", "content": make_child_prompt(parent_chunk)}]
    response = completion(model=MODEL, messages=messages, response_format=ChildChunksLLM)
    raw = ChildChunksLLM.model_validate_json(response.choices[0].message.content)
    return [
        ChildChunk(
            headline=c.headline,
            summary=c.summary,
            child_text=c.child_text,
            parent_id=parent_chunk.parent_id,
        )
        for c in raw.chunks
    ]


def process_hierarchical_document(document: dict) -> tuple[list, list]:
    """
    Process one document into parent and child chunks.
    Returns (parents_with_meta, children_with_meta) where each entry is (chunk, metadata_dict).
    This function is the Pool worker — must be a top-level function for pickling.
    """
    parents = process_parent_chunks(document)
    children = []
    for parent in parents:
        children.extend(process_child_chunks(parent))

    meta = {"source": document["source"], "type": document["type"]}
    return [(p, meta) for p in parents], [(c, meta) for c in children]


def create_hierarchical_chunks(documents: list[dict]) -> tuple[list, list]:
    """
    Parallel processing of all documents into parent and child chunks.
    If you hit rate limits, set WORKERS to 1.
    """
    all_parents = []
    all_children = []
    with Pool(processes=WORKERS) as pool:
        for parents_with_meta, children_with_meta in tqdm(
            pool.imap_unordered(process_hierarchical_document, documents),
            total=len(documents),
        ):
            all_parents.extend(parents_with_meta)
            all_children.extend(children_with_meta)
    return all_parents, all_children


# --- Embedding and storage ---

def create_hierarchical_embeddings(
    all_parents: list[tuple[ParentChunk, dict]],
    all_children: list[tuple[ChildChunk, dict]],
) -> None:
    chroma = PersistentClient(path=DB_NAME)

    # Drop and recreate both collections to ensure a clean slate
    for name in [PARENT_COLLECTION, CHILD_COLLECTION]:
        if name in [c.name for c in chroma.list_collections()]:
            chroma.delete_collection(name)

    # --- Parent collection ---
    parent_texts = [
        p.section_headline + "\n\n" + p.section_text for p, _ in all_parents
    ]
    parent_vectors = [
        e.embedding
        for e in openai.embeddings.create(model=embedding_model, input=parent_texts).data
    ]
    parent_ids = [p.parent_id for p, _ in all_parents]
    parent_metas = [
        {"source": meta["source"], "type": meta["type"], "parent_id": p.parent_id}
        for p, meta in all_parents
    ]

    parent_collection = chroma.get_or_create_collection(PARENT_COLLECTION)
    parent_collection.add(
        ids=parent_ids,
        embeddings=parent_vectors,
        documents=parent_texts,
        metadatas=parent_metas,
    )
    print(f"{PARENT_COLLECTION}: {parent_collection.count()} documents")

    # --- Child collection ---
    child_texts = [
        c.headline + "\n\n" + c.summary + "\n\n" + c.child_text for c, _ in all_children
    ]
    child_vectors = [
        e.embedding
        for e in openai.embeddings.create(model=embedding_model, input=child_texts).data
    ]
    child_ids = [str(i) for i in range(len(all_children))]
    child_metas = [
        {"source": meta["source"], "type": meta["type"], "parent_id": c.parent_id}
        for c, meta in all_children
    ]

    child_collection = chroma.get_or_create_collection(CHILD_COLLECTION)
    child_collection.add(
        ids=child_ids,
        embeddings=child_vectors,
        documents=child_texts,
        metadatas=child_metas,
    )
    print(f"{CHILD_COLLECTION}: {child_collection.count()} documents")


# --- Entry point ---

def ingest_hierarchical() -> None:
    documents = fetch_documents()
    all_parents, all_children = create_hierarchical_chunks(documents)
    create_hierarchical_embeddings(all_parents, all_children)
    print("Hierarchical ingestion complete")


if __name__ == "__main__":
    ingest_hierarchical()
