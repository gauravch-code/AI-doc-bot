import json
import os
import time
import torch
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not set. Did you create your .env file?")

MODEL_DIMENSION = 384
MODEL_NAME = 'all-MiniLM-L6-v2'

INDEX_NAME = "python-docs-ai"

print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found. Creating a new one...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=MODEL_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status['ready']:
        print("Waiting for index to be ready...")
        time.sleep(5)
else:
    print(f"Found existing index: '{INDEX_NAME}'")

index = pc.Index(INDEX_NAME)
print("\n--- Current Index Stats ---")
print(index.describe_index_stats())
print("---------------------------")


print(f"Loading SentenceTransformer model ({MODEL_NAME})...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
embedding_model = SentenceTransformer(MODEL_NAME, device=device)

print("Loading corpus.json...")
try:
    with open('corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)
except FileNotFoundError:
    print("ERROR: corpus.json not found. Please run scraper.py first.")
    exit()

print(f"Loaded {len(corpus)} documents from corpus.")
print("Initializing text splitter...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


print("Starting to process and embed documents...")
start_time = time.time()
docs_to_upsert = []
existing_ids = set() # To track IDs already processed for this run



for i, doc in enumerate(corpus):
    # Skip if doc is missing essential keys
    if not all(k in doc for k in ["library", "source", "url", "text"]):
        print(f"Warning: Skipping malformed doc at index {i}")
        continue

    # Create a unique prefix for this document
    # Simple hash of URL + library should be fairly unique
    doc_hash = hash(doc['url'] + doc['library'])
    doc_id_prefix = f"doc_{doc['library']}_{abs(doc_hash)}"

    # Split the document's text
    chunks = text_splitter.split_text(doc['text'])

    for j, chunk in enumerate(chunks):
        chunk_id = f"{doc_id_prefix}_chunk_{j}"

        if chunk_id in existing_ids: # or chunk_id in existing_ids_in_db:
            continue

        existing_ids.add(chunk_id)

      
        metadata = {
            "library": doc['library'], # <-- ENSURE LIBRARY IS ADDED
            "source": doc['source'],
            "url": doc['url'],
            "text": chunk
        }

        docs_to_upsert.append({
            "id": chunk_id,
            "metadata": metadata
        })

if not docs_to_upsert:
    print("\nNo new documents found to add/update in Pinecone.")
else:
    print(f"\nProcessing {len(docs_to_upsert)} new chunks to upsert.")
    batch_size = 100
    print(f"Embedding and upserting in batches of {batch_size}...")

    for k in tqdm(range(0, len(docs_to_upsert), batch_size)):
        batch = docs_to_upsert[k : k+batch_size]
        texts_to_embed = [d['metadata']['text'] for d in batch]

        embeddings = embedding_model.encode(texts_to_embed).tolist()

        vectors_to_upsert = []
        for d, emb in zip(batch, embeddings):
            vectors_to_upsert.append({
                "id": d['id'],
                "values": emb,
                "metadata": d['metadata'] # Metadata already includes 'library'
            })

        try:
            index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"Error upserting batch starting at index {k}: {e}")
            # Optional: Add retry logic here

    end_time = time.time()
    print(f"\nSuccessfully attempted to upsert {len(docs_to_upsert)} chunks.")
    print(f"Total time: {end_time - start_time:.2f} seconds")

print(f"\nMilestone 3 Complete! 🚀")
print(f"Pinecone index '{INDEX_NAME}' updated.")
print("\n--- Final Index Stats ---")
print(index.describe_index_stats())