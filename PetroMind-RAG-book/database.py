import time
import zlib
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from config import RAGConfig


def _stable_hash(token: str) -> int:
    """Deterministic token → sparse index. Same result every run."""
    return zlib.adler32(token.encode("utf-8")) % (2**31 - 1)

SPARSE_STOPWORDS = {
    "mo", "yr", "day", "week", "hr", "shift",
    "3", "6", "1", "2", "as", "at", "per",
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "be",
    "are", "was", "were", "for", "on", "with", "from", "that", "this",
}


class PineconeRAGDatabase:
    def __init__(self):
        self.pc          = Pinecone(api_key=RAGConfig.PINECONE_API_KEY)
        self.dense_model = SentenceTransformer(RAGConfig.EMBEDDING_MODEL_NAME)
        self.chunk_store: dict = {}  
        self.supports_sparse = False

        existing = [idx["name"] for idx in self.pc.list_indexes()]

        if RAGConfig.PINECONE_INDEX_NAME in existing:
            desc   = self.pc.describe_index(RAGConfig.PINECONE_INDEX_NAME)
            metric = getattr(desc, "metric", None)
            if metric != "dotproduct":
                print(f"Index metric is '{metric}' — must be 'dotproduct' for hybrid. Recreating...")
                self.pc.delete_index(RAGConfig.PINECONE_INDEX_NAME)
                existing.remove(RAGConfig.PINECONE_INDEX_NAME)

        if RAGConfig.PINECONE_INDEX_NAME not in existing:
            print(f"Creating Pinecone index '{RAGConfig.PINECONE_INDEX_NAME}'...")
            self.pc.create_index(
                name=RAGConfig.PINECONE_INDEX_NAME,
                dimension=768,          
                metric="dotproduct",    
                spec=ServerlessSpec(cloud="aws", region=RAGConfig.PINECONE_ENV),
            )
            while not self.pc.describe_index(RAGConfig.PINECONE_INDEX_NAME).status["ready"]:
                time.sleep(1)
            print("Index ready.")

        self.index           = self.pc.Index(RAGConfig.PINECONE_INDEX_NAME)
        self.supports_sparse = True



    def build_sparse_vector(self, text: str) -> dict:
        """TF sparse vector with deterministic hashing. Skips stopwords."""
        tokens = text.lower().split()
        tf: dict[int, float] = {}
        for token in tokens:
            if token in SPARSE_STOPWORDS:
                continue
            idx = _stable_hash(token)
            tf[idx] = tf.get(idx, 0.0) + 1.0
        return {"indices": list(tf.keys()), "values": list(tf.values())}

    def scale_sparse(self, sparse: dict, scale: float) -> dict:
        return {"indices": sparse["indices"], "values": [v * scale for v in sparse["values"]]}

    # Upload 
    def upload_documents(self, chunks: list):
        """
        Embed enriched_text (has context header) but store raw text for retrieval.
        Cache all chunks in memory for fast lookup after retrieval.
        """
        for chunk in chunks:
            self.chunk_store[chunk["id"]] = chunk

        embed_texts = [c["enriched_text"] for c in chunks]
        print(f"Embedding {len(embed_texts)} chunks with {RAGConfig.EMBEDDING_MODEL_NAME}...")
        dense_vecs = self.dense_model.encode(
            embed_texts, show_progress_bar=True, batch_size=32
        ).tolist()

        batch_size = 100
        print("Uploading to Pinecone...")
        for i in range(0, len(chunks), batch_size):
            batch      = chunks[i : i + batch_size]
            batch_vecs = dense_vecs[i : i + batch_size]
            upsert_batch = []

            for chunk, dense in zip(batch, batch_vecs):
                sparse = self.build_sparse_vector(chunk["enriched_text"])
                meta   = dict(chunk["metadata"])
                #store raw text in metadata for reranker access
                meta["text_content"] = chunk["text"]
                # Convert any list fields to strings 
                if "frequency_tags" in meta:
                    meta["frequency_tags"] = ", ".join(meta["frequency_tags"])
                #Remove None values — Pinecone rejects null metadata fields
                meta = {k: v for k, v in meta.items() if v is not None}

                upsert_batch.append({
                    "id":            chunk["id"],
                    "values":        dense,
                    "sparse_values": sparse,
                    "metadata":      meta,
                })

            self.index.upsert(vectors=upsert_batch)

        print(f"Upload complete. {len(chunks)} chunks indexed.")

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.chunk_store.get(chunk_id)
