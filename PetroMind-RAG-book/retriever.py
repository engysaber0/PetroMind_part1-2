import numpy as np
from sentence_transformers import CrossEncoder
from config import RAGConfig


class PineconeHybridRetriever:
    def __init__(self, db):
        self.db           = db
        self.cross_encoder = CrossEncoder(RAGConfig.RERANKER_MODEL_NAME)

    def _build_query_vectors(self, query: str):
        alpha = RAGConfig.HYBRID_ALPHA

        # Dense vector — scaled by alpha
        dense_raw = self.db.dense_model.encode(query).tolist()
        dense     = [v * alpha for v in dense_raw]

        # Sparse vector — scaled by (1 - alpha)
        sparse_raw = self.db.build_sparse_vector(query)
        sparse     = self.db.scale_sparse(sparse_raw, 1.0 - alpha)

        return dense, sparse

    def retrieve(self, query: str) -> list:
        dense, sparse = self._build_query_vectors(query)

        #Pinecone hybrid query 
        if self.db.supports_sparse:
            response = self.db.index.query(
                vector=dense,
                sparse_vector=sparse,
                top_k=RAGConfig.TOP_K_HYBRID,
                include_metadata=True,
            )
        else:
            print("Warning: sparse unavailable — dense-only fallback.")
            response = self.db.index.query(
                vector=dense,
                top_k=RAGConfig.TOP_K_HYBRID,
                include_metadata=True,
            )

        candidates = response.get("matches", [])
        if not candidates:
            print("No candidates returned from Pinecone.")
            return []

        #cross-encoder reranking 
        pairs  = [[query, m["metadata"]["text_content"]] for m in candidates]
        scores = self.cross_encoder.predict(pairs)
        top_indices = np.argsort(scores)[::-1][: RAGConfig.TOP_K_FINAL]

    
        results  = []
        seen_ids = set()

        for idx in top_indices:
            chunk_id = candidates[idx]["id"]
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                chunk = self.db.get_chunk(chunk_id)
                if chunk:
                    results.append(chunk)

        return results
