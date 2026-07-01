import os
from groq import Groq

from parser    import parse_document
from database  import PineconeRAGDatabase
from retriever import PineconeHybridRetriever
from config    import RAGConfig

def build_prompt(query: str, chunks: list) -> str:
    if not chunks:
        return (
            "You are a senior US Army facilities maintenance officer.\n\n"
            "No relevant context was found in TM 5-692-1 for the query below.\n"
            "Reply with: \"Technical guidelines unverified within active document boundaries.\"\n\n"
            f"QUERY: {query}\nANSWER:"
        )

    context_blocks = ""
    for i, chunk in enumerate(chunks, start=1):
        m = chunk["metadata"]
        context_blocks += (
            f"--- CONTEXT BLOCK {i} ---\n"
            f"Section : {m.get('label', 'N/A')}\n"
            f"Pages   : {m.get('page_start', '?')}–{m.get('page_end', '?')}\n"
            f"Content :\n{chunk['text']}\n\n"
        )

    return f"""You are a senior infrastructure facilities officer operating under \
US Army TM 5-692-1 documentation mandates.

INSTRUCTIONS:
1. Answer using ONLY information present in the Context Blocks below.
2. If the context does not contain the answer, reply with: \
"Technical guidelines unverified within active document boundaries."
3. Every statement in your answer must be followed by its Page reference in parentheses, \
e.g. (Page 256).
4. Format your answer as a numbered checklist when appropriate.

[CONTEXT BLOCKS]
{context_blocks}
QUERY: {query}
ANSWER:"""



def call_llm(prompt: str) -> str:
    client = Groq(api_key=RAGConfig.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=RAGConfig.LLM_MODEL,
        max_tokens=RAGConfig.LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content



def run_query(query: str, db: PineconeRAGDatabase, retriever: PineconeHybridRetriever):
    print(f"\nQuery: {query!r}\n")

    chunks = retriever.retrieve(query)
    print(f"Retrieved {len(chunks)} context chunk(s):")
    for c in chunks:
        print(f"  [{c['id']}] {c['metadata']['label']}  "
              f"(pp.{c['metadata']['page_start']}–{c['metadata']['page_end']})")

    prompt = build_prompt(query, chunks)
    answer = call_llm(prompt)

    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(answer)
    print("="*60)
    return answer


def main():
    if not os.path.exists(RAGConfig.PDF_PATH):
        print(f"Error: '{RAGConfig.PDF_PATH}' not found.")
        return

    print("Step 1: Parsing TM 5-692-1 into semantic chunks...")
    chunks = parse_document(RAGConfig.PDF_PATH)
    print(f"  {len(chunks)} chunks produced.")

 
    print("\nStep 2: Connecting to Pinecone...")
    db = PineconeRAGDatabase()

  
    stats = db.index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)

    def _index_is_fresh() -> bool:
        """True only if Pinecone holds vectors that match the current parse."""
        if total_vectors != len(chunks):
            return False
     
        probe_ids = [chunks[0]["id"], chunks[-1]["id"]]
        fetch_result = db.index.fetch(ids=probe_ids)
        found = set(fetch_result.get("vectors", {}).keys())
        return found == set(probe_ids)

    if _index_is_fresh():
        print(f"  Index already contains {total_vectors} matching vectors — skipping upload.")
        for chunk in chunks:
            db.chunk_store[chunk["id"]] = chunk
    else:
        print(f"  Index has {total_vectors} vectors but current parse has {len(chunks)} — re-uploading...")
        db.upload_documents(chunks)

    # Retrieve and answer 
    print("\nStep 3: Initializing retriever...")
    retriever = PineconeHybridRetriever(db)
    queries = [
    
        "What are the rules and guidelines for tool care and inspecting maintenance tools monthly?",
        "How often should cooling tower bearings be lubricated?",
        "What safety precautions apply to electrical maintenance work?",
        "At what temperature should bearing readings be questioned, and what field test indicates a bearing is in distress and must be replaced before seizing?",
        "What happens to freeze protection when antifreeze concentration exceeds 63 percent by volume, and at what concentration does a commercial antifreeze solution protect to minus 32 degrees F?",
        "How often must the boiler safety valve try lever test be performed, what minimum pressure must the boiler be under before the test, and how long should steam be discharged?",
        "What is the acceptable pressure range for a boiler safety valve to pop open, and what action must be taken if it does not open by 17 psi?",
        "At what percentage of continuous electrical overload will motor windings begin to overheat?",
        "What are the precautions that must be taken before a hot work permit can be issued at a C4ISR facility?",
        "What must be sampled in a vessel before a vessel entry permit is issued, and what individual must be posted at the entrance during the work?",
        "What is the maximum rate at which coolant should be added when filling an empty engine cooling system, and why?",
        "How often must standby generators be load-tested, what minimum load percentage is required, and for how long after reaching stable temperature?",
        "What is the maximum interval for repacking bearings with grease during equipment overhauls, and what should be done when new sleeve bearing units are first placed in service?",
    ]

    for query in queries:
        run_query(query, db, retriever)
        print()


if __name__ == "__main__":
    main()