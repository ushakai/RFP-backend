"""
Example: How to integrate hybrid search into RFP question answering
"""
from services.hybrid_search_service import search_knowledge_base
from config.settings import GEMINI_MODEL
import google.generativeai as genai


def answer_rfp_question_with_rag(
    question: str,
    client_id: str,
    rfp_id: str | None = None
) -> dict:
    """
    Answer an RFP question using hybrid search + LLM
    
    This is the recommended pattern for production use.
    """
    # Step 1: Search knowledge base (hybrid: vector + BM25)
    search_results = search_knowledge_base(
        query=question,
        client_id=client_id,
        rfp_id=rfp_id,
        top_k=5,  # Get top 5 most relevant sources
        include_docs=True,  # Include document chunks
        include_qa=True     # Include past Q&A pairs
    )
    
    # Step 2: Build context from search results
    context = search_results['combined_context']
    
    # Step 3: Check if we have relevant context
    if not search_results['documents'] and not search_results['qa_pairs']:
        # No relevant context found
        return {
            "answer": "I don't have enough information to answer this question confidently.",
            "confidence": "low",
            "sources": []
        }
    
    # Step 4: Build LLM prompt with retrieved context
    prompt = f"""You are an expert RFP response writer. Answer the following question using ONLY the provided context.

Question: {question}

Relevant Context from Knowledge Base:
{context}

Instructions:
1. Answer the question comprehensively using information from the context
2. Be specific and cite which sources you used (Q&A or DOC)
3. If the context doesn't fully answer the question, say so
4. Keep the answer professional and concise
5. Don't make up information not in the context

Answer:"""

    # Step 5: Generate answer with LLM
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    response = gemini_model.generate_content(prompt)
    answer = response.text.strip()
    
    # Step 6: Return structured response
    sources = []
    
    # Add Q&A sources
    for qa in search_results['qa_pairs']:
        sources.append({
            "type": "qa",
            "question": qa['question'],
            "score": qa['score']
        })
    
    # Add document sources
    for doc in search_results['documents']:
        sources.append({
            "type": "document",
            "filename": doc['filename'],
            "chunk_index": doc['chunk_index'],
            "score": doc['score']
        })
    
    return {
        "answer": answer,
        "confidence": "high" if sources else "medium",
        "sources": sources,
        "search_results": search_results  # Full results for debugging
    }


# Example usage
if __name__ == "__main__":
    # Simulate answering an RFP question
    result = answer_rfp_question_with_rag(
        question="What is your company's approach to data security and ISO 27001 compliance?",
        client_id="your-client-id",
        rfp_id="your-rfp-id"  # Optional
    )
    
    print("=" * 80)
    print("QUESTION:")
    print(result['answer'])
    print("\n" + "=" * 80)
    print(f"CONFIDENCE: {result['confidence']}")
    print(f"SOURCES ({len(result['sources'])}):")
    for i, source in enumerate(result['sources'], 1):
        if source['type'] == 'qa':
            print(f"  {i}. [Q&A] {source['question'][:60]}... (score: {source['score']:.2f})")
        else:
            print(f"  {i}. [DOC] {source['filename']} chunk {source['chunk_index']} (score: {source['score']:.2f})")
    print("=" * 80)

