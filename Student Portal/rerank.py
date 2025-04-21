from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

def rerank(query, docs, top_n=3):
    pairs = [(query, doc.page_content) for doc in docs]
    
    inputs = tokenizer(
        [f"{q} [SEP] {d}" for q, d in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)

    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)
    top_docs = [docs[i] for i in sorted_indices[:top_n]]

    return top_docs

# Load LLM
llm = ChatOllama(model="mistral")


db = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

query = "What is few shot prompting?"

pdf_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"type": "pdf"}})
transcript_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"type": "transcript"}})

pdf_chunks = pdf_retriever.invoke(query)
transcript_chunks = transcript_retriever.invoke(query)

print(len(pdf_chunks))
print(len(transcript_chunks))
# Load BGE Reranker
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
model.eval()


# Apply reranker
pdf_top = rerank(query, pdf_chunks, top_n=3)
transcript_top = rerank(query, transcript_chunks, top_n=3)

final_docs = pdf_top + transcript_top
# print([doc.metadata['source'] for doc in final_docs])

final_rerank = rerank(query, final_docs, top_n=5)
# print([doc.metadata['source'] for doc in final_rerank])

print([f.page_content for f in final_rerank])
# Now send top_docs to your LLM chain
context = "\n\n".join([
    f"[{doc.metadata['type'].upper()} - {doc.metadata['source']}] {doc.page_content}"
    for doc in final_rerank
])

prompt = f"""
You are a helpful assistant. Based on the following course materials:

{context}

Student asked:
{query}

Respond with a hint or Indirect - Lead them to answer with hints. 
DO NOT GIVE DIRECT ANSWERS!!!!

EXPLAIN LIKE I'M 18 Year Old 
"""

response = llm.invoke(prompt)
print(response.content)
