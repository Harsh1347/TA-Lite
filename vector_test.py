from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

db = FAISS.load_local(
    "teacher_data/faiss_index",
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

query = "What is PEFT?"

docs = retriever.get_relevant_documents(query)
for doc in docs:
    print(f"📄 File: {doc.metadata['source']}, Chunk: {doc.metadata['chunk_id']}")
    print(doc.page_content)
    print("—" * 50)
