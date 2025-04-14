import json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
# from utils.prompts import get_prompt_template
# from utils.parser import extract_text_from_pdf
from utils import get_prompt_template,extract_text_from_pdf

def load_prof_settings():
    with open("config/prof_settings.json", "r") as f:
        return json.load(f)

def prepare_vector_store(doc_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(doc_text)]
    embeddings = OllamaEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_hint_response(question, vector_store, hint_level):
    retriever = vector_store.as_retriever(search_type="similarity", k=3)
    llm = ChatOllama(temperature=0.7)
    prompt_template = get_prompt_template(hint_level)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": None}  # Let it use default
    )

    context = retriever.get_relevant_documents(question)
    prompt = f"""
You are a helpful teaching assistant.

Context from course material:
{''.join([doc.page_content for doc in context])}

Student Question:
{question}

{prompt_template}
"""

    return llm.predict(prompt)
