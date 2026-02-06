# Teaching Assistant Lite Agent 
### A Modular AI Teaching Assistant for Guided Learning

![https://github.com/Harsh1347/TA-Lite/blob/main/assets/TALIA_Thumbnail.png](https://github.com/Harsh1347/TA-Lite/blob/main/assets/TALIA_Thumbnail.png)
---

## 🚀 Project Motivation

In large classroom settings, students often hesitate to ask questions, while professors are overwhelmed with repetitive doubts—especially near exams or assignment deadlines. This creates a communication gap, leading to confusion, disengagement, and shallow learning.

**TA Lite** bridges this gap by providing a scalable, always-available, and instructor-aligned academic assistant powered by LLMs. It is designed not to give direct answers, but to **guide students to think**, reflect, and learn more effectively.

---

## 🧠 Key Features

### 🎓 Student Portal
- **Doubt Solving (RAG-based):** Context-aware, hint-first answers retrieved from course materials.
- **Lecture & Transcript Summarization:** Short, focused summaries for quick revision.
- **Note Generation:** Structured notes from instructor-provided slides and documents.
- **Reference Access:** View curated materials uploaded by instructors.

### 👩‍🏫 Teacher Portal
- **Material Upload:** Lecture slides, transcripts, and external links.
- **Prompt Configuration:** Set tone, hint level, and response style for LLM outputs.
- **Analytics Dashboard:** View most-asked questions and usage trends.

---

## 🔧 System Architecture

![System](https://github.com/Harsh1347/TA-Lite/blob/main/SystemArchitecture.png?raw=true)


- **Two Portals:** Teacher (configuration & upload) and Student (interaction).
- **RAG Pipeline:** Combines vector search (via FAISS) and LLM generation (via Mistral 7B on Ollama).
- **Reranking:** Uses BAAI/bge-reranker-base to improve semantic quality of retrieved context.
- **Chunking Strategy:** Separate chunk sizes for slides (concise) vs. transcripts (long-form).
- **Transparency:** Students see the source document and chunk for every response.

---

## 🛠️ Technologies Used

- **LLM Inference:** Mistral 7B via Ollama
- **Frameworks:** LangChain, Streamlit
- **Embedding Model:** `all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store:** FAISS
- **Reranking Model:** `BAAI/bge-reranker-base`

---

## 📊 Evaluation Highlights

- Manual validation and peer testing from domain students.

---

## 🧭 Design Highlights

- Prompt templating aligned with instructor preferences.
- Modular agent structure (e.g., Summarizer, Note Generator, Exam Prep).
- Hybrid retrieval from both transcripts and slides.
- Reranking ensures better answer composition from mixed sources.

---

## 🔮 Future Scope

- 🎧 **Lecture-to-Podcast:** Auto-generate audio from transcript summaries.
- 🕹️ **Gamification:** Points/streaks for engaging with the tool.
- 🚫 **Exam Integrity:** Flag exam or assignment-related queries.
- 🧩 **Subject-Specific Agents:** Add Math Agent, Code Agent, etc.

---

## 🙌 Acknowledgments

Special thanks to our course instructors and peers for valuable feedback throughout the development of TA Lite.

---

**Link(presentation and demo)**: https://youtu.be/oa860oWLdOo

**Screenshots**: https://github.iu.edu/hagupta/TA-Lite/tree/main/captures

**Team:** 
- Aashi Sharma  
- Harsh Gupta  
- Pranay Bandaru  
- Rakshit Rao  

