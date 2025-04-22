import streamlit as st
import os
from datetime import datetime
from backend import extract_text_from_file, prepare_vector_store, get_ai_response, generate_flashcards, generate_mcq, generate_summary, generate_study_notes, LectureContent
import streamlit.components.v1 as components
import streamlit_confetti as stc
from supabase import create_client, Client
import json
from typing import Optional, Dict, Any, List
import json
import requests
import sys
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from search_agent.main import run_agent

class AIAgent:
    def __init__(self, system_prompt):

        self.full_transcript = [
            {
                "role": "system", 
                "content": f"{system_prompt}"
            }
        ]

    def generate_ai_response(self, quesss):

        self.full_transcript.append({"role": "user", "content": quesss})
        print("AI agent Response:")

        API_KEY="sk-ea09f95e06c740f2b3b983763f702b72"
        URL="https://rc-156-87.rci.uits.iu.edu/api/chat/completions"

        headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                }
        
        messages = {"model": "Ministral-8B-Instruct-2410-GPTQ",
            "messages": self.full_transcript
            }
        json_payload = json.dumps(messages)
        response = requests.post(URL, headers=headers, data=json_payload)
        
        response_json = response.json()
        #print(response_json)
        assistant_reply = response_json["choices"][0]["message"]["content"] 
        self.full_transcript.append({
            "role": "assistant",
            "content": assistant_reply
        })
        print(assistant_reply)
        return assistant_reply

with open('instructor_prompts/instructor_settings_prompt.txt', 'r') as file:
    sys_prompt_captured = file.read()
AI_agent = AIAgent(str(sys_prompt_captured))

CONFIG_FILE = "teacher_data/config/settings_v2.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

llm = ChatOllama(model="mistral")
db = FAISS.load_local(
    r"faiss_index",
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# Supabase Configuration
SUPABASE_URL = "https://sulehfldyspavcngrjaf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN1bGVoZmxkeXNwYXZjbmdyamFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUxMzQxNjUsImV4cCI6MjA2MDcxMDE2NX0.3KkSeyaCOVhPAXpm8RSkSEbmAujwUE4GTRKQjTnKWik"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
# model.eval()


# Hard-coded file mappings
LECTURE_FILES = {
    "01": {
        "slides": "LLMs-01-intro.pdf",
        "transcript": "transcript-01-intro.docx"
    },
    "02": {
        "slides": "LLMs-02-wordrep.pdf",
        "transcript": "transcript-02-w2v.docx"
    },
    "03": {
        "slides": "LLMs-03-language-modeling-2.pdf",
        "transcript": "transcript-03-neuralnet.docx"
    },
    "04": {
        "slides": "LLMs-04-attention-transformers.pdf",
        "transcript": "transcript-04-lm.docx"
    },
    "05": {
        "slides": "LLMs-05-pre-training.pdf",
        "transcript": "transcript-05-transformer.docx"
    },
    "06": {
        "slides": "LLMs-06-fine-tuning.pdf",
        "transcript": "transcript-06-pretraining.docx"
    },
    "07": {
        "slides": "LLMs-07-prompting.pdf",
        "transcript": "transcript-07-finetuning.docx"
    },
    "08": {
        "slides": "LLMs-08-RLHF.pdf",
        "transcript": "transcript-08-prompting.docx"
    },
    "09": {
        "slides": "LLMs-09-DPO.pdf",
        "transcript": "transcript-09-rlhf.docx"
    },
    "10": {
        "slides": "LLMs-10-rag.pdf",
        "transcript": "transcript-10-dpo.docx"
    }
}

LECTURE_TITLES = {
    "01": "Overview and Introduction to LLM",
    "02": "Word Representation and Text Classification",
    "03": "Language modeling",
    "04": "Attention and Transformers",
    "05": "Pre-training and Pre-trained Models",
    "06": "Fine-Tuning and Instruction tuning",
    "07": "Prompting",
    "08": "Reinforcement Learning from Human Feedback",
    "09": "Direct Preference Optimization (DPO) & Group Relative Policy Optimization (GRPO)",
    "10": "Retrieval-Augmented Generation"
}

# Add this after your LECTURE_TITLES dictionary
QA_STORAGE_FILE = "qa_storage.json"

def init_session_state():
    """Initialize session state variables"""
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'flashcards' not in st.session_state:
        st.session_state.flashcards = []
    if 'current_mcq' not in st.session_state:
        st.session_state.current_mcq = None
    if 'query_id' not in st.session_state:
        st.session_state.query_id = None
    if 'student_id' not in st.session_state:
        st.session_state.student_id = None
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = ""
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "upload"
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'notes' not in st.session_state:
        st.session_state.notes = None
    if 'lecture_content' not in st.session_state:
        st.session_state.lecture_content = None
    if 'current_lecture' not in st.session_state:
        st.session_state.current_lecture = None
    if 'show_qa' not in st.session_state:
        st.session_state.show_qa = False
    if 'email' not in st.session_state:
        st.session_state.email = None
    if 'mcq_questions' not in st.session_state:
        st.session_state.mcq_questions = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "study"
    if 'material_type' not in st.session_state:
        st.session_state.material_type = None
    if 'action' not in st.session_state:
        st.session_state.action = None
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

def analyze_query(question: str, answer: str, topic: str) -> dict:
    """Generate summary and analyze difficulty of a question using LLM."""
    try:
        prompt = f"""
        Analyze the following question and its answer:
        
        Topic: {topic}
        Question: {question}
        Answer: {answer}
        
        Please provide a detailed analysis in the following format:
        1. A brief summary (max 100 words)
        2. Difficulty level (1-5, where 1 is easiest and 5 is hardest)
        3. Whether this needs teacher review (true/false)
        4. Key concepts covered (comma-separated list)
        5. Prerequisites needed to understand this (comma-separated list)
        6. Learning objectives addressed (comma-separated list)
        
        Format your response as a JSON object with keys: 
        summary, difficulty, needs_review, key_concepts, prerequisites, learning_objectives
        
        Ensure the difficulty is a number between 1 and 5.
        """
        
        analysis = get_ai_response(prompt)
        result = json.loads(analysis)
        
        # Validate the response format
        required_keys = ['summary', 'difficulty', 'needs_review', 'key_concepts', 'prerequisites', 'learning_objectives']
        if not all(key in result for key in required_keys):
            raise ValueError("Missing required keys in analysis response")
        
        # Validate difficulty range
        result['difficulty'] = max(1, min(5, int(result['difficulty'])))
        
        # Ensure boolean type for needs_review
        result['needs_review'] = bool(result['needs_review'])
        
        return result
    except json.JSONDecodeError:
        st.error("Failed to parse LLM response")
        return {
            "summary": "Summary generation failed",
            "difficulty": 1,
            "needs_review": True,
            "key_concepts": "",
            "prerequisites": "",
            "learning_objectives": ""
        }
    except Exception as e:
        st.error(f"Error in query analysis: {str(e)}")
        return {
            "summary": "Analysis failed",
            "difficulty": 1,
            "needs_review": True,
            "key_concepts": "",
            "prerequisites": "",
            "learning_objectives": ""
        }

def store_query(student_id: str, question: str, answer: str, topic: str) -> Optional[str]:
    """Store a query and its analysis in Supabase."""
    analysis = analyze_query(question, answer, topic)
    
    data = {
        "student_id": student_id,
        "question": question,
        "llm_answer": answer,
        "topic": topic or "General",
        "summary": analysis["summary"],
        "difficulty": analysis["difficulty"],
        "needs_review": analysis["needs_review"],
        "key_concepts": analysis.get("key_concepts", ""),
        "prerequisites": analysis.get("prerequisites", ""),
        "learning_objectives": analysis.get("learning_objectives", ""),
        "source_material": st.session_state.current_file,
        "timestamp": datetime.now().isoformat(),
        "has_mcq": False,
        "has_flashcards": False,
        "engagement_score": 0,
        "feedback_rating": None
    }
    
    try:
        result = supabase.table("student_queries").insert(data).execute()
        query_id = result.data[0]['id']
        return query_id
    except Exception as e:
        st.error(f"Failed to store query: {str(e)}")
        return None

def update_query_metadata(query_id: str, metadata: Dict[str, Any]) -> bool:
    """Update metadata for a stored query."""
    try:
        supabase.table("student_queries").update(metadata).eq('id', query_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to update query metadata: {str(e)}")
        return False

def render_login():
    """Render the login section"""
    if not st.session_state.email:
        email = st.text_input("Logged in as:", placeholder="Enter your university email")
        if email and '@' in email:
            st.session_state.email = email
            st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Logged in as: {st.session_state.email}")
        with col2:
            if st.button("Logout"):
                st.session_state.email = None
                st.rerun()

def render_file_upload():
    """Render the file upload section"""
    st.markdown("### 📚 Upload Course Material")
    uploaded_file = st.file_uploader("Choose a file (PDF or DOCX)", type=['pdf', 'docx'])
    
    if uploaded_file:
        # Save the uploaded file temporarily
        file_name = uploaded_file.name.lower()
        with st.spinner("Processing file..."):
            # Create temporary directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            temp_path = os.path.join("temp", file_name)
            
            # Save the file temporarily
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file
            try:
                text_content = extract_text_from_file(temp_path)
                if text_content:
                    st.session_state.vector_store = prepare_vector_store(text_content)
                    st.session_state.current_file = file_name
                    st.success("File processed successfully!")
                else:
                    st.error("Failed to process file.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

def render_action_buttons():
    """Render the action buttons for summary, notes, and Q&A"""
    if st.session_state.vector_store is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Generate Summary", use_container_width=True):
                with st.spinner("Generating summary..."):
                    summary = generate_summary(st.session_state.vector_store)
                    st.session_state.summary = summary
                    st.session_state.active_tab = "summary"
                    st.rerun()
        
        with col2:
            if st.button("Generate Study Notes", use_container_width=True):
                with st.spinner("Generating study notes..."):
                    notes = generate_study_notes(st.session_state.vector_store)
                    st.session_state.notes = notes
                    st.session_state.active_tab = "notes"
                    st.rerun()
        
        with col3:
            if st.button("Ask Questions", use_container_width=True):
                st.session_state.active_tab = "qa"
                st.rerun()

def load_qa_data() -> Dict:
    """Load Q&A data from JSON file"""
    try:
        if os.path.exists(QA_STORAGE_FILE):
            with open(QA_STORAGE_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading Q&A data: {str(e)}")
        return {}

def save_qa_data(data: Dict):
    """Save Q&A data to JSON file"""
    try:
        with open(QA_STORAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving Q&A data: {str(e)}")

def store_qa_pair(student_email: str, lecture_num: str, material_type: str, question: str, answer: str):
    """Store a Q&A pair in the JSON file"""
    qa_data = load_qa_data()
    
    # Create entry for student if doesn't exist
    if student_email not in qa_data:
        qa_data[student_email] = []
    
    # Add new Q&A pair
    qa_entry = {
        "timestamp": datetime.now().isoformat(),
        "lecture_number": lecture_num,
        "lecture_title": LECTURE_TITLES[lecture_num],
        "material_type": material_type,
        "question": question,
        "answer": answer
    }
    
    qa_data[student_email].append(qa_entry)
    save_qa_data(qa_data)
    return True

def render_qa_section(lecture_num: str):
    """Render the Q&A section"""
    st.markdown(f"### 🤔 Ask Questions About {LECTURE_TITLES[lecture_num]}")
    
    # Display previous Q&A
    for q, a in st.session_state.qa_history:
        st.markdown("---")
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
    
    # New question input
    question = st.text_area("Enter your question:")
    if st.button("Get Answer"):
        if question:
            with st.spinner("🤔 Thinking about your question..."):
                answer = get_ai_response(question, st.session_state.vector_store)
                st.session_state.qa_history.append((question, answer))
                
                # Store the Q&A pair with student email
                if st.session_state.email:
                    material_type = "slides" if st.session_state.material_type == "slides" else "transcript"
                    success = store_qa_pair(
                        student_email=st.session_state.email,
                        lecture_num=lecture_num,
                        material_type=material_type,
                        question=question,
                        answer=answer
                    )
                    if success:
                        st.success("Question and answer saved!")
                
                st.rerun()
        else:
            st.warning("Please enter a question.")

def render_content():
    """Render the main content based on active tab"""
    if st.session_state.active_tab == "summary" and st.session_state.summary:
        st.markdown("### 📝 Document Summary")
        st.write(st.session_state.summary)
        if st.button("← Back"):
            st.session_state.active_tab = "upload"
            st.rerun()
    
    elif st.session_state.active_tab == "notes" and st.session_state.notes:
        st.markdown("### 📚 Study Notes")
        st.write(st.session_state.notes)
        if st.button("← Back"):
            st.session_state.active_tab = "upload"
            st.rerun()
    
    elif st.session_state.active_tab == "qa":
        render_qa_section(st.session_state.current_lecture)
        if st.button("← Back"):
            st.session_state.active_tab = "upload"
            st.rerun()

def format_lecture_number(num: int) -> str:
    """Format lecture number as two digits"""
    return f"{num:02d}"

def generate_test_questions(lecture_num: str, content: str):
    """Generate MCQ questions for self-testing"""
    questions = []
    for _ in range(3):  # Generate 3 MCQ questions
        mcq = generate_mcq(
            topic=LECTURE_TITLES[lecture_num],
            question=f"Generate a question about {LECTURE_TITLES[lecture_num]}",
            context=content
        )
        if 'error' not in mcq:
            questions.append(mcq)
    return questions

def load_lecture_material(lecture_num: str, material_type: str) -> bool:
    """Load and process the selected lecture material"""
    try:
        # Get the correct filename from our mapping
        filename = LECTURE_FILES[lecture_num][material_type]
        
        # Construct full file path
        folder = "materials" if material_type == "slides" else "transcript"
        file_path = os.path.join(folder, filename)
        
        # Show loading animation with custom HTML/CSS
        with st.spinner():
            st.markdown("""
                <div class="loading-spinner">
                    <style>
                        .loading-spinner {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            margin: 20px;
                        }
                        .loading-spinner::after {
                            content: "🤔";
                            font-size: 2em;
                            animation: thinking 1s infinite;
                        }
                        @keyframes thinking {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                    </style>
                </div>
            """, unsafe_allow_html=True)
            
            # Extract text and prepare vector store
            text_content = extract_text_from_file(file_path)
            if text_content:
                st.session_state.vector_store = prepare_vector_store(text_content)
                return True
        return False
    except Exception as e:
        st.error(f"Error loading material: {str(e)}")
        return False

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(page_title="TA-Lite", layout="wide")
    
    # Custom title with blue color
    st.markdown("""
        <h1 style='color: #1E88E5;'>TA-Lite: Academic Assistant</h1>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Login section
    render_login()
    
    # Create sidebar for navigation
    with st.sidebar:
        st.markdown("## What's on your mind?")
        selected_option = st.radio(
            "Choose an option:",
            ["Lecture Review", "Ask Questions", "Test Yourself"],
            index=0
        )
    
    if selected_option == "Lecture Review":
        # Lecture selection (only shown for Lecture Review)
        st.markdown("### 📚 Select Lecture")
        lecture_options = [f"Lecture {format_lecture_number(i)}: {LECTURE_TITLES[format_lecture_number(i)]}" 
                          for i in range(1, 11)]
        
        selected_lecture = st.selectbox(
            "Which lecture would you like help with?",
            lecture_options,
            index=None,
            placeholder="Select a lecture..."
        )
        
        if selected_lecture:
            # Extract lecture number
            lecture_num = format_lecture_number(int(selected_lecture.split(':')[0].split()[-1]))
            st.session_state.current_lecture = lecture_num
            
            # Material type selection
            st.markdown("### 📑 Select Material Type")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Lecture Slides", use_container_width=True):
                    st.session_state.material_type = "slides"
                    if load_lecture_material(lecture_num, "slides"):
                        st.success("Lecture slides loaded successfully!")
                        st.session_state.action = None
                        st.rerun()
            
            with col2:
                if st.button("📝 Lecture Transcript", use_container_width=True):
                    st.session_state.material_type = "transcript"
                    if load_lecture_material(lecture_num, "transcript"):
                        st.success("Lecture transcript loaded successfully!")
                        st.session_state.action = None
                        st.rerun()
            
            # Show content based on selected option
            if st.session_state.vector_store and st.session_state.material_type:
                material_type_display = "slides" if st.session_state.material_type == "slides" else "transcript"
                
                st.markdown("### 📋 Lecture Review")
                st.markdown("Choose what you'd like to review:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📋 Generate Summary", use_container_width=True):
                        st.session_state.action = "📋 Summarize"
                        st.session_state.summary = None
                        st.session_state.notes = None
                        with st.spinner("🤔 Analyzing and summarizing the content..."):
                            summary = generate_summary(st.session_state.vector_store)
                            st.session_state.summary = summary
                        st.rerun()
                
                with col2:
                    if st.button("📚 Generate Study Notes", use_container_width=True):
                        st.session_state.action = "📚 Make Notes"
                        st.session_state.summary = None
                        st.session_state.notes = None
                        with st.spinner("🤔 Generating comprehensive study notes..."):
                            notes = generate_study_notes(st.session_state.vector_store)
                            st.session_state.notes = notes
                        st.rerun()
                
                # Display content based on action
                if st.session_state.action == "📋 Summarize" and st.session_state.summary:
                    st.markdown(f"### 📝 Summary from {material_type_display}")
                    st.markdown("---")
                    st.write(st.session_state.summary)
                
                elif st.session_state.action == "📚 Make Notes" and st.session_state.notes:
                    st.markdown(f"### 📚 Study Notes from {material_type_display}")
                    st.markdown("---")
                    st.write(st.session_state.notes)
    
    elif selected_option == "Ask Questions":
        st.markdown("### ❓ Ask Questions")
        st.markdown("Ask any question about the course content:")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question_asked := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question_asked})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(question_asked)
        
            # Display assistant response in chat message container
            with st.chat_message("assistant"):

#        with st.form(key="Question Form"):
#            question_asked = st.text_area("Your Question:", placeholder="Type your question here...", height=100)
#            submit_button = st.form_submit_button("Get Answer")

        #if submit_button and question_asked:
                with st.spinner("🤔 Thinking about your answer..."):
                    pdf_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"type": "pdf"}})
                    transcript_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"type": "transcript"}})
                    pdf_chunks = pdf_retriever.invoke(question_asked)
                    transcript_chunks = transcript_retriever.invoke(question_asked)
                    pdf_top = rerank(question_asked, pdf_chunks, top_n=3)
                    transcript_top = rerank(question_asked, transcript_chunks, top_n=3)
                    final_docs = pdf_top + transcript_top
                    final_rerank = rerank(question_asked, final_docs, top_n=5)

                    context = "\n\n".join([
                        f"[{doc.metadata['type'].upper()} - {doc.metadata['source']}] {doc.page_content}"
                        for doc in final_rerank
                    ])
                    prompt = f"""
                    You are a helpful assistant. Based on the following course materials:

                    {context}

                    Student asked:
                    {question_asked}

                    Respond with a hint or Indirect - Lead them to answer with hints. 
                    DO NOT GIVE DIRECT ANSWERS!!!!

                    EXPLAIN LIKE I'M 18 Year Old 
                    """

                    response = llm.invoke(prompt)
                    rag_output = response.content
                    sources = [doc.metadata['source'] for doc in final_docs]

                    # Add to QA history
                    qa_pair = {
                        'question': question_asked,
                        'answer': rag_output,
                        'sources': sources,
                        'timestamp': datetime.now().strftime("%I:%M %p")
                    }
                    st.session_state.qa_history.insert(0, qa_pair)  # Add to beginning of list

                    # Display the current answer in an expandable container
                    st.markdown("### 📝 Answer")
                    st.markdown(rag_output)
                    st.markdown("**Sources:**")
                    st.markdown(", ".join(sources))

                    if config['external_ref'] == 'Yes' or config['external_ref'] == "When Unavailable in course content":
                        final_response = AI_agent.generate_ai_response(question_asked+ rag_output)

                        #q2 = input("Your response:\n")
                        #AI_agent.generate_ai_response(str(q2))
                    else:
                        final_response = AI_agent.generate_ai_response(question_asked)

                        #q2 = input("Your response:\n")
                        #AI_agent.generate_ai_response(str(q2))

                    #st.markdown("### 📝 Answer")
                    #st.markdown(final_response)
                    #st.markdown("**Sources:**")
                    #st.markdown(", ".join(sources))
                    resp2 = st.write_stream("### 📝 Answer\n"+final_response+"**Sources:**"+", ".join(sources))
            st.session_state.messages.append({"role": "assistant", "content": resp2})

        # Display QA History
        if st.session_state.qa_history:
            st.markdown("### 📚 Previous Questions & Answers")
            for i, qa in enumerate(st.session_state.qa_history):
                with st.expander(f"Q: {qa['question'][:100]}... ({qa['timestamp']})"):
                    st.markdown("**Question:**")
                    st.markdown(qa['question'])
                    st.markdown("**Answer:**")
                    st.markdown(qa['answer'])
                    st.markdown("**Sources:**")
                    st.markdown(", ".join(qa['sources']))
    
    elif selected_option == "Test Yourself":
        st.markdown("### 📝 Test Yourself")
        st.markdown("Generate questions to test your understanding:")
        
        with st.form(key="Test Form"):
            topic = st.text_area("Topic to test:", placeholder="Enter a specific topic or concept you want to test yourself on...")
            num_questions = st.slider("Number of questions:", min_value=1, max_value=5, value=3)
            submit_test = st.form_submit_button("Generate Questions")

        if submit_test and topic:
            with st.spinner("🤔 Generating test questions..."):
                # Get relevant content for the topic
                pdf_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"type": "pdf"}})
                transcript_retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"type": "transcript"}})
                pdf_chunks = pdf_retriever.invoke(topic)
                transcript_chunks = transcript_retriever.invoke(topic)
                pdf_top = rerank(topic, pdf_chunks, top_n=3)
                transcript_top = rerank(topic, transcript_chunks, top_n=3)
                final_docs = pdf_top + transcript_top
                final_rerank = rerank(topic, final_docs, top_n=5)

                context = "\n\n".join([
                    f"[{doc.metadata['type'].upper()} - {doc.metadata['source']}] {doc.page_content}"
                    for doc in final_rerank
                ])
                
                # Generate questions using Mistral
                prompt = f"""
                Based on the following course materials, generate {num_questions} multiple-choice questions about "{topic}".
                Each question should have 4 options (A, B, C, D) with only one correct answer.
                Include an explanation for the correct answer.
                
                Course materials:
                {context}
                
                Format each question as follows:
                
                Q1: [Question text]
                A) [Option A]
                B) [Option B]
                C) [Option C]
                D) [Option D]
                Correct Answer: [Letter]
                Explanation: [Brief explanation]
                
                ---
                
                Q2: [Question text]
                ...
                """
                
                response = llm.invoke(prompt)
                questions = response.content
                
                # Display questions in an interactive format
                st.markdown("### 📝 Test Questions")
                st.markdown("---")
                
                # Split questions by "---" or "Q" followed by a number
                question_blocks = questions.split("---")
                
                for i, block in enumerate(question_blocks):
                    if not block.strip():
                        continue
                        
                    st.markdown(f"#### Question {i+1}")
                    
                    # Extract question text and options
                    lines = block.strip().split("\n")
                    question_text = lines[0].split(":", 1)[1].strip() if ":" in lines[0] else lines[0].strip()
                    st.markdown(f"**{question_text}**")
                    
                    # Display options as radio buttons
                    options = {}
                    for line in lines[1:5]:
                        if line.startswith(("A)", "B)", "C)", "D)")):
                            option_letter = line[0]
                            option_text = line[2:].strip()
                            options[option_letter] = option_text
                    
                    # Create radio buttons for options
                    selected_option = st.radio(
                        f"Select your answer for Question {i+1}:",
                        options=list(options.keys()),
                        format_func=lambda x: f"{x}) {options[x]}",
                        key=f"q{i}"
                    )
                    
                    # Check if user has selected an answer
                    if selected_option:
                        # Extract correct answer and explanation
                        correct_answer = None
                        explanation = ""
                        
                        for line in lines:
                            if line.startswith("Correct Answer:"):
                                correct_answer = line.split(":", 1)[1].strip()
                            elif line.startswith("Explanation:"):
                                explanation = line.split(":", 1)[1].strip()
                        
                        # Show feedback
                        if correct_answer:
                            if selected_option == correct_answer:
                                st.success("✅ Correct!")
                            else:
                                st.error(f"❌ Incorrect. The correct answer is {correct_answer}.")
                            
                            if explanation:
                                st.info(f"**Explanation:** {explanation}")
                    
                    st.markdown("---")

if __name__ == "__main__":
    main() 