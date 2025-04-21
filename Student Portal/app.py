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
from rerank import rerank

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
    
    # Lecture selection
    st.markdown("### 📚 Select Lecture")
    
    with st.form(key="Doubt Solving"):
        question_asked = st.text_area("Ask You Doubt!")
        submit_button = st.form_submit_button()

    if submit_button:
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
        rag_output = (response.content)
        st.text_area(label="Answer to your question",value=rag_output+f"\nResources:{[doc.metadata['source'] for doc in final_docs]}")

    
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
        
        # Show action selection if material is loaded
        if st.session_state.vector_store and st.session_state.material_type:
            st.markdown("### 🎯 What would you like to do?")
            
            # Action buttons in columns
            col1, col2, col3 = st.columns(3)
            
            material_type_display = "slides" if st.session_state.material_type == "slides" else "transcript"
            
            with col1:
                if st.button("📋 Summarize", use_container_width=True):
                    st.session_state.action = "📋 Summarize"
                    st.session_state.summary = None
                    st.session_state.notes = None
                    with st.spinner("🤔 Analyzing and summarizing the content..."):
                        summary = generate_summary(st.session_state.vector_store)
                        st.session_state.summary = summary
                    st.rerun()
            
            with col2:
                if st.button("📚 Make Notes", use_container_width=True):
                    st.session_state.action = "📚 Make Notes"
                    st.session_state.summary = None
                    st.session_state.notes = None
                    with st.spinner("🤔 Generating comprehensive study notes..."):
                        notes = generate_study_notes(st.session_state.vector_store)
                        st.session_state.notes = notes
                    st.rerun()
            
            with col3:
                if st.button("❓ Ask Questions", use_container_width=True):
                    st.session_state.action = "❓ Ask Questions"
                    st.session_state.summary = None
                    st.session_state.notes = None
                    st.rerun()
            
            st.markdown("---")
            
            # Display content based on action
            if st.session_state.action == "📋 Summarize" and st.session_state.summary:
                st.markdown(f"### 📝 Summary from {material_type_display}")
                st.markdown("---")
                st.write(st.session_state.summary)
            
            elif st.session_state.action == "📚 Make Notes" and st.session_state.notes:
                st.markdown(f"### 📚 Study Notes from {material_type_display}")
                st.markdown("---")
                st.write(st.session_state.notes)
            
            elif st.session_state.action == "❓ Ask Questions":
                render_qa_section(lecture_num)

if __name__ == "__main__":
    main() 