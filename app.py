import streamlit as st
from backend import extract_text_from_pdf, prepare_vector_store, get_hint_response, load_prof_settings

st.title("🧠 LLM-Powered Guided Doubt Solver")

uploaded_file = st.file_uploader("Upload course material (PDF)", type="pdf")
question = st.text_input("Enter your doubt:")

if uploaded_file:
    with open(f"data/course_materials/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("Course material uploaded!")

    text = extract_text_from_pdf(f"data/course_materials/{uploaded_file.name}")
    db = prepare_vector_store(text)

    if question:
        settings = load_prof_settings()
        response = get_hint_response(question, db, settings["hint_level"])
        st.markdown("### 🤖 Guided Hint")
        st.info(response)
