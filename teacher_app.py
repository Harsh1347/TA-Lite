import streamlit as st
import os
import json
from datetime import date

# Ensure folders exist
os.makedirs("teacher_data/config", exist_ok=True)
os.makedirs("teacher_data/materials", exist_ok=True)

CONFIG_FILE = "teacher_data/config/settings.json"

# Sidebar navigation
st.sidebar.title("👩‍🏫 Teacher Portal")
page = st.sidebar.radio("Go to", ["1. Configuration Settings", "2. Upload Class Material"])

# Page 1: Configuration Settings
if page.startswith("1"):
    st.title("⚙️ Configuration Settings")

    # Load existing config
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    else:
        config = {
            "hint_level": 2,
            "custom_prompt": "",
            "workspace_name": "",
            "exam_date": str(date.today())
        }

    st.subheader("Set your preferences:")
    hint_level = st.selectbox("Hint Level (1 = light, 3 = detailed)", [1, 2, 3], index=config["hint_level"] - 1)
    custom_prompt = st.text_area("Custom Prompt Template", value=config["custom_prompt"], height=150)
    workspace_name = st.text_input("Workspace Name", value=config["workspace_name"])
    exam_date = st.date_input("Next Exam Date", value=date.fromisoformat(config["exam_date"]))

    if st.button("💾 Save Configuration"):
        new_config = {
            "hint_level": hint_level,
            "custom_prompt": custom_prompt,
            "workspace_name": workspace_name,
            "exam_date": str(exam_date)
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(new_config, f, indent=4)
        st.success("✅ Configuration saved successfully.")

# Page 2: Upload Class Material
elif page.startswith("2"):
    st.title("📚 Upload Class Material")

    uploaded_files = st.file_uploader(
    "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                file_path = os.path.join("teacher_data/materials", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_file.name}")
