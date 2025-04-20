import streamlit as st
import os
from datetime import datetime
import json
from datetime import date


os.makedirs("teacher_data/config", exist_ok=True)
CONFIG_FILE = "teacher_data/config/settings_v2.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = {
            "goals": "",
            "depth": "",
            "direct_answers":"",
            "directness": "",
            "references": "",
            "external_ref":"",
            "autonomy": 3,
            "directness": ""
    }

st.set_page_config(page_title="Professor AI Preferences", layout="centered")

st.title("🧑‍🏫 AI Assistant Preferences Setup")
st.markdown("Help us configure the AI assistant to align with your teaching philosophy.")

st.header("1. What are your primary goals when helping students?")
goals = st.selectbox(
    "Select one option:",
    options=[
        "Encourage critical thinking",
        "Reinforce lectures",
        "Provide direct answers",
        "Point to resources",
    ]
)

st.header("3. How important is student autonomy in learning?")
autonomy = st.slider("Rate from 1 (Not Important) to 5 (Extremely Important):", 1, 5, 3)

st.header("5. Should the AI provide:")
col1, col2 = st.columns(2)
with col1:
    direct_answers = st.radio("Direct answers or Hints only?", ["Direct answers", "Hints Only", "Direct answers only when needed"], index=2)
    references = st.radio("References to course content?", ["Yes", "No", "Sometimes"], index=2)
with col2:
    questions = st.radio("Thought-provoking questions?", ["Yes", "No", "Sometimes"], index=2)
    external_ref = st.radio("References to external resources?", ["Yes", "No", "When Unavailable in course content"], index=2)

st.header("6. Depth of explanation?")

depth_preferences = st.radio("Do you want the responses based on student's level of understanding?", ["Yes", "No"], index=1)

st.header("7. Directness?")
directness = st.radio("How direct should the AI be when responding to a confused student?", [ "Direct - Just give the answer",
        "Indirect - Lead them to answer with hints",
        "Diagnostic - Try to identify misconceptions and correct them"], index=2)

st.markdown("---")
if st.button("Submit"):
    
    st.success("✅ Preferences Submitted")
    st.subheader("Here’s a summary of your preferences:")

#     prompt_template = f"""
# You are an AI teaching assistant helping university students.
# Your guidance style should follow these professor preferences:

# - Teaching goals: {goals}.
# - Student autonomy importance (on a scale of 1-5): {autonomy}.
# - AI should:
#     - Provide direct answers: {direct_answers.lower()}.
#     - Ask thought-provoking questions: {questions.lower()}.
#     - Refer to lecture slides: {references.lower()}.
#     - Recommend external resources (e.g., videos/papers): {external_ref.lower()}.
# - Depth of explanation should adapt to: {depth_preferences}.
# - Preferred style when student is confused: {directness}.

# You can choose to use the RAG agent we have to fetch any answers from the course material.
# Your responses should align with these preferences to balance between promoting self-learning and giving direct answers.
# """

#     st.text_area("📄 Generated Prompt", prompt_template, height=300)

#     # Optional: Save prompt to file
#     prompt_filename = f"instructor_settings_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
#     with open('instructor_prompts/'+prompt_filename, "w") as f:
#         f.write(prompt_template)

    # Build sections with conditionals
    ai_behavior_lines = []

    # Teaching goals
    goal_line = f"- Teaching goals: {goals}."
    
    # Autonomy rating
    autonomy_line = f"- Student autonomy importance (on a scale of 5): {autonomy}."

    # AI behavior
    if direct_answers == "Hints Only":
        ai_behavior_lines.append("    - Provide **hints but make sure not to include the direct answer**.")
    elif direct_answers == "Direct answers":
        ai_behavior_lines.append("    - Provide **direct answers**.")
    elif direct_answers == "Direct answers only when needed":
        ai_behavior_lines.append("    - Provide **direct answers only when needed**.")

    if questions == "Yes":
        ai_behavior_lines.append("    - Ask thought-provoking questions without deviating from the topic.")
    elif questions == "Sometimes":
        ai_behavior_lines.append("    - Sometimes ask thought-provoking questions.")

    if references == "Yes":
        ai_behavior_lines.append("    - **Make sure to refer to lecture slides.**")
    if references == "No":
        ai_behavior_lines.append("    - Don't refer to lecture slides.")
    elif references == "Sometimes":
        ai_behavior_lines.append("    - Sometimes refer to lecture slides.")

    if external_ref == "Yes":
        ai_behavior_lines.append("    - **Make sure to Recommend external resources.**")
    if external_ref == "No":
        ai_behavior_lines.append("    - Don't Recommend external resources.")
    elif external_ref == "When Unavailable in course content":
        ai_behavior_lines.append("    - Use external resources only when unavailable in course content.")

    # Depth of explanation
    depth_line = ""
    if depth_preferences == "Yes":
        depth_line = "- Responses should adapt to the student's level of understanding."

    # Directness style
    directness_line = f"- When student is confused, the approach should be: {directness}."

    # Final full prompt
    final_prompt = f"""You are an AI teaching assistant helping university students.
Your guidance style should follow these professor preferences:

{goal_line}
{autonomy_line}
- AI should:
{chr(10).join(ai_behavior_lines)}
{depth_line}
{directness_line}

You can choose to use the RAG agent we have to fetch any answers from the course material.
**YOUR MAIN FOCUS IS TO HELP THE STUDENT UNDERSTAND THE TOPIC, NOT TO ANSWER THE QUESTION**
Your responses should align with these preferences to balance between promoting self-learning and giving direct answers.
"""
    st.markdown("### ✨ Generated Prompt")
    st.text_area("LLM Prompt", final_prompt, height=350)
    
    #Save prompt to config

    new_config = {
        "goals": goals,
        "depth": depth_preferences,
        "direct_answers":direct_answers,
        "directness": directness,
        "references": references,
        "external_ref":external_ref,
        "autonomy": autonomy,
        "directness": directness
        #"exam_date": str(exam_date)
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(new_config, f, indent=4)
    st.success("✅ Configuration saved successfully.")

    #Save prompt to file
    prompt_filename = f"instructor_settings_prompt.txt"
    with open('instructor_prompts/'+prompt_filename, "w") as f:
        f.write(final_prompt)
