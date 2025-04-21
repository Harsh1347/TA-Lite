import json
import requests
import sys
import streamlit as st
import os
from datetime import datetime

print("Python executable:", sys.executable)

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
        print("GPT-4o Response:")

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
        print(response_json)
        assistant_reply = response_json["choices"][0]["message"]["content"] 
        self.full_transcript.append({
            "role": "assistant",
            "content": assistant_reply
        })
        print(json.dumps(self.full_transcript))
        return assistant_reply

st.set_page_config(page_title="Exam Preperation Help", layout="centered")
st.title("Exam Preperation Help")
st.markdown("Let's get you exam ready.")
file_path = 'example.txt'

with open('teacher_data/materials/course_content.txt', 'r') as file:
    course_content = str(file.read())

question = ""
topic = st.text_input("Which topic do you want help in?", value=question)
if st.button("Let's test how much you know"):
    AI_ques_gen = AIAgent(str(f"Help me with my final prep by asking me 5 questions (NOT multiple choice) about the below topic and then evaluate my responses based on correctness. \n**LIMIT YOUR RESPONSES TO THE BELOW CONTENTS:\n**{course_content}\n\nHere is the topic name:"))
    agent_response = AI_ques_gen.generate_ai_response(topic)

AI_ques_gen = AIAgent(str(f"Help me with my final prep by asking me 5 questions (NOT multiple choice) about the below topic and then evaluate my responses based on correctness. \n**DO NOT EXCEED THE SCOPE OF THE BELOW CONTENTS:\n**{course_content}\n\nHere is the topic name:"))

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_container = st.container()

# Display all previous messages
with chat_container:
    for entry in st.session_state.chat_history:
        role, message = entry
        if role == "user":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 AI:** {message}")

# Input container at the bottom
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", placeholder="Type a message...")
    submit = st.form_submit_button("Send")

# On form submission
counter = 0
if submit and user_input:
    # Save user message
    st.session_state.chat_history.append(("user", user_input))
    
    # Get response from your model
    if counter < 1:
        ai_response = AI_ques_gen.generate_ai_response(user_input)
    else:
        ai_response = AI_ques_gen.generate_ai_response("Now only evaluate these responses. ** DO NOT GENERATE MORE QUESTIONS** \nRESPONSES:\n" + user_input)
    counter += 1
    # Save AI response
    st.session_state.chat_history.append(("ai", ai_response))
    
    # Rerun to show updated conversation
    st.rerun()
