import json
import requests
import sys
import os
import json

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

#### QUESTION HERE ####
question_asked = "What is the primary advantage of LoRA, and how does LoRA modify the behavior of pre-trained models during fine-tuning?"

######## EXTRACTING SEARCH KEYWORD FROM DOUBT QUERY ##########
os.makedirs("teacher_data/config", exist_ok=True)
CONFIG_FILE = "teacher_data/config/settings_v2.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
print("\n\n######## EXTRACTING SEARCH KEYWORD FROM DOUBT QUERY ##########")

if config['external_ref'] == 'Yes' or config['external_ref'] == "When Unavailable in course content":
    AI_agent.generate_ai_response(question_asked+ rag_output)

    q2 = input("Your response:\n")
    AI_agent.generate_ai_response(str(q2))
else:
    AI_agent.generate_ai_response(question_asked)

    q2 = input("Your response:\n")
    AI_agent.generate_ai_response(str(q2))

if config['external_ref'] == "Yes":
    AI_agent_search_key = AIAgent(str("Examine the question and tell me what is the most important concept a student needs to understand to answer this question? \n **DO NOT ANSWER THE QUESTION. ONLY give direct answer with no extra details**\n Examples responses: 'Neural Networks', 'Transformers', 'Encoding', 'Tokenization'"))
    search_key = AI_agent_search_key.generate_ai_response(question_asked)
    print(f"Search key Identified: {search_key}")

