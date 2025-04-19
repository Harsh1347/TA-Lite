import json
import requests
import sys
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
        print(assistant_reply)


with open('instructor_prompts/instructor_settings_prompt_20250418_204033.txt', 'r') as file:
    sys_prompt_captured = file.read()
AI_agent = AIAgent(str(sys_prompt_captured))

AI_agent.generate_ai_response("What is the primary advantage of LoRA, and how does LoRA modify the behavior of pre-trained models during fine-tuning?")

q2 = input("prompt")
AI_agent.generate_ai_response(str(q2))

