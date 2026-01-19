import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(override=True)

# --- CONFIGURATION ---
API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("GEMINI_BASE_URL")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
USER_NAME = os.getenv("USER_NAME")
USER_ROLE = os.getenv("USER_ROLE")
MODEL_NAME = os.getenv("SYSTEM_MODEL")

if not API_KEY:
    print("WARNING: API Key is missing.")

# --- TOOLS ---
def google_search(query):
    print(f"Searching Google for: {query}")
    url = "https://google.serper.dev/search"
    payload = json.dumps({'q': query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        result = response.json().get("organic", [])[:2]
        return json.dumps(result)
    except Exception as e:
        return f"Search Error: {e}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Use this to search for company info, recent news, or technical concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]

# --- DATA LOADER ---
def load_context():
    context = ""
    data_folder = "data"
    if not os.path.exists(data_folder): os.makedirs(data_folder)
    
    files = os.listdir(data_folder)
    print(f"Loading data for {USER_NAME}...")
    for filename in files:
        file_path = os.path.join(data_folder, filename)
        try:
            if filename.endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages: context += page.extract_text() + "\n"
            elif filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f: context += f.read()
        except Exception as e: print(f"Error loading {filename}: {e}")
            
    return context

GLOBAL_CONTEXT = load_context()

# --- BOT LOGIC ---
client = OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL,
    default_headers={"HTTP-Referer": "http://localhost:7860", "X-Title": "Interview Bot"},
    timeout=120.0
)

def system_prompt():
    return f"""
    You are acting as {USER_NAME}, a {USER_ROLE}.
    
    ## DUAL GOAL
    1. **Candidate:** Answer impressively using the RESUME CONTEXT.
    2. **Coach:** After the answer, explain WHY it is good.

    ## INSTRUCTIONS
    - Speak in First Person ("I", "Me").
    - Use the STAR Method (Situation, Task, Action, Result).
    - If asked about a company, use 'google_search' first.
    - End with a 'Coach's Note'.

    ## RESUME CONTEXT
    {GLOBAL_CONTEXT}
    """

def chat(message, history):
    # 1. Build Messages
    messages = [{"role": "system", "content": system_prompt()}]
    for entry in history:
        if isinstance(entry, (list, tuple)):
            messages.append({"role": "user", "content": entry[0]})
            if len(entry) > 1 and entry[1]: messages.append({"role": "assistant", "content": entry[1]})
        elif isinstance(entry, dict):
            messages.append(entry) 
    messages.append({"role": "user", "content": message})

    try:
        # First Call
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages, 
            tools=tools
        )
        msg = response.choices[0].message

        # 2. Check for Tools
        if msg.tool_calls:
            messages.append(msg)
            
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "google_search":
                    args = json.loads(tool_call.function.arguments)
                    res = google_search(**args)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res})
            
            # Second Call
            final_res = client.chat.completions.create(model=MODEL_NAME, messages=messages)
            return final_res.choices[0].message.content
        
        else:
            return msg.content

    except Exception as e:
        print(f"ERROR: {e}")
        return f"System Error: {e}"

# --- LAUNCH (Text Only) ---
if __name__ == "__main__":
    print("🚀 Launching Interface...")
    gr.ChatInterface(
        fn=chat,
        title=f"Interview with {USER_NAME}",
        description="Ask me anything. I will answer and provide a Coach's Critique."
    ).launch()