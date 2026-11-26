import os
import json
import re
import requests
import dotenv

dotenv.load_dotenv('C:/Users/Sunil/OneDrive/Desktop/mirror/.env')
key = os.environ.get('GROQ_API_KEY')
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

def parse_agent_response(resp_obj):
    # Case 1: Standard 200 OK
    if resp_obj.status_code == 200:
        raw = resp_obj.json()["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {"intent": "GENERAL_CHAT", "answer": raw}

    # Case 2: json_validate_failed but failed_generation has the exact text
    try:
        err_data = resp_obj.json().get('error', {})
        if 'failed_generation' in err_data:
            fg = err_data['failed_generation'].strip()
            match = re.search(r'\{.*\}', fg, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {"intent": "GENERAL_CHAT", "answer": fg}
    except Exception:
        pass

    return {"intent": "GENERAL_CHAT", "answer": "I am right here with you, buddy! How can I help?"}

system_prompt = """You are Jarvis, a smart, warm, enthusiastic everyday companion and best friend talking through the user's Smart Mirror.
CONVERSATIONAL STYLE & PERSONALITY:
- Talk naturally, warmly, and casually like two close friends chatting in everyday life.
- Always be supportive, energetic, and engaging. Never be robotic.
- Understand normal casual sentences (e.g., "I'm getting ready for college", "Tell me a joke", "What are you doing?").
- Context Memory: Remember recent conversation turns and provide helpful follow-ups without repeating.
- Short Spoken Responses: Keep voice answers concise (1 to 3 lively sentences, maximum 35 words).
- Multilingual & Romanized Indic Intelligence:
  * Romanized Kannada (Kanglish, e.g. "hegiddiya", "oota aaytha", "enu samachara") -> Reply in warm Romanized Kannada ("Chennagiddini buddy! Neenu hegiddiya?").
  * Native Kannada script -> Reply in fluent Kannada.
  * Romanized Telugu (e.g. "ela unnavu", "tinnava") -> Reply in friendly Telugu ("Bagunnanu buddy! Nuvvu ela unnavu?").
  * Romanized Hindi / Hinglish ("kaise ho", "kya chal raha hai") -> Reply in friendly Hindi/Hinglish ("Badhiya hoon buddy! Aap batao?").
  * English -> Reply in friendly, casual English.

Always output a valid JSON object strictly matching this schema:
{
  "intent": "GENERAL_CHAT" | "MUSIC_PLAY" | "IMAGE_OUTFIT" | "WEATHER" | "TIME" | "NEWS",
  "answer": "short spoken text",
  "song_query": null
}
"""

test_dialogue = [
    "Hey Jarvis!",
    "I am getting ready for college.",
    "What is the weather today?",
    "Suggest something comfortable to wear.",
    "Jarvis, hegiddiya?"
]

messages = [{"role": "system", "content": system_prompt}]

for turn in test_dialogue:
    messages.append({"role": "user", "content": turn})
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 120
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=8)
    ans_json = parse_agent_response(r)
    print(f"User: {turn}")
    print(f"JARVIS: {ans_json.get('answer')}")
    print(f"Intent: {ans_json.get('intent')}")
    print("-" * 50)
    messages.append({"role": "assistant", "content": ans_json.get("answer", "")})
