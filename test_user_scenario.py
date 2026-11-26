import urllib.request
import json

def chat_turn(msg, history, lang='en', weather='28°C Partly Cloudy in Chennai'):
    url = "http://127.0.0.1:5000/api/agent/chat"
    payload = {
        "message": msg,
        "text": msg,
        "active_language": lang,
        "persona": "Jarvis",
        "weather": weather,
        "history": history
    }
    req = urllib.request.Request(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload).encode('utf-8'))
    with urllib.request.urlopen(req, timeout=10) as res:
        data = json.loads(res.read().decode('utf-8'))
        return data

print("==================================================")
print("TESTING USER SCENARIO CONVERSATION SEQUENCE")
print("==================================================")

history = []

turns = [
    ("User: Hey Jarvis!", "Hey Jarvis!", "en"),
    ("User: I'm getting ready for college.", "I'm getting ready for college.", "en"),
    ("User: What is the weather?", "What is the weather?", "en"),
    ("User: Suggest something to wear.", "Suggest something to wear.", "en"),
    ("User: Jarvis, hegiddiya?", "Jarvis, hegiddiya?", "kn"),
    ("User: Jarvis, play believer", "Jarvis, play believer", "en")
]

for label, text, lang in turns:
    res = chat_turn(text, history, lang)
    ans = res.get('answer', '')
    intent = res.get('intent', '')
    song = res.get('song_query', '')
    print(label)
    print(f"JARVIS ({intent}): {ans}")
    if song:
        print(f"  🎵 Music Query Triggered: {song}")
    print("-" * 50)
    history.append({"role": "user", "text": text})
    history.append({"role": "assistant", "text": ans})

print("==================================================")
