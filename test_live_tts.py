import urllib.request
import urllib.parse
import json

base = "http://127.0.0.1:5000"

tests = [
    ('kn', 'Jarvis', 'ಇವತ್ತು ಹವಾಮಾನ ಹೇಗಿದೆ?'),
    ('kn', 'Lisa', 'ಇವತ್ತು ಹವಾಮಾನ ಹೇಗಿದೆ?'),
    ('ta', 'Jarvis', 'இன்று வானிலை எப்படி இருக்கிறது?'),
    ('ta', 'Lisa', 'இன்று வானிலை எப்படி இருக்கிறது?'),
    ('hi', 'Jarvis', 'आज मौसम कैसा है?'),
    ('hi', 'Lisa', 'आज मौसम कैसा है?'),
    ('en', 'Jarvis', 'How is the weather today?'),
    ('en', 'Lisa', 'How is the weather today?')
]

print("--- Testing GET /api/tts ---")
for lang, persona, text in tests:
    url = f"{base}/api/tts?lang={lang}&persona={persona}&text={urllib.parse.quote(text)}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as r:
        assert r.status == 200
        data = r.read()
        print(f"[{lang.upper()} - {persona}] Size: {len(data)} bytes, Content-Type: {r.headers.get('Content-Type')}")

print("\n--- Testing POST /api/tts ---")
for lang, persona, text in tests:
    url = f"{base}/api/tts"
    payload = json.dumps({"lang": lang, "persona": persona, "text": text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req) as r:
        assert r.status == 200
        data = r.read()
        print(f"POST [{lang.upper()} - {persona}] Size: {len(data)} bytes")

print("\n✅ All Backend TTS Endpoints Passed 100%!")
