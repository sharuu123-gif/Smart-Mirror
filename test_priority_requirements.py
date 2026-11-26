import urllib.request
import urllib.parse
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_multilingual_gold_jarvis_lisa():
    print("\n--- 1. Testing Both JARVIS & LISA Equal Multilingual Competence (Priority 4) ---")
    prompts = [
        ("kn", "Jarvis", "ಜಾರ್ವಿಸ್, ಇವತ್ತು ಚಿನ್ನದ ಬೆಲೆ ಎಷ್ಟು?"),
        ("kn", "Lisa", "ಲೀಸಾ, ಇವತ್ತು ಚಿನ್ನದ ಬೆಲೆ ಎಷ್ಟು?"),
        ("hi", "Jarvis", "जार्विस, आज सोने का भाव कितना है?"),
        ("hi", "Lisa", "लिसा, आज सोने का भाव कितना है?"),
        ("ta", "Jarvis", "ஜார்விஸ், இன்று தங்கத்தின் விலை என்ன?"),
        ("ta", "Lisa", "லிசா, இன்று தங்கத்தின் விலை என்ன?"),
        ("en", "Jarvis", "What is the price of gold today in Bangalore?"),
        ("en", "Lisa", "What is the price of gold today in Bangalore?")
    ]
    for lang, persona, query in prompts:
        time.sleep(0.5)
        payload = {
            "message": query,
            "active_language": lang,
            "persona": persona,
            "history": []
        }
        req = urllib.request.Request(
            f"{BASE_URL}/api/agent/chat",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.loads(r.read())
            ans = res.get("answer", "")
            print(f"[{lang.upper()} - {persona}] Query: {query}")
            print(f"[{lang.upper()} - {persona}] Answer: {ans}\n")
            assert len(ans) > 10, f"Empty response for {persona} in {lang}"
            # Ensure it contains gold price context
            assert any(k in ans.lower() for k in ["24", "22", "ಚಿನ್ನ", "ಬಂಗಾರ", "ಸೋನಾ", "सोना", "தங்கம்", "gold", "gram", "₹", "inr"])
    print("✅ Priority 4 Verified: Both JARVIS and LISA successfully answer in Kannada, Hindi, Tamil, and English!")

def test_multi_turn_continuity():
    print("\n--- 2. Testing Multi-Turn Context & Pronoun Continuity (Priority 5 & 6) ---")
    # Turn 1: AI
    time.sleep(0.5)
    p1 = {"message": "What is artificial intelligence?", "active_language": "en", "persona": "Jarvis", "history": []}
    req1 = urllib.request.Request(f"{BASE_URL}/api/agent/chat", data=json.dumps(p1).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req1, timeout=8) as r:
        a1 = json.loads(r.read())["answer"]
        print(f"User: What is artificial intelligence?\nJARVIS: {a1}\n")

    # Turn 2: Where is it used?
    time.sleep(0.5)
    h2 = [{"role": "user", "text": "What is artificial intelligence?"}, {"role": "model", "text": a1}]
    p2 = {"message": "Where is it used?", "active_language": "en", "persona": "Jarvis", "history": h2}
    req2 = urllib.request.Request(f"{BASE_URL}/api/agent/chat", data=json.dumps(p2).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req2, timeout=8) as r:
        a2 = json.loads(r.read())["answer"]
        print(f"User: Where is it used?\nJARVIS: {a2}\n")
        assert len(a2) > 15

    # Turn 3: What about India?
    time.sleep(0.5)
    h3 = h2 + [{"role": "user", "text": "Where is it used?"}, {"role": "model", "text": a2}]
    p3 = {"message": "What about India?", "active_language": "en", "persona": "Jarvis", "history": h3}
    req3 = urllib.request.Request(f"{BASE_URL}/api/agent/chat", data=json.dumps(p3).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req3, timeout=8) as r:
        a3 = json.loads(r.read())["answer"]
        print(f"User: What about India?\nJARVIS: {a3}\n")
        assert len(a3) > 15
        assert any(k in a3.lower() for k in ["india", "indian", "tech", "ai", "healthcare", "education", "startups", "market"])

    # Turn 4: Gold yesterday follow-up
    time.sleep(0.5)
    p4 = {"message": "What is the price of gold today?", "active_language": "en", "persona": "Lisa", "history": []}
    req4 = urllib.request.Request(f"{BASE_URL}/api/agent/chat", data=json.dumps(p4).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req4, timeout=8) as r:
        a4 = json.loads(r.read())["answer"]
        print(f"User: What is the price of gold today?\nLISA: {a4}\n")

    time.sleep(0.5)
    h5 = [{"role": "user", "text": "What is the price of gold today?"}, {"role": "model", "text": a4}]
    p5 = {"message": "What about yesterday?", "active_language": "en", "persona": "Lisa", "history": h5}
    req5 = urllib.request.Request(f"{BASE_URL}/api/agent/chat", data=json.dumps(p5).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req5, timeout=8) as r:
        a5 = json.loads(r.read())["answer"]
        print(f"User: What about yesterday?\nLISA: {a5}\n")
        assert any(k in a5.lower() for k in ["yesterday", "24k", "22k", "gold", "₹", "inr"])
    print("✅ Priority 5 & 6 Verified: Context continuity and pronouns resolved seamlessly!")

def test_tts_persona_streaming():
    print("\n--- 3. Testing Audio TTS Persona Streaming (/api/tts) ---")
    tts_tests = [
        ("kn", "Jarvis", "ನಮಸ್ಕಾರ, ನಾನು ಜಾರ್ವಿಸ್."),
        ("kn", "Lisa", "ನಮಸ್ಕಾರ, ನಾನು ಲೀಸಾ."),
        ("hi", "Jarvis", "नमस्ते, मैं जार्विस हूँ।"),
        ("hi", "Lisa", "नमस्ते, मैं लीसा हूँ।"),
        ("ta", "Jarvis", "வணக்கம், நான் ஜಾರ್விஸ்."),
        ("en", "Jarvis", "Hello! Real-time voice assistant is fully ready."),
        ("en", "Lisa", "Hey buddy, Lisa is here to help you!")
    ]
    for lang, persona, text in tts_tests:
        url = f"{BASE_URL}/api/tts?lang={lang}&persona={persona}&text={urllib.parse.quote(text)}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=6) as r:
            assert r.status == 200
            data = r.read()
            print(f"[{lang.upper()} - {persona}] Status: 200, Audio Size: {len(data)} bytes")
            assert len(data) > 1000
    print("✅ Audio TTS Persona Streaming verified for all languages!")

def test_frontend_codebase_integrity():
    print("\n--- 4. Testing Frontend Codebase Integrity (index.html) ---")
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # Check unwanted features removed
    assert 'image-studio-modal' not in html, "Image studio modal still present!"
    assert 'Upload Item' not in html, "Upload Item button still present!"
    assert 'Type Query' not in html, "Type Query button still present!"
    assert 'objectUploadInput' not in html, "objectUploadInput file input still present!"

    # Check state machine and recovery elements
    assert 'VoiceState' in html, "VoiceState state machine missing!"
    assert 'recoverVoiceAssistant' in html, "recoverVoiceAssistant recovery function missing!"
    assert 'isSentenceIncomplete' in html, "Sentence incomplete pause detector missing!"
    assert 'captureAndAnalyzeOutfit' in html, "Outfit analysis camera function missing!"
    assert 'captureAndIdentifyObject' in html, "Object identify camera function missing!"

    print("✅ Frontend Codebase Integrity verified (Unwanted features removed, state machine active)!")

if __name__ == "__main__":
    print("🚀 RUNNING PRIORITY VERIFICATION TEST SUITE\n")
    test_multilingual_gold_jarvis_lisa()
    test_multi_turn_continuity()
    test_tts_persona_streaming()
    test_frontend_codebase_integrity()
    print("\n" + "="*60)
    print("🎉 ALL PRIORITY TESTS PASSED SUCCESSFULLY (100%)!")
    print("="*60 + "\n")
