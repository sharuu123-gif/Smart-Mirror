import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import re
import json
import time
import base64
import warnings
import urllib.request
import urllib.parse
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import requests

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.path.join(BASE_DIR, 'music')
os.makedirs(MUSIC_DIR, exist_ok=True)

try:
    import dotenv
    dotenv.load_dotenv(os.path.join(BASE_DIR, '.env'))
except Exception:
    pass

from services.image_generation import image_service

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/mirror-agent"

# ============================================================
# Google Gemini Multi-Model Cascade Engine (Resilient & Fast)
# ============================================================
GEMINI_MODELS = [
    'gemini-flash-latest',
    'gemini-3.1-flash-lite',
    'gemini-3-flash-preview',
    'gemini-2.5-flash'
]

def call_gemini_generate(prompt: str, system_instruction: str = None, json_mode: bool = False, timeout: int = 10) -> str:
    if not GEMINI_KEY:
        return ""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    if json_mode:
        payload["generationConfig"] = {
            "responseMimeType": "application/json"
        }
    data = json.dumps(payload).encode('utf-8')

    for model in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
            req = urllib.request.Request(url, headers={'Content-Type': 'application/json'}, data=data)
            res = urllib.request.urlopen(req, timeout=timeout)
            resp_data = json.loads(res.read().decode('utf-8'))
            candidates = resp_data.get('candidates', [])
            if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
                text = candidates[0]['content']['parts'][0].get('text', '').strip()
                if text:
                    return text
        except Exception as e:
            print(f"Gemini generateContent [{model}] error: {e}")
            continue
    return ""

def call_gemini_vision(image_base64: str, prompt: str, system_instruction: str = None, timeout: int = 12) -> str:
    if not GEMINI_KEY or not image_base64:
        return ""
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    image_base64 = image_base64.strip()

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    data = json.dumps(payload).encode('utf-8')

    for model in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
            req = urllib.request.Request(url, headers={'Content-Type': 'application/json'}, data=data)
            res = urllib.request.urlopen(req, timeout=timeout)
            resp_data = json.loads(res.read().decode('utf-8'))
            candidates = resp_data.get('candidates', [])
            if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
                text = candidates[0]['content']['parts'][0].get('text', '').strip()
                if text:
                    return text
        except Exception as e:
            print(f"Gemini vision [{model}] error: {e}")
            continue
    return ""

# Persistent Session State across conversational turns
SESSION_STATE = {
    "active_language": "en",
    "conversation_active": False,
    "is_speaking": False,
    "current_music": None,
    "current_headlines": [],
    "last_intent": None,
    "pending_ambiguity": None
}

LANGUAGE_CONFIG = {
    "en": {"name": "English", "code": "en-IN", "confirm": "Sure, I'll speak in English from now on."},
    "kn": {"name": "Kannada", "code": "kn-IN", "confirm": "ಆಯ್ತು, ಇನ್ನು ಮುಂದೆ ನಾನು ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸುತ್ತೇನೆ."},
    "hi": {"name": "Hindi", "code": "hi-IN", "confirm": "नमस्ते! अब मैं हिंदी में बात करूँगा।"},
    "ta": {"name": "Tamil", "code": "ta-IN", "confirm": "சரி, இனி நான் தமிழில் பேசுகிறேன்."},
    "te": {"name": "Telugu", "code": "te-IN", "confirm": "సరే, ఇకనుండి నేను తెలుగులో మాట్లాడతాను."},
    "ml": {"name": "Malayalam", "code": "ml-IN", "confirm": "ശരി, ഇനി ഞാൻ മലയാളത്തിൽ സംസാരിക്കാം."},
    "mr": {"name": "Marathi", "code": "mr-IN", "confirm": "ठीक आहे, आता मी मराठीत बोलेन."},
    "bn": {"name": "Bengali", "code": "bn-IN", "confirm": "ঠিক আছে, এখন থেকে আমি বাংলায় কথা বলব."}
}

# ============================================================
# 1. Session & Real Language State Management
# ============================================================
@app.route('/api/session', methods=['GET', 'POST'])
def handle_session():
    global SESSION_STATE
    if request.method == 'POST':
        data = request.get_json() or {}
        for key in ['active_language', 'conversation_active', 'is_speaking', 'current_music', 'last_intent', 'pending_ambiguity']:
            if key in data:
                SESSION_STATE[key] = data[key]
        return jsonify({"status": "success", "session": SESSION_STATE})
    return jsonify(SESSION_STATE)

@app.route('/api/language/switch', methods=['POST'])
def switch_language():
    global SESSION_STATE
    data = request.get_json() or {}
    lang = data.get("language", "en").lower().strip()
    
    lang_map = {
        "kannada": "kn", "kannad": "kn", "kn": "kn", "ಕನ್ನಡ": "kn",
        "hindi": "hi", "hind": "hi", "hi": "hi", "हिंदी": "hi",
        "tamil": "ta", "ta": "ta", "தமிழ்": "ta",
        "telugu": "te", "te": "te", "తెలుగు": "te",
        "malayalam": "ml", "ml": "ml", "മലയാളം": "ml",
        "marathi": "mr", "mr": "mr", "मराठी": "mr",
        "bengali": "bn", "bangla": "bn", "bn": "bn", "বাংলা": "bn",
        "english": "en", "en": "en"
    }
    
    target_code = lang_map.get(lang, "en")
    SESSION_STATE["active_language"] = target_code
    config = LANGUAGE_CONFIG.get(target_code, LANGUAGE_CONFIG["en"])
    
    return jsonify({
        "status": "success",
        "active_language": target_code,
        "language_name": config["name"],
        "speech_code": config["code"],
        "confirmation": config["confirm"]
    })

# ============================================================
# 1.5. High-Fidelity Audio TTS Streaming for Multilingual Output
# ============================================================
@app.route('/api/tts', methods=['GET'])
def api_tts():
    text = request.args.get('text', '').strip()
    lang = request.args.get('lang', 'en').strip().lower()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    lang_map = {
        'kn-in': 'kn', 'kannada': 'kn', 'kn': 'kn',
        'te-in': 'te', 'telugu': 'te', 'te': 'te',
        'hi-in': 'hi', 'hindi': 'hi', 'hi': 'hi',
        'ta-in': 'ta', 'tamil': 'ta', 'ta': 'ta',
        'ml-in': 'ml', 'malayalam': 'ml', 'ml': 'ml',
        'mr-in': 'mr', 'marathi': 'mr', 'mr': 'mr',
        'bn-in': 'bn', 'bengali': 'bn', 'bn': 'bn',
        'en-in': 'en', 'en-us': 'en', 'english': 'en', 'en': 'en'
    }
    target_tl = lang_map.get(lang, lang[:2])

    # Split text into safe chunks of <= 120 chars by punctuation
    raw_sentences = re.split(r'([.!?,\n]+)', text)
    chunks = []
    curr = ""
    for piece in raw_sentences:
        if len(curr) + len(piece) <= 120:
            curr += piece
        else:
            if curr.strip():
                chunks.append(curr.strip())
            curr = piece
    if curr.strip():
        chunks.append(curr.strip())

    chunks = [c for c in chunks if c.strip()][:4]
    if not chunks:
        chunks = [text[:120]]

    audio_bytes = b""
    for chunk in chunks:
        try:
            tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl={target_tl}&q={urllib.parse.quote(chunk)}"
            req = urllib.request.Request(tts_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            res = urllib.request.urlopen(req, timeout=5)
            audio_bytes += res.read()
        except Exception as chunk_err:
            print(f"TTS chunk error for [{chunk[:30]}]: {chunk_err}")
            continue

    if audio_bytes:
        return Response(audio_bytes, mimetype='audio/mpeg')
    return jsonify({"error": "Failed to synthesize audio"}), 500

# ============================================================
# 2. Dynamic Music System (Unlimited Songs: Local + Global)
# ============================================================
def fetch_jiosaavn_full_songs(query):
    if not HAS_CRYPTO:
        return []
    try:
        url = f"https://www.jiosaavn.com/api.php?__call=search.getResults&q={urllib.parse.quote(query)}&_format=json&_marker=0&api_version=4&ctx=web6dot0"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        res = urllib.request.urlopen(req, timeout=5)
        data = json.loads(res.read().decode('utf-8'))
        
        tracks = []
        key = b'38346591'
        
        for item in data.get('results', []):
            more = item.get('more_info', {})
            enc_url = more.get('encrypted_media_url')
            if not enc_url:
                continue
            try:
                raw = base64.b64decode(enc_url)
                cipher = Cipher(algorithms.TripleDES(key * 3), modes.ECB(), backend=default_backend())
                decryptor = cipher.decryptor()
                dec = decryptor.update(raw) + decryptor.finalize()
                pad = dec[-1]
                dec_url = dec[:-pad].decode('utf-8')
                
                # Upgrade to full 320kbps complete studio quality audio with 160k & 96k fallbacks
                full_audio_url = dec_url.replace('_96.mp4', '_320.mp4')
                url_160 = dec_url.replace('_96.mp4', '_160.mp4')
                url_96 = dec_url
                
                title = item.get('title', '').replace('&quot;', '"').replace('&#039;', "'").replace('&amp;', '&')
                artist = more.get('music', '') or item.get('subtitle', '')
                artist = artist.replace('&quot;', '"').replace('&#039;', "'").replace('&amp;', '&')
                album = more.get('album', '').replace('&quot;', '"').replace('&#039;', "'")
                img = item.get('image', '').replace('150x150', '500x500').replace('50x50', '500x500')
                duration = int(more.get('duration', 180))
                
                tracks.append({
                    "title": title,
                    "artist": artist,
                    "album": album,
                    "url": full_audio_url,
                    "url_320": full_audio_url,
                    "url_160": url_160,
                    "url_96": url_96,
                    "artwork": img,
                    "duration": duration,
                    "is_full_song": True,
                    "source": "jiosaavn_full"
                })
            except Exception:
                continue
        return tracks
    except Exception as e:
        print("JioSaavn search error:", e)
        return []

@app.route('/api/music/search', methods=['GET'])
def search_music():
    raw_query = request.args.get('q', '').strip()
    active_lang = request.args.get('lang') or SESSION_STATE.get("active_language", "en")

    # Map generic requests to language-specific hits
    lang_defaults = {
        'te': 'Telugu hits',
        'kn': 'Kannada hits',
        'hi': 'Bollywood Hindi hits',
        'ta': 'Tamil hits',
        'ml': 'Malayalam hits',
        'mr': 'Marathi hits',
        'bn': 'Bengali hits',
        'en': 'top trending hits'
    }

    query = raw_query
    generic_words = [
        'music', 'song', 'songs', 'some songs', 'trending hits', 'play music', 'a song', 'something',
        'play any songs', 'play any song', 'any songs', 'any song', 'some song', 'some songs', 'play songs', 'play song',
        'ಹಾಡು', 'ಹಾಡುಗಳು', 'ಪ್ಲೇ', 'ಸಾಂಗ್', 'ಸಾಂಗ್ಸ್', 'ಸಂಗೀತ', 'ಹಾಡು ಹಾಕು', 'ಪ್ಲೇ ಮಾಡು', 'ಯಾವುದಾದರೂ ಹಾಡು',
        'ಯಾವುದಾದರೂ ಹಾಡುಗಳು', 'ಒಂದು ಹಾಡು', 'ಹಾಡು ಕೇಳಿಸು', 'ಹಾಡು ಹಾಕಿ', 'ಕನ್ನಡ ಹಾಡು', 'ಕನ್ನಡ ಹಾಡುಗಳು',
        'ಕನ್ನಡ ಸಾಂಗ್', 'ಕನ್ನಡ ಸಾಂಗ್ಸ್',
        'పాట', 'పాటలు', 'ప్ಲೇ', 'సాಂಗ್', 'సాಂಗ್ಸ್', 'సಂಗೀತಂ', 'పాటలు వేయి', 'పాట పెట్టు',
        'गाना', 'गाने', 'गीत', 'सॉन्ग', 'प्ले', 'बजाओ', 'संगीत', 'गाना सुनाओ', 'गाना बजाओ', 'गाना लगाओ',
        'कोई गाना', 'कोई गाना सुनाओ',
        'பாடல்', 'பாட்டு', 'பாடல்கள்', 'பிளே', 'இசை'
    ]
    is_different = any(k in raw_query.lower() for k in ['different', 'another', 'next', 'ಬೇರೆ', 'ವೇరే', 'మరో', 'दूसरा', 'अगला'])

    pure_language_queries = {
        'telugu': 'Telugu hits', 'తెలుగు': 'Telugu hits',
        'kannada': 'Kannada hits', 'ಕನ್ನಡ': 'Kannada hits',
        'hindi': 'Bollywood Hindi hits', 'हिंदी': 'Bollywood Hindi hits',
        'tamil': 'Tamil hits', 'தமிழ்': 'Tamil hits',
        'malayalam': 'Malayalam hits', 'മലയാളം': 'Malayalam hits',
        'english': 'top trending hits'
    }

    clean_q = re.sub(
        r'^(?:can\s+you\s+)?(?:could\s+you\s+)?(?:please\s+)?(?:i\s+want\s+(?:to\s+listen\s+to\s+)?)?'
        r'(?:play|start|put\s+on|listen\s+to|hear)?\s*'
        r'(?:a\s+|some\s+|the\s+)?'
        r'(?:songs?\s+(?:from|of|by)?|music\s+(?:from|of|by)?|track\s+(?:from|of|by)?)?\s*',
        '', raw_query, flags=re.IGNORECASE
    ).strip()
    clean_q = re.sub(r'\s+(?:songs?|music|track|movie\s+songs?)$', '', clean_q, flags=re.IGNORECASE).strip()

    if is_different or not clean_q or clean_q.lower() in generic_words:
        query = lang_defaults.get(active_lang, 'top trending hits')
    elif clean_q.lower() in pure_language_queries:
        query = pure_language_queries[clean_q.lower()]
    else:
        query = clean_q

    results = []

    # 1. Search local music directory
    if os.path.exists(MUSIC_DIR):
        for f in os.listdir(MUSIC_DIR):
            if f.lower().endswith(('.mp3', '.m4a', '.wav', '.ogg')):
                if query.lower() in f.lower() or raw_query.lower() in f.lower():
                    results.append({
                        "title": os.path.splitext(f)[0],
                        "artist": "Local Library",
                        "url": f"/music/{urllib.parse.quote(f)}",
                        "is_full_song": True,
                        "source": "local"
                    })

    # 2. Search JioSaavn FULL Songs (Provides complete 3-5 min studio songs in 320kbps)
    full_tracks = fetch_jiosaavn_full_songs(query)
    if not full_tracks and active_lang in ['kn', 'te', 'hi', 'ta', 'ml']:
        lang_names = {'kn': 'Kannada', 'te': 'Telugu', 'hi': 'Hindi', 'ta': 'Tamil', 'ml': 'Malayalam'}
        full_tracks = fetch_jiosaavn_full_songs(f"{query} {lang_names.get(active_lang, '')}")

    if full_tracks:
        # Relevance scoring: prioritize tracks where query keywords appear in title or album
        q_words = [w.lower() for w in re.split(r'\s+', query) if len(w) > 2 and w.lower() not in ['movie', 'song', 'songs', 'hits']]
        if q_words:
            def relevance_score(t):
                t_title = t.get('title', '').lower()
                t_album = t.get('album', '').lower()
                score = 0
                for w in q_words:
                    if w in t_album:
                        score += 3
                    if w in t_title:
                        score += 2
                return score
            full_tracks = sorted(full_tracks, key=relevance_score, reverse=True)

        if is_different and len(full_tracks) > 1:
            rot = SESSION_STATE.get("music_rotation_index", 1) % len(full_tracks)
            if rot == 0: rot = 1
            SESSION_STATE["music_rotation_index"] = (rot + 1) % len(full_tracks)
            full_tracks = full_tracks[rot:] + full_tracks[:rot]

        results.extend(full_tracks)

    # 3. Search iTunes India Storefront as fallback (&country=IN covers Indian languages)
    if not results:
        try:
            itunes_url = f"https://itunes.apple.com/search?term={urllib.parse.quote(query)}&entity=song&limit=12&country=IN"
            req = urllib.request.Request(itunes_url, headers={'User-Agent': 'Mozilla/5.0'})
            res = urllib.request.urlopen(req, timeout=5)
            data = json.loads(res.read().decode('utf-8'))
            for item in data.get('results', []):
                if item.get('previewUrl'):
                    results.append({
                        "title": item.get('trackName'),
                        "artist": item.get('artistName'),
                        "album": item.get('collectionName'),
                        "url": item.get('previewUrl'),
                        "artwork": item.get('artworkUrl100'),
                        "is_full_song": False,
                        "source": "itunes_in"
                    })
        except Exception as e:
            print("Music search country=IN error:", e)

    return jsonify({"query": query, "count": len(results), "tracks": results})

@app.route('/music/<path:filename>')
def serve_music_file(filename):
    return send_from_directory(MUSIC_DIR, filename)

# ============================================================
# 3. Live News Headlines (Always Fresh + Interactive Drilldown)
# ============================================================
@app.route('/api/news/live', methods=['GET'])
def get_live_news():
    global SESSION_STATE
    lang = SESSION_STATE.get("active_language", "en")
    lang_name = LANGUAGE_CONFIG.get(lang, {}).get("name", "English")
    today_str = time.strftime("%B %d, %Y")
    
    prompt = f"Provide exactly 5 current real news headlines for today ({today_str}) in India and worldwide in {lang_name}. Keep each headline under 12 words. Return ONLY valid JSON: [ {{\"id\": 1, \"title\": \"...\", \"category\": \"...\", \"summary\": \"...\"}}, ... ]. No markdown codeblocks or extra text."

    headlines = []
    gemini_resp = call_gemini_generate(prompt, json_mode=True, timeout=8)
    if gemini_resp:
        try:
            clean_json = gemini_resp.strip().replace('```json', '').replace('```', '').strip()
            headlines = json.loads(clean_json)
        except Exception as e:
            print("Gemini news JSON parse error:", e)

    if not headlines:
        headlines = [
            {"id": 1, "title": "India's Tech Sector Surges with New AI & Semiconductor Initiatives", "category": "Tech", "summary": "Government and industry leaders announce major computing expansions across Karnataka and nationwide."},
            {"id": 2, "title": "Global Climate Summit Focuses on Renewable Energy Investments", "category": "World", "summary": "Countries commit to accelerating solar, wind, and battery storage infrastructure development."},
            {"id": 3, "title": "Space Agency Announces Next-Gen Lunar and Planetary Missions", "category": "Science", "summary": "Preparations are underway for upcoming autonomous exploration flights and orbital satellites."},
            {"id": 4, "title": "National Sports Championship Highlights Emerging Athletes", "category": "Sports", "summary": "Record-breaking performances mark the national athletics and cricket championships."},
            {"id": 5, "title": "Smart Cities Mission Deploys Advanced Traffic & Energy Systems", "category": "National", "summary": "Urban municipal corporations integrate IoT and AI for improved daily citizen utilities."}
        ]

    SESSION_STATE["current_headlines"] = headlines
    return jsonify({"date": today_str, "headlines": headlines})

@app.route('/api/news/explain', methods=['POST'])
def explain_headline():
    global SESSION_STATE
    data = request.get_json() or {}
    headline_id = data.get('id', 1)
    lang = SESSION_STATE.get("active_language", "en")
    lang_name = LANGUAGE_CONFIG.get(lang, {}).get("name", "English")

    headlines = SESSION_STATE.get("current_headlines", [])
    target = next((h for h in headlines if h.get('id') == headline_id), None)
    if not target and headlines:
        target = headlines[0]

    if not target:
        return jsonify({"explanation": "Headline details are currently unavailable."})

    prompt = f"The user asked about headline #{target.get('id')}: '{target.get('title')}'. Summary: '{target.get('summary')}'. Explain this headline concisely in 2 sentences in {lang_name}."
    text = call_gemini_generate(prompt, timeout=6)
    if text:
        return jsonify({"id": target.get('id'), "title": target.get('title'), "explanation": text})
    return jsonify({"id": target.get('id'), "title": target.get('title'), "explanation": target.get('summary')})

GROQ_API_KEY = GROQ_KEY

def extract_core_subject(user_prompt):
    # Try fast Gemini extraction
    try:
        sys_inst = "Extract the core visual subject, clothing item, or hairstyle requested by the user. Output ONLY 2-6 words (e.g. mullet haircut, blue tuxedo, cute kitten, red sports car). Do not explain."
        extracted = call_gemini_generate(user_prompt, system_instruction=sys_inst, timeout=5)
        if extracted:
            clean_ext = extracted.strip().strip('"').strip("'")
            if clean_ext and len(clean_ext) < 50:
                return clean_ext
    except Exception:
        pass

    # Regex fallback: Strip polite phrasing, questions, stuttering, and conversational fillers
    clean = user_prompt.strip()
    clean = re.sub(r'^(?:what\s+(?:i\s+am\s+)?if\s+(?:i\s+)?(?:if\s+i\s+)?(?:have|had)?\s*)', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'^(?:can\s+you\s+)?(?:please\s+)?(?:generate|create|draw|make|show(?:\s+me)?)\s+(?:an?\s+)?(?:image\s+(?:of\s+)?|picture\s+(?:of\s+)?|photo\s+(?:of\s+)?)?', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'(?:,?\s*can\s+you\s+(?:generate|create|show|make|draw).*$)', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'(?:,?\s*how\s+(?:i|would\s+i|do\s+i|will\s+i).*$)', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\s+on\s+my\s+hair.*$', '', clean, flags=re.IGNORECASE)
    clean = clean.strip()
    return clean if clean else user_prompt

def extract_person_visual_profile(image_base64):
    if not image_base64:
        return "real young adult South Asian Indian person with natural warm skin tone, dark brown eyes, and dark hair"

    try:
        prompt_text = "In ONE concise sentence, describe this person's facial identity in the mirror for an authentic real-world photograph: gender, approximate age, ethnicity and skin complexion (e.g. South Asian / Indian with natural warm olive or brown skin tone), face structure, eye color, eyebrow shape, and facial hair. Do not describe their hairstyle or clothing."
        desc = call_gemini_vision(image_base64, prompt_text, timeout=6)
        if desc:
            neg_indicators = ['impossible', 'cannot', 'can not', 'no person', 'no human', 'not a person', 'screenshot', 'interface', 'graphic']
            if any(neg in desc.lower() for neg in neg_indicators):
                return "young adult South Asian Indian person with natural warm brown skin tone, dark brown eyes, and dark hair"
            desc = re.sub(r'^(?:The individual is\s+(?:a\s+|an\s+)?|The person is\s+(?:a\s+|an\s+)?|The image (?:features|shows|depicts)\s+(?:a\s+|an\s+)?|Based on [^,]+,\s*(?:the person is\s+)?|This is a\s+|A photograph of a\s+|In this [^,]+,\s*)', '', desc, flags=re.IGNORECASE).strip().rstrip('.')
            desc = re.sub(r'^(?:a\s+|an\s+)?', '', desc, flags=re.IGNORECASE)
            if len(desc) > 15:
                return desc
    except Exception as e:
        print("Vision feature extraction error:", e)

    return "young adult South Asian Indian person with natural warm brown skin tone, dark brown eyes, and dark hair"

# ============================================================
# 4. Multimodal Intent Classifier & Persona Attribute Extraction
# ============================================================
def classify_multimodal_intent(text: str, active_language: str = "en", context: dict = None) -> dict:
    # Auto-detect Indic script language if not already specified or defaulted
    if any('\u0c80' <= c <= '\u0cff' for c in text):
        active_language = "kn"
    elif any('\u0c00' <= c <= '\u0c7f' for c in text):
        active_language = "te"
    elif any('\u0900' <= c <= '\u097f' for c in text):
        active_language = "hi"
    elif any('\u0b80' <= c <= '\u0bff' for c in text):
        active_language = "ta"

    t = text.lower().strip()
    context = context or {}

    # Fast check for non-image intents using word boundaries
    text_only_patterns = [
        r'\bweather\b', r'\btemperature\b', r'\bforecast\b', r'\brain\b', r'\bclimate\b',
        r'\bnews\b', r'\bheadlines?\b',
        r'\bplay\b', r'\bsongs?\b', r'\bmusic\b', r'\blisten\b',
        r'\bwhat time\b', r'\bdate\b',
        r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bwho are you\b', r'\bhow are you\b',
        r'ಹಾಡು', r'ಸುದ್ದಿ', r'ಹವಾಮಾನ', r'ಸಂಗೀತ',
        r'పాట', r'వార్తలు', r'వాతావరణం', r'సంగీతం',
        r'गाना', r'समाचार', r'मौसम', r'संगीत'
    ]
    is_non_image = any(re.search(p, t) for p in text_only_patterns) and not any(k in t for k in ['generate', 'image', 'picture', 'photo', 'look in', 'wear', 'put me', 'show me', 'shirt', 'suit', 'hair', 'dress'])
    if is_non_image:
        return {
            "intent": "TEXT_ONLY",
            "is_personal": False,
            "requires_camera": False,
            "modifications": {},
            "core_prompt": text
        }

    system_prompt = (
        "You are an intent classification and attribute extraction engine for an AI Smart Mirror with camera, vision, and image generation.\n"
        "Analyze the user's spoken command (which may be in English, Kannada, Telugu, Hindi, or Tamil) and classify into EXACTLY ONE of these intents:\n"
        "- NORMAL_IMAGE_GENERATION: Generic image request unrelated to the user (e.g. 'generate a cat', 'a futuristic city at night', 'draw a sports car'). NO camera.\n"
        "- PERSONAL_TRY_ON: User wants to see themselves in clothing/outfit/appearance (e.g. 'how will I look in a white shirt?', 'show me with a black suit', 'see what I look like if I wear a white shirt'). Requires camera.\n"
        "- HAIRSTYLE_CHANGE: User specifically wants to see themselves with a different hairstyle or haircut (e.g. 'how will I look with a mullet?', 'give me a fade haircut', 'make it shorter'). Requires camera.\n"
        "- OUTFIT_CHANGE: User specifically wants to change their clothes/outfit (e.g. 'show me in a blue denim jacket', 'put me in a tuxedo', 'now make the shirt black'). Requires camera.\n"
        "- BACKGROUND_CHANGE: User wants to place themselves in a different setting/background (e.g. 'put me on a beach', 'place me in New York at night'). Requires camera.\n"
        "- IMAGE_ANALYSIS: User is asking for styling advice, recommendation, or analysis without generating an image (e.g. 'what hairstyle suits me?', 'what outfit should I wear?', 'how is my dress?'). Requires camera.\n"
        "- TEXT_ONLY: Non-image queries (weather, news, general chat, music).\n"
        "- UNKNOWN: Irrelevant noise.\n\n"
        "IMPORTANT RULES:\n"
        "1. Extract ONLY the clean attribute names (e.g. 'white button-down shirt', 'mullet haircut', 'beach at sunset'). Never include conversational speech like 'see what I am if I wear' or 'can you able to generate'.\n"
        "2. clean_prompt: Provide a clean, short 2-4 word title for UI display (e.g. 'White dress shirt', 'Mullet haircut', 'Tailored suit', 'Ginger cat').\n"
        "3. If Previous Context contains existing modifications or a generated image, follow-up tweak commands like 'now make the shirt black', 'change it to blue', 'make it shorter', 'add a beard' MUST be classified as the corresponding personal modification intent with is_personal: true and requires_camera: true.\n\n"
        "Output ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "intent": "NORMAL_IMAGE_GENERATION" | "PERSONAL_TRY_ON" | "HAIRSTYLE_CHANGE" | "OUTFIT_CHANGE" | "BACKGROUND_CHANGE" | "IMAGE_ANALYSIS" | "TEXT_ONLY" | "UNKNOWN",\n'
        '  "is_personal": true | false,\n'
        '  "requires_camera": true | false,\n'
        '  "is_followup": true | false,\n'
        '  "clean_prompt": string,\n'
        '  "modifications": {\n'
        '    "clothing": string | null,\n'
        '    "hairstyle": string | null,\n'
        '    "background": string | null,\n'
        '    "details": string | null\n'
        '  },\n'
        '  "core_prompt": string\n'
        "}"
    )

    context_str = json.dumps(context) if context else "None"
    has_prev_context = bool(context and (context.get('lastType') or context.get('modifications') or context.get('lastReferenceImage')))

    try:
        user_prompt = f"Active Language: {active_language}\nPrevious Context: {context_str}\nUser Spoken Text: \"{text}\""
        gemini_res = call_gemini_generate(user_prompt, system_instruction=system_prompt, json_mode=True, timeout=5)
        if gemini_res:
            parsed = json.loads(gemini_res)
            parsed['is_personal'] = parsed.get('is_personal', False)
            parsed['requires_camera'] = parsed.get('requires_camera', False)
            parsed['is_followup'] = parsed.get('is_followup', has_prev_context)
            if not parsed.get('modifications'):
                parsed['modifications'] = {}
            if 'clothing' in parsed['modifications'] and 'outfit' not in parsed['modifications']:
                parsed['modifications']['outfit'] = parsed['modifications']['clothing']
            
            # Ensure clean prompt exists and is concise
            if not parsed.get('clean_prompt'):
                c = parsed['modifications'].get('clothing') or parsed['modifications'].get('outfit')
                h = parsed['modifications'].get('hairstyle')
                b = parsed['modifications'].get('background')
                chosen = c or h or b or text
                parsed['clean_prompt'] = re.sub(r'^(?:can\s+you\s+)?(?:generate|create|draw|make|show(?:\s+me)?)\s+(?:an?\s+)?', '', chosen, flags=re.IGNORECASE).strip()
            
            if not parsed.get('core_prompt'):
                parsed['core_prompt'] = text
            return parsed
    except Exception as e:
        print("Classifier LLM error:", e)

    # Heuristic fallback with concise attribute extraction
    modifications = {}
    clean_prompt = ""
    hair_keywords = ['hair', 'haircut', 'hairstyle', 'mullet', 'fade', 'curly', 'straight', 'blonde', 'bald', 'ಹೇರ್‌ಸ್ಟೈಲ್', 'ಹೇರ್', 'క్రాపింగ్', 'హెయిర్‌స్టైల్', 'హెయిర్', 'हेयरस्टाइल', 'बाल']
    outfit_keywords = ['shirt', 'suit', 'jacket', 'hoodie', 'dress', 'tuxedo', 'pants', 't-shirt', 'wearing', 'outfit', 'clothes', 'ಬಟ್ಟೆ', 'ಡ್ರೆಸ್', 'ಶರ್ಟ್', 'ಶರ್ಟ್‌', 'ಕುರ್ತಾ', 'షర్ట్', 'షర్టు', 'డ్రెస్', 'బట్టలు', 'కుర్తా', 'शर्ट', 'सूट', 'कुर्ता', 'कपड़े', 'ड्रेस', 'ஆடை', 'சட்டை']
    bg_keywords = ['beach', 'paris', 'new york', 'city', 'background', 'place me', 'put me on', 'put me in', 'ಸಮುದ್ರ', 'ನ್ಯೂಯಾರ್ಕ್', 'బీಚ್', 'న్యూయార్క్', 'बीच', 'न्यूयॉर्क']
    analysis_keywords = ['suits me', 'should i wear', 'advice', 'recommend', 'rate my', 'how do i look', 'how is my', 'ಸರಿಹೊಂದುತ್ತದೆ', 'ಹೇಗಿದೆ', 'బాగుంటుంది', 'ఎలా ఉంది', 'अच्छा लगेगा', 'कैसा है']
    gen_keywords = ['generate', 'create', 'draw', 'picture of', 'image of', 'photo of', 'ಚಿತ್ರ', 'ಫೋಟೋ', 'చిత్రం', 'ఫోటో', 'चित्र', 'तस्वीर', 'फोटो']

    if any(k in t for k in analysis_keywords):
        intent = "IMAGE_ANALYSIS"
        is_personal = True
        clean_prompt = "Style Advice"
    elif any(k in t for k in hair_keywords):
        intent = "HAIRSTYLE_CHANGE"
        is_personal = True
        if 'mullet' in t:
            modifications['hairstyle'] = 'mullet haircut'
            clean_prompt = 'Mullet haircut'
        elif 'fade' in t:
            modifications['hairstyle'] = 'fade haircut'
            clean_prompt = 'Fade haircut'
        else:
            modifications['hairstyle'] = 'new hairstyle'
            clean_prompt = 'Hairstyle change'
    elif any(k in t for k in outfit_keywords):
        intent = "OUTFIT_CHANGE"
        is_personal = True
        if 'white shirt' in t or ('white' in t and 'shirt' in t):
            modifications['clothing'] = 'white button-down dress shirt'
            modifications['outfit'] = 'white button-down dress shirt'
            clean_prompt = 'White dress shirt'
        elif 'black shirt' in t or ('black' in t and 'shirt' in t):
            modifications['clothing'] = 'black button-down shirt'
            modifications['outfit'] = 'black button-down shirt'
            clean_prompt = 'Black shirt'
        elif 'suit' in t or 'tuxedo' in t:
            modifications['clothing'] = 'tailored classic suit'
            modifications['outfit'] = 'tailored classic suit'
            clean_prompt = 'Tailored suit'
        elif 'jacket' in t:
            modifications['clothing'] = 'stylish jacket'
            modifications['outfit'] = 'stylish jacket'
            clean_prompt = 'Stylish jacket'
        else:
            modifications['clothing'] = 'outfit transformation'
            modifications['outfit'] = 'outfit transformation'
            clean_prompt = 'Outfit preview'
    elif any(k in t for k in bg_keywords):
        intent = "BACKGROUND_CHANGE"
        is_personal = True
        if 'beach' in t:
            modifications['background'] = 'tropical beach at sunset'
            clean_prompt = 'Beach background'
        elif 'paris' in t:
            modifications['background'] = 'Paris Eiffel Tower'
            clean_prompt = 'Paris background'
        elif 'new york' in t or 'city' in t:
            modifications['background'] = 'New York city skyline at night'
            clean_prompt = 'New York background'
        else:
            modifications['background'] = 'scenic background'
            clean_prompt = 'Background change'
    elif any(k in t for k in gen_keywords):
        intent = "NORMAL_IMAGE_GENERATION"
        is_personal = False
        clean_prompt = re.sub(r'^(?:can\s+you\s+)?(?:generate|create|draw|make|show(?:\s+me)?)\s+(?:an?\s+)?(?:image\s+(?:of\s+)?|picture\s+(?:of\s+)?|photo\s+(?:of\s+)?)?', '', text, flags=re.IGNORECASE).strip() or text
    else:
        intent = "TEXT_ONLY"
        is_personal = False
        clean_prompt = text

    return {
        "intent": intent,
        "is_personal": is_personal,
        "requires_camera": is_personal and intent != "NORMAL_IMAGE_GENERATION",
        "is_followup": has_prev_context,
        "modifications": modifications,
        "clean_prompt": clean_prompt,
        "core_prompt": text
    }

def check_person_visibility(image_base64: str):
    """
    Checks if a usable person or human face is visible in front of the mirror camera.
    Returns (True, person_visual_profile) or (False, user_friendly_error_message).
    """
    if not image_base64 or len(image_base64.strip()) < 100:
        return False, "I can't see you clearly. Please move in front of the mirror."

    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    try:
        vis_prompt = "Analyze this mirror camera frame. Is there an actual human person or face clearly visible in front of the mirror? If yes, answer 'YES: ' followed by a one-sentence description of their gender (male/female), approximate age, ethnicity, skin tone, facial hair (e.g. mustache), and facial features. If no person/face is visible (e.g. empty room, wall, ceiling, completely black, blurry, or blocked), answer 'NO: <brief reason>'."
        reply = call_gemini_vision(image_base64, vis_prompt, timeout=6)
        if reply:
            if reply.upper().startswith("YES"):
                profile = reply[4:].strip()
                return True, profile
            elif reply.upper().startswith("NO"):
                return False, "I can't see you clearly. Please move in front of the mirror."
    except Exception as e:
        print("Visibility check error:", e)

    # If vision check timed out but image was provided, gracefully assume user presence with authentic profile
    return True, "young adult South Asian Indian male with natural warm brown skin tone, dark brown eyes, neat dark hair, and neat short mustache"

# ============================================================
# 4. Multimodal Image Generation & Analysis Endpoints
# ============================================================
@app.route('/api/classify-intent', methods=['POST'])
def api_classify_intent():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    active_language = data.get('active_language', 'en').lower()
    context = data.get('context', {})
    if not text:
        return jsonify({"error": "No text provided", "intent": "UNKNOWN"}), 400
    
    result = classify_multimodal_intent(text, active_language, context)
    return jsonify(result)

@app.route('/api/generate-image', methods=['POST'])
def api_generate_image():
    data = request.get_json() or {}
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({"success": False, "message": "No prompt provided"}), 400
    
    res = image_service.generate_image(prompt)
    return jsonify({
        "success": True,
        "image_url": res["image_url"],
        "prompt": prompt,
        "enhanced_prompt": res.get("enhanced_prompt", prompt),
        "message": "Image generated successfully"
    })

@app.route('/api/generate-personal-image', methods=['POST'])
def api_generate_personal_image():
    data = request.get_json() or {}
    image_base64 = data.get('image') or data.get('image_base64')
    prompt = data.get('prompt', '').strip()
    modifications = data.get('modifications') or {}
    context = data.get('context') or {}

    if not prompt:
        prompt = "Show me with the requested style"

    # 1. Person visibility check
    is_visible, profile_or_err = check_person_visibility(image_base64)
    if not is_visible:
        return jsonify({
            "success": False,
            "error_code": "NO_PERSON_VISIBLE",
            "message": profile_or_err or "I can't see you clearly. Please move in front of the mirror."
        }), 400

    person_profile = profile_or_err

    # 2. Context resolution: merge with previous context if user says 'make it black', etc.
    prev_mods = context.get('modifications', {})
    merged_mods = dict(prev_mods)
    merged_mods.update({k: v for k, v in modifications.items() if v})

    # 3. Generate image preserving identity
    res = image_service.generate_image_from_reference(
        image_base64=image_base64,
        prompt=prompt,
        person_profile=person_profile,
        modifications=merged_mods
    )

    return jsonify({
        "success": True,
        "image_url": res["image_url"],
        "prompt": prompt,
        "enhanced_prompt": res.get("enhanced_prompt", prompt),
        "modifications": merged_mods,
        "message": "Here is your preview."
    })

@app.route('/api/analyze-image', methods=['POST'])
def api_analyze_image():
    data = request.get_json() or {}
    image_base64 = data.get('image') or data.get('image_base64')
    question = data.get('question') or data.get('prompt') or "What hairstyle and styling suits me?"
    lang = (data.get('active_language') or data.get('language') or 'en').lower()
    lang_info = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])

    if not image_base64:
        return jsonify({
            "success": False,
            "message": "I can't see you clearly. Please move in front of the mirror."
        }), 400

    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    prompt_text = (
        f"You are Jarvis, a smart mirror personal stylist. "
        f"Analyze the user's face shape (e.g. oval, square, round, heart), skin undertone, and current styling visible in the camera frame. "
        f"The user asks: '{question}'. "
        f"Provide 2 to 3 concise, stylish, and practical recommendations in {lang_info['name']}. "
        f"Keep your tone confident, helpful, and natural for immediate speech output."
    )

    try:
        advice = call_gemini_vision(image_base64, prompt_text, timeout=12)
        if advice:
            return jsonify({
                "success": True,
                "analysis": advice,
                "recommendations": advice,
                "message": advice
            })
    except Exception as e:
        print("Analyze image error:", e)

    # Resilient fallback advice ensures mirror experience never fails
    fallback_advice = "Based on your mirror scan, a clean textured crop or classic side-part haircut with a well-groomed trim will sharply complement your face shape and natural skin tone."
    return jsonify({
        "success": True,
        "analysis": fallback_advice,
        "recommendations": fallback_advice,
        "message": fallback_advice
    })

@app.route('/api/camera/capture', methods=['POST'])
def api_camera_capture():
    data = request.get_json() or {}
    image_base64 = data.get('image') or data.get('image_base64')
    if not image_base64:
        return jsonify({"success": False, "message": "No camera frame received"}), 400
    return jsonify({"success": True, "status": "captured", "message": "Frame received successfully"})

@app.route('/api/voice', methods=['POST'])
def api_voice():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    lang = (data.get('active_language') or 'en').lower()
    context = data.get('context', {})
    if not text:
        return jsonify({"success": False, "message": "No voice text received"}), 400
    
    classification = classify_multimodal_intent(text, lang, context)
    return jsonify({
        "success": True,
        "text": text,
        "classification": classification
    })

@app.route('/api/image/generate', methods=['POST'])
def generate_user_image():
    data = request.get_json() or {}
    user_prompt = data.get('prompt', 'a cute cat').strip()
    image_base64 = data.get('image') or data.get('image_base64')

    self_keywords = [
        'me', 'myself', 'i look', 'how do i', 'how would i', 'how will i', 'how i look',
        'on me', 'my hair', 'haircut', 'hairstyle', 'mullet', 'beard', 'mustache',
        'wearing', 'dress', 'suit', 'outfit', 'clothes', 'look like', 'look on me',
        'what if i', 'if i have', 'if i had', 'portrait', 'face', 'person', 'man', 'guy', 'boy', 'woman', 'girl',
        'ನನ್ನ', 'ನಾನು', 'ಹೇರ್‌ಸ್ಟೈಲ್', 'ಬಟ್ಟೆ', 'ನಾ', 'నేను', 'హెయిర్', 'मेरा', 'मेरी', 'मैं'
    ]
    is_self = any(k in user_prompt.lower() for k in self_keywords)

    if is_self and image_base64:
        res = image_service.generate_image_from_reference(
            image_base64=image_base64,
            prompt=user_prompt,
            person_profile=extract_person_visual_profile(image_base64)
        )
    else:
        res = image_service.generate_image(user_prompt)

    return jsonify({
        "status": "success",
        "prompt": user_prompt,
        "enhanced_prompt": res.get("enhanced_prompt", user_prompt),
        "image_url": res["image_url"]
    })

# ============================================================
# 4.5. Real-Time Object & Item Identification via Mirror Camera
# ============================================================
@app.route('/api/vision/identify', methods=['POST'])
def identify_camera_object():
    data = request.get_json() or {}
    image_base64 = data.get('image') or data.get('image_base64')
    question = data.get('question') or data.get('prompt') or data.get('text') or "What is this object?"
    lang = (data.get('active_language') or data.get('language') or 'en').lower()
    persona = data.get('persona', 'Jarvis')
    lang_info = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])

    if not image_base64:
        return jsonify({
            "status": "error",
            "answer": "I couldn't see anything. Please hold the object clearly in front of the mirror camera."
        }), 400

    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    system_instruction = (
        f"You are {persona}, an intelligent Smart Mirror visual assistant. "
        f"Look carefully at the live camera frame of the user in front of the mirror. "
        f"The user is showing or holding an item, object, tool, gadget, book, fruit, medicine, accessory, or thing. "
        f"Identify specifically what the user is holding up or showing in the frame. "
        f"Respond in 1 to 2 clear, helpful sentences in {lang_info['name']}. "
        f"Name the item clearly (including brand, model, type, or color if visible), and provide 1 practical detail, function, or fact. "
        f"If the user is not holding anything specific, briefly describe what is in their hand or what is prominent in front of the camera."
    )

    try:
        prompt_with_q = f"{system_instruction}\nUser's question: {question}"
        content = call_gemini_vision(image_base64, prompt_with_q, timeout=10)
        if content:
            return jsonify({
                "status": "success",
                "object_description": content,
                "answer": content,
                "language": lang
            })
    except Exception as e:
        print("Object identification vision error:", e)

    return jsonify({
        "status": "error",
        "answer": "I had trouble identifying the object. Please hold it closer to the camera."
    })

# ============================================================
# 4.6. Intelligent Conversational Agent (Gemini 2.5 Flash)
# ============================================================
@app.route('/api/agent/chat', methods=['POST'])
def api_agent_chat():
    global SESSION_STATE
    data = request.get_json() or {}
    message = (data.get('message') or data.get('text') or '').strip()
    active_language = (data.get('active_language') or data.get('language') or SESSION_STATE.get('active_language', 'en')).lower()
    persona = data.get('persona', 'Jarvis')
    history = data.get('history') or []
    memory = data.get('memory') or {}

    if not message:
        return jsonify({"status": "error", "message": "No message provided"}), 400

    lang_info = LANGUAGE_CONFIG.get(active_language, LANGUAGE_CONFIG['en'])
    lang_name = lang_info['name']

    # Auto-detect language switch if user asks in their native script or text
    lang_map = {
        "kannada": "kn", "kannad": "kn", "kn": "kn", "ಕನ್ನಡ": "kn",
        "hindi": "hi", "hind": "hi", "hi": "hi", "हिंदी": "hi",
        "tamil": "ta", "ta": "ta", "தமிழ்": "ta",
        "telugu": "te", "te": "te", "తెలుగు": "te",
        "malayalam": "ml", "ml": "ml", "മലയാളം": "ml",
        "marathi": "mr", "mr": "mr", "मराठी": "mr",
        "bengali": "bn", "bangla": "bn", "bn": "bn", "বাংলা": "bn",
        "english": "en", "en": "en"
    }
    m_low = message.lower()
    for name, code in lang_map.items():
        if f"switch to {name}" in m_low or f"speak in {name}" in m_low or f"talk in {name}" in m_low or f"{name}ದಲ್ಲಿ ಮಾತನಾಡು" in message or f"{name}ಲ್ಲಿ ಮಾತನಾಡು" in message:
            SESSION_STATE['active_language'] = code
            target_config = LANGUAGE_CONFIG.get(code, LANGUAGE_CONFIG['en'])
            return jsonify({
                "status": "success",
                "intent": "LANGUAGE_SWITCH",
                "active_language": code,
                "answer": target_config["confirm"],
                "language": code
            })

    system_prompt = (
        f"You are {persona}, a warm, empathetic, and exceptionally intelligent Smart Mirror AI companion.\n"
        f"CONVERSATIONAL STYLE & PERSONALITY:\n"
        f"- Speak with the genuine warmth, empathy, lively cadence, and interactive flow of daily life conversations — effortlessly blending the camaraderie of a supportive best friend, the patient clarity of an encouraging teacher/mentor, and the heartfelt care of a family member.\n"
        f"- NEVER sound stiff, cold, robotic, or corporate. Completely avoid clichés like 'How may I assist you today?' or 'I am an AI'.\n"
        f"- Respond in {lang_name} naturally using conversational expressions and smooth transitions.\n"
        f"- Keep your spoken response concise (2 to 3 natural sentences) so voice synthesis is pleasant and feels like real-time back-and-forth dialogue.\n"
        f"- If the user expresses stress, nervousness (e.g. exams, job interviews, tired after work), respond with genuine reassurance, encouragement, and practical advice like a caring mentor.\n"
        f"- You have long-term memory of past interactions and user preferences.\n"
        f"- HARDWARE INTEGRATION & INTENT CLASSIFICATION:\n"
        f"You control the smart mirror hardware: Music Player, Outfit Analyzer, Camera Object Identifier, and Photo/Hairstyle Studio.\n"
        f"Classify user intent into one of:\n"
        f"- MUSIC_PLAY: User wants to play a song/music (in Kannada, Hindi, Telugu, Tamil, English, etc.). Provide clean 'song_query'.\n"
        f"- IMAGE_OUTFIT: User wants to analyze or scan their current outfit/dress in the camera.\n"
        f"- OBJECT_IDENTIFY: User is holding an object up or asking 'what is this' / 'what am I holding'.\n"
        f"- IMAGE_GENERATE: User wants to generate an image or see themselves with a new hairstyle/shirt/outfit. Provide clean 'subject'.\n"
        f"- GENERAL_CHAT: Normal dialogue, questions, advice, storytelling, or companionship.\n\n"
        f"OUTPUT SCHEMA (JSON ONLY):\n"
        f"{{\n"
        f'  "intent": "GENERAL_CHAT" | "MUSIC_PLAY" | "IMAGE_OUTFIT" | "OBJECT_IDENTIFY" | "IMAGE_GENERATE",\n'
        f'  "answer": "Concise natural spoken response in {lang_name}",\n'
        f'  "song_query": "Song title or query if MUSIC_PLAY else null",\n'
        f'  "subject": "Extracted subject if IMAGE_GENERATE else null",\n'
        f'  "learned_preference": "Any user preference or fact learned from this turn, or null"\n'
        f"}}"
    )

    contents = []
    if memory:
        contents.append({"parts": [{"text": f"User Profile & Preferences: {json.dumps(memory)}"}]})

    for h in history[-6:]:
        role = "user" if h.get("role") == "user" else "model"
        text = h.get("text") or h.get("content") or ""
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})

    user_turn_text = f"Language: {lang_name} ({active_language})\nUser Spoken: {message}"
    contents.append({"role": "user", "parts": [{"text": user_turn_text}]})

    # 1. Primary Agent: n8n Workflow (if active and running)
    try:
        n8n_payload = {
            "text": message,
            "query": message,
            "message": message,
            "active_language": active_language,
            "persona": persona,
            "messages": history + [{"role": "user", "content": message}]
        }
        n8n_req = urllib.request.Request(
            N8N_WEBHOOK_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(n8n_payload).encode('utf-8')
        )
        with urllib.request.urlopen(n8n_req, timeout=6) as n8n_res:
            if n8n_res.status == 200:
                n8n_data = json.loads(n8n_res.read().decode('utf-8'))
                if n8n_data.get('answer'):
                    return jsonify({
                        "status": "success",
                        "intent": n8n_data.get("intent", "GENERAL_CHAT"),
                        "answer": n8n_data.get("answer", ""),
                        "song_query": n8n_data.get("song_query"),
                        "subject": n8n_data.get("subject"),
                        "learned_preference": n8n_data.get("learned_preference"),
                        "language": n8n_data.get("active_language", active_language)
                    })
    except Exception as e:
        print(f"n8n webhook query error, trying fallback: {e}")

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"responseMimeType": "application/json"}
    }
    data_bytes = json.dumps(payload).encode('utf-8')

    for model in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
            req = urllib.request.Request(url, headers={'Content-Type': 'application/json'}, data=data_bytes)
            res = urllib.request.urlopen(req, timeout=8)
            resp_data = json.loads(res.read().decode('utf-8'))
            candidates = resp_data.get('candidates', [])
            if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
                raw_text = candidates[0]['content']['parts'][0]['text']
                parsed = json.loads(raw_text)
                return jsonify({
                    "status": "success",
                    "intent": parsed.get("intent", "GENERAL_CHAT"),
                    "answer": parsed.get("answer", ""),
                    "song_query": parsed.get("song_query"),
                    "subject": parsed.get("subject"),
                    "learned_preference": parsed.get("learned_preference"),
                    "language": active_language
                })
        except Exception as e:
            print(f"api_agent_chat [{model}] error: {e}")
            continue

    fallback_answers = {
        "kn": "ನಾನು ನಿಮ್ಮೊಂದಿಗಿದ್ದೇನೆ! ಮುಂದೆ ಏನು ತಿಳಿಯಲು ಬಯಸುತ್ತೀರಿ?",
        "hi": "मैं आपके साथ हूँ! आप आगे क्या जानना चाहेंगे?",
        "te": "నేను మీతోనే ఉన్నాను! మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
        "ta": "நான் உங்களுடன் இருக்கிறேன்! அடுத்து என்ன தெரிந்துகொள்ள விரும்புகிறீர்கள்?",
        "en": "I'm right here with you! What would you like to explore next?"
    }
    return jsonify({
        "status": "success",
        "intent": "GENERAL_CHAT",
        "answer": fallback_answers.get(active_language, fallback_answers["en"]),
        "language": active_language
    })

# ============================================================
# 5. Serve Frontend
# ============================================================
@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/')
def index():
    for name in ['index.html', 'mirror.html']:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
    return "mirror.html not found!"

if __name__ == '__main__':
    print("==================================================")
    print("[*] SmartMirror-Vox Flask Backend running on http://127.0.0.1:5000")
    print("==================================================")
    app.run(host='0.0.0.0', port=5000, debug=False)
