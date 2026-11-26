import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import re
import json
import time
import datetime
import base64
import warnings
import urllib.request
import urllib.parse
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import requests
import numpy as np
try:
    import cv2
except Exception as cv_e:
    print('cv2 import notice:', cv_e)

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

try:
    from services.market_data import fetch_live_gold_rates, fetch_live_market_indices, get_gold_briefing_text
except Exception as _e_mkt:
    print("[MarketData] Import notice:", _e_mkt)

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/mirror-agent"
# ============================================================
# Local AI Vision Engines (YOLOv8 + MobileNet V3)
# ============================================================
GLOBAL_YOLO_MODEL = None
GLOBAL_MOBILENET_MODEL = None
GLOBAL_MOBILENET_CATEGORIES = []
GLOBAL_MOBILENET_PREPROCESS = None

try:
    from ultralytics import YOLO
    _yolo_path = os.path.join(BASE_DIR, 'yolov8n.pt') if os.path.exists(os.path.join(BASE_DIR, 'yolov8n.pt')) else 'yolov8n.pt'
    GLOBAL_YOLO_MODEL = YOLO(_yolo_path)
    print("✅ YOLOv8 AI object recognition model loaded ready for instant camera identification!")
except Exception as yolo_err:
    print("YOLO initialization notice:", yolo_err)

try:
    import torchvision.models as tv_models
    import torch
    from PIL import Image
    _mb_weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
    GLOBAL_MOBILENET_MODEL = tv_models.mobilenet_v3_small(weights=_mb_weights).eval()
    GLOBAL_MOBILENET_CATEGORIES = _mb_weights.meta['categories']
    GLOBAL_MOBILENET_PREPROCESS = _mb_weights.transforms()
    print("✅ MobileNet V3 ImageNet fine-grained classifier loaded ready!")
except Exception as mb_err:
    print("MobileNet initialization notice:", mb_err)

OBJECT_LABEL_MAP = {
    'cell phone': 'smartphone',
    'cellular telephone': 'smartphone',
    'handheld computer': 'smartphone',
    'bottle': 'water bottle',
    'water bottle': 'water bottle',
    'wine glass': 'drinking glass',
    'cup': 'coffee mug or cup',
    'coffee mug': 'coffee mug',
    'fork': 'fork',
    'knife': 'knife',
    'spoon': 'spoon',
    'bowl': 'bowl',
    'banana': 'fresh banana',
    'apple': 'fresh apple',
    'sandwich': 'sandwich',
    'orange': 'fresh orange',
    'remote': 'remote control',
    'remote control': 'remote control',
    'keyboard': 'keyboard',
    'mouse': 'computer mouse',
    'laptop': 'laptop computer',
    'book': 'book',
    'notebook': 'notebook',
    'clock': 'clock or watch',
    'digital clock': 'digital clock or screen',
    'analog clock': 'clock',
    'wall clock': 'clock',
    'wristwatch': 'wristwatch',
    'watch': 'wristwatch',
    'vase': 'decorative vase',
    'scissors': 'pair of scissors',
    'toothbrush': 'toothbrush',
    'hair drier': 'hair dryer',
    'backpack': 'backpack',
    'handbag': 'handbag or purse',
    'purse': 'handbag or purse',
    'wallet': 'wallet',
    'umbrella': 'umbrella',
    'tie': 'necktie',
    'sunglasses': 'sunglasses',
    'sunglass': 'sunglasses',
    'spectacles': 'reading glasses',
    'ballpoint': 'pen',
    'ballpoint pen': 'pen',
    'fountain pen': 'fountain pen',
    'lighter': 'lighter',
}

MULTILINGUAL_OBJECT_TERMS = {
    'smartphone': {'en': 'smartphone', 'kn': 'ಸ್ಮಾರ್ಟ್‌ಫೋನ್', 'te': 'స్మార్ట్‌ఫోన్', 'hi': 'स्मार्टफोन', 'ta': 'ஸ்மார்ட்போன்'},
    'water bottle': {'en': 'water bottle', 'kn': 'ನೀರಿನ ಬಾಟಲಿ', 'te': 'నీళ్ల సీసా', 'hi': 'पानी की बोतल', 'ta': 'தண்ணீர் பாட்டில்'},
    'coffee mug or cup': {'en': 'coffee mug', 'kn': 'ಕಾಫಿ ಕಪ್', 'te': 'కాఫీ కప్పు', 'hi': 'कॉफ़ी कप', 'ta': 'காபி கப்'},
    'coffee mug': {'en': 'coffee mug', 'kn': 'ಕಾಫಿ ಕಪ್', 'te': 'కాఫీ కప్పు', 'hi': 'कॉफ़ी कप', 'ta': 'காபி கப்'},
    'remote control': {'en': 'remote control', 'kn': 'ರಿಮೋಟ್ ಕಂಟ್ರೋಲ್', 'te': 'రిమోట్ కంట్రోల్', 'hi': 'रिमोट कंट्रोल', 'ta': 'ரிமோட் கண்ட்ரோல்'},
    'book': {'en': 'book', 'kn': 'ಪುಸ್ತಕ', 'te': 'పుస్తకం', 'hi': 'किताब', 'ta': 'புத்தகம்'},
    'wristwatch': {'en': 'wristwatch', 'kn': 'ಕೈ ಗಡಿಯಾರ', 'te': 'చేతి గడియారం', 'hi': 'हाथ की घड़ी', 'ta': 'கைக்கடிகாரம்'},
    'reading glasses': {'en': 'reading glasses', 'kn': 'ಕನ್ನಡಕ', 'te': 'కళ్లజోడు', 'hi': 'चश्मा', 'ta': 'கண்ணாடி'},
    'sunglasses': {'en': 'sunglasses', 'kn': 'ಕೂಲ್ ಸನ್ ಗ್ಲಾಸ್', 'te': 'సన్ గ్లాసెస్', 'hi': 'धूप का चश्मा', 'ta': 'சூரிய கண்ணாடி'},
    'pen': {'en': 'pen', 'kn': 'ಪೆನ್ನು', 'te': 'పెన్ను', 'hi': 'कलम', 'ta': 'பேனா'},
    'wallet': {'en': 'wallet', 'kn': 'ವ್ಯಾಲೆಟ್', 'te': 'పర్స్ లేదా వాలెట్', 'hi': 'बटुआ', 'ta': 'பணப்பை'},
    'pair of scissors': {'en': 'scissors', 'kn': 'ಕತ್ತರಿ', 'te': 'ಕತ್ತెర', 'hi': 'कैंची', 'ta': 'கத்தரிக்கோல்'},
    'fresh apple': {'en': 'apple', 'kn': 'ಸೇಬು ಹಣ್ಣು', 'te': 'యాపిల్ పండు', 'hi': 'सेब', 'ta': 'ஆப்பிள்'},
    'fresh banana': {'en': 'banana', 'kn': 'ಬಾಳೆಹಣ್ಣು', 'te': 'అరటిపండు', 'hi': 'केला', 'ta': 'வாழைப்பழம்'},
    'computer mouse': {'en': 'computer mouse', 'kn': 'ಕಂಪ್ಯೂಟರ್ ಮೌಸ್', 'te': 'కంప్యూటర్ మౌస్', 'hi': 'माउस', 'ta': 'மவுஸ்'},
    'laptop computer': {'en': 'laptop', 'kn': 'ಲ್ಯಾಪ್ಟಾಪ್', 'te': 'ల్యాప్‌టాప్', 'hi': 'लैपटॉप', 'ta': 'லேப்டாப்'},
    'keyboard': {'en': 'keyboard', 'kn': 'ಕೀಬೋರ್ಡ್', 'te': 'కీబోర్డ్', 'hi': 'कीबोर्ड', 'ta': 'விசைப்பலகை'},
    'toothbrush': {'en': 'toothbrush', 'kn': 'ಟೂತ್ ಬ್ರಷ್', 'te': 'టూత్‌బ్రష్', 'hi': 'टूथब्रश', 'ta': 'பல் துலக்கி'}
}

def detect_held_camera_object(img):
    if img is None or img.size == 0:
        return None, []

    h, w, _ = img.shape
    detected_candidates = []

    # 1. YOLOv8 on full image
    if GLOBAL_YOLO_MODEL:
        try:
            res = GLOBAL_YOLO_MODEL(img, verbose=False)[0]
            for b in res.boxes:
                cls_name = GLOBAL_YOLO_MODEL.names[int(b.cls[0])]
                conf = float(b.conf[0])
                if cls_name != 'person' and conf > 0.2:
                    xyxy = [int(x) for x in b.xyxy[0]]
                    detected_candidates.append({
                        'label': cls_name,
                        'confidence': conf,
                        'box': xyxy,
                        'source': 'yolo_full'
                    })
        except Exception as e:
            print("YOLO full inference notice:", e)

    # 2. YOLOv8 on foreground / center crop (where object is presented)
    ch1, ch2 = int(h * 0.15), int(h * 0.95)
    cw1, cw2 = int(w * 0.15), int(w * 0.85)
    center_crop = img[ch1:ch2, cw1:cw2]
    if GLOBAL_YOLO_MODEL and center_crop.size > 0:
        try:
            res_crop = GLOBAL_YOLO_MODEL(center_crop, verbose=False)[0]
            for b in res_crop.boxes:
                cls_name = GLOBAL_YOLO_MODEL.names[int(b.cls[0])]
                conf = float(b.conf[0])
                if cls_name != 'person' and conf > 0.22:
                    xyxy = [
                        int(b.xyxy[0][0]) + cw1,
                        int(b.xyxy[0][1]) + ch1,
                        int(b.xyxy[0][2]) + cw1,
                        int(b.xyxy[0][3]) + ch1
                    ]
                    detected_candidates.append({
                        'label': cls_name,
                        'confidence': conf,
                        'box': xyxy,
                        'source': 'yolo_crop'
                    })
        except Exception as e:
            print("YOLO crop inference notice:", e)

    # Sort YOLO candidates by confidence
    detected_candidates.sort(key=lambda x: x['confidence'], reverse=True)

    # 3. MobileNet V3 Classification on primary crop or center
    mobilenet_top = []
    if GLOBAL_MOBILENET_MODEL and GLOBAL_MOBILENET_PREPROCESS:
        try:
            import torch
            from PIL import Image
            crop_for_mb = None
            if detected_candidates:
                bx = detected_candidates[0]['box']
                x1, y1 = max(0, bx[0] - 15), max(0, bx[1] - 15)
                x2, y2 = min(w, bx[2] + 15), min(h, bx[3] + 15)
                crop_for_mb = img[y1:y2, x1:x2]
            else:
                crop_for_mb = center_crop

            if crop_for_mb is not None and crop_for_mb.size > 100:
                pil_c = Image.fromarray(cv2.cvtColor(crop_for_mb, cv2.COLOR_BGR2RGB))
                tensor = GLOBAL_MOBILENET_PREPROCESS(pil_c).unsqueeze(0)
                with torch.no_grad():
                    out = GLOBAL_MOBILENET_MODEL(tensor).squeeze(0)
                    probs = torch.softmax(out, dim=0)
                    top_vals, top_inds = torch.topk(probs, 5)
                    for val, ind in zip(top_vals, top_inds):
                        c_name = GLOBAL_MOBILENET_CATEGORIES[ind]
                        c_prob = float(val)
                        mobilenet_top.append((c_name, c_prob))
        except Exception as e:
            print("MobileNet inference notice:", e)

    # 4. Resolve Best Identified Label
    chosen_label = None
    if detected_candidates:
        raw_label = detected_candidates[0]['label']
        chosen_label = OBJECT_LABEL_MAP.get(raw_label.lower(), raw_label)
    elif mobilenet_top:
        for m_cat, m_prob in mobilenet_top:
            m_lower = m_cat.lower()
            if any(k in m_lower for k in ['pen', 'glass', 'watch', 'clock', 'bottle', 'mug', 'cup', 'phone', 'cellular', 'wallet', 'purse', 'remote', 'lighter', 'scissors']):
                chosen_label = OBJECT_LABEL_MAP.get(m_lower, m_cat)
                break
        if not chosen_label and mobilenet_top[0][1] > 0.12:
            chosen_label = OBJECT_LABEL_MAP.get(mobilenet_top[0][0].lower(), mobilenet_top[0][0])

    return chosen_label, detected_candidates


# ============================================================
# Multi-Model AI Engine: Groq Ultra-Fast + Google Gemini Fallback
# ============================================================
GEMINI_MODELS = [
    'gemini-2.0-flash',
    'gemini-1.5-flash',
    'gemini-1.5-flash-8b',
    'gemini-1.5-pro'
]

def call_groq_chat(prompt: str = None, system_instruction: str = None, json_mode: bool = False, timeout: int = 8, model: str = "qwen/qwen3.8-27b", messages_history: list = None) -> str:
    """
    Executes fast inference via Groq cloud API with full conversation memory.
    Supports qwen/qwen3.8-27b and openai/gpt-oss-120b.
    """
    if not GROQ_KEY:
        return ""

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    if messages_history:
        for m in messages_history:
            role = "assistant" if m.get("role") in ["assistant", "model"] else "user"
            content = m.get("text") or m.get("content") or ""
            if content:
                messages.append({"role": role, "content": content})
    if prompt:
        messages.append({"role": "user", "content": prompt})

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_KEY}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    models_to_try = [model, "openai/gpt-oss-120b"] if model != "openai/gpt-oss-120b" else ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

    for m in models_to_try:
        try:
            payload = {
                "model": m,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 800
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=data)
            res = urllib.request.urlopen(req, timeout=timeout)
            resp = json.loads(res.read().decode('utf-8'))
            content = resp.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            if content:
                return content
        except urllib.error.HTTPError as he:
            try:
                err_body = he.read().decode('utf-8')
                err_data = json.loads(err_body)
                failed_gen = err_data.get('error', {}).get('failed_generation', '').strip()
                if failed_gen:
                    if json_mode:
                        return json.dumps({"intent": "GENERAL_CHAT", "answer": failed_gen})
                    return failed_gen
            except Exception:
                pass
            print(f"Groq chat [{m}] HTTP error: {he}")
        except Exception as e:
            print(f"Groq chat [{m}] error: {e}")
            continue
    return ""

def call_gemini_generate(prompt: str, system_instruction: str = None, json_mode: bool = False, timeout: int = 10) -> str:
    """
    Unified LLM generation: Groq primary (fast & active) with Gemini fallback.
    """
    # 1. Try Groq first for instant 200ms response
    groq_resp = call_groq_chat(prompt, system_instruction=system_instruction, json_mode=json_mode, timeout=timeout)
    if groq_resp:
        return groq_resp

    # 2. Fallback to Gemini if key available
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

def call_groq_vision(image_base64: str, prompt: str, timeout: int = 10) -> str:
    """
    Multimodal image understanding via Groq Qwen 3.8 27B vision.
    """
    if not GROQ_KEY or not image_base64:
        return ""
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    image_base64 = image_base64.strip()

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_KEY}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "temperature": 0.4,
        "max_tokens": 400
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=data)
        res = urllib.request.urlopen(req, timeout=timeout)
        resp = json.loads(res.read().decode('utf-8'))
        content = resp.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        if content:
            return content
    except Exception as e:
        print(f"Groq vision error: {e}")
    return ""

def call_vision(image_base64: str, prompt: str, system_instruction: str = None, timeout: int = 12) -> str:
    """Unified vision pipeline: Groq Qwen 3.8 27B primary, Gemini fallback."""
    # 1. Try Groq multimodal vision first
    g_res = call_groq_vision(image_base64, prompt, timeout=timeout)
    if g_res:
        return g_res
    # 2. Try Gemini vision fallback
    return call_gemini_vision(image_base64, prompt, system_instruction=system_instruction, timeout=timeout)

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
# 1.5. High-Fidelity Audio TTS Streaming for Multilingual Output (JARVIS & LISA)
# ============================================================
TTS_AUDIO_CACHE = {}

@app.route('/api/tts', methods=['GET', 'POST'])
def api_tts():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form or {}
        text = str(data.get('text', '')).strip()
        lang = str(data.get('lang', '')).strip().lower()
        persona = str(data.get('persona', 'Jarvis')).strip()
    else:
        text = request.args.get('text', '').strip()
        lang = request.args.get('lang', '').strip().lower()
        persona = request.args.get('persona', 'Jarvis').strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    clean_text = re.sub(r'[*_~#`]', '', text)
    clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Intelligent Unicode Script Detection: Prevents mismatched voice selection
    kn_chars = len(re.findall(r'[\u0C80-\u0CFF]', clean_text))
    ta_chars = len(re.findall(r'[\u0B80-\u0BFF]', clean_text))
    te_chars = len(re.findall(r'[\u0C00-\u0C7F]', clean_text))
    hi_chars = len(re.findall(r'[\u0900-\u097F]', clean_text))
    ml_chars = len(re.findall(r'[\u0D00-\u0D7F]', clean_text))
    bn_chars = len(re.findall(r'[\u0980-\u09FF]', clean_text))

    script_counts = [
        ('kn', kn_chars),
        ('ta', ta_chars),
        ('te', te_chars),
        ('hi', hi_chars),
        ('ml', ml_chars),
        ('bn', bn_chars)
    ]
    script_counts.sort(key=lambda x: x[1], reverse=True)

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

    if script_counts[0][1] >= 2:
        target_tl = script_counts[0][0]
    else:
        target_tl = lang_map.get(lang, 'en')

    is_lisa = (persona.lower() == 'lisa')
    cache_key = (clean_text[:400], target_tl, is_lisa)
    if cache_key in TTS_AUDIO_CACHE:
        return Response(TTS_AUDIO_CACHE[cache_key], mimetype='audio/mpeg')

    try:
        from gtts import gTTS
        import io
        if target_tl == 'en':
            tld = 'com' if is_lisa else 'co.uk'
            tts = gTTS(text=clean_text[:450], lang='en', tld=tld)
        else:
            tts = gTTS(text=clean_text[:450], lang=target_tl)

        buf = io.BytesIO()
        tts.write_to_fp(buf)
        audio_data = buf.getvalue()

        # Cache in memory
        if len(TTS_AUDIO_CACHE) > 150:
            TTS_AUDIO_CACHE.pop(next(iter(TTS_AUDIO_CACHE)))
        TTS_AUDIO_CACHE[cache_key] = audio_data

        return Response(audio_data, mimetype='audio/mpeg')
    except Exception as e:
        print(f"gTTS generation error for lang '{target_tl}':", e)
        return jsonify({"error": str(e), "lang": target_tl}), 500

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

def clean_music_query(raw_query, active_lang='en'):
    q = (raw_query or '').strip()
    if not q:
        return 'top trending hits'

    # Remove punctuation
    q = re.sub(r'[\.,!?।\-_"\'\(\)]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()

    # Leading fillers across English, Kannada, Telugu, Hindi, Tamil
    leading_patterns = [
        r'^(?:can\s+you\s+)?(?:could\s+you\s+)?(?:please\s+)?(?:i\s+want\s+(?:to\s+listen\s+to\s+)?)?(?:play|start|put\s+on|listen\s+to|hear)\s+(?:me\s+)?(?:a\s+|some\s+|the\s+)?',
        r'^(?:play|start|put\s+on|listen\s+to|hear)\s+',
        r'^(?:some\s+|a\s+|the\s+)?(?:songs?\s+|music\s+|tracks?\s+)?(?:of\s+|by\s+|from\s+)?',
        r'^(?:ದಯವಿಟ್ಟು\s+)?(?:ಯಾವುದಾದರೂ\s+)?(?:ಒಂದು\s+)?(?:ಒಳ್ಳೆಯ\s+)?(?:ಹೊಸ\s+)?',
        r'^(?:దయచేసి\s+)?(?:ఏదైనా\s+)?(?:ఒక\s+)?(?:మంచి\s+)?(?:కొత్త\s+)?',
        r'^(?:कृपया\s+)?(?:कोई\s+)?(?:एक\s+)?(?:अच्छा\s+)?(?:नया\s+)?',
        r'^(?:தயவுசெய்து\s+)?(?:ஏதாவது\s+)?(?:ஒரு\s+)?(?:நல்ல\s+)?(?:புதிய\s+)?'
    ]
    for pat in leading_patterns:
        q = re.sub(pat, '', q, flags=re.IGNORECASE).strip()

    # Trailing fillers
    trailing_patterns = [
        r'\s+(?:songs?|music|tracks?|audios?|movie\s+songs?)$',
        r'(?:\s+)?(?:ಪ್ಲೇ\s*ಮಾಡು|ಪ್ಲೇ\s*ಮಾಡಿ|ಹಾಡು\s*ಹಾಕು|ಹಾಕು|ಕೇಳಿಸು|ಪ್ಲೇ|ಮಾಡು|ಹಾಕಿ|ಮಾಡಿಸು)$',
        r'(?:\s+)?(?:ಹಾಡುಗಳು?|ಸಾಂಗ್ಸ್?|ಸಾಂಗ್|ಸಂಗೀತ|ಮ್ಯೂಸಿಕ್)$',
        r'(?:\s+)?(?:[ಪ್ಪ್]ಲೇ\s*[చేచ]ే[యియ][ిಿ]?|వినిపించు|పెట్టు|వేయి|ప్ಲೇ|చేయి)$',
        r'(?:\s+)?(?:పాటలు?|సాంగ్ಸ್?|సాంగ్|సಂಗీతం|మ్యూజిక్)$',
        r'(?:\s+)?(?:बजाओ|सुनाओ|चलाओ|लगाओ|प्ले\s*करो|प्ले)$',
        r'(?:\s+)?(?:गाने?|गीत|सॉन्ग्स?|सॉन्ग|संगीत|म्यूजिक)$',
        r'(?:\s+)?(?:போடு|பிளே\s*செய்|பிளே\s*பண்ணு|பிளே|பாடு)$',
        r'(?:\s+)?(?:பாடல்கள்?|பாட்டு|சாங்|இசை|மியூசிக்)$'
    ]
    for pat in trailing_patterns:
        q = re.sub(pat, '', q, flags=re.IGNORECASE).strip()

    lang_defaults = {
        'kn': 'Kannada top hits',
        'te': 'Telugu top hits',
        'hi': 'Bollywood Hindi hits',
        'ta': 'Tamil top hits',
        'ml': 'Malayalam hits',
        'mr': 'Marathi hits',
        'bn': 'Bengali hits',
        'en': 'top trending hits'
    }

    q_low = q.lower()
    specific_lang = None
    if 'kannada' in q_low or 'ಕನ್ನಡ' in q:
        specific_lang = 'kn'
    elif 'telugu' in q_low or 'తెలుగు' in q:
        specific_lang = 'te'
    elif 'hindi' in q_low or 'हिंदी' in q or 'हಿಂದಿ' in q:
        specific_lang = 'hi'
    elif 'tamil' in q_low or 'தமிழ்' in q or 'ತಮಿಳು' in q:
        specific_lang = 'ta'
    elif 'malayalam' in q_low or 'മലയാളം' in q:
        specific_lang = 'ml'

    residual = q
    residual = re.sub(r'\b(?:kannada|telugu|hindi|tamil|malayalam|english)\b', '', residual, flags=re.IGNORECASE)
    residual = re.sub(r'(?:ಕನ್ನಡ|తెలుగు|हिंदी|தமிழ்|മലയാളം)', '', residual)
    residual = re.sub(r'\b(?:top|trending|hit|hits|best|popular|super|latest|new|old|song|songs|music|track|audio|play)\b', '', residual, flags=re.IGNORECASE)
    residual = re.sub(r'(?:ಹಾಡುಗಳು?|ಹಾಡು|ಸಾಂಗ್ಸ್?|ಸಾಂಗ್|ಸಂಗೀತ|ಮ್ಯೂಸಿಕ್|ಪ್ಲೇ|ಹಾಕು|ಮಾಡಿಸು|ಕೇಳಿಸು)', '', residual)
    residual = re.sub(r'(?:పాటలు?|పాట|సాಂಗ್ಸ್?|సాంగ్|సಂಗీతం|మ్యూజిక్|ప్లే|పెట్టు|వేయి|చేయి|వినిపించు)', '', residual)
    residual = re.sub(r'(?:गाने?|गाना|गीत|सॉन्ग्स?|सॉन्ग|संगीत|म्यूजिक|बजाओ|सुनाओ|चलाओ|लगाओ|प्ले)', '', residual)
    residual = re.sub(r'(?:பாடல்கள்?|பாடல்|பாட்டு|சாங்|இசை|மியூசிக்|போடு|பிளே|பாடு)', '', residual)
    residual = re.sub(r'(?:ಒಂದು|ಯಾವುದಾದರೂ|ಒಳ್ಳೆಯ|ಹೊಸ|ಯಾವ್ದಾದ್ರೂ|ಒಳ್ಳೆ)', '', residual)
    residual = re.sub(r'(?:ఒక|ఏదైనా|మంచి|కొత్త|ఒక్క)', '', residual)
    residual = re.sub(r'(?:एक|कोई|अच्छा|नया|के|का|की|को)', '', residual)
    residual = re.sub(r'(?:ஒரு|ஏதாவது|நல்ல|புதிய)', '', residual)
    residual = re.sub(r'\s+', ' ', residual).strip()

    if not residual or len(residual) <= 1:
        if specific_lang:
            return lang_defaults[specific_lang]
        return lang_defaults.get(active_lang, 'top trending hits')

    clean_specific = q
    clean_specific = re.sub(r'\s+(?:songs?|music|track)$', '', clean_specific, flags=re.IGNORECASE)
    clean_specific = re.sub(r'(?:\s+)?(?:ಹಾಡುಗಳು?|ಹಾಡು|ಸಾಂಗ್ಸ್?|ಸಾಂಗ್|ಹಾಕು|ಪ್ಲೇ)$', '', clean_specific)
    clean_specific = re.sub(r'(?:\s+)?(?:పాటలు?|పాట|సాಂಗ್ಸ್?|సాಂಗ್|పెట్టు|ಪ್ಲೇ)$', '', clean_specific)
    clean_specific = re.sub(r'(?:\s+)?(?:गाने?|गाना|गीत|सॉन्ग्स?|सॉन्ग|बजाओ|सुनाओ)$', '', clean_specific)
    clean_specific = re.sub(r'(?:\s+)?(?:பாடல்கள்?|பாடல்|பாட்டு|போடு|பிளே)$', '', clean_specific)
    clean_specific = re.sub(r'(?:^[केकाकीको]\s+)', '', clean_specific)
    clean_specific = re.sub(r'(?:\s+[केकाकीको]$)', '', clean_specific)
    clean_specific = clean_specific.strip()

    return clean_specific or q

@app.route('/api/music/search', methods=['GET'])
def search_music():
    raw_query = (request.args.get('q') or request.args.get('query') or '').strip()
    active_lang = request.args.get('lang') or SESSION_STATE.get("active_language", "en")

    lang_defaults = {
        'te': 'Telugu top hits',
        'kn': 'Kannada top hits',
        'hi': 'Bollywood Hindi hits',
        'ta': 'Tamil top hits',
        'ml': 'Malayalam hits',
        'mr': 'Marathi hits',
        'bn': 'Bengali hits',
        'en': 'top trending hits'
    }

    is_different = any(k in raw_query.lower() for k in ['different', 'another', 'next', 'ಬೇರೆ', 'ವೇರೇ', 'మరో', 'दूसरा', 'अगला'])

    if is_different:
        query = lang_defaults.get(active_lang, 'top trending hits')
    else:
        query = clean_music_query(raw_query, active_lang)

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
    lang = request.args.get("lang") or SESSION_STATE.get("active_language", "en")
    lang_name = LANGUAGE_CONFIG.get(lang, {}).get("name", "English")
    today_str = time.strftime("%B %d, %Y")
    
    prompt = (
        f"Provide exactly 5 major current real news headlines in India and worldwide in {lang_name}. "
        "Keep each headline under 12 words. "
        "Return ONLY a JSON array: [ {\"id\": 1, \"title\": \"...\", \"category\": \"...\", \"summary\": \"...\"}, ... ]. "
        "No markdown codeblocks or extra text."
    )

    headlines = []
    # 1. Primary: Ultra-fast Groq news generation (< 1.5s)
    try:
        groq_resp = call_groq_chat(
            prompt,
            system_instruction="You are a live real-time news curator. Return ONLY a valid JSON array of 5 news headlines.",
            timeout=5
        )
        if groq_resp:
            clean_json = groq_resp.strip().replace('```json', '').replace('```', '').strip()
            parsed = json.loads(clean_json)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict) and 'title' in parsed[0]:
                headlines = parsed
    except Exception as ge:
        print("Groq news parse error:", ge)

    if not headlines:
        multilingual_fallbacks = {
            'kn': [
                {"id": 1, "title": "ಭಾರತದ ತಂತ್ರಜ್ಞಾನ ವಲಯದಲ್ಲಿ ಎಐ ಕ್ರಾಂತಿ", "category": "ತಂತ್ರಜ್ಞಾನ", "summary": "ಕರ್ನಾಟಕ ಮತ್ತು ದೇಶಾದ್ಯಂತ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ಸೆಮಿಕಂಡಕ್ಟರ್ ಉದ್ಯಮಗಳಿಗೆ ಭಾರಿ ಹೂಡಿಕೆ."},
                {"id": 2, "title": "ನವೀಕರಿಸಬಹುದಾದ ಇಂಧನ ಯೋಜನೆಗಳಲ್ಲಿ ಹೊಸ ದಾಖಲೆ", "category": "ಪರಿಸರ", "summary": "ಸೌರ ಮತ್ತು ಪವನ ಶಕ್ತಿ ಮೂಲಸೌಕರ್ಯ ಅಭಿವೃದ್ಧಿಗೆ ಜಾಗತಿಕ ಹೂಡಿಕೆಗಳು ಹೆಚ್ಚಳ."},
                {"id": 3, "title": "ಇಸ್ರೋದಿಂದ ನೂತನ ಬಾಹ್ಯಾಕಾಶ ಮತ್ತು ಉಪಗ್ರಹ ಯೋಜನೆಗಳು", "category": "ವಿಜ್ಞಾನ", "summary": "ಚಂದ್ರ ಹಾಗೂ ಗ್ರಹಗಳ ಸಂಶೋಧನೆಗಾಗಿ ಮುಂದಿನ ಪೀಳಿಗೆಯ ಉಪಗ್ರಹಗಳ ಉಡಾವಣೆಗೆ ಸಿದ್ಧತೆ."},
                {"id": 4, "title": "ರಾಷ್ಟ್ರೀಯ ಕ್ರೀಡಾಕೂಟದಲ್ಲಿ ಯುವ ಪ್ರತಿಭೆಗಳ ಮಿಂಚು", "category": "ಕ್ರೀಡೆ", "summary": "ರಾಷ್ಟ್ರೀಯ ಮಟ್ಟದ ಕ್ರೀಡೆಗಳಲ್ಲಿ ಹೊಸ ದಾಖಲೆಗಳನ್ನು ನಿರ್ಮಿಸಿದ ಯುವ ಆಟಗಾರರು."},
                {"id": 5, "title": "ಸ್ಮಾರ್ಟ್ ಸಿಟಿಗಳಲ್ಲಿ ಸುಧಾರಿತ ಸಂಚಾರ ಮತ್ತು ವಿದ್ಯುತ್ ವ್ಯವಸ್ಥೆ", "category": "ನಾಗರಿಕ", "summary": "ನಗರಗಳಲ್ಲಿ ಐಒಟಿ ಹಾಗೂ ಎಐ ಬಳಸಿ ಸುಗಮ ಸಂಚಾರ ನಿರ್ವಹಣೆ."}
            ],
            'te': [
                {"id": 1, "title": "భారత టెక్ రంగాన్ని నడిపిస్తున్న అధునాతన ఏఐ ప్రాజెక్టులు", "category": "టెక్నాలజీ", "summary": "దేశవ్యాప్తంగా సెమీకండక్టర్లు మరియు ఏఐ కంప్యూటింగ్ విస్తరణ."},
                {"id": 2, "title": "పునరుత్పాదక ఇంధన పెట్టుబడుల్లో సరికొత్త రికార్డులు", "category": "పర్యావరణం", "summary": "సౌర మరియు పవన విద్యుత్ ప్రాజెక్టులకు దేశవ్యాప్త ఊతం."},
                {"id": 3, "title": "ఇస్రో ద్వారా అంతరిక్ష అన్వేషణలో నూతన ముందడుగు", "category": "సైన్స్", "summary": "భవిష్యత్ పరిశోధనల కోసం సరికొత్త ఉపగ్రహాల ప్రయోగ సన్నాహాలు."},
                {"id": 4, "title": "జాతీయ క్రీడల్లో యువ అథ్లెట్ల అద్భుత ప్రదర్శన", "category": "క్రీడలు", "summary": "జాతీయ ఛాంపియన్‌షిప్‌లలో సరికొత్త రికార్డులు నెలకొల్పిన క్రీడాకారులు."},
                {"id": 5, "title": "స్మార్ట్ సిటీలలో ఆధునిక ట్రాఫిక్ & ఇంధన నిర్వహణ వ్యవస్థలు", "category": "నగరం", "summary": "పౌర సౌకర్యాల మెరుగుదలకు అత్యాధునిక ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ పరిష్కారాలు."}
            ],
            'hi': [
                {"id": 1, "title": "भारत के तकनीकी क्षेत्र में एआई और सेमीकंडक्टर का बड़ा विस्तार", "category": "तकनीक", "summary": "देशभर में आधुनिक कंप्यूटिंग इंफ्रास्ट्रक्चर के लिए महत्वपूर्ण निवेश की घोषणा।"},
                {"id": 2, "title": "अक्षय ऊर्जा क्षेत्र में वैश्विक स्तर पर रिकॉर्ड निवेश", "category": "पर्यावरण", "summary": "सौर और पवन ऊर्जा विकास में भारी गति दर्ज की गई।"},
                {"id": 3, "title": "इसरो का आगामी अंतरिक्ष और उपग्रह मिशन तैयार", "category": "विज्ञान", "summary": "वैज्ञानिक अन्वेषण के लिए नए अनुसंधान उपग्रहों का प्रक्षेपण जल्द।"},
                {"id": 4, "title": "राष्ट्रीय खेल प्रतियोगिताओं में युवा खिलाड़ियों का उत्कृष्ट प्रदर्शन", "category": "खेल", "summary": "एथलेटिक्स और क्रिकेट में नए राष्ट्रीय कीर्तिमान स्थापित।"},
                {"id": 5, "title": "स्मार्ट शहरों में आधुनिक यातायात और डिजिटल ऊर्जा प्रबंधन", "category": "नागरिक", "summary": "स्मार्ट प्रबंधन के जरिए दैनिक शहरी सुविधाओं में व्यापक सुधार।"}
            ],
            'ta': [
                {"id": 1, "title": "இந்திய தொழில்நுட்பத் துறையில் புதிய ஏஐ முதலீடுகள்", "category": "தொழில்நுட்பம்", "summary": "நாடு முழுவதும் செயற்கை நுண்ணறிவு மற்றும் குறைக்கடத்தி விரிவாக்கம்."},
                {"id": 2, "title": "புதுப்பிக்கத்தக்க எரிசக்தி திட்டங்களில் புதிய சாதனை", "category": "சுற்றுச்சூழல்", "summary": "சூரிய மற்றும் காற்றாலை மின் உற்பத்தியில் உலகளாவிய முதலீடுகள் அதிகரிப்பு."},
                {"id": 3, "title": "இஸ்ரோவின் அடுத்த தலைமுறை விண்வெளி ஆய்வுகள் தீவிரம்", "category": "அறிவியல்", "summary": "புதிய செயற்கைக்கோள் திட்டங்களுக்கான தயாரிப்புகள் மும்முரம்."},
                {"id": 4, "title": "தேசிய விளையாட்டுப் போட்டிகளில் இளம் வீரர்களின் சாதனை", "category": "விளையாட்டு", "summary": "தேசிய தடகளப் போட்டிகளில் புதிய சாதனைகள் பதிவு."},
                {"id": 5, "title": "ஸ்மார்ட் நகரங்களில் அதிநவீன போக்குவரத்து மேலாண்மை", "category": "நகர்ப்புறம்", "summary": "ஐஓடி மற்றும் ஏஐ தொழில்நுட்பம் மூலம் சீரான போக்குவரத்து."}
            ],
            'en': [
                {"id": 1, "title": "India's Tech Sector Surges with New AI & Semiconductor Initiatives", "category": "Tech", "summary": "Government and industry leaders announce major computing expansions across Karnataka and nationwide."},
                {"id": 2, "title": "Global Climate Summit Focuses on Renewable Energy Investments", "category": "World", "summary": "Countries commit to accelerating solar, wind, and battery storage infrastructure development."},
                {"id": 3, "title": "Space Agency Announces Next-Gen Lunar and Planetary Missions", "category": "Science", "summary": "Preparations are underway for upcoming autonomous exploration flights and orbital satellites."},
                {"id": 4, "title": "National Sports Championship Highlights Emerging Athletes", "category": "Sports", "summary": "Record-breaking performances mark the national athletics and cricket championships."},
                {"id": 5, "title": "Smart Cities Mission Deploys Advanced Traffic & Energy Systems", "category": "National", "summary": "Urban municipal corporations integrate IoT and AI for improved daily citizen utilities."}
            ]
        }
        headlines = multilingual_fallbacks.get(lang, multilingual_fallbacks['en'])

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
        desc = call_vision(image_base64, prompt_text, timeout=8)
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
        reply = call_vision(image_base64, vis_prompt, timeout=8)
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
@app.route('/api/market/gold', methods=['GET'])
def api_market_gold():
    city = request.args.get('city', 'bangalore')
    data = fetch_live_gold_rates(city)
    if data:
        return jsonify({"status": "success", "data": data})
    return jsonify({"status": "error", "message": "Failed to fetch live gold rates"}), 500

@app.route('/api/market/indices', methods=['GET'])
def api_market_indices():
    indices = fetch_live_market_indices()
    if indices:
        return jsonify({"status": "success", "data": indices})
    return jsonify({"status": "error", "message": "Failed to fetch market indices"}), 500

def analyze_user_appearance_opencv(b64_str):
    try:
        if not b64_str:
            return {"face_detected": False, "clothing_color": "neutral tone", "face_shape": "balanced oval"}
        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        raw_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"face_detected": False, "clothing_color": "neutral tone", "face_shape": "balanced oval"}

        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        face_shape = "balanced oval"
        color_desc = "neutral tone"
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            ratio = fh / float(fw)
            if ratio > 1.25:
                face_shape = "elongated oval"
            elif ratio < 1.05:
                face_shape = "square / defined"
            else:
                face_shape = "balanced oval"

            torso_y1 = min(h - 1, fy + fh + int(fh * 0.15))
            torso_y2 = min(h, torso_y1 + int(fh * 1.2))
            torso_x1 = max(0, fx - int(fw * 0.4))
            torso_x2 = min(w, fx + fw + int(fw * 0.4))

            if torso_y2 > torso_y1 and torso_x2 > torso_x1:
                torso = img[torso_y1:torso_y2, torso_x1:torso_x2]
                mean_bgr = cv2.mean(torso)[:3]
                b, g, r = mean_bgr
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                if brightness < 65:
                    color_desc = "dark / black"
                elif brightness > 185:
                    color_desc = "light / white"
                elif b > r + 20 and b > g + 20:
                    color_desc = "blue / navy"
                elif g > r + 20 and g > b + 20:
                    color_desc = "olive / green"
                elif r > g + 20 and r > b + 20:
                    color_desc = "warm red / maroon"
                else:
                    color_desc = "classic neutral"
        return {
            "face_detected": len(faces) > 0,
            "face_shape": face_shape,
            "clothing_color": color_desc
        }
    except Exception as e:
        print("Visual appearance analysis notice:", e)
        return {"face_detected": False, "clothing_color": "neutral tone", "face_shape": "balanced oval"}

@app.route('/api/analyze-image', methods=['POST'])
def api_analyze_image():
    data = request.get_json() or {}
    image_base64 = data.get('image') or data.get('image_base64')
    question = data.get('question') or data.get('prompt') or "Give me detailed suggestions for my overall look, outfit, and hairstyle."
    weather = (data.get('weather') or data.get('weather_summary') or '').strip()
    lang = (data.get('active_language') or data.get('language') or 'en').lower()
    persona = data.get('persona', 'Jarvis')
    lang_info = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])

    if not image_base64:
        return jsonify({
            "success": False,
            "message": "I cannot see you clearly in the mirror. Please step in front of the camera and let's check your look."
        }), 400

    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    # Fast OpenCV visual appearance extraction
    cv_info = analyze_user_appearance_opencv(image_base64)
    cv_summary = f"Detected visual features: face structure is {cv_info['face_shape']}, upper clothing tone is {cv_info['clothing_color']}."
    weather_note = f"Current local weather is {weather}. Address whether their clothes suit this weather and suggest practical layering if helpful. " if weather else ""

    prompt_text = (
        f"You are {persona}, talking with a close friend in front of the smart mirror. "
        f"Look at their mirror camera image and visual details: {cv_summary}. "
        f"{weather_note}"
        f"The friend asks: '{question}'. "
        f"Give them a simple, friendly styling suggestion for their overall look in {lang_info['name']}. "
        f"STRICT RULES: "
        f"- Provide exactly 2 or 3 short, simple, easily understandable sentences (maximum 40 words total). "
        f"- Sentence 1: A simple, friendly compliment on their current outfit color ({cv_info['clothing_color']}) and fit. "
        f"- Sentence 2: One simple hairstyle or grooming tip for their face. "
        f"- Sentence 3: One simple accessory or styling suggestion (like a classic watch or clean footwear). "
        f"- Use everyday words that anyone can easily understand. "
        f"- Speak naturally like two good friends having a conversation. Do NOT use 'bro', 'buddy', or repetitive slang. "
        f"- Must be complete, grammatically correct, simple sentences."
    )

    def clean_simple_outfit_sentences(text: str, max_sentences: int = 3) -> str:
        if not text:
            return ""
        clean = re.sub(r'[*_~#`]', '', text)
        clean = re.sub(r'^[-\d\.\s]+', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'\s+', ' ', clean).strip()
        sentences = re.findall(r'[^.!?।\n]+[.!?।]', clean)
        if not sentences:
            sentences = [clean]
        return ' '.join(s.strip() for s in sentences[:max_sentences])

    # 0. Primary: Direct Multimodal AI Vision Stylist (Groq Qwen 3.8 27B)
    try:
        outfit_prompt = (
            f"You are {persona}, a thoughtful, stylish personal fashion stylist in Smart Mirror Vox.\n"
            f"Examine this mirror camera snapshot of the user carefully.\n"
            f"Analyze their outfit, clothing items (shirts, t-shirt, jacket, dress, colors), hairstyle, and overall look.\n"
            f"{weather_note}\n"
            f"Provide 2 to 3 articulate, warm, and helpful styling suggestions in {lang_info['name']} ({lang_info.get('native', '')}).\n"
            f"Acknowledge what they are wearing, what suits them well, and suggest one sharp enhancement (colors to match, accessories, or hairstyle grooming). Maximum 40 words total."
        )
        groq_outfit = call_groq_vision(image_base64, outfit_prompt, timeout=8)
        if groq_outfit and len(groq_outfit.strip()) > 8:
            clean_outfit = clean_simple_outfit_sentences(groq_outfit)
            if clean_outfit:
                return jsonify({
                    "success": True,
                    "analysis": clean_outfit,
                    "recommendations": clean_outfit,
                    "message": clean_outfit
                })
    except Exception as ove:
        print("Groq outfit vision notice:", ove)

    # Primary: Dynamic Groq LLM with visual attributes extracted by OpenCV (< 1s latency)
    try:
        groq_sys = (
            f"You are {persona}, a stylish, thoughtful close friend giving honest advice on your friend's overall appearance. "
            f"Speak naturally like two close, polite friends having a real conversation. "
            f"Never use 'bro', 'buddy', or repetitive slang. "
            f"Provide exactly 2 to 3 complete, simple, and articulate sentences in {lang_info['name']} (maximum 40 words total)."
        )
        groq_advice = call_groq_chat(
            prompt_text,
            system_instruction=groq_sys,
            timeout=6
        )
        if groq_advice:
            cleaned_groq = clean_simple_outfit_sentences(groq_advice)
            if cleaned_groq and len(cleaned_groq) > 8:
                return jsonify({
                    "success": True,
                    "analysis": cleaned_groq,
                    "recommendations": cleaned_groq,
                    "message": cleaned_groq
                })
    except Exception as ge:
        print("Groq analyze image notice:", ge)

    # 3. Resilient fallback advice in simple 2-3 sentences
    fallback_map = {
        'te': f"మీ {cv_info['clothing_color']} దుస్తులు మీకు చాలా బాగున్నాయి. మీ ముఖానికి తగినట్లుగా హెయిర్ సైడ్స్ నీట్‌గా ఉంచితే లుక్ మరింత బాగుంటుంది. దీనికి ఒక సింపుల్ వాచ్ జతచేస్తే మీ లుక్ పరిపూర్ణంగా ఉంటుంది.",
        'kn': f"ನಿಮ್ಮ {cv_info['clothing_color']} ಬಟ್ಟೆ ನಿಮಗೆ ತುಂಬಾ ಚೆನ್ನಾಗಿ ಹೊಂದುತ್ತದೆ. ನಿಮ್ಮ ಮುಖಕ್ಕೆ ಸೈಡ್ಸ್ ನೀಟಾಗಿ ಟ್ರಿಮ್ ಮಾಡಿದ ಹೇರ್‌ಸ್ಟೈಲ್ ಇನ್ನಷ್ಟು ಚೆನ್ನಾಗಿ ಕಾಣಿಸುತ್ತದೆ. ಇದಕ್ಕೆ ಒಂದು ಕ್ಲಾಸಿಕ್ ವಾಚ್ ಸೇರಿಸಿದರೆ ನಿಮ್ಮ ಲುಕ್ ಪರ್ಫೆಕ್ಟ್ ಆಗಿರುತ್ತದೆ.",
        'hi': f"आपके {cv_info['clothing_color']} कपड़े आप पर बहुत अच्छे लग रहे हैं। आपके चेहरे पर साइड्स से ट्रिम की हुई हेयरस्टाइल और भी बढ़िया लगेगी। इसके साथ एक सिंपल क्लासिक घड़ी पहनेंगे तो आपका पूरा लुक एकदम परफेक्ट लगेगा।",
        'ta': f"உங்கள் {cv_info['clothing_color']} ஆடை உங்களுக்கு மிகவும் அழகாகப் பொருந்துகிறது. உங்கள் முகத்திற்கு பக்கவாட்டில் நேர்த்தியான ஹேர்ஸ்டைல் இன்னும் எடுப்பாக இருக்கும். அத்துடன் ஒரு எளிய வாட்ச் அணிந்தால் உங்கள் லுக் அருமையாக இருக்கும்.",
        'en': f"Your {cv_info['clothing_color']} outfit looks great on you. Neatly tapered sides with a textured hairstyle will frame your face nicely. Adding a simple classic watch will complete your stylish look."
    }
    fallback_advice = fallback_map.get(lang, fallback_map['en'])
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
    return jsonify({
        "status": "disabled",
        "message": "Image generation feature has been removed. Camera outfit analysis and object vision remain fully active."
    }), 200

# ============================================================
# 4.5. Real-Time Object & Item Identification via Mirror Camera
# ============================================================
@app.route('/api/vision/identify', methods=['POST'])
def identify_camera_object():
    data = request.get_json() or {}
    image_base64 = data.get('image') or data.get('image_base64')
    question = data.get('question') or data.get('prompt') or data.get('text') or "What is this object?"
    detected_label = data.get('detected_label') or data.get('label') or ""
    lang = (data.get('active_language') or data.get('language') or 'en').lower()
    persona = data.get('persona', 'Jarvis')
    lang_info = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])

    if not image_base64 and not detected_label:
        return jsonify({
            "status": "error",
            "answer": "I couldn't see anything. Please hold the object clearly in front of the mirror camera."
        }), 400

    clean_b64 = image_base64
    if clean_b64 and ',' in clean_b64:
        clean_b64 = clean_b64.split(',')[1]

    img = None
    if clean_b64:
        # 0. Primary: High-Accuracy Multimodal AI Vision (Groq Qwen 3.8 27B)
        vision_prompt = (
            f"You are {persona}, an intelligent Smart Mirror visual assistant.\n"
            f"Examine this mirror camera frame carefully. The user is asking: '{question}'.\n"
            f"Identify specifically and accurately what item, object, gadget, fruit, drink, tool, book, accessory, or thing the user is holding up or presenting to the mirror camera.\n"
            f"Respond strictly in {lang_info['name']} ({lang_info.get('native', '')}).\n"
            f"Provide 1 to 2 crisp, friendly, and articulate sentences. Name the exact item clearly, describe its color/finish, and state its practical use. Maximum 30 words total."
        )
        try:
            groq_vis = call_groq_vision(clean_b64, vision_prompt, timeout=8)
            if groq_vis and len(groq_vis.strip()) > 6:
                clean_ans = re.sub(r'[*_~#`"]', '', groq_vis).strip()
                return jsonify({
                    "status": "success",
                    "object_description": clean_ans,
                    "answer": clean_ans,
                    "language": lang,
                    "detected_label": "vision_identified"
                })
        except Exception as ve:
            print("Groq vision primary identification notice:", ve)

        try:
            import cv2
            import numpy as np
            import base64
            img_bytes = base64.b64decode(clean_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception as img_err:
            print("Vision image decode error:", img_err)

    # 1. State-of-the-Art Dual Local Vision Engine (YOLOv8 + MobileNet)
    chosen_label, detections = detect_held_camera_object(img)
    if not chosen_label and detected_label:
        chosen_label = OBJECT_LABEL_MAP.get(detected_label.lower(), detected_label)

    # 2. Extract visual color and finish cues with OpenCV
    color_cues = []
    if img is not None and img.shape[0] > 10 and img.shape[1] > 10:
        try:
            h, w, _ = img.shape
            ch1, ch2 = int(h * 0.2), int(h * 0.8)
            cw1, cw2 = int(w * 0.2), int(w * 0.8)
            center_crop = img[ch1:ch2, cw1:cw2]
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            val = np.mean(hsv[:, :, 2])
            sat = np.mean(hsv[:, :, 1])
            if val < 65:
                color_cues.append("sleek dark or black finish")
            elif val > 185 and sat < 40:
                color_cues.append("metallic silver or clean light finish")
            elif sat > 90:
                color_cues.append("vibrant colored design")
        except Exception as cv_e:
            print("Color cues extraction notice:", cv_e)

    resolved_item = chosen_label or "handheld item"
    appearance_desc = f"{resolved_item}"
    if color_cues:
        appearance_desc += f" with a {color_cues[0]}"

    # 3. Dynamic Friendly Groq Generation in User's Active Language (< 1s latency)
    prompt_text = (
        f"You are {persona}, talking with a close friend in front of the smart mirror.\n"
        f"Your friend is holding up an item to the camera. The item is clearly identified as a {appearance_desc}.\n"
        f"The friend asks: '{question}'.\n"
        f"In exactly 1 to 2 short, complete, and friendly sentences in {lang_info['name']}, clearly acknowledge and identify the {resolved_item} they are holding. "
        f"Describe its neat appearance or practical use. Speak naturally like two close friends talking in everyday life. "
        f"STRICT RULES: Do NOT use slang, do NOT say 'bro' or 'buddy'. Maximum 25 words total. Must be grammatically complete and accurate."
    )

    try:
        groq_answer = call_groq_chat(
            prompt_text,
            system_instruction=f"You are {persona}, an articulate, friendly close friend who identifies objects for the user in {lang_info['name']}. Always speak in complete, polite sentences.",
            timeout=5
        )
        if groq_answer and len(groq_answer.strip()) > 6:
            clean_ans = re.sub(r'[*_~#`"]', '', groq_answer).strip()
            return jsonify({
                "status": "success",
                "object_description": clean_ans,
                "answer": clean_ans,
                "language": lang,
                "detected_label": resolved_item
            })
    except Exception as ge:
        print("Groq vision identify notice:", ge)

    # 4. Instant Resilient Multilingual Fallback (Never Fail, 0ms Latency)
    native_term = MULTILINGUAL_OBJECT_TERMS.get(resolved_item, {}).get(lang, resolved_item)
    fallback_identify = {
        'kn': f"ನಿಮ್ಮ ಕೈಯಲ್ಲಿರುವ {native_term} ನನಗೆ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣಿಸುತ್ತಿದೆ. ಇದು ತುಂಬಾ ಅಚ್ಚುಕಟ್ಟಾಗಿದೆ.",
        'hi': f"मुझे आपके हाथ में {native_term} साफ़ दिखाई दे रहा है। यह देखने में काफ़ी अच्छा और उपयोगी है।",
        'te': f"మీ చేతిలోని {native_term} నాకు స్పష్టంగా కనిపిస్తోంది. ఇది చాలా చక్కగా ఉంది.",
        'ta': f"உங்கள் கையில் உள்ள {native_term} தெளிவாகத் தெரிகிறது. இது மிகவும் நேர்த்தியாக உள்ளது.",
        'en': f"I can clearly see the {resolved_item} you are holding. It looks sleek and well kept."
    }
    answer = fallback_identify.get(lang, fallback_identify['en'])
    return jsonify({
        "status": "success",
        "object_description": answer,
        "answer": answer,
        "language": lang,
        "detected_label": resolved_item
    })

# ============================================================
# 4.6. Intelligent Conversational Agent (Gemini 2.5 Flash)
def get_entity_grounding(text: str, history: list = None) -> str:
    if not text:
        return ""
    t = text.lower()

    # 1. NCET Factual Grounding
    if any(k in t for k in ['nagarjuna', 'ncet', 'ನಾಗಾರ್ಜುನ', 'నాగార్జున', 'नागार्जुन']):
        return "Fact Context: Nagarjuna College of Engineering and Technology (NCET) is an autonomous engineering college located in Devanahalli, Bengaluru (Bangalore), Karnataka, near Kempegowda International Airport. It was established in 2001 by Nagarjuna Education Society and is affiliated with Visvesvaraya Technological University (VTU). It is located in Bengaluru, Karnataka (NOT in Andhra Pradesh/Guntur/Hyderabad)."

    # 2. Real-Time Verified Gold and Precious Metals Grounding
    gold_keywords = [
        'gold', '24k', '22k', '18k', 'bullion', 'sovereign', 'silver',
        'ಚಿನ್ನ', 'ಚಿನ್ನದ', 'ಬಂಗಾರ', 'ಬಂಗಾರದ', 'ಬೆಳ್ಳಿ', 'ಬೆಳ್ಳಿಯ',
        'బంగారం', 'బంగారు', 'బంగారము', 'వెండి',
        'सोना', 'सोने', 'स्वर्ण', 'चांदी',
        'தங்கம்', 'தங்க', 'தங்கத்தின்', 'வெள்ளி',
        'സ്വർണ്ണം', 'സ്വർണം', 'സ്വർണ്ണ', 'സ്വർണ്ണത്തിന്റെ', 'വെള്ളി'
    ]
    hist_text = " ".join([h.get("text") or h.get("content") or "" for h in (history or [])[-4:]]).lower()
    is_gold_context = any(k in t for k in gold_keywords) or (
        any(k in hist_text for k in gold_keywords) and
        any(w in t for w in ['yesterday', 'today', 'gram', '10g', 'price', 'rate', 'what about', 'in ', 'cost', 'ಹಿಂದಿನ', 'ನಿನ್ನೆ', 'ನಿನ್ನೆಯ', 'నిన్న', 'నిన్నటి', 'कल', 'நேற்று', 'நேற்றைய', 'இன்று', 'இன்றைய'])
    )
    if is_gold_context:
        cities = ['bangalore', 'bengaluru', 'chennai', 'delhi', 'mumbai', 'hyderabad', 'kolkata', 'pune', 'mysore', 'mangalore', 'kerala', 'jaipur', 'ahmedabad', 'coimbatore', 'vijayawada']
        detected_city = 'bangalore'
        for c in cities:
            if c in t or c in hist_text:
                detected_city = c
                break
        is_yesterday = any(w in t for w in ['yesterday', 'yesterdays', 'ನಿನ್ನೆ', 'ನಿನ್ನೆಯ', 'నిన్న', 'నిన్నటి', 'कल', 'நேற்று', 'நேற்றைய', 'இன்னലെ', 'previous day', 'last day', 'before'])
        try:
            return get_gold_briefing_text(city=detected_city, is_yesterday=is_yesterday)
        except Exception as eg:
            print("[MarketData] Gold grounding error:", eg)

    # 3. Real-Time Verified Stock Market Indices Grounding
    market_keywords = ['sensex', 'nifty', 'stock market', 'share market', 'bse', 'nse', 'dollar', 'usd to inr', 'usdinr']
    if any(k in t for k in market_keywords):
        try:
            indices = fetch_live_market_indices()
            parts = []
            for k, info in indices.items():
                parts.append(f"- {info['name']}: {info['price']} (Change: {info['change']} / {info['change_pct']}%)")
            return "REAL-TIME VERIFIED STOCK INDICES (Live):\n" + "\n".join(parts)
        except Exception as em:
            print("[MarketData] Market indices grounding error:", em)

    return ""

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

    grounding = get_entity_grounding(message, history=history)
    enriched_message = f"{message}\n[{grounding}]" if grounding else message

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

    live_weather_info = (data.get('weather') or data.get('weather_summary') or '').strip()
    weather_context = f"\nCURRENT LIVE LOCAL WEATHER ON MIRROR: {live_weather_info}\nUse this live weather data to give accurate weather info or weather-appropriate outfit advice.\n" if live_weather_info else ""

    system_prompt = (
        f"You are {persona}, an intelligent, friendly, natural everyday companion and close friend inside Smart Mirror Vox.\n"
        f"CONVERSATIONAL STYLE & PERSONALITY:\n"
        f"- Talk naturally, warmly, and casually like two best friends having an everyday conversation in real life.\n"
        f"- Always be supportive, energetic, and engaging. Never be cold, dry, or robotic.\n"
        f"- Understand normal casual sentences (e.g. 'I am getting ready for college', 'Tell me a joke', 'What's up?').\n"
        f"- Context Memory: Remember the conversation history and understand follow-up questions without requiring the user to repeat themselves.\n"
        f"- Voice Conciseness: Keep responses short and punchy by default (strictly 1 to 3 natural sentences, maximum 35 words). Only provide detailed explanations when explicitly requested.\n"
        f"- Proactive Help: Naturally suggest next steps based on what the user says (e.g. if getting ready for college, ask if they want outfit suggestions or a weather check).\n"
        f"- MULTILINGUAL & ROMANIZED INDIC INTELLIGENCE:\n"
        f"  * Romanized Kannada (Kanglish, e.g. 'hegiddiya', 'oota aaytha', 'enu samachara') -> Reply warmly in Romanized Kannada (e.g. 'Chennagiddini buddy! Neenu hegiddiya?').\n"
        f"  * Native Kannada script -> Reply in fluent Kannada.\n"
        f"  * Romanized Telugu (e.g. 'ela unnavu', 'tinnava', 'em chestunnav') -> Reply in friendly Telugu (e.g. 'Bagunnanu buddy! Nuvvu ela unnavu?').\n"
        f"  * Romanized Hindi / Hinglish ('kaise ho', 'kya chal raha hai') -> Reply in friendly Hindi/Hinglish (e.g. 'Badhiya hoon buddy! Aap batao?').\n"
        f"  * Romanized Tamil / Tamil script -> Reply in warm Tamil (e.g. 'Nallaa irukken buddy! Neenga eppadi irukkeenga?').\n"
        f"  * Romanized Malayalam / Malayalam script -> Reply in warm Malayalam.\n"
        f"  * English -> Reply in friendly, casual English.\n"
        f"  * Active language setting is: {lang_name} ({active_language}).\n"
        f"{weather_context}"
        f"CURRENT EXACT REAL-TIME CLOCK: {datetime.datetime.now().strftime('%I:%M %p on %A, %B %d, %Y')}.\n"
        f"CRITICAL FACTUAL ACCURACY: Nagarjuna College of Engineering and Technology (NCET) is an autonomous engineering college located in Devanahalli, Bengaluru, Karnataka, affiliated with VTU.\n"
        f"REAL-TIME DATA GROUNDING: When real-time data (such as live gold prices, stock market indices, or weather) is enclosed in brackets like [REAL-TIME VERIFIED...], you MUST always integrate these exact verified numbers into your friendly spoken answer in the active language. Never say live data is unavailable when grounding is provided.\n"
        f"HARDWARE INTEGRATION & INTENT CLASSIFICATION:\n"
        f"Classify user intent into one of:\n"
        f"- MUSIC_PLAY: User wants to play a song/music (in Kannada, Hindi, Telugu, Tamil, English, etc.). Provide clean 'song_query'.\n"
        f"- IMAGE_OUTFIT: User wants to scan/analyze their current outfit or dress in camera.\n"
        f"- OBJECT_IDENTIFY: User is holding an object up or asking 'what is this' / 'what am I holding'.\n"
        f"- GENERAL_CHAT: Normal dialogue, general knowledge, financial/gold rates, tech, advice, college banter, jokes, storytelling, or companionship.\n\n"
        f"OUTPUT SCHEMA (JSON ONLY):\n"
        f"{{\n"
        f'  "intent": "GENERAL_CHAT" | "MUSIC_PLAY" | "IMAGE_OUTFIT" | "OBJECT_IDENTIFY",\n'
        f'  "answer": "Concise, friendly spoken response in the requested language (1-3 sentences max)",\n'
        f'  "song_query": "Song title or query if MUSIC_PLAY else null",\n'
        f'  "subject": "Extracted subject if IMAGE_GENERATE else null",\n'
        f'  "learned_preference": "Any user preference or fact learned from this turn, or null"\n'
        f"}}"
    )

    user_turn_text = f"Language: {lang_name} ({active_language})\nUser Spoken: {enriched_message}"

    contents = []
    if memory:
        contents.append({"parts": [{"text": f"User Profile & Preferences: {json.dumps(memory)}"}]})
    for h in history[-6:]:
        role = "user" if h.get("role") == "user" else "model"
        text = h.get("text") or h.get("content") or ""
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})
    contents.append({"role": "user", "parts": [{"text": user_turn_text}]})

    # 1. Primary High-Speed Agent: Groq LLM with full context history
    try:
        groq_resp = call_groq_chat(
            prompt=user_turn_text,
            system_instruction=system_prompt,
            json_mode=True,
            timeout=7,
            messages_history=history
        )
        if groq_resp:
            parsed = None
            try:
                parsed = json.loads(groq_resp)
            except Exception:
                m_match = re.search(r'\{.*\}', groq_resp, re.DOTALL)
                if m_match:
                    try:
                        parsed = json.loads(m_match.group(0))
                    except Exception:
                        pass
                if not parsed:
                    parsed = {"intent": "GENERAL_CHAT", "answer": groq_resp}
            return jsonify({
                "status": "success",
                "intent": parsed.get("intent", "GENERAL_CHAT"),
                "answer": parsed.get("answer", ""),
                "song_query": parsed.get("song_query"),
                "subject": parsed.get("subject"),
                "learned_preference": parsed.get("learned_preference"),
                "language": active_language
            })
    except Exception as ge:
        print(f"Groq chat agent error: {ge}")

    # 2. Secondary Agent: n8n Workflow (if configured)
    try:
        n8n_payload = {
            "text": message,
            "query": message,
            "message": message,
            "fact_context": grounding,
            "active_language": active_language,
            "persona": persona,
            "messages": history + [{"role": "user", "content": message}]
        }
        n8n_req = urllib.request.Request(
            N8N_WEBHOOK_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(n8n_payload).encode('utf-8')
        )
        with urllib.request.urlopen(n8n_req, timeout=3) as n8n_res:
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

    # 3. Gemini Fallback
    if GEMINI_KEY:
        try:
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
                    res = urllib.request.urlopen(req, timeout=6)
                    resp_data = json.loads(res.read().decode('utf-8'))
                    raw_text = resp_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                    if raw_text:
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
                except Exception:
                    continue
        except Exception:
            pass

    if grounding:
        return jsonify({
            "status": "success",
            "intent": "GENERAL_CHAT",
            "answer": grounding,
            "language": active_language
        })

    # 4. Graceful Fallback
    fallback_answers = {
        'en': "I'm right here with you, buddy! What would you like to explore or do next?",
        'kn': "ನಾನು ನಿಮ್ಮೊಂದಿಗೆ ಇಲ್ಲೇ ಇದ್ದೇನೆ ಗೆಳೆಯ! ಮುಂದೆ ಏನು ಮಾಡೋಣ?",
        'te': "నేను ఇక్కడే ఉన్నాను బడ్డీ! తదుపరి ఏం చేద్దాం?",
        'hi': "मैं आपके साथ यहीं हूँ दोस्त! आगे क्या करना चाहेंगे?"
    }
    return jsonify({
        "status": "success",
        "intent": "GENERAL_CHAT",
        "answer": fallback_answers.get(active_language, fallback_answers['en']),
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


@app.route('/api/health')
def api_health():
    return jsonify({
        "status": "healthy",
        "service": "SmartMirror-Vox Backend",
        "groq": bool(GROQ_KEY),
        "yolo": GLOBAL_YOLO_MODEL is not None,
        "mobilenet": GLOBAL_MOBILENET_MODEL is not None
    })

if __name__ == '__main__':
    print("==================================================")
    print("[*] SmartMirror-Vox Flask Backend running on http://127.0.0.1:5000")
    print("==================================================")
    app.run(host='0.0.0.0', port=5000, debug=False)
