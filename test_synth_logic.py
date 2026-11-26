import re
from gtts import gTTS
import io

def synthesize(text, lang='en', persona='Jarvis'):
    clean_text = re.sub(r'[*_~#`]', '', text)
    clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    kn_chars = len(re.findall(r'[\u0C80-\u0CFF]', clean_text))
    ta_chars = len(re.findall(r'[\u0B80-\u0BFF]', clean_text))
    te_chars = len(re.findall(r'[\u0C00-\u0C7F]', clean_text))
    hi_chars = len(re.findall(r'[\u0900-\u097F]', clean_text))
    ml_chars = len(re.findall(r'[\u0D00-\u0D7F]', clean_text))
    bn_chars = len(re.findall(r'[\u0980-\u09FF]', clean_text))

    script_counts = [('kn', kn_chars), ('ta', ta_chars), ('te', te_chars), ('hi', hi_chars), ('ml', ml_chars), ('bn', bn_chars)]
    script_counts.sort(key=lambda x: x[1], reverse=True)

    if script_counts[0][1] >= 2:
        target_tl = script_counts[0][0]
    else:
        target_tl = lang

    is_lisa = (persona.lower() == 'lisa')
    tld = 'com' if is_lisa else 'co.uk' if target_tl == 'en' else None

    if tld:
        tts = gTTS(text=clean_text[:450], lang=target_tl, tld=tld)
    else:
        tts = gTTS(text=clean_text[:450], lang=target_tl)

    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return len(buf.getvalue()), target_tl

tests = [
    ('ಇವತ್ತು ಹವಾಮಾನ ಹೇಗಿದೆ?', 'en', 'Jarvis'),
    ('ಇವತ್ತು ಹವಾಮಾನ ಹೇಗಿದೆ?', 'en', 'Lisa'),
    ('இன்று வானிலை எப்படி இருக்கிறது?', 'en', 'Jarvis'),
    ('இன்று வானிலை எப்படி இருக்கிறது?', 'en', 'Lisa'),
    ('आज मौसम कैसा है?', 'en', 'Jarvis'),
    ('आज मौसम कैसा है?', 'en', 'Lisa'),
    ('How is the weather today?', 'en', 'Jarvis'),
    ('How is the weather today?', 'en', 'Lisa')
]

for t, l, p in tests:
    sz, detected = synthesize(t, l, p)
    print(f'[{p}] "{t}" -> Detected Lang: {detected}, Audio size: {sz} bytes')
