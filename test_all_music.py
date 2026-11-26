import urllib.request
import json

tests = [
    ("Kannada generic", "ಒಂದು ಹಾಡು ಪ್ಲೇ ಮಾಡು", "kn"),
    ("Kannada specific", "KGF ಹಾಡು ಹಾಕು", "kn"),
    ("Telugu generic", "ఒక పాట ప్లే చేయి", "te"),
    ("Telugu specific", "RRR పాట పెట్టు", "te"),
    ("Hindi generic", "कोई गाना बजाओ", "hi"),
    ("Hindi specific", "Arijit Singh का गाना बजाओ", "hi"),
    ("English specific", "play Believer by Imagine Dragons", "en"),
]

for label, text, lang in tests:
    req = urllib.request.Request(
        "http://localhost:5678/webhook/mirror-agent",
        data=json.dumps({"text": text, "active_language": lang, "persona": "Jarvis"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        d = json.loads(res.read().decode("utf-8"))
        intent = d.get("intent")
        sq = d.get("song_query")
        print(f"{label}: intent={intent} | song_query={ascii(sq)}")
