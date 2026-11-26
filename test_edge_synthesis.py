import asyncio
import edge_tts

async def test_audio():
    tests = [
        ('kn', 'kn-IN-GaganNeural', 'ಇವತ್ತು ಹವಾಮಾನ ಹೇಗಿದೆ?'),
        ('kn', 'kn-IN-SapnaNeural', 'ಇವತ್ತು ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ.'),
        ('ta', 'ta-IN-ValluvarNeural', 'இன்று வானிலை எப்படி இருக்கிறது?'),
        ('ta', 'ta-IN-PallaviNeural', 'இன்று வானிலை மிகவும் அருமையாக உள்ளது.'),
        ('hi', 'hi-IN-MadhurNeural', 'आज मौसम कैसा है?'),
        ('hi', 'hi-IN-SwaraNeural', 'आज मौसम बहुत अच्छा है।'),
        ('en', 'en-GB-RyanNeural', 'How is the weather today?'),
        ('en', 'en-IN-NeerjaNeural', 'The weather is great today!')
    ]
    for lang, voice, text in tests:
        communicate = edge_tts.Communicate(text, voice)
        data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                data.extend(chunk["data"])
        print(f"✅ {lang} ({voice}): generated {len(data)} bytes of audio!")

if __name__ == '__main__':
    asyncio.run(test_audio())
