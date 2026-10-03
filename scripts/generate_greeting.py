import asyncio
import os
import json
import base64
import websockets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

async def generate():
    api_key = os.getenv("SARVAM_API_KEY")
    uri = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true"
    greeting = "Namaste, main Rudra Infotek se baat kar rahi hoon. Main aapki kya madad kar sakti hoon?"
    
    mp3_file = open("greeting.mp3", "wb")
    
    async with websockets.connect(uri, additional_headers={"api-subscription-key": api_key}) as sarvam_ws:
        config_msg = {
            "type": "config",
            "data": {
                "language_code": "hi-IN",
                "speaker": "ritu",
                "model": "bulbul:v3",
                "speech_sample_rate": 16000
            }
        }
        await sarvam_ws.send(json.dumps(config_msg))
        await sarvam_ws.send(json.dumps({"type": "text", "data": {"text": greeting}}))
        await sarvam_ws.send(json.dumps({"type": "flush"}))
        
        async for msg_str in sarvam_ws:
            msg = json.loads(msg_str)
            if msg.get("type") == "audio":
                chunk = base64.b64decode(msg["data"]["audio"])
                mp3_file.write(chunk)
            elif msg.get("type") == "event" and msg.get("data", {}).get("event_type") == "final":
                break

    mp3_file.close()
    
    # Convert mp3 to pcm
    os.system("ffmpeg -y -i greeting.mp3 -f s16le -ar 16000 -ac 1 greeting.pcm")
    print("Generated greeting.pcm successfully.")

asyncio.run(generate())
