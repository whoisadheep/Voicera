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
    
    raw_data = bytearray()
    
    async with websockets.connect(uri, additional_headers={"api-subscription-key": api_key}) as sarvam_ws:
        config_msg = {
            "type": "config",
            "data": {
                "language_code": "hi-IN",
                "speaker": "kavya",
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
                raw_data.extend(chunk)
            elif msg.get("type") == "event" and msg.get("data", {}).get("event_type") == "final":
                break

    # Save exactly what Sarvam gave us
    with open("sarvam_output.bin", "wb") as f:
        f.write(raw_data)
    print("Saved sarvam_output.bin")

asyncio.run(generate())
