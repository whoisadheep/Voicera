import os, asyncio, json, base64
import websockets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

async def generate_greeting():
    uri = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true"
    api_key = os.getenv("SARVAM_API_KEY")
    audio_data = b""
    
    try:
        async with websockets.connect(uri, additional_headers={"api-subscription-key": api_key}, ping_interval=None) as sarvam_ws:
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
            
            text_msg = {
                "type": "text",
                "data": {
                    "text": "Hello, Rudra Infotek. Kahiye, kya kaam tha?"
                }
            }
            await sarvam_ws.send(json.dumps(text_msg))
            
            eos_msg = {
                "type": "flush"
            }
            await sarvam_ws.send(json.dumps(eos_msg))
            
            while True:
                try:
                    msg_str = await sarvam_ws.recv()
                    msg = json.loads(msg_str)
                    if msg.get("type") == "audio":
                        chunk = base64.b64decode(msg["data"]["audio"])
                        audio_data += chunk
                    elif msg.get("type") == "event" and msg.get("data", {}).get("event_type") == "final":
                        break
                except websockets.exceptions.ConnectionClosed:
                    break
    finally:
        if audio_data:
            import subprocess
            out_16k = os.path.join(ASSETS_DIR, "greeting.pcm")
            out_8k = os.path.join(ASSETS_DIR, "greeting_8k.pcm")
            
            # Generate 16kHz PCM
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "mp3", "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1", out_16k],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            proc.communicate(input=audio_data)
            print(f"Generated {out_16k} (16kHz s16le PCM, {os.path.getsize(out_16k)} bytes)")
            
            # Generate 8kHz PCM
            subprocess.run(
                ["ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", out_16k, "-f", "s16le", "-ar", "8000", "-ac", "1", out_8k],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Generated {out_8k} (8kHz s16le PCM, {os.path.getsize(out_8k)} bytes)")
        else:
            print("Failed to generate audio data!")

asyncio.run(generate_greeting())
