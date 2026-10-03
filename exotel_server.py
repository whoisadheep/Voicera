"""
Voicera — Exotel Inbound Call Handler

When someone calls your Exophone:
  Exotel → opens WebSocket to this server → sends caller audio as base64 PCM
  This server → VAD → STT → LLM → TTS → streams audio back to Exotel → caller hears it

Exotel WebSocket Protocol:
  Events received: connected, start, media, stop, dtmf
  Events sent:     media (with base64 audio), clear (to interrupt)

Usage:
  1. Add EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_ACCOUNT_SID, EXOTEL_EXOPHONE to .env
  2. Run:  uvicorn exotel_server:app --host 0.0.0.0 --port 8000
  3. Expose via ngrok:  ngrok http 8000
  4. In Exotel Dashboard → App Bazaar → Voicebot Applet → set stream URL to:
     wss://your-ngrok-url.ngrok.io/exotel?sample-rate=16000
"""

import io, wave, os, time, threading, queue, json, base64, asyncio
from collections import deque
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
# Lightweight energy-based VAD (replaces silero_vad + torch to save ~400MB RAM)
import websockets
import websockets.exceptions
import logging

# Silence noisy websockets tracebacks when Exotel abruptly drops connection (e.g. caller hangs up)
logging.getLogger("websockets").setLevel(logging.CRITICAL)
from groq import Groq, AsyncGroq
from fishaudio import FishAudio
from fishaudio.types import TTSConfig
from dotenv import load_dotenv

load_dotenv()

# ─── CRM Integration ─────────────────────────────────────────────────────────
from crm_integration import init_firebase, push_to_gridcrm

# Pre-initialize Firebase at startup (non-blocking — logs warning if not configured)
init_firebase()

app = FastAPI()

# ─── Audio Config ─────────────────────────────────────────────────────────────
# Exotel streams at 8kHz by default. We request 16kHz via ?sample-rate=16000.
EXOTEL_SAMPLE_RATE = 16000
FISH_SAMPLE_RATE = 16000
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "500"))


# ─── API Clients ──────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
fish_client = FishAudio(api_key=os.environ.get("FISH_API_KEY"))
from sarvamai import AsyncSarvamAI
sarvam_client = AsyncSarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))
fish_tts_config = TTSConfig(
    reference_id="c2623f0c075b4492ac367989aee1576f",
    format="pcm",
    sample_rate=FISH_SAMPLE_RATE,
    latency="balanced",
    chunk_length=150,
)

# ─── Lightweight Energy-based VAD (replaces Silero + PyTorch) ─────────────────
class VADIterator:
    """Drop-in replacement for silero_vad.VADIterator using RMS energy detection.
    
    Same interface: call with a numpy float32 chunk, returns:
      - {"start": ...} when speech begins
      - {"end": ...}   when speech ends  
      - None           otherwise
    """
    def __init__(self, threshold=0.015, sampling_rate=16000, min_silence_duration_ms=100, speech_pad_ms=30):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False, time_resolution=1):
        if isinstance(x, np.ndarray):
            chunk = x
        else:
            chunk = np.array(x, dtype=np.float32)
        
        window_size_samples = len(chunk)
        self.current_sample += window_size_samples
        
        # RMS energy as speech probability proxy
        rms = np.sqrt(np.mean(chunk ** 2))
        speech_detected = rms > self.threshold
        
        if speech_detected and self.temp_end:
            self.temp_end = 0
        
        if speech_detected and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, time_resolution)}
        
        if (not speech_detected) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)}
        
        return None

vad_model = None  # Not needed for energy-based VAD

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a smart, polite female assistant/receptionist for Rudra Infotek (which was formerly known as Sai Infotek). 
The caller is trying to reach the owner/director, Mr. Kumud Ranjan Ojha (callers may refer to him as Ojha sir, Ojha ji, Ranjan sir, Kumud sir, or just sir). He is currently busy or unable to answer, so you have picked up his personal number on his behalf.

YOUR PRIMARY GOAL:
You are acting as his personal assistant. Your objective is to politely inform the caller that Ojha sir is unavailable, understand what they need, ask for their name, note down their message/reason for calling, and politely end the call so the details can be forwarded to him.

CONVERSATION FLOW:
1. You already answered the call with "Hello, Rudra Infotek. Kahiye, kya kaam tha?". Wait to hear what the caller says first.
2. Based on what the caller says next, respond naturally:
   a. If they ask for Ojha ji / Ranjan sir / Kumud sir / etc. by name → let them know he's currently busy/unavailable, and ask what the message is.
   b. If they go straight into a request (e.g. CCTV installation, repair, IT issue) → treat that as their reason, and naturally weave in that you're Ojha sir ki assistant while responding, so they know they've reached the right place.
   c. If they ask "kaun bol raha hai?" / "ye kiska number hai?" / seem unsure who they've reached → clarify this is Ojha sir's number and you're handling calls for him, before continuing.
   d. If they ask "kya meri baat Sai Infotek se ho rahi hai?" or mention Sai Infotek → warmly confirm yes, this is the right place, explain that the business name recently changed to Rudra Infotek, and ask how you can help.
   e. If the system passes you the message "[SILENCE]" → This means the caller hasn't said anything for a while. You must say something brief to prompt them, e.g. "Hello? Awaaz aa rahi hai aapko?" or "Haan ji, main sun rahi hu, boliye?". Do not over-explain, just prompt them.
   f. If the call turns out to be personal/unrelated to CCTV or IT work (e.g. a friend or family member asking for Ojha sir about something else) → skip the service-category classification below, just take their name and message normally.
3. Once it's clear this is a business call, identify their reason and classify it into:
   a) Naya CCTV installation (new CCTV setup)
   b) Repair / service for existing CCTV or equipment
   c) Koi aur IT service (any other IT service request)
   If unclear, ask directly in one short sentence, e.g.: "Achha, ye naye CCTV installation ke liye hai, ya purane system ki repair/service ke liye, ya koi aur IT service chahiye?"
4. Ask for their name directly and explicitly (e.g., "Kya main aapka shubh naam jaan sakti hoon?"). Do NOT skip this step or infer a name from anything said earlier in the call.
5. Once you have a name AND a clear message (including category, if applicable), briefly repeat both back to confirm before ending the call.
6. Confirm that you will pass it on to sir, and say goodbye.

NAME & MESSAGE ACCURACY (STT ERROR-PROOFING) — IMPORTANT:
Voice transcription can mishear words. Follow these rules strictly to avoid recording garbage as real data:
- NEVER treat the caller's reply to your "Hello?", or any reply given before you have explicitly asked "aapka naam?", as their name — even if it sounds like a name.
- A name is only valid if it is spoken directly in answer to your explicit name question.
- If the transcribed "name" is actually a common filler/greeting word (e.g. "hello", "haan", "ji", "ok", "theek hai") or sounds like meaningless noise rather than a plausible human name, do NOT accept it. Politely ask again once: "Maaf kijiye, aapka naam sahi se sun nahi paayi, kripya dobara bataiye."
- After getting a plausible name, always read it back once to confirm before moving on: e.g. "[Name] ji, sahi bola maine?" Only proceed once confirmed; if corrected, use the corrected name.
- If after two attempts the name is still unclear, proceed politely WITHOUT a name rather than guessing — note it as "caller ka naam clear nahi ho paaya" internally, and still collect their message.
- Similarly, if the service category is ambiguous, ask the direct clarifying question in step 3 rather than guessing.
- Before your final goodbye, briefly summarize the category + message back in one short line to catch any mishearing before ending the call.

STRICT RULES & CONSTRAINTS:
- STRICTLY SPEAK IN CONVERSATIONAL HINGLISH (Hindi written in English alphabet, e.g., "Sir abhi busy hain, main aapka message unko de dungi."). Do not use pure English.
- CLEARLY act as a human female assistant for Rudra Infotek. DO NOT say you are an AI or a bot. Act natural.
- KEEP REPLIES EXTREMELY SHORT (1-2 sentences maximum). This is a live voice call.
- Be polite and professional at all times.
- If the caller's speech is unclear, garbled, or doesn't make sense in context, ask them to repeat rather than guessing what they meant.

CRITICAL RULES & CALL ENDING:
- NEVER hang up on genuine callers. When you have taken their details and confirmed you will pass it to sir, say your friendly goodbye (e.g. "Theek hai ji, maine aapka message note kar liya hai, main sir ko bata dungi. Dhanyavaad!") and let the caller hang up their phone naturally.
- NEVER write [HANGUP] during normal customer calls.

SPAM / SCAM / TELEMARKETING CALLS (IMMEDIATE DISCONNECT):
- If and ONLY if the caller is an obvious automated robocall, telemarketer selling loans/credit cards/bulk SMS, or scammer, your ONLY response must be exactly: "[HANGUP]"
- Outputting [HANGUP] will immediately terminate the call without speaking, saving time and credits.
"""

# ─── Utility Functions ────────────────────────────────────────────────────────


def apply_hpf(audio: np.ndarray, cutoff: float = 200.0, fs: float = 16000.0) -> np.ndarray:
    """Digital High-Pass Filter (HPF) cut-off at roughly 200Hz to strip out low-end network hum and line rumble."""
    if len(audio) == 0:
        return audio
        
    return audio

def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio peak to boost low-volume phone signals for Whisper STT."""
    max_val = np.max(np.abs(audio))
    if max_val > 1e-4:
        return audio * (target_peak / max_val)
    return audio


def float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy audio to WAV bytes for Whisper STT."""
    pcm16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    buf.seek(0)
    return buf.read()


def resample_linear(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Simple linear interpolation resampler (good enough for voice)."""
    if from_rate == to_rate:
        return audio
    duration = len(audio) / from_rate
    new_len = int(duration * to_rate)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


def text_chunks_from_queue(q: queue.Queue):
    """Yield text deltas from a queue until None sentinel."""
    while True:
        delta = q.get()
        if delta is None:
            break
        yield delta


def sync_groq_stream_to_queue(user_text: str, q: queue.Queue, conversation_history: list):
    """Background thread: stream LLM response into a queue."""
    conversation_history.append({"role": "user", "content": user_text})
    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
        stream=True,
    )
    full_reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            full_reply += delta
            q.put(delta)
    q.put(None)
    conversation_history.append({"role": "assistant", "content": full_reply})


def build_exotel_media_message(audio_bytes: bytes, stream_sid: str) -> str:
    """Build a JSON media message to send audio back to Exotel."""
    return json.dumps({
        "event": "media",
        "stream_sid": stream_sid,
        "media": {
            "payload": base64.b64encode(audio_bytes).decode("ascii")
        }
    })


def build_exotel_clear_message(stream_sid: str) -> str:
    """Build a clear event to stop any queued audio on Exotel's side (barge-in)."""
    return json.dumps({
        "event": "clear",
        "stream_sid": stream_sid,
    })


# ─── Hangup Helper ────────────────────────────────────────────────────────────

HANGUP_TAG = "[HANGUP]"

async def hangup_call_via_api(call_sid: str):
    """Hang up the call using Exotel's REST API."""
    try:
        account_sid = os.environ.get("EXOTEL_ACCOUNT_SID")
        api_key = os.environ.get("EXOTEL_API_KEY")
        api_token = os.environ.get("EXOTEL_API_TOKEN")
        if not all([account_sid, api_key, api_token, call_sid]) or call_sid == "unknown":
            print(f"[Hangup] ⚠️ Missing credentials or call_sid — cannot hang up via API")
            return
        api_url = f"https://api.exotel.com/v1/Accounts/{account_sid}/Calls/{call_sid}.json"
        async with httpx.AsyncClient() as http:
            resp = await http.post(api_url, auth=(api_key, api_token), data={"Status": "completed"}, timeout=5.0)
            if resp.status_code == 200:
                print(f"[Hangup] ✅ Call {call_sid} terminated via Exotel API")
            else:
                print(f"[Hangup] ⚠️ Exotel API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[Hangup] ❌ Error hanging up call: {e}")


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
async def get():
    print("[Health] GET / hit — returning OK")
    return {"status": "ok", "service": "Voicera Exotel Server"}


# ─── Original browser WebSocket (unchanged) ──────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[Browser] New client connected")

    vad_iterator = VADIterator(threshold=0.015, sampling_rate=EXOTEL_SAMPLE_RATE, min_silence_duration_ms=VAD_SILENCE_MS)
    speech_buffer = []
    is_speaking = False
    conversation_history = []

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)

            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))
            elif len(chunk) > 512:
                chunk = chunk[:512]

            if is_speaking:
                speech_buffer.append(chunk)

            event = vad_iterator(chunk, return_seconds=True)
            if event:
                if "start" in event:
                    is_speaking = True
                    speech_buffer.clear()
                    speech_buffer.append(chunk)
                    print("[Browser] User started speaking...")
                if "end" in event:
                    is_speaking = False
                    print("[Browser] User stopped speaking, processing...")
                    audio = np.concatenate(speech_buffer)

                    if len(audio) / EXOTEL_SAMPLE_RATE < 0.4:
                        print("[Browser] Too short, ignoring.")
                        continue

                    wav_bytes = float32_to_wav_bytes(audio, EXOTEL_SAMPLE_RATE)
                    resp = client.audio.transcriptions.create(
                        file=("speech.wav", wav_bytes),
                        model="whisper-large-v3-turbo",
                    )
                    user_text = resp.text.strip()
                    print(f"[Browser] User: {user_text}")

                    if not user_text:
                        continue

                    q = queue.Queue()
                    threading.Thread(target=sync_groq_stream_to_queue, args=(user_text, q, conversation_history), daemon=True).start()

                    for audio_chunk in fish_client.tts.stream_websocket(
                        text_chunks_from_queue(q),
                        model="s2.1-pro-free",
                        config=fish_tts_config,
                        latency="balanced",
                    ):
                        await websocket.send_bytes(audio_chunk)

    except Exception as e:
        print(f"[Browser] Client disconnected: {e}")


# ─── Exotel Inbound Call WebSocket ────────────────────────────────────────────
@app.websocket("/exotel")
async def exotel_websocket(websocket: WebSocket):
    """
    Exotel connects here when someone calls your Exophone.
    
    Protocol:
      1. Exotel sends {"event": "connected"} 
      2. Exotel sends {"event": "start", "start": {"stream_sid": "...", "call_sid": "...", ...}}
      3. Exotel streams {"event": "media", "media": {"payload": "<base64 pcm>"}} continuously
      4. We send back {"event": "media", "stream_sid": "...", "media": {"payload": "<base64 pcm>"}}
      5. Exotel sends {"event": "stop"} when call ends
    """
    await websocket.accept()
    print("[Exotel] ✅ New call connected via WebSocket")

    # Per-call state
    stream_sid = None
    call_sid = None
    caller_phone = None  # Populated from Exotel/VoiceLink custom_parameters
    exotel_sr = 8000  # Default Exotel sample rate (8000Hz unless 16000Hz detected)
    vad_iterator = VADIterator(threshold=0.015, sampling_rate=16000, min_silence_duration_ms=VAD_SILENCE_MS)
    speech_buffer: list[np.ndarray] = []
    pre_roll_buffer: deque = deque(maxlen=8)  # ~256ms pre-roll buffer to prevent cutting off first syllable
    is_speaking = False
    conversation_history = []
    is_agent_speaking = False  # Track if we're currently streaming TTS back
    agent_stopped_speaking_time = 0.0  # Track when TTS finished for echo cooldown

    # Asyncio event loop reference for sending from threads
    loop = asyncio.get_event_loop()

    async def send_greeting():
        """Send an initial greeting when the call connects."""
        nonlocal is_agent_speaking, agent_stopped_speaking_time
        if not stream_sid:
            return

        is_agent_speaking = True
        greeting = "Hello, Rudra Infotek. Kahiye, kya kaam tha?"
        conversation_history.append({"role": "assistant", "content": greeting})

        try:
            greeting_filename = "greeting.pcm" if exotel_sr == 16000 else "greeting_8k.pcm"
            base_dir = os.path.dirname(os.path.abspath(__file__))
            greeting_file = os.path.join(base_dir, "assets", greeting_filename)
            if not os.path.exists(greeting_file):
                greeting_file = os.path.join(base_dir, greeting_filename)
            with open(greeting_file, "rb") as f:
                pcm_data = f.read()
            
            chunk_size = 640 if exotel_sr == 16000 else 320  # 20ms of audio (16kHz=640B, 8kHz=320B)
            bytes_per_sec = 32000 if exotel_sr == 16000 else 16000
            
            # Send 0.5s of silence to keep Exotel WS alive while carrier line opens
            silent_chunk = b'\x00' * chunk_size
            for _ in range(25):  # 25 * 20ms = 500ms
                await websocket.send_text(build_exotel_media_message(silent_chunk, stream_sid))
                await asyncio.sleep(0.02)
            
            start_time = time.time()
            total_sent = 0
            for i in range(0, len(pcm_data), chunk_size):
                subchunk = pcm_data[i:i+chunk_size]
                if len(subchunk) < chunk_size:
                    subchunk += b'\x00' * (chunk_size - len(subchunk))
                exotel_msg = build_exotel_media_message(subchunk, stream_sid)
                await websocket.send_text(exotel_msg)
                total_sent += len(subchunk)
                
                expected_time = total_sent / float(bytes_per_sec)
                elapsed = time.time() - start_time
                sleep_needed = expected_time - elapsed
                if sleep_needed > 0.001:
                    await asyncio.sleep(sleep_needed)

            print(f"[Exotel] ✅ Greeting sent ({exotel_sr}Hz)")
        except Exception as e:
            print(f"[Exotel] ⚠️ Error sending greeting: {e}")
        finally:
            is_agent_speaking = False
            agent_stopped_speaking_time = time.time()

    async def process_speech(user_text: str, t0: float, t1: float):
        """Process a complete utterance: STT → LLM → TTS → stream back to Exotel."""
        nonlocal is_agent_speaking, agent_stopped_speaking_time

        if not stream_sid:
            return

        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        llm_first_token = False
        tts_first_byte = False
        exotel_first_audio = False
        should_hangup = False  # Will be set True if LLM outputs [HANGUP]

        # 2. Send a "clear" event to stop any previous audio still playing
        try:
            await websocket.send_text(build_exotel_clear_message(stream_sid))
        except Exception:
            return

        # 3. Process LLM -> Sarvam TTS -> Exotel
        is_agent_speaking = True
        try:
            conversation_history.append({"role": "user", "content": user_text})

            text_q = asyncio.Queue()

            async def generate_text():
                nonlocal should_hangup
                try:
                    stream = await async_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
                        stream=True,
                    )
                    full_reply = ""
                    async for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            nonlocal llm_first_token, t2
                            if not llm_first_token:
                                t2 = time.time()
                                llm_first_token = True
                            full_reply += delta
                            # Strip [HANGUP] from the text sent to TTS
                            clean_delta = delta.replace(HANGUP_TAG, "")
                            if clean_delta:
                                await text_q.put(clean_delta)
                    await text_q.put(None)
                    # Detect scam / spam hangup signal
                    if HANGUP_TAG in full_reply:
                        cleaned = full_reply.replace(HANGUP_TAG, "").strip()
                        should_hangup = True
                        print(f"[Exotel] 🚫 Scam/Spam detected — LLM signaled instant HANGUP")
                        full_reply = cleaned
                    conversation_history.append({"role": "assistant", "content": full_reply})
                    print(f"[Exotel] 🤖 Agent replied: {full_reply}")
                except asyncio.CancelledError:
                    await text_q.put(None)
                except Exception as e:
                    if "connect" in str(e).lower() or "timeout" in str(e).lower():
                        print(f"[Exotel] 🔌 Network Error in LLM stream: Connection lost.")
                    else:
                        print(f"[Exotel] ⚠️ Error in LLM stream: {e.__class__.__name__}")
                    await text_q.put(None)

            llm_task = asyncio.create_task(generate_text())

            api_key = os.getenv("SARVAM_API_KEY")
            uri = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true"
            
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

                llm_finished = False
                async def send_text_to_sarvam():
                    nonlocal llm_finished
                    buffer = ""
                    try:
                        while True:
                            text_chunk = await text_q.get()
                            if text_chunk is None:
                                if any(c.isalnum() for c in buffer):
                                    await sarvam_ws.send(json.dumps({"type": "text", "data": {"text": buffer}}))
                                await sarvam_ws.send(json.dumps({"type": "flush"}))
                                llm_finished = True
                                break
                            buffer += text_chunk
                            if any(c.isalnum() for c in buffer) and buffer[-1] in " \n\t.!?,;:-।":
                                await sarvam_ws.send(json.dumps({"type": "text", "data": {"text": buffer}}))
                                buffer = ""
                    except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError, asyncio.CancelledError):
                        pass
                    except Exception as e:
                        print(f"[Exotel] ⚠️ Error sending text to Sarvam: {e}")

                async def receive_audio_from_sarvam():
                    ffmpeg_proc = await asyncio.create_subprocess_exec(
                        'ffmpeg', '-f', 'mp3', '-i', 'pipe:0', '-f', 's16le', '-ar', str(exotel_sr), '-ac', '1', 'pipe:1',
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL
                    )

                    async def pump_mp3():
                        try:
                            while True:
                                try:
                                    msg_str = await asyncio.wait_for(sarvam_ws.recv(), timeout=3.0 if llm_finished else None)
                                except asyncio.TimeoutError:
                                    if llm_finished:
                                        break
                                    continue
                                except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError, asyncio.CancelledError):
                                    break
                                except Exception:
                                    break
                                    
                                msg = json.loads(msg_str)
                                if msg.get("type") == "error":
                                    print("[Exotel] Sarvam WS Error:", msg)
                                    break
                                if msg.get("type") == "event" and msg.get("data", {}).get("event_type") == "final":
                                    break
                                if msg.get("type") == "audio":
                                    nonlocal tts_first_byte, t3
                                    if not tts_first_byte:
                                        t3 = time.time()
                                        tts_first_byte = True
                                    chunk = base64.b64decode(msg["data"]["audio"])
                                    if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.is_closing():
                                        ffmpeg_proc.stdin.write(chunk)
                                        await ffmpeg_proc.stdin.drain()
                        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError, asyncio.CancelledError):
                            pass
                        except Exception as e:
                            print(f"[Exotel] ⚠️ Error pumping MP3: {e}")
                        finally:
                            try:
                                if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.is_closing():
                                    ffmpeg_proc.stdin.close()
                            except Exception:
                                pass

                    async def read_pcm():
                        nonlocal exotel_first_audio, t4
                        chunk_size = 640 if exotel_sr == 16000 else 320  # 20ms of audio (16kHz=640B, 8kHz=320B)
                        bytes_per_sec = 32000 if exotel_sr == 16000 else 16000
                        pcm_buffer = bytearray()
                        stream_start_time = None
                        total_bytes_sent = 0
                        try:
                            while True:
                                raw = await ffmpeg_proc.stdout.read(4096)
                                if not raw:
                                    break
                                pcm_buffer.extend(raw)
                                
                                while len(pcm_buffer) >= chunk_size:
                                    subchunk = bytes(pcm_buffer[:chunk_size])
                                    del pcm_buffer[:chunk_size]
                                    
                                    if stream_start_time is None:
                                        stream_start_time = time.time()
                                        t4 = time.time()
                                        exotel_first_audio = True
                                        
                                    exotel_msg = build_exotel_media_message(subchunk, stream_sid)
                                    await websocket.send_text(exotel_msg)
                                    total_bytes_sent += chunk_size
                                    
                                    expected_time = total_bytes_sent / float(bytes_per_sec)
                                    elapsed = time.time() - stream_start_time
                                    sleep_needed = expected_time - elapsed
                                    if sleep_needed > 0.001:
                                        await asyncio.sleep(sleep_needed)
                                        
                            if len(pcm_buffer) > 0:
                                remainder = len(pcm_buffer) % (320 if exotel_sr == 16000 else 160)
                                if remainder != 0:
                                    pcm_buffer.extend(b'\x00' * ((320 if exotel_sr == 16000 else 160) - remainder))
                                exotel_msg = build_exotel_media_message(bytes(pcm_buffer), stream_sid)
                                await websocket.send_text(exotel_msg)
                                total_bytes_sent += len(pcm_buffer)
                                
                                expected_time = total_bytes_sent / float(bytes_per_sec)
                                elapsed = time.time() - stream_start_time if stream_start_time else 0
                                sleep_needed = expected_time - elapsed
                                if sleep_needed > 0.001:
                                    await asyncio.sleep(sleep_needed)
                        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                            pass
                        except Exception as e:
                            print(f"[Exotel] ⚠️ Error sending PCM to Exotel: {e}")

                    await asyncio.gather(pump_mp3(), read_pcm(), return_exceptions=True)

                await asyncio.gather(llm_task, send_text_to_sarvam(), receive_audio_from_sarvam(), return_exceptions=True)

            print(f"[Exotel] ✅ Response streamed ({time.time() - t0:.2f}s total)")
            print(f"[Exotel] ⏱️ TTFA breakdown — STT:{t1-t0:.2f}s | LLM-first-token:{t2-t1:.2f}s | TTS-first-byte:{t3-t2:.2f}s | sent-to-exotel:{t4-t3:.2f}s | TOTAL:{t4-t0:.2f}s")

            # ── Scam / Spam instant hangup ──────────────────────────────
            if should_hangup:
                print(f"[Exotel] 🚫 Dropping spam/scam call immediately via API...")
                # Terminate the call via Exotel REST API
                await hangup_call_via_api(call_sid)
                # Close the WebSocket from our side
                try:
                    await websocket.send_text(json.dumps({"event": "stop", "stream_sid": stream_sid}))
                except Exception:
                    pass
                print(f"[Exotel] ✅ Spam call terminated (stream_sid={stream_sid})")
                return  # Exit process_speech — the main loop will handle WS close

        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError, WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            print(f"[Exotel] ⚠️ Error streaming response: {e}")
        finally:
            is_agent_speaking = False
            agent_stopped_speaking_time = time.time()

    try:
        stt_queue = None
        stt_task = None
        stt_t0_ref = [0.0]

        async def run_stt_stream(q: asyncio.Queue, t0_ref: list) -> str:
            try:
                async with sarvam_client.speech_to_text_streaming.connect(
                    language_code="hi-IN",
                    model="saaras:v3"
                ) as stt_ws:
                    while True:
                        chunk_f32 = await q.get()
                        if chunk_f32 is None:
                            await stt_ws.flush()
                            async for msg in stt_ws:
                                if msg.type == "data":
                                    return msg.data.transcript.strip()
                            return ""
                        
                        pcm_int16 = (chunk_f32 * 32767).astype(np.int16)
                        b64 = base64.b64encode(pcm_int16.tobytes()).decode('ascii')
                        await stt_ws.transcribe(b64)
            except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
                print(f"[STT Stream] Silently returning empty due to: {e.__class__.__name__}")
                return ""
            except Exception as e:
                if str(e).strip():
                    print(f"[STT Stream] Error: {e}")
                return ""

        while True:
            # All Exotel messages are JSON text
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                print("[Exotel] 📞 WebSocket connected, waiting for stream start...")

            elif event == "start":
                stream_sid = msg["start"]["stream_sid"]
                call_sid = msg["start"].get("call_sid", "unknown")
                custom_params = msg["start"].get("custom_parameters", {})
                media_format = msg["start"].get("media_format", {})
                
                # Exotel strictly requires 'sample-rate' (with hyphen) in URL to switch to 16kHz; 'samplerate' without hyphen is ignored by Exotel and stays 8kHz
                sr_param = str(custom_params.get("sample-rate", ""))
                if media_format.get("sample_rate") == 16000 or sr_param == "16000":
                    exotel_sr = 16000
                else:
                    exotel_sr = 8000
                print(f"[Exotel] 🎙️  Stream started — stream_sid={stream_sid}, call_sid={call_sid}, sample_rate={exotel_sr}Hz")
                print(f"[Exotel]    Custom params: {custom_params}")
                # Method 1: Check custom_parameters (works with VoiceLink or custom Exotel flows)
                caller_phone = (
                    custom_params.get("From")
                    or custom_params.get("from")
                    or custom_params.get("caller_id")
                    or custom_params.get("caller_number")
                    or custom_params.get("CallFrom")
                    or custom_params.get("callfrom")
                )

                if caller_phone in ["01409082082", "1409082082", "+911409082082", "+9101409082082"]:
                    print(f"[Exotel] 🚫 Known spam number ({caller_phone}) detected from params. Hanging up immediately.")
                    await hangup_call_via_api(call_sid)
                    try:
                        await websocket.send_text(json.dumps({"event": "stop", "stream_sid": stream_sid}))
                    except Exception:
                        pass
                    return

                # Send greeting IMMEDIATELY if not spam
                asyncio.create_task(send_greeting())

                # Method 2: Fetch from Exotel REST API in background (doesn't block anything)
                if not caller_phone and call_sid and call_sid != "unknown":
                    async def fetch_caller_phone():
                        nonlocal caller_phone
                        try:
                            account_sid = os.environ.get("EXOTEL_ACCOUNT_SID")
                            api_key = os.environ.get("EXOTEL_API_KEY")
                            api_token = os.environ.get("EXOTEL_API_TOKEN")
                            if account_sid and api_key and api_token:
                                api_url = f"https://api.exotel.com/v1/Accounts/{account_sid}/Calls/{call_sid}.json"
                                async with httpx.AsyncClient() as http:
                                    resp = await http.get(api_url, auth=(api_key, api_token), timeout=5.0)
                                    if resp.status_code == 200:
                                        call_details = resp.json()
                                        caller_phone = (
                                            call_details.get("Call", {}).get("From", None)
                                            or call_details.get("Call", {}).get("CallerNumber", None)
                                        )
                                        if caller_phone:
                                            print(f"[Exotel] 📱 Caller phone: {caller_phone}")
                                            if caller_phone in ["01409082082", "1409082082", "+911409082082", "+9101409082082"]:
                                                print("[Exotel] 🚫 Known spam number detected. Hanging up immediately.")
                                                await hangup_call_via_api(call_sid)
                                                try:
                                                    await websocket.send_text(json.dumps({"event": "stop", "stream_sid": stream_sid}))
                                                except Exception:
                                                    pass
                                                return
                                        else:
                                            print(f"[Exotel] ⚠️ Phone not in API response. Keys: {list(call_details.get('Call', {}).keys())}")
                                    else:
                                        print(f"[Exotel] ⚠️ Call details API returned {resp.status_code}: {resp.text[:200]}")
                        except httpx.RequestError as e:
                            print(f"[Exotel] 🔌 Network Error: Could not reach Exotel API to fetch phone.")
                        except Exception as e:
                            print(f"[Exotel] ⚠️ Could not fetch caller phone from API: {e.__class__.__name__}")
                    asyncio.create_task(fetch_caller_phone())
                elif caller_phone:
                    print(f"[Exotel] 📱 Caller phone: {caller_phone}")

            elif event == "media":
                if is_agent_speaking or time.time() - agent_stopped_speaking_time < 0.5:
                    # Skip incoming audio while agent is speaking and for 500ms after (echo prevention)
                    continue

                # If caller has been silent for 8 seconds since we last spoke, prompt them
                if not is_speaking and agent_stopped_speaking_time > 0 and (time.time() - agent_stopped_speaking_time > 8.0):
                    print("[Exotel] 🤫 Caller is silent for 8 seconds, injecting [SILENCE] to LLM.")
                    agent_stopped_speaking_time = time.time()  # Reset to prevent multiple immediate triggers
                    asyncio.create_task(process_speech("[SILENCE]", time.time(), time.time()))
                    continue


                # Decode base64 PCM audio from Exotel
                payload = msg["media"]["payload"]
                pcm_bytes = base64.b64decode(payload)

                # Auto-detect sample rate from media packet length if not already set
                if len(pcm_bytes) <= 320 and exotel_sr != 8000:
                    exotel_sr = 8000
                elif len(pcm_bytes) >= 640 and exotel_sr != 16000:
                    exotel_sr = 16000

                pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                if exotel_sr == 8000:
                    # 2x linear upsample to 16kHz for VAD and STT
                    pcm_int16 = np.repeat(pcm_int16, 2)

                # Convert to float32 for Silero VAD (expects float32 in [-1, 1] at 16kHz)
                chunk_f32 = pcm_int16.astype(np.float32) / 32768.0

                # Silero VAD expects exactly 512 samples at 16kHz
                # Process in 512-sample windows
                offset = 0
                while offset < len(chunk_f32):
                    remaining = len(chunk_f32) - offset
                    if remaining >= 512:
                        vad_chunk = chunk_f32[offset:offset + 512]
                    else:
                        vad_chunk = np.pad(chunk_f32[offset:], (0, 512 - remaining))

                    current_slice = chunk_f32[offset:offset + min(512, remaining)]
                    pre_roll_buffer.append(current_slice)

                    if is_speaking:
                        speech_buffer.append(current_slice)
                        if stt_queue is not None:
                            stt_queue.put_nowait(current_slice)

                    vad_event = vad_iterator(vad_chunk, return_seconds=True)
                    if vad_event:
                        if "start" in vad_event:
                            is_speaking = True
                            speech_buffer.clear()
                            
                            stt_queue = asyncio.Queue()
                            stt_t0_ref[0] = time.time()
                            
                            # Prepend pre-roll buffer so initial syllables (e.g. "Cap-") are never chopped off
                            speech_buffer.extend(list(pre_roll_buffer))
                            for slice_arr in pre_roll_buffer:
                                stt_queue.put_nowait(slice_arr)
                                
                            stt_task = asyncio.create_task(run_stt_stream(stt_queue, stt_t0_ref))
                            print("[Exotel] 🟢 Caller started speaking...")

                        if "end" in vad_event:
                            is_speaking = False
                            print("[Exotel] 🔴 Caller stopped speaking, processing...")
                            pre_roll_buffer.clear()
                            if speech_buffer:
                                audio = np.concatenate(speech_buffer)

                                if len(audio) / EXOTEL_SAMPLE_RATE < 0.4:
                                    print("[Exotel] ⏭️  Too short, skipping.")
                                    if stt_task and not stt_task.done():
                                        stt_task.cancel()
                                    stt_queue = None
                                else:
                                    if stt_queue is not None:
                                        stt_queue.put_nowait(None)
                                        
                                        async def process_wrapper(task, t0):
                                            try:
                                                t_end = time.time()
                                                text = await task
                                                t_done = time.time()
                                                if text:
                                                    print(f"[Exotel] 🗣️  Caller: {text}  (Streaming STT finalized {t_done - t_end:.2f}s after speech ended)")
                                                    spam_keywords = ["bulk sms", "बल्क एसएमएस", "credit card", "loan", "लोन", "क्रेडिट कार्ड", "मैसेजिंग सेवा", "आरसीएस", "rcs", "टेलीमार्केटिंग"]
                                                    if any(k in text.lower() for k in spam_keywords):
                                                        print("[Exotel] 🚫 SPAM DETECTED. Hanging up immediately to save credits.")
                                                        await hangup_call_via_api(call_sid)
                                                        try:
                                                            await websocket.send_text(json.dumps({"event": "stop", "stream_sid": stream_sid}))
                                                        except Exception:
                                                            pass
                                                        return
                                                    await process_speech(text, t0, t_done)
                                            except asyncio.CancelledError:
                                                pass
                                            except Exception as e:
                                                if str(e).strip():
                                                    print(f"Error in STT task: {e}")
                                                
                                        asyncio.create_task(process_wrapper(stt_task, stt_t0_ref[0]))
                                        stt_queue = None

                    offset += 512

            elif event == "stop":
                print(f"[Exotel] 📴 Call ended (stream_sid={stream_sid})")
                if stt_task and not stt_task.done():
                    stt_task.cancel()
                
                # Push call data to GridCRM
                if conversation_history:
                    print("[CRM] 📤 Pushing call data to GridCRM...")
                    try:
                        await push_to_gridcrm(
                            conversation_history=conversation_history,
                            caller_phone=caller_phone,
                        )
                    except Exception as e:
                        print(f"[CRM] ❌ Error pushing to CRM: {e}")
                else:
                    print("[CRM] ⏭️ No conversation data — skipping CRM push")
                
                break

            elif event == "dtmf":
                digit = msg.get("dtmf", {}).get("digit", "?")
                print(f"[Exotel] 🔢 DTMF pressed: {digit}")

            else:
                print(f"[Exotel] ❓ Unknown event: {event}")

    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError):
        if stt_task and not stt_task.done():
            stt_task.cancel()
        print(f"[Exotel] 📴 WebSocket disconnected (stream_sid={stream_sid})")
        # Also push to CRM on unexpected disconnect
        if conversation_history:
            try:
                await push_to_gridcrm(conversation_history=conversation_history, caller_phone=caller_phone)
            except Exception as e:
                print(f"[CRM] ❌ Error pushing to CRM on disconnect: {e}")
    except Exception as e:
        if stt_task and not stt_task.done():
            stt_task.cancel()
        error_str = str(e).lower()
        if any(x in error_str for x in ["timeout", "connection", "network", "host", "resolve", "unreachable"]):
            print(f"[Exotel] 🔌 Network Error: Internet connection lost ({e.__class__.__name__})")
        else:
            print(f"[Exotel] ❌ Unexpected Error: {e.__class__.__name__} - {str(e)}")


# ─── Trigger an outbound call (optional utility) ─────────────────────────────
@app.post("/call")
async def make_outbound_call():
    """
    Optional: Trigger an outbound call via Exotel's Connect Voice AI API.
    The call will connect back to this server's /exotel WebSocket.
    
    You'll need to set your ngrok URL in the request.
    """
    import httpx

    account_sid = os.environ.get("EXOTEL_ACCOUNT_SID")
    api_key = os.environ.get("EXOTEL_API_KEY")
    api_token = os.environ.get("EXOTEL_API_TOKEN")
    exophone = os.environ.get("EXOTEL_EXOPHONE")
    # The phone number to call — you'd pass this as a query param in practice
    to_number = os.environ.get("EXOTEL_TEST_NUMBER", "+919999999999")
    # Your public ngrok/cloudflared URL
    stream_url = os.environ.get("EXOTEL_STREAM_URL", "wss://your-ngrok-url.ngrok.io/exotel?sample-rate=16000")

    url = f"https://{api_key}:{api_token}@api.in.exotel.com/v1/accounts/{account_sid}/calls/connect"

    async with httpx.AsyncClient() as http:
        resp = await http.post(url, data={
            "from": to_number,
            "callerid": exophone,
            "streamurl": stream_url,
            "streamtype": "bidirectional",
        })

    return {"status": resp.status_code, "body": resp.text}
