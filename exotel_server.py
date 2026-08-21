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
from silero_vad import load_silero_vad, VADIterator
import websockets
from groq import Groq, AsyncGroq
from fishaudio import FishAudio
from fishaudio.types import TTSConfig
from scipy.signal import butter, lfilter
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ─── Audio Config ─────────────────────────────────────────────────────────────
# Exotel streams at 8kHz by default. We request 16kHz via ?sample-rate=16000.
EXOTEL_SAMPLE_RATE = 16000
FISH_SAMPLE_RATE = 16000
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "500"))
_HPF_B, _HPF_A = butter(2, 200.0 / (0.5 * 16000.0), btype='high', analog=False)

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

# ─── Silero VAD ───────────────────────────────────────────────────────────────
vad_model = load_silero_vad()

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional and friendly inbound customer support and sales executive for "Rudra Infotek", an IT services company.
You specialize in CCTV installations, Networking, and IT services.
You must speak STRICTLY in conversational Hinglish (Hindi written in the English alphabet, e.g., "Aap kaise ho?").
Do NOT use pure English sentences. Do NOT use Devnagari script.

Your goal is to assist callers and qualify them as leads:
1. When they state their need, ask clarifying questions (e.g., if they want CCTV, ask how many cameras and if it's for home or office).
2. Answer basic questions confidently but do not explain deep technical details.
3. Once you have their basic requirements, let them know a senior engineer will follow up with a detailed quotation.

Rules:
- Keep every reply very short (1-2 sentences max). This is a voice call.
- Be polite and professional."""


# ─── Utility Functions ────────────────────────────────────────────────────────


def apply_hpf(audio: np.ndarray, cutoff: float = 200.0, fs: float = 16000.0) -> np.ndarray:
    """Digital High-Pass Filter (HPF) cut-off at roughly 200Hz to strip out low-end network hum and line rumble."""
    if len(audio) == 0:
        return audio
        
    return lfilter(_HPF_B, _HPF_A, audio).astype(audio.dtype)

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

    vad_iterator = VADIterator(vad_model, sampling_rate=EXOTEL_SAMPLE_RATE, threshold=0.6, min_silence_duration_ms=VAD_SILENCE_MS)
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
    vad_iterator = VADIterator(vad_model, sampling_rate=EXOTEL_SAMPLE_RATE, threshold=0.6, min_silence_duration_ms=VAD_SILENCE_MS)
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
        greeting = "Namaste! Rudra Infotek mein aapka swagat hai. Main aapki kaise madad kar sakti hoon?"
        conversation_history.append({"role": "assistant", "content": greeting})

        try:
            api_key = os.getenv("SARVAM_API_KEY")
            uri = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true"
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

                # Sarvam returns MP3 audio — decode to PCM via ffmpeg (same as response pipeline)
                ffmpeg_proc = await asyncio.create_subprocess_exec(
                    'ffmpeg', '-f', 'mp3', '-i', 'pipe:0', '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL
                )

                async def pump_greeting_mp3():
                    try:
                        async for msg_str in sarvam_ws:
                            msg = json.loads(msg_str)
                            if msg.get("type") == "error":
                                print(f"[Exotel] Sarvam greeting error: {msg}")
                                break
                            if msg.get("type") == "event" and msg.get("data", {}).get("event_type") == "final":
                                break
                            if msg.get("type") == "audio":
                                chunk = base64.b64decode(msg["data"]["audio"])
                                ffmpeg_proc.stdin.write(chunk)
                                await ffmpeg_proc.stdin.drain()
                    finally:
                        ffmpeg_proc.stdin.close()

                async def read_greeting_pcm():
                    chunk_size = 3200
                    while True:
                        subchunk = await ffmpeg_proc.stdout.read(chunk_size)
                        if not subchunk:
                            break
                        remainder = len(subchunk) % 320
                        if remainder != 0:
                            subchunk += b'\x00' * (320 - remainder)
                        exotel_msg = build_exotel_media_message(subchunk, stream_sid)
                        await websocket.send_text(exotel_msg)
                        await asyncio.sleep(0.01)

                await asyncio.gather(pump_greeting_mp3(), read_greeting_pcm())

            print("[Exotel] ✅ Greeting sent")
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

        # 2. Send a "clear" event to stop any previous audio still playing
        await websocket.send_text(build_exotel_clear_message(stream_sid))

        # 3. Process LLM -> Sarvam TTS -> Exotel
        is_agent_speaking = True
        try:
            conversation_history.append({"role": "user", "content": user_text})

            text_q = asyncio.Queue()

            async def generate_text():
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
                            await text_q.put(delta)
                    await text_q.put(None)
                    conversation_history.append({"role": "assistant", "content": full_reply})
                    print(f"[Exotel] 🤖 Agent replied.")
                except Exception as e:
                    print(f"Error in LLM stream: {e}")
                    await text_q.put(None)

            llm_task = asyncio.create_task(generate_text())

            api_key = os.getenv("SARVAM_API_KEY")
            uri = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true"
            
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

                llm_finished = False
                async def send_text_to_sarvam():
                    nonlocal llm_finished
                    buffer = ""
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

                async def receive_audio_from_sarvam():
                    ffmpeg_proc = await asyncio.create_subprocess_exec(
                        'ffmpeg', '-f', 'mp3', '-i', 'pipe:0', '-f', 's16le', '-ar', '16000', '-ac', '1', 'pipe:1',
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
                                except Exception:
                                    break
                                    
                                msg = json.loads(msg_str)
                                if msg.get("type") == "error":
                                    print("Sarvam WS Error:", msg)
                                    break
                                if msg.get("type") == "event" and msg.get("data", {}).get("event_type") == "final":
                                    break
                                if msg.get("type") == "audio":
                                    nonlocal tts_first_byte, t3
                                    if not tts_first_byte:
                                        t3 = time.time()
                                        tts_first_byte = True
                                    chunk = base64.b64decode(msg["data"]["audio"])
                                    ffmpeg_proc.stdin.write(chunk)
                                    await ffmpeg_proc.stdin.drain()
                        finally:
                            ffmpeg_proc.stdin.close()

                    async def read_pcm():
                        nonlocal exotel_first_audio, t4
                        chunk_size = 3200
                        while True:
                            subchunk = await ffmpeg_proc.stdout.read(chunk_size)
                            if not subchunk:
                                break
                            remainder = len(subchunk) % 320
                            if remainder != 0:
                                subchunk += b'\x00' * (320 - remainder)
                            exotel_msg = build_exotel_media_message(subchunk, stream_sid)
                            await websocket.send_text(exotel_msg)
                            
                            if not exotel_first_audio:
                                t4 = time.time()
                                exotel_first_audio = True
                                
                            await asyncio.sleep(0.01)

                    await asyncio.gather(pump_mp3(), read_pcm())

                await asyncio.gather(llm_task, send_text_to_sarvam(), receive_audio_from_sarvam())

            print(f"[Exotel] ✅ Response streamed ({time.time() - t0:.2f}s total)")
            print(f"[Exotel] ⏱️ TTFA breakdown — STT:{t1-t0:.2f}s | LLM-first-token:{t2-t1:.2f}s | TTS-first-byte:{t3-t2:.2f}s | sent-to-exotel:{t4-t3:.2f}s | TOTAL:{t4-t0:.2f}s")
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
            except Exception as e:
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
                print(f"[Exotel] 🎙️  Stream started — stream_sid={stream_sid}, call_sid={call_sid}")
                print(f"[Exotel]    Custom params: {custom_params}")

                # Send greeting when call starts
                asyncio.create_task(send_greeting())

            elif event == "media":
                if is_agent_speaking or time.time() - agent_stopped_speaking_time < 0.5:
                    # Skip incoming audio while agent is speaking and for 500ms after (echo prevention)
                    continue

                # Decode base64 PCM audio from Exotel
                payload = msg["media"]["payload"]
                pcm_bytes = base64.b64decode(payload)
                pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

                # Convert to float32 for Silero VAD (expects float32 in [-1, 1])
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
                                                    await process_speech(text, t0, t_done)
                                            except Exception as e:
                                                print(f"Error in STT task: {e}")
                                                
                                        asyncio.create_task(process_wrapper(stt_task, stt_t0_ref[0]))
                                        stt_queue = None

                    offset += 512

            elif event == "stop":
                print(f"[Exotel] 📴 Call ended (stream_sid={stream_sid})")
                break

            elif event == "dtmf":
                digit = msg.get("dtmf", {}).get("digit", "?")
                print(f"[Exotel] 🔢 DTMF pressed: {digit}")

            else:
                print(f"[Exotel] ❓ Unknown event: {event}")

    except WebSocketDisconnect:
        print(f"[Exotel] 📴 WebSocket disconnected (stream_sid={stream_sid})")
    except Exception as e:
        print(f"[Exotel] ❌ Error: {e}")
        import traceback
        traceback.print_exc()


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
