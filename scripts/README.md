# Utility and Helper Scripts

- `start_server_loop.sh`: Starts the Uvicorn server in a persistent auto-restart loop for local development.
- `outbound.py`: Triggers an outbound call via Exotel REST API to test agent voice flows.
- `gen_greeting.py`: Calls Sarvam AI TTS WebSocket API to generate `assets/greeting.pcm` (16kHz) and `assets/greeting_8k.pcm` (8kHz).
- `generate_greeting.py`: Alternative/legacy greeting audio generator using Sarvam AI TTS.
- `generate_hello.py`: Quick test script to fetch TTS audio for a short "Hello?" snippet.
- `test_sarvam.py`: Test script for Sarvam AI TTS connectivity and audio packet decoding.
