"""
Example Client for Whisper Streaming Server

This example demonstrates:
- Simultaneous microphone and system audio capture
- Two parallel WebSocket connections to the transcription server
- Color-coded output for each audio source

Usage:
    Local server:
        python example_client.py
    
    Docker server:
        python example_client.py --server ws://localhost:9000/ws/transcription
    
    Remote server:
        python example_client.py --server ws://192.168.1.100:9000/ws/transcription
"""

import websocket
import argparse
import queue
import threading
import numpy as np
import soundcard as sc
import json
import time
import warnings
import io
import sys

warnings.filterwarnings("ignore", category=Warning)

# ANSI colors for output
class Colors:
    USER = "\033[94m"      # Blue for user/mic
    SYSTEM = "\033[93m"    # Yellow for system audio
    RESET = "\033[0m"
    BOLD = "\033[1m"

# Configuration
DEFAULT_SERVER_URL = "ws://localhost:9000/ws/transcription"
SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE * 0.1)  # 100ms chunks


def main(server_url: str = DEFAULT_SERVER_URL):
    print(f"\n{Colors.BOLD}Whisper Streaming Server - Example Client{Colors.RESET}")
    print("=" * 50)
    print(f"Server: {server_url}")
    print("=" * 50)
    
    # Create two websocket connections
    ws_user = websocket.WebSocket()
    ws_system = websocket.WebSocket()
    
    try:
        ws_user.connect(f"{server_url}?client_id=user")
        print(f"{Colors.USER}✓ Connected: User (Microphone){Colors.RESET}")
    except Exception as e:
        print(f"Failed to connect user WebSocket: {e}")
        return
        
    try:
        ws_system.connect(f"{server_url}?client_id=system")
        print(f"{Colors.SYSTEM}✓ Connected: System (Loopback){Colors.RESET}")
    except Exception as e:
        print(f"Failed to connect system WebSocket: {e}")
        ws_user.close()
        return

    print("=" * 40)
    print("Listening... (Ctrl+C to stop)\n")

    # Queues for audio
    mic_queue = queue.Queue()
    sys_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start audio capture threads
    mic_thread = threading.Thread(
        target=capture_microphone, 
        args=(mic_queue, stop_event),
        daemon=True
    )
    sys_thread = threading.Thread(
        target=capture_system_audio, 
        args=(sys_queue, stop_event),
        daemon=True
    )
    
    # Start receive threads for transcriptions
    recv_user_thread = threading.Thread(
        target=receive_transcripts, 
        args=(ws_user, "user"),
        daemon=True
    )
    recv_sys_thread = threading.Thread(
        target=receive_transcripts, 
        args=(ws_system, "system"),
        daemon=True
    )
    
    mic_thread.start()
    sys_thread.start()
    recv_user_thread.start()
    recv_sys_thread.start()

    try:
        while True:
            # Send microphone audio
            send_audio_chunks(ws_user, mic_queue)
            
            # Send system audio
            send_audio_chunks(ws_system, sys_queue)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.BOLD}Stopping...{Colors.RESET}")
        stop_event.set()
        
    finally:
        ws_user.close()
        ws_system.close()
        print("Done.")


def send_audio_chunks(ws, audio_queue):
    """Collect and send audio chunks from queue"""
    chunks = []
    while not audio_queue.empty():
        try:
            chunks.append(audio_queue.get_nowait())
        except queue.Empty:
            break

    if chunks:
        try:
            data = np.concatenate(chunks, axis=0)
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
            data = data.flatten().astype(np.float32)

            buf = io.BytesIO()
            np.save(buf, data, allow_pickle=False)
            ws.send(buf.getvalue(), opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(f"[Send Error] {e}")


def receive_transcripts(ws, source):
    """Receive and display transcripts with color coding"""
    color = Colors.USER if source == "user" else Colors.SYSTEM
    label = "🎤 User" if source == "user" else "🔊 System"
    
    while True:
        try:
            message = ws.recv()
            data = json.loads(message)
            text = data.get("text", "")
            if text:
                print(f"{color}{label}: {text}{Colors.RESET}")
        except Exception:
            break


def capture_microphone(queue_out, stop_event):
    """Capture audio from default microphone"""
    try:
        mic = sc.default_microphone()
        print(f"  Microphone: {mic.name}")
        
        with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
            while not stop_event.is_set():
                data = recorder.record(numframes=BLOCK_SIZE)
                if len(data) > 0:
                    queue_out.put(data)
    except Exception as e:
        print(f"[Mic Error] {e}")


def capture_system_audio(queue_out, stop_event):
    """Capture system audio via loopback device"""
    try:
        # Find loopback device matching default speaker
        default_speaker = sc.default_speaker()
        loopback = None
        
        for mic in sc.all_microphones(include_loopback=True):
            if mic.isloopback and mic.name == default_speaker.name:
                loopback = mic
                break
        
        if not loopback:
            # Fallback: first available loopback
            for mic in sc.all_microphones(include_loopback=True):
                if mic.isloopback:
                    loopback = mic
                    break
        
        if loopback:
            print(f"  System Audio: {loopback.name} (Loopback)")
            with loopback.recorder(samplerate=SAMPLE_RATE) as recorder:
                while not stop_event.is_set():
                    data = recorder.record(numframes=BLOCK_SIZE)
                    if len(data) > 0:
                        queue_out.put(data)
        else:
            print("  System Audio: No loopback device found")
            
    except Exception as e:
        print(f"[System Audio Error] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Whisper Streaming Server - Example Client"
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        default=DEFAULT_SERVER_URL,
        help=f"WebSocket server URL (default: {DEFAULT_SERVER_URL})"
    )
    args = parser.parse_args()
    main(server_url=args.server)
