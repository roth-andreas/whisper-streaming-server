# Whisper Streaming Server

A real-time speech-to-text transcription server with **multi-client support** and **automatic context switching**. Stream audio from multiple sources simultaneously and receive low-latency transcriptions via WebSocket.

## ✨ Features

- 🎤 **Multi-Client Transcription** - Connect multiple audio sources (microphone, system audio, etc.) simultaneously
- 🔄 **Automatic Context Switching** - Seamlessly switch between clients with full state preservation
- ⚡ **Real-Time Streaming** - Low-latency transcription using the AlignAtt simultaneous policy
- 🌐 **WebSocket API** - Simple integration via standard WebSocket connections
- 🗣️ **Voice Activity Detection** - Built-in Silero VAD for intelligent speech detection
- 🤖 **Auto Model Download** - Whisper models are downloaded automatically on first use

## 🚀 Installation

### Option A: Local Installation

```bash
pip install -r requirements.txt
```

### Option B: Docker (Recommended)

```bash
docker-compose up -d
```

## 📖 Quick Start

### 1. Start the Server

**Local:**
```bash
uvicorn src.server:app --host 0.0.0.0 --port 9000
```

**Docker:**
```bash
docker-compose up -d
```

The WebSocket endpoint is available at `ws://localhost:9000/ws/transcription`.

### 2. Run the Example Client

The example client demonstrates multi-source transcription with both microphone and system audio:

```bash
# Connect to local server
python examples/example_client.py

# Connect to Docker server (same URL by default)
python examples/example_client.py --server ws://localhost:9000/ws/transcription

# Connect to remote server
python examples/example_client.py --server ws://192.168.1.100:9000/ws/transcription
```

You'll see color-coded output:
- 🎤 **Blue** = Your microphone (user)
- 🔊 **Yellow** = System audio (what's playing on your computer)

## 🐳 Docker

### Build & Run

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### GPU Support

The Docker setup includes NVIDIA GPU support. Requirements:
- NVIDIA GPU with CUDA support
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_PATH` | `turbo.pt` | Whisper model filename |
| `WHISPER_LAN` | `de` | Language code |
| `WHISPER_LOG_LEVEL` | `INFO` | Logging level |

## 🔌 WebSocket API

### Connecting

```
ws://localhost:9000/ws/transcription?client_id=<unique_id>
```

Each client needs a unique `client_id`. Examples:
- `user` - Microphone input
- `system` - System audio / loopback
- `guest` - Additional participant

### Sending Audio

Audio must be sent as binary numpy arrays:
- **Sample Rate:** 16,000 Hz
- **Channels:** Mono
- **Format:** float32

```python
import numpy as np
import io

# Your audio chunk (16kHz, mono, float32)
audio_chunk = np.array([...], dtype=np.float32)

# Serialize and send
buf = io.BytesIO()
np.save(buf, audio_chunk, allow_pickle=False)
websocket.send(buf.getvalue())
```

### Receiving Transcripts

The server sends JSON messages with transcription results:

```json
{
  "type": "transcript",
  "client_id": "user",
  "text": "Hello, this is a test.",
  "timestamp": 1706614425.123
}
```

## ⚙️ Configuration

Server configuration options (in `server.py`):

| Option | Default | Description |
|--------|---------|-------------|
| `model_path` | `turbo.pt` | Whisper model (auto-downloaded) |
| `beams` | `1` | Beam search size (1 = greedy) |
| `decoder` | `greedy` | Decoder type (`greedy` or `beam`) |

Available Whisper models:
- `tiny.pt` - Fastest, lowest quality
- `small.pt` - Good balance
- `medium.pt` - Better quality
- `large-v3.pt` - Best quality
- `turbo.pt` - Optimized for speed (recommended)

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Whisper Streaming Server                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│   │ Client A │   │ Client B │   │ Client C │  ...       │
│   │  (user)  │   │ (system) │   │ (guest)  │            │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘            │
│        │              │              │                   │
│        └──────────────┼──────────────┘                   │
│                       ▼                                  │
│        ┌──────────────────────────────┐                  │
│        │      Context Manager         │                  │
│        │  • Audio buffer per client   │                  │
│        │  • Model state per client    │                  │
│        │  • Automatic switching       │                  │
│        └──────────────┬───────────────┘                  │
│                       ▼                                  │
│        ┌──────────────────────────────┐                  │
│        │     Shared Whisper Model     │                  │
│        │  • Single GPU instance       │                  │
│        │  • State save/restore        │                  │
│        └──────────────────────────────┘                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**How Context Switching Works:**

1. When Client A sends audio, the server processes it
2. If Client B sends audio while A is being processed, B's audio is buffered
3. When A's processing completes, the server:
   - Saves A's model state (tokens, position, etc.)
   - Loads B's saved state (or initializes fresh)
   - Processes B's buffered audio
4. Each client maintains its own transcription context

## 🙏 Credits

This project is built on the excellent work of:

- **[SimulStreaming](https://github.com/ufal/SimulStreaming)** by Charles University - Simultaneous streaming for Whisper
- **[Simul-Whisper](https://github.com/backspacetg/simul_whisper/)** - AlignAtt policy implementation
- **[Whisper-Streaming](https://github.com/ufal/whisper_streaming)** - Streaming interface for Whisper
- **[OpenAI Whisper](https://github.com/openai/whisper)** - The underlying speech recognition model

Extended with multi-client WebSocket server and context switching for real-time applications.

## 📄 License

MIT License - see [LICENCE.txt](LICENCE.txt)
