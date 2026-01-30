from contextlib import asynccontextmanager
import numpy as np
import io
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.simulstreaming_whisper import simul_asr_factory
import logging
from pydantic_settings import BaseSettings
import asyncio 
import torch
from src.utils.vad import VADProcessor

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextManager:
    def __init__(self):
        self.current_source = None
        self.contexts = {}

    def has_most_chunks(self, source):
        if not self.contexts.get(source, {}).get("buffer_has_speech"):
            return False

        chunks = len(self.contexts[source]["audio_buffer"])
        if chunks < 6:
            return False

        return not any(ctx["buffer_has_speech"] and len(ctx["audio_buffer"]) > chunks for cid, ctx in self.contexts.items() if cid != source)

    def add_context(self, new_source):
        self.contexts[new_source] = {
                "online_state": None,
                "audio_buffer": [],
                "buffer_has_speech": False,
            }

    def remove_context(self, source):
        if source in self.contexts:
            del self.contexts[source]
        if self.current_source == source:
            self.current_source = None

    def switch(self, online, old_source, new_source):
        if old_source != new_source:
            if old_source is not None:
                self.contexts[old_source]["online_state"] = online.save_state()            
            if self.contexts[new_source]["online_state"] is not None:
                online.load_state(self.contexts[new_source]["online_state"])
            else:
                online.init()
            self.current_source = new_source
        

class Settings(BaseSettings):
    # Configuration
    model_path: str = "turbo.pt" 
    log_level: str = "ERROR" 
    # BEAM SEARCH
    beams: int = 1
    decoder: str = "greedy"
    
    audio_max_len: float = 10.0
    audio_min_len: float = 0.0
    frame_threshold: int = 25
    cif_ckpt_path: str = ""
    never_fire: bool = False
    init_prompt: str = ""
    static_init_prompt: str = ""
    max_context_tokens: int = 0
    
    # Language
    lan: str = "de" 
    min_chunk_size: float = 0.5
    task: str = "transcribe"
    
    # VAD Enabled
    vac: bool = True
    vac_chunk_size: float = 1.0 # Standard chunk size for VAD
    logdir: str = "logs"
    
    class Config:
        env_prefix = "WHISPER_"
        env_file = ".env"
        extra = "ignore"

# Load model on app startup
@asynccontextmanager
async def lifespan(app: FastAPI): 
    args = Settings()

    logger.info("--- SimulStreaming Microphone Demo (German | Beam-Search | VAD) ---")
    logger.info(f"Initializing Model (may download '{args.model_path}')...")

    try:
        asr, online = simul_asr_factory(args)            
    except Exception as e:
        logger.error(f"CRITICAL ERROR initializing model: {e}")

    logger.info("Model verification complete")
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, online.finish)
    except Exception:
        logger.info("Problem during finish")


    app.state.context_manager = ContextManager()
    app.state.asr = asr
    app.state.online = online
    app.state.model_lock = asyncio.Lock()
    app.state.task_running = False
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)    

async def process_and_send(client_id, buffer, websocket, ctx_mgr, online):
    try:
        async with app.state.model_lock:
            ctx_mgr.switch(online, ctx_mgr.current_source, client_id)
            online.audio_chunks = buffer
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, online.process_iter)
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        res = None
    finally:
        app.state.task_running = False     

    text = res.get('text', '') if res else None
    
    if text:
        logger.info("\n[Transcript] " + text)
        await websocket.send_json({
            "type": "transcript",
            "client_id": client_id,
            "text": text,
            "timestamp": time.time()
        })

@app.websocket("/ws/transcription")
async def websocket_endpoint(websocket: WebSocket, client_id: str = "guest") -> None:
    
    ctx_mgr = app.state.context_manager

    if client_id in ctx_mgr.contexts:
        logger.info(f"Client {client_id} already connected")
        await websocket.close(code=1008, reason="Client already connected")
        return

    ctx_mgr.add_context(client_id)

    await websocket.accept()

    logger.info(f"Client {client_id} connected: Loading model...")

    vad = VADProcessor(threshold=0.5)
    online = app.state.online

    logger.info("Model loaded successfully! Speak now (Deutsch).")

    speaking = False
    MAX_CHUNKS = 200
    
    try:
        while True:
            try:
                audio_bytes = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.1)
                audio_chunk = np.load(io.BytesIO(audio_bytes), allow_pickle=False)

                ctx_mgr.contexts[client_id]["audio_buffer"].append(torch.from_numpy(audio_chunk))
                buffer = ctx_mgr.contexts[client_id]["audio_buffer"]
                if len(buffer) > MAX_CHUNKS:
                    ctx_mgr.contexts[client_id]["audio_buffer"] = buffer[-MAX_CHUNKS:]

                vad_result = vad.process_chunk(audio_chunk)
                if vad_result['event'] == 'start':
                    print(f"START signal for {client_id}")
                    logger.info(f"VAD: {vad_result}, chunk_min={audio_chunk.min():.4f}, chunk_max={audio_chunk.max():.4f}")                
                    speaking = True
                if vad_result['event'] == 'end':
                    speaking = False     

                if speaking:
                    ctx_mgr.contexts[client_id]["buffer_has_speech"] = True
            except asyncio.TimeoutError:
                pass
                
            if ctx_mgr.has_most_chunks(client_id) and not app.state.task_running: 
                total_samples = len(ctx_mgr.contexts[client_id]['audio_buffer'])
                if total_samples >= 6:
                    app.state.task_running = True
                    print(f"Processing audio for {client_id} and {total_samples} total samples")                                
                    buffer = ctx_mgr.contexts[client_id]["audio_buffer"].copy()
                    ctx_mgr.contexts[client_id]["audio_buffer"] = []
                    asyncio.create_task(process_and_send(client_id, buffer, websocket, ctx_mgr, online))
                    ctx_mgr.contexts[client_id]["buffer_has_speech"] = False

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
                
    except Exception as e:
        logger.error(f"\n[Processing Error] {e}")

    finally:
        async with app.state.model_lock:
            ctx_mgr.remove_context(client_id)
