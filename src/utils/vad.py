"""
VAD (Voice Activity Detection) Utilities

Provides helper functions and classes for voice activity detection
using Silero VAD model.
"""

import torch
import numpy as np
from src.whisper_streaming.silero_vad_iterator import FixedVADIterator

SAMPLING_RATE = 16000


def load_vad_model():
    """Load and return the Silero VAD model."""
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    return model


def create_vad_iterator(model=None, threshold=0.5):
    """
    Create a VAD iterator for processing audio chunks.
    
    Args:
        model: Pre-loaded Silero VAD model. If None, loads a new one.
        threshold: Speech detection threshold (0.0 to 1.0).
                   Higher = less sensitive, fewer false positives.
    
    Returns:
        FixedVADIterator instance
    """
    if model is None:
        model = load_vad_model()
    return FixedVADIterator(model, threshold=threshold)


class VADProcessor:
    """
    Voice Activity Detection processor for streaming audio.
    
    Tracks speech state and provides callbacks for speech start/end events.
    """
    
    def __init__(self, threshold=0.5):
        """
        Initialize VAD processor.
        
        Args:
            threshold: Speech detection threshold (0.0 to 1.0)
        """
        self.model = load_vad_model()
        self.vad = FixedVADIterator(self.model, threshold=threshold)
        self.is_speaking = False
        self.speech_start_time = None
    
    def reset(self):
        """Reset VAD state."""
        self.vad.reset_states()
        self.is_speaking = False
        self.speech_start_time = None
    
    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process an audio chunk and detect speech activity.
        
        Args:
            audio_chunk: Audio data as numpy array (16kHz, mono, float32)
        
        Returns:
            dict with keys:
                - 'event': 'start', 'end', 'continue', or None
                - 'is_speaking': current speech state
                - 'vad_result': raw VAD result
        """
        vad_result = self.vad(audio_chunk)
        event = None
        
        if vad_result is not None:
            if 'start' in vad_result and not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = vad_result.get('start')
                event = 'start'
            elif 'end' in vad_result and self.is_speaking:
                self.is_speaking = False
                event = 'end'
        elif self.is_speaking:
            event = 'continue'
        
        return {
            'event': event,
            'is_speaking': self.is_speaking,
            'vad_result': vad_result
        }
