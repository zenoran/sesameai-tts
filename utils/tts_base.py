"""
Base class for SesameAI Text-to-Speech applications.

This module provides a common base class for TTS functionality
shared between different applications to avoid code duplication.
"""
import logging
import re
import time
import numpy as np
from tts_service import TTS, DEFAULT_VOICE
from utils.tts_utils import clean_text_for_tts
import threading

logger = logging.getLogger(__name__)

class TTSBaseApp:
    def __init__(self, voice: str = DEFAULT_VOICE):
        default_voice = voice
        self.tts = TTS(device="cuda")
        self.tts.load_model()
        
        try:
            self.tts.load_voice(default_voice)
            self.current_voice = default_voice
            self.current_status = f"Ready. Using voice: {default_voice}"
        except Exception as e:
            logger.error(f"Error loading default voice: {e}")
            self.current_voice = None
            self.current_status = "Ready. TTS initialized with no voice."
            
        # Common state variables
        self.sentences = []
        self.current_sample_rate = None
        self.lock = threading.Lock()
        
    def split_text_into_sentences(self, text):
        """Split text into sentences and return list of processed sentences."""
        cleaned_text = clean_text_for_tts(text)
        split_pattern = r"([.!?])(\s+|$)"
        parts = re.split(split_pattern, cleaned_text)
        
        new_sentences_raw = []
        current_sentence = ""
        for i in range(0, len(parts), 3):
            part = parts[i] if i < len(parts) else ""
            delimiter = parts[i+1] if i + 1 < len(parts) else ""
            
            if part:
                current_sentence += part + delimiter
                if delimiter:  # End of a sentence found
                    if current_sentence.strip():
                        new_sentences_raw.append(current_sentence.strip())
                    current_sentence = ""  # Reset for next sentence
        
        if current_sentence.strip():
            new_sentences_raw.append(current_sentence.strip())
            
        return [s for s in new_sentences_raw if s]
        
    def change_voice(self, new_voice_name):
        """Changes the TTS voice and updates status."""
        print(f"Attempting to change voice to: {new_voice_name}")
        with self.lock:
            try:
                self.tts.load_voice(new_voice_name)
                self.current_voice = new_voice_name
                self.current_status = f"Voice changed to {new_voice_name}. Ready."
                print(f"Successfully changed voice to {new_voice_name}.")
            except Exception as e:
                logger.exception(f"Error changing voice to {new_voice_name}: {e}")
                self.current_status = f"Error changing voice to {new_voice_name}: {e}"
                print(self.current_status)
            return self.current_status
    
    def generate_audio_for_sentence_index(self, sentence_index, temperature=0.8, topk=40, speed_factor=1.0, speaker: int = 1):
        """Generate audio for a specific sentence index and return audio data for streaming"""
        audio_data = None  # Default to None
        
        with self.lock:
            if sentence_index >= len(self.sentences):
                status = f"Sentence index {sentence_index} out of bounds (total: {len(self.sentences)})"
                logger.warning(status)
                return status, None
            
            sentence = self.sentences[sentence_index]
            total_sentences = len(self.sentences)
            status = f"Generating audio for sentence {sentence_index+1}/{total_sentences}: {sentence[:50]}..."
            self.current_status = status
            print(self.current_status)
        
        try:
            logger.info(f"Generating audio for sentence {sentence_index+1}/{total_sentences}")
            start_time = time.time()
            
            # Generate audio for this sentence
            audio_segment = self.tts.generate_audio_segment(
                sentence,
                speaker=speaker,
                temperature=temperature,
                topk=topk,
                fade_duration=50,
                start_silence_duration=150,
                end_silence_duration=150,
            )
            
            # Apply speed adjustment if needed
            if speed_factor != 1.0:
                audio_segment = audio_segment.speedup(playback_speed=speed_factor)
            
            with self.lock:
                # Set sample rate if not already set
                if self.current_sample_rate is None:
                    self.current_sample_rate = audio_segment.frame_rate
                
                # Each child class will have its own way of storing audio_segments
                self._store_audio_segment(audio_segment, sentence_index)
            
            # Convert pydub AudioSegment to numpy array for Gradio
            raw_samples = audio_segment.get_array_of_samples()
            audio_np_raw = np.array(raw_samples)
            
            # Normalize to float32 between -1.0 and 1.0 as expected by Gradio
            if audio_np_raw.dtype == np.int16:
                audio_np = audio_np_raw.astype(np.float32) / 32768.0
            elif audio_np_raw.dtype != np.float32:
                max_val = np.iinfo(audio_np_raw.dtype).max
                audio_np = audio_np_raw.astype(np.float32) / max_val
            else:
                audio_np = audio_np_raw
            
            # Create audio data tuple (sample_rate, audio_array) for Gradio
            audio_data = (audio_segment.frame_rate, audio_np)
            
            duration = audio_segment.duration_seconds
            process_time = time.time() - start_time
            
            next_status = f"Processed sentence {sentence_index+1}/{total_sentences} " + \
                         f"({duration:.1f}s audio / {process_time:.1f}s proc)"
            
            with self.lock:
                is_last = sentence_index == len(self.sentences) - 1
                if not is_last:
                    next_status += ". Generating next..."
                else:
                    next_status += ". All sentences processed."
                self.current_status = next_status
            
            return self.current_status, audio_data
            
        except Exception as e:
            error_msg = f"Error generating audio for sentence {sentence_index+1}: {e}"
            logger.exception(error_msg)
            with self.lock:
                self.current_status = f"Error on sentence {sentence_index+1}/{len(self.sentences)}. Skipping."
            return self.current_status, None

    def _store_audio_segment(self, audio_segment, sentence_index):
        """
        Store the generated audio segment. 
        This is meant to be overridden by child classes with specific storage methods.
        """
        pass
        
    def list_available_voices(self):
        """Return the list of available TTS voices."""
        try:
            return self.tts.list_voices()
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return ["Error loading voices"] 