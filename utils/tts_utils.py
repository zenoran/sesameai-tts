import re
import tempfile
import os
import numpy as np
from scipy.io.wavfile import write as write_wav
import logging

logger = logging.getLogger(__name__)

def clean_text_for_tts(text):
    """
    Clean text to make it suitable for TTS processing by:
    1. Removing markdown formatting
    2. Removing special characters
    3. Converting to plain sentences
    """
    if not isinstance(text, str):
        text = str(text) # Ensure input is string
        
    text = text.replace("—", "...")
    # Remove code blocks first
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code
    text = re.sub(r'`[^`]*`', '', text)
    # Remove markdown links, keeping the text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bold/italics
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove characters not suitable for TTS (allow basic punctuation)
    # Keep letters, numbers, spaces, and .,!?:;'"-
    text = re.sub(r'[^\w\s.,!?:;\'"-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize punctuation (avoid sequences like "!!")
    text = re.sub(r'([.,!?:;-])\1+', r'\1', text)
    # Add space after punctuation if followed by a word character
    text = re.sub(r'([.,!?:;-])(\w)', r'\1 \2', text)

    return text.strip()

def generate_tts_audio(text, tts_instance, voice_wav="", language="en", temperature=0.7, top_k=None, top_p=None):
    """
    Generates TTS audio from text using the provided TTS instance,
    saves it to a temporary WAV file, and returns the file path.
    Returns None if TTS generation fails or text is empty.
    """
    cleaned_text = clean_text_for_tts(text)
    if not cleaned_text:
        logger.warning("Skipping TTS generation for empty or invalid text.")
        return None

    try:
        logger.info(f"Generating TTS for: '{cleaned_text[:100]}...'")
        # Assuming tts_instance has a 'generate' method
        output = tts_instance.generate(
            cleaned_text,
            voice_wav=voice_wav,
            language=language,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        if output is None or 'audio_wav' not in output or 'sample_rate' not in output:
             logger.error("TTS generation failed or returned unexpected output format.")
             return None

        audio_wav = output['audio_wav']
        sample_rate = output['sample_rate']
        
        if not isinstance(audio_wav, np.ndarray) or audio_wav.size == 0:
            logger.error("TTS generated empty or non-numpy audio array.")
            return None

        # Ensure audio data is in int16 format for WAV saving
        if audio_wav.dtype == np.float32:
            # Assuming float audio is in range [-1.0, 1.0]
            audio_wav = (audio_wav * 32767).astype(np.int16)
        elif audio_wav.dtype != np.int16:
             logger.warning(f"Unexpected audio dtype: {audio_wav.dtype}. Attempting conversion to int16.")
             max_val = np.max(np.abs(audio_wav))
             if max_val > 0:
                 audio_wav = (audio_wav / max_val * 32767).astype(np.int16)
             else:
                 audio_wav = audio_wav.astype(np.int16)

        # Create a temporary file to save the audio
        # Use a context manager for robust temporary file handling
        fd, file_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd) # Close the file descriptor as write_wav opens the file path
        
        try:
            write_wav(file_path, sample_rate, audio_wav)
            logger.info(f"TTS audio successfully saved to temporary file: {file_path}")
            return file_path
        except Exception as write_e:
            logger.exception(f"Failed to write WAV file: {write_e}")
            # Clean up the temp file if writing failed
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
                except OSError as remove_e:
                    logger.error(f"Error removing temporary file {file_path}: {remove_e}")
            return None

    except Exception as e:
        logger.exception(f"Error during TTS generation process: {e}")
        # Attempt cleanup even if generation fails before writing
        if 'file_path' in locals() and os.path.exists(file_path):
             try:
                 os.remove(file_path)
                 logger.info(f"Cleaned up temporary file after generation error: {file_path}")
             except OSError as remove_e:
                 logger.error(f"Error removing temporary file {file_path} after generation error: {remove_e}")
        return None 