import re
import tempfile
import os
import numpy as np
from scipy.io.wavfile import write as write_wav
import logging
from tts_service import TTS
import pydub


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

def generate_tts_audio(text: str, tts_instance: TTS, temperature=0.7, top_k=None):
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
        # Call the TTS service to get an AudioSegment
        audio_segment = tts_instance.generate_audio_segment(
            cleaned_text,
            temperature=temperature,
            fade_duration=50, 
            start_silence_duration=100, 
            end_silence_duration=100,
        )

        # Verify we got a valid AudioSegment
        if audio_segment is None or not isinstance(audio_segment, pydub.AudioSegment):
            logger.error(f"TTS generation failed or returned unexpected type: {type(audio_segment)}")
            return None
        
        if len(audio_segment) == 0:
            logger.error("TTS generated empty audio segment.")
            return None

        # Create a temporary file to save the audio
        fd, file_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # Close the file descriptor
        
        try:
            # Export the AudioSegment directly to WAV
            audio_segment.export(file_path, format="wav")
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
        # Attempt cleanup if file_path was created before the error
        if 'file_path' in locals() and os.path.exists(file_path):
             try:
                 os.remove(file_path)
                 logger.info(f"Cleaned up temporary file after generation error: {file_path}")
             except OSError as remove_e:
                 logger.error(f"Error removing temporary file {file_path} after generation error: {remove_e}")
        return None 