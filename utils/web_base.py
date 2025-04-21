import abc
import logging
import os
import time

from ask_llm.core import AskLLM
from ask_llm.utils.config import config as llm_config
from ask_llm.model_manager import ModelManager
from .tts_base import TTSBaseApp

logger = logging.getLogger(__name__)


class WebAppBase(TTSBaseApp, abc.ABC):
    def __init__(self, voice: str, model: str):
        self.temp_audio_files = []
        self.audio_segments = []

        self.model_manager = ModelManager(llm_config)

        llm_config.VERBOSE = False

        requested_alias = model
        self.current_resolved_alias = self.model_manager.resolve_model_alias(
            requested_alias
        )

        if not self.current_resolved_alias:
            raise ValueError(f"Could not resolve initial model alias: {requested_alias}")
        else:
            print(f"Resolved initial model alias: {self.current_resolved_alias}")
            try:
                self.llm = AskLLM(resolved_model_alias=self.current_resolved_alias, config=llm_config)
            except Exception as e:
                print(
                    f"[Fatal Error] Failed to initialize AskLLM with {self.current_resolved_alias}: {e}"
                )
                raise

        self.available_models = llm_config.get_model_options()
        self.current_model = self.current_resolved_alias
        super().__init__(voice=voice)

    def get_answer(self, query: str):
        pass

    def update_status(self, message: str):
        pass

    def stream_audio_response(self, audio_chunk):
        pass

    def clear_ui(self):
        pass

    def _store_audio_segment(self, audio_segment, sentence_index):
        self.audio_segments.append(audio_segment)

    def interrupt_and_reset(self):
        logger.info("Interrupting any ongoing TTS generation.")
        with self.lock:
            self.sentences = []
            self.audio_segments = []
        self.update_status("Interrupted previous response.")
        # Subclasses should handle UI state reset (e.g., processing_active flag, audio player)

    def sentence_generator_loop(
        self, start_index, end_index, active, temperature=0.7, speed_factor=1.2, topk=40, speaker: int = 1
    ):
        if not active:
            logger.info("Generator triggered but not active.")
            self.update_status("Processing stopped.")
            yield False, None  # Yield active=False, audio=None
            return

        logger.info(
            f"Starting sentence generator loop from index {start_index} to {end_index} with speed {speed_factor}"
        )
        current_index = start_index

        while True:
            with self.lock:
                total_sentences = len(self.sentences)
                is_within_bounds = (
                    current_index < total_sentences and current_index < end_index
                )

            if not active or not is_within_bounds:
                final_status = "Processing stopped."
                if active and not is_within_bounds:
                    final_status = "All sentences processed. Audio playback complete."

                logger.info(f"Generator loop finished. Status: {final_status}")
                self.update_status(final_status)
                yield False, None  # Yield active=False, audio=None
                return

            status, audio_tuple = self.generate_audio_for_sentence_index(
                current_index, temperature, topk=topk, speed_factor=speed_factor, speaker=speaker
            )
            next_index = current_index + 1

            if "Error" in status and audio_tuple is None:
                logger.error(
                    f"Error processing sentence {current_index + 1}. Will stop."
                )
                self.update_status(status)
                yield False, None  # Yield active=False, audio=None
                return

            self.update_status(status)
            self.stream_audio_response(audio_tuple)
            yield active, audio_tuple  # Keep active, yield audio
            current_index = next_index
            time.sleep(0.05)

    def clear_session(self, clear_history: bool = True):
        logger.info("Clearing base session state...")
        
        # Clear LLM history if requested
        if clear_history:
            if hasattr(self.llm, 'history_manager') and self.llm.history_manager is not None:
                try:
                    self.llm.history_manager.clear_history()
                    logger.info("LLM history cleared.")
                except Exception as e:
                    logger.error(f"Error clearing LLM history: {e}")
            else:
                logger.warning("LLM object or history manager not found, skipping history clear.")
                
        # Clean up temporary audio files
        for audio_path in self.temp_audio_files:
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info(f"Removed temporary audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Error removing temp file {audio_path}: {e}")

        self.temp_audio_files = []

        # Reset sentence processing state
        with self.lock:
            self.sentences = []
            self.audio_segments = []
            self.current_sample_rate = None

        self.clear_ui()  # Call subclass UI clearing
        logger.info("Base session state cleared.")
        # Subclass should handle history, UI messages, and status updates

    def _load_llm_history(self, minutes: int = 30):
        """Helper method to load LLM history from the manager."""
        logger.info(f"Attempting to load LLM history from the last {minutes} minutes...")
        if not hasattr(self.llm, 'load_history') or not callable(self.llm.load_history):
            logger.error("LLM object does not have a callable 'load_history' method.")
            raise AttributeError("LLM object missing required 'load_history' method.")
            
        try:
            raw_history = self.llm.load_history(since_minutes=minutes)
            logger.info(f"Retrieved {len(raw_history)} messages from history.")
            return raw_history
        except Exception as e:
            logger.exception(f"Error loading history via llm.load_history: {e}")
            raise # Re-raise the exception for the caller to handle

    def change_model(self, new_model_requested):
        print(f"Attempting to change model to: {new_model_requested}")
        status_update = ""
        with self.lock:
            resolved_new_alias = self.model_manager.resolve_model_alias(
                new_model_requested
            )
            if not resolved_new_alias:
                error_msg = f"Error: Could not resolve requested model alias '{new_model_requested}'."
                print(error_msg)
                status_update = error_msg
            else:
                print(
                    f"Resolved '{new_model_requested}' to '{resolved_new_alias}'. Initializing..."
                )
                try:
                    new_llm = AskLLM(
                        resolved_model_alias=resolved_new_alias, config=llm_config
                    )
                    self.llm = new_llm
                    self.current_resolved_alias = resolved_new_alias
                    self.current_model = resolved_new_alias
                    status_update = f"Model changed to {resolved_new_alias}. Ready."
                    print(f"Successfully changed model to {resolved_new_alias}.")
                except Exception as e:
                    error_msg = (
                        f"Error initializing AskLLM for {resolved_new_alias}: {e}"
                    )
                    print(error_msg)
                    status_update = error_msg

        return self.update_status(status_update)

    # Override change_voice to use update_status
    def change_voice(self, voice: str):
        super().change_voice(voice)  # Call the TTSBase method
        status = f"Voice changed to {self.current_voice}"
        return self.update_status(status)
