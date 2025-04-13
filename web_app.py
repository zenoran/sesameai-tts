#!/usr/bin/env python3
"""
SesameAI Text-to-Speech Web Interface

This script provides a web interface for interacting with the SesameAI Text-to-Speech model,
allowing users to generate stories with text and audio output, built with Gradio.
"""
import logging
import gradio as gr
import torch
import numpy as np
import re
from ask_llm.utils.config import config
from ask_llm.main import AskLLM
from runme import TTS
import time
import threading
import tempfile
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def clean_text_for_tts(text):
    """
    Clean text to make it suitable for TTS processing by:
    1. Removing markdown formatting
    2. Removing special characters
    3. Converting to plain sentences
    """
    text = text.replace("—", "...")
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^\w\s.,!?:;\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?:;-])\1+', r'\1', text)
    text = re.sub(r'([.,!?:;-])([\w])', r'\1 \2', text)
    
    return text.strip()

class StorytellerApp:
    def __init__(self):
        # LLM Setup
        config.SYSTEM_MESSAGE = """
        You are a storyteller. You paint vivid images in the reader's mind. You are a master of description and detail.
        Your response will be spoken via a text-to-speech system, so you should only include words to be spoken in your response. Do not use any emojis or annotations. Do not use parentheticals or action lines. Remember to only respond with words to be spoken. Write out and normalize text, rather than using abbreviations, numbers, and so on. For example, $2.35 should be two dollars and thirty-five cents, MPH should be miles per hour, and so on. Mathematical formulae should be written out as a human would speak it. Use only standard English alphabet characters [A-Z] along with basic punctuation. Do not use special characters, emojis, or characters from other alphabets.
        Your response should not use quotes to indicate dialogue. Sentences should be complete and stand alone.
        """
        huggingface_model = "PygmalionAI/pygmalion-3-12b"  # Or any other model you want to use
        config.DEFAULT_MODEL = huggingface_model
        config.VERBOSE = True
        
        if huggingface_model not in config.HUGGINGFACE_MODELS:
            config.HUGGINGFACE_MODELS.append(huggingface_model)

        self.llm = AskLLM()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        default_voice = "melina"
        self.tts = TTS(device=device)
        self.tts.load_model()
        self.tts.load_voice(default_voice)

        self.available_voices = self.tts.list_voices()
        self.current_voice = default_voice
        
        # State Variables
        self.sentences = []
        self.current_sentence = ""
        self.current_status = f"Idle. Ready for story or text input. (Voice: {self.current_voice})"
        self.prompt_audio_segments = [] # List of lists: [[seg1, seg2], [seg3], ...]
        self.current_sample_rate = None
        self.generated_prompt_wav_paths = []
        self.generated_full_story_paths = []
        
        self.lock = threading.Lock()
    
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
    
    def process_llm_query(self, query, history, is_continuation=False, temperature=0.8, topk=40):
        """Process query, append to history/sentences. If not continuation, clear first."""
        if not is_continuation:
            self._clear_internal_state()
            history = [] # Start new history for Gradio component

        history.append((query, ""))
        
        print(f"Processing query: {query} (Continuation: {is_continuation}, temp={temperature}, topk={topk})")
        self.current_status = f"Querying LLM (temp={temperature:.2f}, topk={topk})..."
        yield history, self.current_status, 0, 0, False, None, self.generated_prompt_wav_paths

        try:
            response = self.llm.query(query, plaintext_output=True)
            history[-1] = (query, response)
        except Exception as e:
            print(f"LLM Query failed: {e}")
            error_msg = f"Error during LLM query: {e}"
            history[-1] = (query, f"ERROR: {error_msg}")
            self.current_status = error_msg
            yield history, self.current_status, 0, 0, False, None, self.generated_prompt_wav_paths
            return
        
        cleaned_response = clean_text_for_tts(response)
        
        # Split into sentences
        split_pattern = r"([.!?])(\s+|$)"
        parts = re.split(split_pattern, cleaned_response)
        
        new_sentences_raw = []
        current_sentence = ""
        for i in range(0, len(parts), 3):
            part = parts[i]
            delimiter = parts[i+1] if i + 1 < len(parts) else ""
            
            if part: # Skip empty parts resulting from split
                current_sentence += part + delimiter
                if delimiter: # End of a sentence found
                     if current_sentence.strip():
                         new_sentences_raw.append(current_sentence.strip())
                     current_sentence = "" # Reset for next sentence
            
        if current_sentence.strip():
            new_sentences_raw.append(current_sentence.strip())
            
        new_sentences = [s for s in new_sentences_raw if s]
        print(f"Split LLM response into {len(new_sentences)} sentences:")
        for idx, sent in enumerate(new_sentences):
            print(f"  {idx+1}: {sent[:80]}{'...' if len(sent) > 80 else ''}")

        start_idx_of_new = 0
        end_idx_of_new = 0
        with self.lock:
            start_idx_of_new = len(self.sentences)
            self.sentences.extend(new_sentences)
            end_idx_of_new = len(self.sentences) # End index for the generator loop
            total_len = len(self.sentences)
            
            if not is_continuation or not self.prompt_audio_segments:
                 self.prompt_audio_segments.append([])
            elif new_sentences: # Only add if new sentences were actually added for continuation
                 self.prompt_audio_segments.append([])
                 
            print(f"Added new prompt group. Total groups: {len(self.prompt_audio_segments)}")
            
            llm_status = "Processing prompt with LLM..."
            self.current_status = llm_status
            
            if not new_sentences:
                self.current_status = "LLM responded, but no new sentences found."
            elif is_continuation:
                 self.current_status = f"Added {len(new_sentences)} sentences. Total: {total_len}. Resuming..."
            else:
                 self.current_status = f"Processing 1/{total_len} sentences..."
        
        print(f"Appended {len(new_sentences)} sentences. Total now: {total_len}")
        
        # Get initial audio from previous segments (for continuation)
        initial_audio_to_send = None
        if is_continuation:
            with self.lock:
                if self.prompt_audio_segments and self.current_sample_rate:
                    try:
                        # Use all segments except the last empty one we just added
                        all_segments = [seg for prompt_list in self.prompt_audio_segments[:-1] for seg in prompt_list]
                        if all_segments:
                            combined_seg = all_segments[0]
                            for seg in all_segments[1:]:
                                combined_seg += seg
                            
                            initial_np = np.array(combined_seg.get_array_of_samples())
                            if initial_np.dtype == np.int16:
                                initial_np = initial_np.astype(np.float32) / 32768.0
                            elif initial_np.dtype != np.float32:
                                max_val = np.iinfo(initial_np.dtype).max
                                initial_np = initial_np.astype(np.float32) / max_val
                                
                            initial_audio_to_send = (self.current_sample_rate, initial_np)
                            print(f"Sending initial combined audio for continuation ({len(all_segments)} segments, {combined_seg.duration_seconds:.2f}s)")
                    except Exception as e:
                        print(f"Error combining initial audio for continuation: {e}")
                        initial_audio_to_send = None
        
        yield history, self.current_status, start_idx_of_new, end_idx_of_new, True, initial_audio_to_send, self.generated_prompt_wav_paths

    def process_pasted_text(self, pasted_text, temperature=0.8, topk=40, is_continuation=False):
        """Process pasted text directly through TTS without LLM."""
        if not pasted_text:
            print("Empty text pasted, doing nothing.")
            yield self.current_status, 0, 0, False, None, self.generated_prompt_wav_paths
            return
            
        print(f"Processing pasted text with temp={temperature}, topk={topk}: '{pasted_text[:50]}...'")
        self.current_status = f"Processing pasted text (temp={temperature:.2f}, topk={topk})..."
        yield self.current_status, 0, 0, False, None, self.generated_prompt_wav_paths
        
        # If not continuation, clear state first
        if not is_continuation:
            self._clear_internal_state()
        
        cleaned_text = clean_text_for_tts(pasted_text)
        
        # Split into sentences using the same logic as for LLM response
        split_pattern = r"([.!?])(\s+|$)"
        parts = re.split(split_pattern, cleaned_text)
        
        new_sentences_raw = []
        current_sentence = ""
        for i in range(0, len(parts), 3):
            part = parts[i]
            delimiter = parts[i+1] if i + 1 < len(parts) else ""
            
            if part: # Skip empty parts resulting from split
                current_sentence += part + delimiter
                if delimiter: # End of a sentence found
                     if current_sentence.strip():
                         new_sentences_raw.append(current_sentence.strip())
                     current_sentence = "" # Reset for next sentence
            
        if current_sentence.strip():
            new_sentences_raw.append(current_sentence.strip())
            
        new_sentences = [s for s in new_sentences_raw if s]
        print(f"Split pasted text into {len(new_sentences)} sentences:")
        for idx, sent in enumerate(new_sentences):
            print(f"  {idx+1}: {sent[:80]}{'...' if len(sent) > 80 else ''}")
            
        start_idx_of_new = 0
        end_idx_of_new = 0
        with self.lock:
            start_idx_of_new = len(self.sentences)
            self.sentences.extend(new_sentences)
            end_idx_of_new = len(self.sentences) # End index for the generator loop
            total_len = len(self.sentences)
            
            # Always add a new prompt group for the new pasted text
            self.prompt_audio_segments.append([])
                 
            print(f"Added new prompt group for pasted text. Total groups: {len(self.prompt_audio_segments)}")
            
            if not new_sentences:
                self.current_status = "No sentences found in pasted text after cleaning."
            else:
                 self.current_status = f"Processing 1/{len(new_sentences)} sentences from pasted text..."
        
        print(f"Appended {len(new_sentences)} sentences from pasted text. Total now: {total_len}")
        
        # Get initial audio from previous segments (for continuity)
        initial_audio_to_send = None
        with self.lock:
            if self.prompt_audio_segments and len(self.prompt_audio_segments) > 1 and self.current_sample_rate:
                try:
                    # Use all segments except the last empty one we just added
                    all_segments = [seg for prompt_list in self.prompt_audio_segments[:-1] for seg in prompt_list]
                    if all_segments:
                        combined_seg = all_segments[0]
                        for seg in all_segments[1:]:
                            combined_seg += seg
                        
                        initial_np = np.array(combined_seg.get_array_of_samples())
                        if initial_np.dtype == np.int16:
                            initial_np = initial_np.astype(np.float32) / 32768.0
                        elif initial_np.dtype != np.float32:
                            max_val = np.iinfo(initial_np.dtype).max
                            initial_np = initial_np.astype(np.float32) / max_val
                            
                        initial_audio_to_send = (self.current_sample_rate, initial_np)
                        print(f"Sending initial combined audio for pasted text ({len(all_segments)} segments)")
                except Exception as e:
                    print(f"Error combining initial audio for pasted text: {e}")
                    initial_audio_to_send = None
        
        yield self.current_status, start_idx_of_new, end_idx_of_new, True, initial_audio_to_send, self.generated_prompt_wav_paths
    
    def generate_audio_for_sentence_index(self, sentence_index, temperature=0.8, topk=40):
        """Generate audio for a specific sentence index and RETURN data for Gradio."""
        audio_data = None # Default to None
        with self.lock:
            total_sentences = len(self.sentences)
            if sentence_index >= total_sentences:
                 status = self.current_status
                 print(f"generate_audio: Index {sentence_index} out of bounds ({total_sentences}). Status: {status}")
                 return status, audio_data # Return status and None audio
            
            sentence = self.sentences[sentence_index]
            self.current_sentence = sentence
            status_update = f"Generating {sentence_index+1}/{total_sentences}: {sentence[:50]}..."
            self.current_status = status_update
            
        try:
            print(f"Generating audio for sentence {sentence_index+1}/{total_sentences}: {sentence}")
            start_time = time.time()
            
            seg = self.tts.generate_audio_segment(
                sentence, 
                fade_duration=50, 
                start_silence_duration=100, 
                end_silence_duration=100,
                temperature=temperature,
                topk=topk
            )
            
            with self.lock:
                if self.current_sample_rate is None:
                    self.current_sample_rate = seg.frame_rate
                # Append to the *last* list in prompt_audio_segments
                if self.prompt_audio_segments:
                    self.prompt_audio_segments[-1].append(seg)
                    # print(f"Appended segment to prompt group {len(self.prompt_audio_segments)}. Group size: {len(self.prompt_audio_segments[-1])}")
                else:
                     print("Warning: prompt_audio_segments is empty, cannot append segment.")
                
            raw_samples = seg.get_array_of_samples()
            audio_np_raw = np.array(raw_samples)
            
            # Normalize to float32 between -1.0 and 1.0
            if audio_np_raw.dtype == np.int16:
                audio_np = audio_np_raw.astype(np.float32) / 32768.0
            elif audio_np_raw.dtype != np.float32: # Handle other potential integer types
                max_val = np.iinfo(audio_np_raw.dtype).max
                audio_np = audio_np_raw.astype(np.float32) / max_val
            else: # Already float32
                audio_np = audio_np_raw

            audio_data = (seg.frame_rate, audio_np)
            duration = seg.duration_seconds if seg else 0
            process_time = time.time() - start_time
            print(f"Generated segment {sentence_index+1}/{total_sentences} ({duration:.2f}s audio in {process_time:.1f}s)")
            
            next_status = f"Processed {sentence_index+1}/{total_sentences} " + \
                         f"({duration:.1f}s audio / {process_time:.1f}s proc)"
            
            with self.lock:
                 total_sentences_now = len(self.sentences) # Re-check in case more were added
                 is_last_currently = (sentence_index == total_sentences_now - 1)
                 if not is_last_currently:
                     next_status += ". Generating next..."
                 else:
                     next_status += ". Reached end of current list."
                 self.current_status = next_status
                 
            return self.current_status, audio_data # Return status AND audio data
                
        except Exception as e:
            error_msg = f"Error generating audio for sentence {sentence_index+1}: {e}"
            print(error_msg)
            with self.lock:
                 total_sentences_in_exc = len(self.sentences)
                 self.current_status = f"Error on sentence {sentence_index+1}/{total_sentences_in_exc}. Skipping."
            return self.current_status, None # Return error status and None audio
    
    def _clear_internal_state(self):
        """Internal method to clear state without returning UI values."""
        print("Clearing internal session state...")
        self.llm.history_manager.clear_history()
        with self.lock:
            self.sentences = []
            self.current_sentence = ""
            # Reset status including current voice
            self.current_status = f"Session cleared. Ready for new story or text input. (Voice: {self.current_voice})"
            self.prompt_audio_segments = []
            self.current_sample_rate = None
            self.generated_prompt_wav_paths = []
            self.generated_full_story_paths = []
            
    def clear_session_for_ui(self):
         """Method called by Gradio button to clear state and return UI defaults."""
         self._clear_internal_state()
         # Return defaults for all relevant UI components
         return (
             [], # chatbot history
             self.current_status, # status_output
             0, # sentence_index_to_process
             False, # processing_active
             None, # streaming_audio_output - set to None to clear audio
             [], # generated_files_output 
             [], # full_story_download_output
             "", # query_input 
             "", # pasted_text_input
         )

    def _save_audio_for_prompt(self, prompt_index):
        """Combines segments for a specific prompt index, saves to WAV, updates state list."""
        print(f"Saving audio for prompt index {prompt_index}...")
        
        with self.lock:
            if prompt_index >= len(self.prompt_audio_segments) or prompt_index < 0:
                print(f"Error: Prompt index {prompt_index} out of bounds.")
                return self.generated_prompt_wav_paths # Return current list on error
            
            segments_to_save = list(self.prompt_audio_segments[prompt_index])
            sample_rate = self.current_sample_rate
        
        if not segments_to_save or sample_rate is None:
            print(f"No segments available or sample rate unknown for prompt {prompt_index+1}.")
            self.current_status = f"Cannot save Prompt {prompt_index+1}: No audio segments generated."
            return self.generated_prompt_wav_paths # Return current list

        print(f"Combining {len(segments_to_save)} segments for prompt {prompt_index+1}...")
        try:
            combined_audio = segments_to_save[0]
            for seg in segments_to_save[1:]:
                combined_audio += seg
        except Exception as e:
            error_msg = f"Error combining audio segments for prompt {prompt_index+1}: {e}"
            print(error_msg)
            self.current_status = error_msg
            return self.generated_prompt_wav_paths # Return current list on error

        output_path = None
        try:
            # Create a temp directory for organized storage
            temp_dir = os.path.join(tempfile.gettempdir(), "storyteller_audio")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Use temp dir and consistent naming
            num_digits = max(2, len(str(len(self.prompt_audio_segments))))
            filename = f"prompt_{prompt_index+1:0{num_digits}d}.wav"
            output_path = os.path.join(temp_dir, filename)
            
            print(f"Exporting combined prompt audio to: {output_path}")
            combined_audio.export(output_path, format="wav")
            
            success_msg = f"Prompt {prompt_index+1} audio saved ({combined_audio.duration_seconds:.2f}s)."
            print(success_msg)
            # Avoid duplicates
            if output_path not in self.generated_prompt_wav_paths:
                self.generated_prompt_wav_paths.append(output_path)
            self.current_status = success_msg
            return self.generated_prompt_wav_paths
            
        except Exception as e:
            error_msg = f"Error exporting prompt {prompt_index+1} audio to WAV: {e}"
            print(error_msg)
            self.current_status = error_msg
            # Clean up partial file
            if output_path and os.path.exists(output_path):
                 try:
                     os.remove(output_path)
                     print(f"Cleaned up partially created file: {output_path}")
                 except OSError as rm_err:
                     print(f"Error removing partial file {output_path}: {rm_err}")
            return self.generated_prompt_wav_paths # Return current list on error
            
    def generate_and_save_full_story(self):
        """Combines ALL stored audio segments and saves to a temporary WAV file."""
        print("Starting full story audio generation for download...")
        self.current_status = "Combining all segments for full story..."
        yield list(self.generated_full_story_paths), self.current_status # Initial update

        all_segments = []
        sample_rate = None
        with self.lock:
            # Flatten the list of lists
            all_segments = [seg for prompt_list in self.prompt_audio_segments for seg in prompt_list]
            sample_rate = self.current_sample_rate
        
        if not all_segments or sample_rate is None:
            print("No segments available or sample rate unknown for full story.")
            self.current_status = "No audio generated yet to create a full story file."
            yield list(self.generated_full_story_paths), self.current_status
            return

        print(f"Combining {len(all_segments)} audio segments for full story...")
        try:
            combined_audio = all_segments[0]
            for seg in all_segments[1:]:
                combined_audio += seg
        except Exception as e:
            error_msg = f"Error combining audio segments for full story: {e}"
            print(error_msg)
            self.current_status = error_msg
            yield list(self.generated_full_story_paths), self.current_status
            return

        output_path = None
        try:
            # Create a temp directory for organized storage
            temp_dir = os.path.join(tempfile.gettempdir(), "storyteller_audio")
            os.makedirs(temp_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"full_story_{timestamp}.wav"
            output_path = os.path.join(temp_dir, filename)

            print(f"Exporting full story audio to: {output_path}")
            combined_audio.export(output_path, format="wav")
            
            success_msg = f"Full story audio saved ({combined_audio.duration_seconds:.2f}s)."
            print(success_msg)
            self.generated_full_story_paths.append(output_path)
            self.current_status = success_msg
            yield self.generated_full_story_paths, self.current_status 
            
        except Exception as e:
            error_msg = f"Error exporting full story audio to WAV: {e}"
            print(error_msg)
            self.current_status = error_msg
            # Clean up partial file
            if output_path and os.path.exists(output_path):
                 try:
                     os.remove(output_path)
                     print(f"Cleaned up partially created file: {output_path}")
                 except OSError as rm_err:
                     print(f"Error removing partial file {output_path}: {rm_err}")
            yield list(self.generated_full_story_paths), self.current_status


# --- Gradio UI ---
def main():
    storyteller = StorytellerApp()

    with gr.Blocks(title="Storyteller TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 Storyteller TTS")
        
        # --- Shared Components (Outside Tabs) ---
        with gr.Row():
            status_output = gr.Textbox(label="Status", lines=1, interactive=False, value=storyteller.current_status, scale=8)
            reset_btn = gr.Button("Reset Session", variant="stop", scale=1, min_width=150)

        with gr.Row():
             voice_selector = gr.Radio(
                 label="Select Voice",
                 choices=storyteller.available_voices,
                 value=storyteller.current_voice,
                 interactive=True,
                 scale=4
             )
             streaming_audio_output = gr.Audio(
                 label="Narration Stream",
                 autoplay=True,
                 streaming=True,    
                 show_label=True,
                 show_download_button=False, 
                 interactive=False, 
                 elem_id="narration_audio",
                 value=None,  # Start with no audio
                 scale=6 # Adjust scale
             )

        # Add TTS generation parameters
        with gr.Row():
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.05, value=0.8,
                label="Temperature (Creativity)", 
                info="Lower = more predictable, Higher = more creative",
                scale=3
            )
            topk_slider = gr.Slider(
                minimum=10, maximum=100, step=5, value=40,
                label="Top-K (Variety)",
                info="Lower = more focused, Higher = more varied options",
                scale=3
            )

        # State variables
        sentence_index_to_process = gr.State(value=0)
        sentence_end_index = gr.State(value=0)
        processing_active = gr.State(value=False)

        # --- Tabs for Input Modes ---
        with gr.Tabs():
            # --- LLM Story Tab ---
            with gr.TabItem("LLM Story Generation"):
                query_input = gr.Textbox(
                    placeholder="Start or continue a story...",
                    label="LLM Prompt", lines=1, show_label=True)
                
                with gr.Row():
                    generate_btn = gr.Button("✨ Start New Story", variant="primary", size="sm")
                    continue_btn = gr.Button("➡️ Continue Story", variant="secondary", size="sm")

                chatbot = gr.Chatbot(label="LLM Conversation", height=300) 

            # --- Pasted Text Tab ---
            with gr.TabItem("Pasted Text Input"):
                pasted_text_input = gr.Textbox(label="Paste Text Here", lines=10)
                
                with gr.Row():
                    process_text_btn = gr.Button("✨ Start New Text", variant="primary", size="sm")
                    continue_text_btn = gr.Button("➡️ Append Text", variant="secondary", size="sm")


        # --- Shared Output Components (Below Tabs) ---
        with gr.Row():
            generated_files_output = gr.File(
                label="Segment Audio Files (WAV)", 
                file_count="multiple", 
                interactive=False,
                height=100,
            )

        with gr.Row():
            download_full_story_btn = gr.Button("💾 Generate & Download Full Story (WAV)", scale=1)
            full_story_download_output = gr.File(
                label="Full Story Download(s)", 
                file_count="multiple", 
                interactive=False,
                scale=3,
                height=40
            )

        # --- Event Handlers ---

        # Voice Selector Change Handler
        voice_selector.change(
            fn=storyteller.change_voice,
            inputs=[voice_selector],
            outputs=[status_output] # Update status on voice change
        )

        def sentence_generator_loop(start_index, end_index, active, temperature, topk):
            """Generator that processes sentences in sequence and yields audio."""
            if not active:
                print("Generator triggered but not active.")
                # Yield 5 values: status, index, active, audio, files
                yield storyteller.current_status, start_index, False, None, storyteller.generated_prompt_wav_paths
                return

            print(f"Generator starting loop from index: {start_index} up to {end_index} (temp={temperature}, topk={topk})") 
            current_index = start_index
            prompt_index_to_save = -1
            
            # Get initial prompt group index - always target the last one added
            with storyteller.lock:
                if storyteller.prompt_audio_segments:
                    prompt_index_to_save = len(storyteller.prompt_audio_segments) - 1
                    print(f"Generator will target prompt group {prompt_index_to_save+1}")

            while True:
                with storyteller.lock:
                    total_sentences = len(storyteller.sentences)
                    is_within_bounds = current_index < total_sentences

                if not active or not is_within_bounds:
                    # We're done - determine final status and save audio file
                    final_status = storyteller.current_status # Default to current status
                    final_paths = storyteller.generated_prompt_wav_paths # Default to current paths
                    
                    # Only save if we reached the end naturally (not cancelled)
                    if active and not is_within_bounds and prompt_index_to_save >= 0:
                        print(f"Generator loop for prompt {prompt_index_to_save+1} finished. Saving audio...")
                        final_paths = storyteller._save_audio_for_prompt(prompt_index_to_save)
                        final_status = storyteller.current_status # Use updated status after save
                        # Add completion message if appropriate
                        if "Error" not in final_status and "saved" in final_status:
                            final_status += " Ready for next input."
                    elif not active:
                        final_status = "Processing stopped by user (Reset)."
                    elif "Error" not in final_status and "Reached end" not in final_status:
                        final_status += " Processing complete."

                    print(f"Generator loop finished. Final status: {final_status}")
                    # Yield final 5 values
                    yield final_status, current_index, False, None, final_paths
                    
                    # Show a toast notification for completion
                    if "Error" in final_status:
                        gr.Warning(f"TTS Processing ended with error: {final_status}")
                    elif active and not is_within_bounds:
                        gr.Info("TTS Processing Complete!")
                    return

                # Process the current sentence
                status, audio_tuple = storyteller.generate_audio_for_sentence_index(current_index, temperature, topk)
                next_index = current_index + 1
                
                # Did we get an error? If so, might need to stop
                if "Error" in status and audio_tuple is None:
                    print(f"Error occurred processing sentence {current_index+1}. Will stop.")
                    yield status, next_index, False, None, storyteller.generated_prompt_wav_paths
                    gr.Warning(f"TTS Processing error: {status}")
                    return

                # Yield status update and continue
                yield status, next_index, active, audio_tuple, storyteller.generated_prompt_wav_paths
                current_index = next_index
                time.sleep(0.05)

        # --- Basic LLM button handlers ---

        # Generate New Story button
        generate_btn.click(
            fn=storyteller.process_llm_query,
            inputs=[query_input, chatbot, gr.State(value=False), temperature_slider, topk_slider], # is_continuation=False
            outputs=[chatbot, status_output, sentence_index_to_process, sentence_end_index, 
                     processing_active, streaming_audio_output, generated_files_output]
        ).then(
            fn=lambda: "", # Clear query input
            outputs=[query_input]
        ).then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, sentence_end_index, processing_active, 
                    temperature_slider, topk_slider],
            outputs=[status_output, sentence_index_to_process, processing_active, 
                     streaming_audio_output, generated_files_output]
        )

        # Continue Story button
        continue_btn.click(
            fn=storyteller.process_llm_query,
            inputs=[query_input, chatbot, gr.State(value=True), temperature_slider, topk_slider], # is_continuation=True 
            outputs=[chatbot, status_output, sentence_index_to_process, sentence_end_index, 
                     processing_active, streaming_audio_output, generated_files_output]
        ).then(
            fn=lambda: "", # Clear query input
            outputs=[query_input]
        ).then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, sentence_end_index, processing_active,
                    temperature_slider, topk_slider],
            outputs=[status_output, sentence_index_to_process, processing_active, 
                     streaming_audio_output, generated_files_output]
        )
        
        # Enter key in query input - acts like "New Story"
        query_input.submit(
            fn=storyteller.process_llm_query,
            inputs=[query_input, chatbot, gr.State(value=False), temperature_slider, topk_slider], # is_continuation=False
            outputs=[chatbot, status_output, sentence_index_to_process, sentence_end_index, 
                     processing_active, streaming_audio_output, generated_files_output]
        ).then(
            fn=lambda: "", # Clear query input
            outputs=[query_input]
        ).then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, sentence_end_index, processing_active,
                    temperature_slider, topk_slider],
            outputs=[status_output, sentence_index_to_process, processing_active, 
                     streaming_audio_output, generated_files_output]
        )

        # Pasted Text button
        process_text_btn.click(
            fn=storyteller.process_pasted_text,
            inputs=[pasted_text_input, temperature_slider, topk_slider, gr.State(value=False)], # is_continuation=False
            outputs=[status_output, sentence_index_to_process, sentence_end_index, 
                     processing_active, streaming_audio_output, generated_files_output]
        ).then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, sentence_end_index, processing_active,
                    temperature_slider, topk_slider],
            outputs=[status_output, sentence_index_to_process, processing_active, 
                     streaming_audio_output, generated_files_output]
        )
        # Note: intentionally NOT clearing pasted_text_input

        # Continue Text button for pasted text
        continue_text_btn.click(
            fn=storyteller.process_pasted_text,
            inputs=[pasted_text_input, temperature_slider, topk_slider, gr.State(value=True)], # is_continuation=True
            outputs=[status_output, sentence_index_to_process, sentence_end_index, 
                     processing_active, streaming_audio_output, generated_files_output]
        ).then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, sentence_end_index, processing_active,
                    temperature_slider, topk_slider],
            outputs=[status_output, sentence_index_to_process, processing_active, 
                     streaming_audio_output, generated_files_output]
        )

        # Reset Button with custom handler for audio cleanup
        def reset_handler():
            # First call storyteller's clear session
            result = storyteller.clear_session_for_ui()
            # Create a completely new empty audio output to replace the old one
            # This forces Gradio to completely reset the audio component
            return result

        reset_btn.click(
            fn=reset_handler, 
            outputs=[chatbot, status_output, sentence_index_to_process, processing_active, 
                     streaming_audio_output, generated_files_output, full_story_download_output,
                     query_input, pasted_text_input],
        )
        
        # Full Story Download Button
        download_full_story_btn.click(
            fn=storyteller.generate_and_save_full_story,
            inputs=[], # No inputs needed, uses internal state
            outputs=[full_story_download_output, status_output] # Output file path list and status
        )
    
    # Clean up old temp files on start
    temp_dir = os.path.join(tempfile.gettempdir(), "storyteller_audio")
    if os.path.exists(temp_dir):
        print(f"Cleaning temp directory: {temp_dir}")
        for filename in os.listdir(temp_dir):
            if filename.endswith(".wav"):
                 try:
                      os.remove(os.path.join(temp_dir, filename))
                 except OSError:
                      pass # Ignore if file is locked or doesn't exist
                      
    demo.queue().launch(server_name="0.0.0.0") # Use queue for handling multiple clicks

if __name__ == "__main__":
    main() 