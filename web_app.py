#!/usr/bin/env python3
"""
SesameAI Text-to-Speech Web Interface

This script provides a web interface for interacting with the SesameAI Text-to-Speech model,
allowing users to generate stories with text and audio output, built with Gradio.
"""
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
    text = re.sub(r'[^\w\s.,!?:;-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?:;-])\1+', r'\1', text)
    text = re.sub(r'([.,!?:;-])(\w)', r'\1 \2', text)
    
    return text.strip()

class StorytellerApp:
    def __init__(self):
        config.SYSTEM_MESSAGE = """
        You are a storyteller.
        You will respond with clear and concise sentences without any formatting.
        Don't use any special characters or quotes, just alphabet characters and punctuation to designate pauses and flowing sentences.
        """
        huggingface_model = "PygmalionAI/pygmalion-3-12b"  # Or any other model you want to use
        config.DEFAULT_MODEL = huggingface_model
        
        if huggingface_model not in config.HUGGINGFACE_MODELS:
            config.HUGGINGFACE_MODELS.append(huggingface_model)


        self.llm = AskLLM()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(device=device)
        self.tts.load_model()
        self.tts.generate_audio_segment("Warming up!")
        
        
        self.sentences = []
        self.current_sentence = ""
        self.current_status = "Idle."
        self.prompt_audio_segments = [] # List of lists: [[seg1, seg2], [seg3], ...]
        self.current_sample_rate = None
        self.generated_prompt_wav_paths = []
        self.generated_full_story_paths = []
        
        self.lock = threading.Lock()
        
    
    def process_query(self, query, history, is_continuation=False):
        """Process query, append to history/sentences. If not continuation, clear first."""
        if not is_continuation:
            self._clear_internal_state()
            history = [] # Start new history for Gradio component

        history.append((query, ""))
        
        print(f"Processing query: {query} (Continuation: {is_continuation})")
        response = self.llm.query(query, plaintext_output=True)
        
        cleaned_response = clean_text_for_tts(response)
        
        history[-1] = (query, response)
        
        split_pattern = r"([.!?])(\s+|$)"
        parts = re.split(split_pattern, cleaned_response)
        
        new_sentences_raw = []
        current_sentence = ""
        for i in range(0, len(parts), 3):
            part = parts[i]
            delimiter = parts[i+1] if i + 1 < len(parts) else ""
            whitespace = parts[i+2] if i + 2 < len(parts) else ""
            
            if part: # Skip empty parts resulting from split
                current_sentence += part + delimiter
                if delimiter: # End of a sentence found
                     if current_sentence.strip():
                         new_sentences_raw.append(current_sentence.strip())
                     current_sentence = "" # Reset for next sentence
            
        if current_sentence.strip():
            new_sentences_raw.append(current_sentence.strip())
            
        new_sentences = [s for s in new_sentences_raw if s]

        
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
        return history, llm_status, start_idx_of_new, end_idx_of_new, True, None, self.generated_prompt_wav_paths
    
    def generate_audio(self, sentence_index):
        """Generate audio for a specific sentence index and RETURN data for Gradio."""
        audio_data = None # Default to None
        with self.lock:
            total_sentences = len(self.sentences)
            if sentence_index >= total_sentences:
                 status = self.current_status
                 return status, audio_data # Return status and None audio
            
            sentence = self.sentences[sentence_index]
            self.current_sentence = sentence
            status_update = f"Processing {sentence_index+1}/{total_sentences}: {sentence[:50]}..."
            self.current_status = status_update
            
        try:
            print(f"Generating audio for sentence {sentence_index+1}/{total_sentences}: {sentence}")
            start_time = time.time()
            
            seg = self.tts.generate_audio_segment(
                sentence, 
                fade_duration=50, 
                start_silence_duration=500 if sentence_index == 0 else 100,
                end_silence_duration=100
            )
            
            with self.lock:
                if self.current_sample_rate is None:
                    self.current_sample_rate = seg.frame_rate
                if not isinstance(self.prompt_audio_segments, list):
                    self.prompt_audio_segments = []
                if self.prompt_audio_segments:
                    self.prompt_audio_segments[-1].append(seg)
                    print(f"Appended segment to prompt group {len(self.prompt_audio_segments)}. Group size: {len(self.prompt_audio_segments[-1])}")
                else:
                     print("Warning: prompt_audio_segments is empty, cannot append segment.")
                
            raw_samples = seg.get_array_of_samples()
            audio_np_raw = np.array(raw_samples)
            
            if audio_np_raw.dtype == np.int16:
                audio_np = audio_np_raw.astype(np.float32) / 32768.0
            elif audio_np_raw.dtype != np.float32: # Handle other potential types if necessary
                max_val = np.iinfo(audio_np_raw.dtype).max
                audio_np = audio_np_raw.astype(np.float32) / max_val
            else: # Already float32
                audio_np = audio_np_raw

            audio_data = (seg.frame_rate, audio_np)
            print(f"Generated segment {sentence_index+1}/{total_sentences} ({seg.duration_seconds:.2f}s)")
            
            duration = seg.duration_seconds if seg else 0
            process_time = time.time() - start_time
            next_status = f"Processed {sentence_index+1}/{total_sentences} " + \
                         f"({duration:.1f}s audio generated in {process_time:.1f}s)"
            
            with self.lock:
                is_last_currently = (sentence_index == len(self.sentences) - 1)
                if not is_last_currently:
                    next_status += f". Processing next..."
                else:
                    next_status += f". Reached end of current list."
                self.current_status = next_status
                
            return self.current_status, audio_data # Return status AND audio data
                
        except Exception as e:
            error_msg = f"Error generating audio for sentence {sentence_index+1}: {e}"
            print(error_msg)
            with self.lock:
                 total_sentences_in_exc = len(self.sentences)
                 self.current_status = f"Error processing sentence {sentence_index+1}/{total_sentences_in_exc}. Skipping."
            return self.current_status, None # Return error status and None audio
    
    def _clear_internal_state(self):
        """Internal method to clear state without returning UI values."""
        print("Clearing internal session state...")
        self.llm.history_manager.clear_history()
        with self.lock:
            self.sentences = []
            self.current_sentence = ""
            self.current_status = "Session cleared. Ready for new story."
            self.prompt_audio_segments = []
            self.current_sample_rate = None
            self.generated_prompt_wav_paths = []
            self.generated_full_story_paths = []
            
    def clear_session_for_ui(self):
         """Method called by Gradio button to clear state and return UI defaults."""
         self._clear_internal_state()
         return [], "Session cleared. Ready for new story.", 0, False, None, None, None

    def _save_audio_for_prompt(self, prompt_index):
        """Combines segments for a specific prompt, saves to WAV, updates state list."""
        print(f"Saving audio for prompt index {prompt_index}...")
        
        with self.lock:
            if prompt_index >= len(self.prompt_audio_segments):
                print(f"Error: Prompt index {prompt_index} out of bounds.")
                return self.generated_prompt_wav_paths # Return current list on error
            
            segments_to_save = list(self.prompt_audio_segments[prompt_index])
            sample_rate = self.current_sample_rate
        
        if not segments_to_save or sample_rate is None:
            print("No segments available or sample rate unknown for this prompt.")
            self.current_status = f"Cannot save audio for prompt {prompt_index+1}: No segments generated."
            return self.generated_prompt_wav_paths # Return current list

        print(f"Combining {len(segments_to_save)} segments for prompt {prompt_index+1}...")
        try:
            combined_audio = segments_to_save[0]
            for seg in segments_to_save[1:]:
                combined_audio += seg
        except Exception as e:
            error_msg = f"Error combining audio segments: {e}"
            print(error_msg)
            self.current_status = error_msg
            return self.generated_prompt_wav_paths # Return current list on error

        try:
            num_digits = len(str(len(self.prompt_audio_segments))) # For padding
            filename = f"prompt_{prompt_index+1:0{num_digits}d}.wav"
            output_path = tempfile.mktemp(suffix=filename) # Gets a unique path
            print(f"Exporting combined audio to temporary file: {output_path}")
            combined_audio.export(output_path, format="wav")
            
            success_msg = f"Prompt {prompt_index+1} audio saved ({combined_audio.duration_seconds:.2f}s)."
            print(success_msg)
            self.generated_prompt_wav_paths.append(output_path)
            self.current_status = success_msg
            return self.generated_prompt_wav_paths
            
        except Exception as e:
            error_msg = f"Error exporting combined audio to WAV: {e}"
            print(error_msg)
            self.current_status = error_msg
            if 'output_path' in locals() and os.path.exists(output_path):
                 os.remove(output_path)
            return self.generated_prompt_wav_paths # Return current list on error
            
    def generate_and_save_full_story(self):
        """Combines ALL stored audio segments and saves to a temporary WAV file."""
        print("Starting full story audio generation for download...")
        self.current_status = "Combining all segments for full story..."
        
        with self.lock:
            all_segments = [seg for prompt_list in self.prompt_audio_segments for seg in prompt_list]
            sample_rate = self.current_sample_rate
        
        if not all_segments or sample_rate is None:
            print("No segments available or sample rate unknown.")
            self.current_status = "No audio generated yet to create a full story file."
            return self.generated_full_story_paths, self.current_status

        print(f"Combining {len(all_segments)} audio segments...")
        try:
            combined_audio = all_segments[0]
            for seg in all_segments[1:]:
                combined_audio += seg
        except Exception as e:
            error_msg = f"Error combining audio segments for full story: {e}"
            print(error_msg)
            self.current_status = error_msg
            return self.generated_full_story_paths, self.current_status

        output_path = None
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"full_story_{timestamp}.wav"
            output_path = tempfile.mktemp(suffix=filename) # Gets a unique path
            print(f"Exporting full story audio to temporary file: {output_path}")
            combined_audio.export(output_path, format="wav")
            
            success_msg = f"Full story audio saved ({combined_audio.duration_seconds:.2f}s)."
            print(success_msg)
            self.generated_full_story_paths.append(output_path)
            self.current_status = success_msg
            return self.generated_full_story_paths, self.current_status 
            
        except Exception as e:
            error_msg = f"Error exporting full story audio to WAV: {e}"
            print(error_msg)
            self.current_status = error_msg
            if output_path and os.path.exists(output_path):
                 os.remove(output_path)
            return self.generated_full_story_paths, self.current_status

def main():
    storyteller = StorytellerApp()

    with gr.Blocks(title="Storyteller", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 Storyteller")

        with gr.Row():
            with gr.Column(scale=6):
                query_input = gr.Textbox(
                    placeholder="Tell me a story about... or continue with...",
                    label="Your prompt", lines=1, show_label=False)
            with gr.Column(scale=1, min_width=100):
                generate_btn = gr.Button("New", variant="primary")
            with gr.Column(scale=1, min_width=100):
                continue_btn = gr.Button("Continue", variant="secondary")
            with gr.Column(scale=1, min_width=150):
                 reset_btn = gr.Button("Reset", variant="stop")

        with gr.Row():
             status_output = gr.Textbox(label="Status", lines=1, interactive=False, value="Idle.")

        with gr.Row():
             streaming_audio_output = gr.Audio(
                 label="Narration Stream", 
                 autoplay=True,
                 streaming=True,    
                 show_label=True,
                 show_download_button=False, # Disable default download
                 interactive=False, 
                 elem_id="narration_audio")

        with gr.Row():
             chatbot = gr.Chatbot(height=400)

        with gr.Row():
            generated_files_output = gr.File(
                label="Generated Prompt Audio Files", 
                file_count="multiple", 
                interactive=False,
                height=200 # Adjust height as needed
            )

        with gr.Row():
            with gr.Column(scale=1):
                 download_full_story_btn = gr.Button("Generate & Download Full Story (WAV)")
            with gr.Column(scale=3):
                 full_story_download_output = gr.File(label="Full Story Download(s)", file_count="multiple", interactive=False)

        sentence_index_to_process = gr.State(value=0)
        processing_active = gr.State(value=False)


        def handle_submit(query, history, is_continuation):
            """Handles Generate/Continue: Process query, return initial state AND initial audio."""
            updated_history, status, start_idx, end_idx, activate_loop, initial_audio_to_send, generated_files = storyteller.process_query(query, history, is_continuation)
            
            initial_audio_to_send = None
            with storyteller.lock:
                if storyteller.prompt_audio_segments and storyteller.current_sample_rate:
                    combined_initial_seg = None
                    try:
                        all_initial_segments = [seg for prompt_list in storyteller.prompt_audio_segments for seg in prompt_list]
                        if len(all_initial_segments) > 0:
                             combined_initial_seg = all_initial_segments[0]
                             for seg in all_initial_segments[1:]:
                                  combined_initial_seg += seg
                        
                        if combined_initial_seg:
                             initial_np = np.array(combined_initial_seg.get_array_of_samples())
                             if initial_np.dtype == np.int16:
                                  initial_np = initial_np.astype(np.float32) / 32768.0
                             elif initial_np.dtype != np.float32:
                                  max_val = np.iinfo(initial_np.dtype).max
                                  initial_np = initial_np.astype(np.float32) / max_val
                                  
                             initial_audio_to_send = (storyteller.current_sample_rate, initial_np)
                             print(f"handle_submit: Sending initial combined audio ({len(all_initial_segments)} segments from {len(storyteller.prompt_audio_segments)} prompts, {combined_initial_seg.duration_seconds:.2f}s)")
                        else:
                             print("handle_submit: No existing audio segments to combine and send initially.")
                             initial_audio_to_send = None
                             
                    except Exception as e:
                        print(f"Error combining initial audio in handle_submit: {e}")
                        initial_audio_to_send = None # Avoid sending broken audio
                else:
                     print("handle_submit: No existing audio segments or sample rate for initial send.")
                     initial_audio_to_send = None

            print(f"handle_submit returning: next_gen_idx={start_idx}, activate={activate_loop}")
            return updated_history, status, start_idx, end_idx, activate_loop, initial_audio_to_send, generated_files

        def sentence_generator_loop(start_index, end_index, active):
            """Generator that processes sentences and streams the NEW audio segment for each."""
            if end_index is None:
                with storyteller.lock:
                    end_index = len(storyteller.sentences) # Default to all sentences if end_index not provided
                    print(f"end_index was None, setting to total sentences: {end_index}")

            if not active:
                print("Generator triggered but not active.")
                # Yield 5 values: status, index, active, audio, files
                yield storyteller.current_status, start_index, False, None, storyteller.generated_prompt_wav_paths
                return

            print(f"Generator starting loop from index: {start_index} up to (but not including) {end_index}")
            current_index = start_index
            silence_threshold = 1e-6

            while True:
                with storyteller.lock:
                    total_sentences = len(storyteller.sentences)
                    is_within_bounds = current_index < total_sentences
                    is_active_state = active 

                if not is_within_bounds or not is_active_state:
                    prompt_index_to_save = -1
                    with storyteller.lock:
                        if storyteller.prompt_audio_segments:
                             storyteller.current_status = f"Finished TTS for prompt {len(storyteller.prompt_audio_segments)}. Saving file..."
                             prompt_index_to_save = len(storyteller.prompt_audio_segments) - 1
                        
                    generated_files = storyteller.generated_prompt_wav_paths # Default to current list
                    if prompt_index_to_save != -1:
                        print(f"Generator loop for prompt {prompt_index_to_save+1} finished. Saving audio...")
                        generated_files = storyteller._save_audio_for_prompt(prompt_index_to_save)
                        print(f"Updated generated files list: {generated_files}")
                    
                    final_status = storyteller.current_status
                    if not is_active_state:
                        final_status = "Processing stopped by user (Clear)."
                    elif "Reached end" not in final_status and "Error" not in final_status:
                        final_status += " Processing complete."

                    print(f"Generator loop finished for batch up to index {end_index}. Final status: {final_status}")
                    # Yield final 5 values
                    yield final_status, current_index, False, None, generated_files
                    return

                status, audio_tuple = storyteller.generate_audio(current_index)
                next_index = min(current_index + 1, total_sentences)
                
                audio_to_yield = None
                if audio_tuple:
                    sr, new_audio_np_float = audio_tuple
                    if np.abs(new_audio_np_float).max() > silence_threshold:
                        audio_to_yield = audio_tuple 
                        print(f"Sentence {current_index+1}: Yielding NEW audio segment ({len(new_audio_np_float)/sr:.2f}s)")
                    else:
                         print(f"Sentence {current_index+1}: Audio generated but below silence threshold. Skipping yield.")
                         audio_to_yield = None # Explicitly set to None if not yielding
                else:
                     print(f"Sentence {current_index+1}: No audio generated.")
                     audio_to_yield = None # Explicitly set to None

                yield status, next_index, True, audio_to_yield, storyteller.generated_prompt_wav_paths

                current_index = next_index
                time.sleep(0.1)

        def show_completion_toast(status):
            if status and "Processing complete." in status:
                gr.Info("TTS processing for the latest prompt is complete!")
            elif status and "Error" in status:
                 gr.Warning(f"TTS processing finished with an error: {status}")
            return None


        generate_event = generate_btn.click(
            fn=handle_submit,
            inputs=[query_input, chatbot, gr.State(value=False)], 
            outputs=[chatbot, status_output, sentence_index_to_process, gr.State(), processing_active, streaming_audio_output, generated_files_output],
        ).then(fn=lambda: "", outputs=[query_input])

        generate_event.then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, gr.State(), processing_active],
            outputs=[status_output, sentence_index_to_process, processing_active, streaming_audio_output, generated_files_output]
        ).then(
            fn=show_completion_toast,
            inputs=[status_output], # Input is the final status from the generator
            outputs=None # No output, just side effect
        )

        continue_event = continue_btn.click(
            fn=handle_submit,
            inputs=[query_input, chatbot, gr.State(value=True)], 
            outputs=[chatbot, status_output, sentence_index_to_process, gr.State(), processing_active, streaming_audio_output, generated_files_output],
        ).then(fn=lambda: "", outputs=[query_input])

        continue_event.then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, gr.State(), processing_active],
            outputs=[status_output, sentence_index_to_process, processing_active, streaming_audio_output, generated_files_output]
        ).then(
            fn=show_completion_toast,
            inputs=[status_output], # Input is the final status from the generator
            outputs=None # No output, just side effect
        )

        enter_event = query_input.submit(
            fn=handle_submit,
            inputs=[query_input, chatbot, gr.State(value=False)],
            outputs=[chatbot, status_output, sentence_index_to_process, gr.State(), processing_active, streaming_audio_output, generated_files_output],
        ).then(fn=lambda: "", outputs=[query_input])

        enter_event.then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, gr.State(), processing_active],
            outputs=[status_output, sentence_index_to_process, processing_active, streaming_audio_output, generated_files_output]
        ).then(
            fn=show_completion_toast,
            inputs=[status_output], # Input is the final status from the generator
            outputs=None # No output, just side effect
        )

        reset_btn.click(
            fn=storyteller.clear_session_for_ui, 
            outputs=[chatbot, status_output, sentence_index_to_process, processing_active, streaming_audio_output, generated_files_output, full_story_download_output],
        )
        
        download_full_story_btn.click(
            fn=storyteller.generate_and_save_full_story,
            inputs=[], # No inputs needed, uses internal state
            outputs=[full_story_download_output, status_output] # Output file path list and status
        )
        
    
    demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main() 