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
from pydub import AudioSegment
import time
import threading
import queue

def clean_text_for_tts(text):
    """
    Clean text to make it suitable for TTS processing by:
    1. Removing markdown formatting
    2. Removing special characters
    3. Converting to plain sentences
    """
    # Replace em dashes with ellipses
    text = text.replace("—", "...")
    
    # Remove code blocks (```...```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove inline code (`)
    text = re.sub(r'`[^`]*`', '', text)
    
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove markdown formatting (**, *, __, _, etc.)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?:;-]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'([.,!?:;-])\1+', r'\1', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.,!?:;-])(\w)', r'\1 \2', text)
    
    return text.strip()

class StorytellerApp:
    def __init__(self):
        # Configure the LLM to be a storyteller
        config.SYSTEM_MESSAGE = """
        You are a storyteller.
        You will respond with clear and concise sentences without any formatting.
        Don't use any special characters or quotes, just alphabet characters and punctuation to designate pauses and flowing sentences.
        """
        
        # Initialize the LLM and TTS engines
        self.llm = AskLLM()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(device=device)
        self.tts.load_model()
        self.tts.generate_audio_segment("Warming up!")
        
        # Generate a warmup segment to initialize everything
        # self.tts.generate_audio_segment("Storyteller initialized and ready.")
        
        # State variables
        self.sentences = []
        self.current_sentence = ""
        self.current_status = "Idle."
        # NEW: Store generated audio data (numpy arrays)
        self.segments_audio_np = [] 
        self.current_sample_rate = None
        
        # Lock for thread safety (especially around self.sentences)
        self.lock = threading.Lock()
        
        # Removed playback queue and thread
        # self.audio_queue = queue.Queue()
        # self.stop_event = threading.Event()
        # self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        # self.playback_thread.start()
    
    def process_query(self, query, history, is_continuation=False):
        """Process query, append to history/sentences. If not continuation, clear first."""
        if not is_continuation:
            self._clear_internal_state()
            history = [] # Start new history for Gradio component

        # Record the user's query (appends to current history list)
        history.append((query, ""))
        
        # Get response from LLM (AskLLM maintains its internal history)
        print(f"Processing query: {query} (Continuation: {is_continuation})")
        response = self.llm.query(query)
        
        # Clean the response text for TTS
        cleaned_response = clean_text_for_tts(response)
        
        # Update history display with the complete response
        history[-1] = (query, response)
        
        # Append new sentences to the list
        new_sentences = [s for s in re.split(r"(?<=[.!?])\s+", cleaned_response) if s.strip()]
        
        start_idx_of_new = 0
        with self.lock:
            start_idx_of_new = len(self.sentences)
            self.sentences.extend(new_sentences)
            total_len = len(self.sentences)
            
            if not new_sentences:
                self.current_status = "LLM responded, but no new sentences found."
            elif is_continuation:
                 # Status reflects adding to existing queue
                 self.current_status = f"Added {len(new_sentences)} sentences. Total: {total_len}. Resuming..."
            else:
                 # Status for a new story start
                 self.current_status = f"Processing 1/{total_len} sentences..."
        
        print(f"Appended {len(new_sentences)} sentences. Total now: {total_len}")
        # Return updated history (for Gradio) and status
        return history, self.current_status, start_idx_of_new
    
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
            
            # Convert to numpy array for Gradio playback
            if seg:
                sample_rate = seg.frame_rate
                audio_np = np.array(seg.get_array_of_samples())
                # Convert to float32 normalized between -1 and 1
                audio_np = audio_np.astype(np.float32) / 32768.0
                audio_data = (sample_rate, audio_np)
                print(f"Generated segment {sentence_index+1}/{total_sentences} ({seg.duration_seconds:.2f}s)")
            else:
                 print(f"Segment {sentence_index+1} generation resulted in None.")
            
            # Update status after generation attempt
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
            # NEW: Clear audio state
            self.segments_audio_np = []
            self.current_sample_rate = None
            
    def clear_session_for_ui(self):
         """Method called by Gradio button to clear state and return UI defaults."""
         self._clear_internal_state()
         # Return empty/default values for the UI components bound to the clear button
         # history, status, index, active_flag
         return [], "Session cleared. Ready for new story.", 0, False 

    def generate_full_story_audio(self):
        """Generates and combines audio for the entire story currently stored."""
        print("Starting full story audio generation...")
        audio_segments = []
        sample_rate = None

        with self.lock:
            sentences_to_process = list(self.sentences) # Create a copy
            total_sentences = len(sentences_to_process)
        
        if not sentences_to_process:
            print("No sentences available to generate full story audio.")
            return None # Or return empty audio data?

        for i, sentence in enumerate(sentences_to_process):
            print(f"Generating full story audio for sentence {i+1}/{total_sentences}: {sentence[:50]}...")
            try:
                # Use similar settings as generate_audio, maybe longer silences for full export?
                seg = self.tts.generate_audio_segment(
                    sentence, 
                    fade_duration=50, 
                    start_silence_duration=500 if i == 0 else 200, # Longer pause at start, medium between
                    end_silence_duration=500 if i == total_sentences - 1 else 200 # Longer pause at end
                )
                if seg:
                    audio_segments.append(seg)
                    if sample_rate is None:
                         sample_rate = seg.frame_rate # Get sample rate from first segment
            except Exception as e:
                print(f"Error generating audio for sentence {i+1} during full export: {e}. Skipping.")
                # Optionally add a short silence segment as a placeholder on error
                # if sample_rate: audio_segments.append(AudioSegment.silent(duration=500, frame_rate=sample_rate))
        
        if not audio_segments:
             print("No audio segments were generated for the full story.")
             return None

        # Combine segments
        print("Combining audio segments...")
        combined_audio = audio_segments[0]
        for seg in audio_segments[1:]:
            combined_audio += seg
        
        # Convert to numpy array for Gradio File output
        # Ensure sample rate is set (should be from the first segment)
        if sample_rate is None:
            print("Warning: Sample rate not determined. Cannot format audio for download.")
            return None 
            
        # Convert to float32 normalized between -1 and 1
        audio_np = np.array(combined_audio.get_array_of_samples())
        audio_np = audio_np.astype(np.float32) / 32768.0  # Convert int16 to float32 [-1.0, 1.0]
        print(f"Download audio array: shape={audio_np.shape}, dtype={audio_np.dtype}, min={audio_np.min()}, max={audio_np.max()}")
        print(f"Combined audio ready: {combined_audio.duration_seconds:.2f}s")
        
        # Gradio File component expects (sample_rate, numpy_array)
        return (sample_rate, audio_np)

def main():
    storyteller = StorytellerApp()

    with gr.Blocks(title="Storyteller", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 Interactive Storyteller with AI Voice")
        gr.Markdown("Enter prompts to start or continue a story.")

        # --- Input & Controls Row --- 
        with gr.Row():
            with gr.Column(scale=6):
                query_input = gr.Textbox(
                    placeholder="Tell me a story about... or continue with...",
                    label="Your prompt", lines=1, show_label=False)
            with gr.Column(scale=1, min_width=100):
                generate_btn = gr.Button("Generate New", variant="primary")
            with gr.Column(scale=1, min_width=100):
                continue_btn = gr.Button("Continue", variant="secondary")
            with gr.Column(scale=1, min_width=150):
                 # Moved Clear Button to controls row
                 clear_btn = gr.Button("Start New Story", variant="stop")

        # --- Status Row --- 
        with gr.Row():
             status_output = gr.Textbox(label="Status", lines=1, interactive=False, value="Idle.")

        # --- Narration Audio --- 
        with gr.Row():
             audio_output = gr.Audio(
                 label="Narration", 
                 autoplay=True,
                 streaming=True,    
                 show_label=True,
                 show_download_button=True,
                 interactive=False, 
                 elem_id="narration_audio")

        # --- Chat History --- 
        with gr.Row():
             chatbot = gr.Chatbot(label="Story", height=400)

        # Hidden states remain the same
        sentence_index_to_process = gr.State(value=0)
        processing_active = gr.State(value=False)

        # --- Event Handlers --- 

        def handle_submit(query, history, is_continuation):
            """Handles Generate/Continue: Process query, return initial state AND initial audio."""
            # Process query (clears state if not continuation)
            updated_history, status, start_idx = storyteller.process_query(query, history, is_continuation)
            
            # Prepare initial audio to send immediately
            initial_audio_to_send = None
            with storyteller.lock:
                activate_loop = start_idx < len(storyteller.sentences)
                # Combine existing audio segments if any exist
                if storyteller.segments_audio_np and storyteller.current_sample_rate:
                    if len(storyteller.segments_audio_np) > 0:
                        try:
                            combined_initial = np.concatenate(storyteller.segments_audio_np)
                            initial_audio_to_send = (storyteller.current_sample_rate, combined_initial)
                            print(f"handle_submit: Sending initial combined audio ({len(storyteller.segments_audio_np)} segments)")
                        except Exception as e:
                            print(f"Error combining initial audio in handle_submit: {e}")
                            initial_audio_to_send = None # Avoid sending broken audio
                    else:
                         print("handle_submit: No existing audio segments to send initially.")
                else:
                     print("handle_submit: No existing audio segments or sample rate.")

            print(f"handle_submit returning: next_gen_idx={start_idx}, activate={activate_loop}")
            # Return history, status, STARTING index for generator, activation flag, AND initial audio
            return updated_history, status, start_idx, activate_loop, initial_audio_to_send

        def sentence_generator_loop(start_index, active):
            """Generator that processes sentences and streams the NEW audio segment for each."""
            if not active:
                print("Generator triggered but not active.")
                yield storyteller.current_status, start_index, False, None
                return

            print(f"Generator starting loop from index: {start_index}")
            current_index = start_index
            # Threshold to consider audio silent and avoid Gradio processing errors
            silence_threshold = 1e-6 
            
            while True: 
                with storyteller.lock:
                    total_sentences = len(storyteller.sentences)
                    is_within_bounds = current_index < total_sentences
                    is_active_state = active 

                if not is_within_bounds or not is_active_state:
                    # Generator loop ended 
                    final_status = storyteller.current_status
                    if not is_active_state: 
                        final_status = "Processing stopped by user (Clear)."
                    elif "Reached end" not in final_status and "Error" not in final_status:
                        final_status += " Processing complete."
                        
                    print(f"Generator loop finished. Final status: {final_status}")
                    yield final_status, current_index, False, None # Send final status, no more audio
                    return

                # Process current sentence to get the full audio segment
                status, audio_tuple = storyteller.generate_audio(current_index)
                next_index = min(current_index + 1, total_sentences)
                
                # Store the new audio segment in the app state if valid
                audio_to_yield = None
                if audio_tuple:
                    sr, new_audio_np = audio_tuple
                    if np.abs(new_audio_np).max() > silence_threshold:
                        with storyteller.lock:
                            if storyteller.current_sample_rate is None:
                                storyteller.current_sample_rate = sr
                            # Ensure list exists before appending
                            if not isinstance(storyteller.segments_audio_np, list):
                                 storyteller.segments_audio_np = [] 
                            storyteller.segments_audio_np.append(new_audio_np)
                            print(f"Appended segment {len(storyteller.segments_audio_np)} to state.")
                        
                        # Yield ONLY the NEW audio segment
                        audio_to_yield = audio_tuple 
                        print(f"Sentence {current_index+1}: Yielding NEW audio segment ({len(new_audio_np)/sr:.2f}s)")
                    else:
                         print(f"Sentence {current_index+1}: Audio generated but below silence threshold. Skipping append/yield.")
                else:
                     print(f"Sentence {current_index+1}: No audio generated.")

                # Yield status, index, active flag, and ONLY the NEW audio segment (or None)
                yield status, next_index, True, audio_to_yield
                
                current_index = next_index
                # Add a small delay to allow UI to update, etc.
                time.sleep(0.1)

        # ---- Button Click/Submit Logic ----

        # GENERATE NEW button
        generate_event = generate_btn.click(
            fn=handle_submit,
            inputs=[query_input, chatbot, gr.State(value=False)], 
            # Now ALSO outputs to audio_output initially
            outputs=[chatbot, status_output, sentence_index_to_process, processing_active, audio_output],
        ).then(fn=lambda: "", outputs=[query_input])

        generate_event.then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, processing_active],
            # Generator still outputs its new segments to audio_output
            outputs=[status_output, sentence_index_to_process, processing_active, audio_output] 
        )

        # CONTINUE button
        continue_event = continue_btn.click(
            fn=handle_submit,
            inputs=[query_input, chatbot, gr.State(value=True)], 
            # Now ALSO outputs to audio_output initially
            outputs=[chatbot, status_output, sentence_index_to_process, processing_active, audio_output],
        ).then(fn=lambda: "", outputs=[query_input])

        continue_event.then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, processing_active],
            # Generator still outputs its new segments to audio_output
            outputs=[status_output, sentence_index_to_process, processing_active, audio_output]
        )

        # ENTER key submit (acts like Generate New)
        enter_event = query_input.submit(
            fn=handle_submit,
            inputs=[query_input, chatbot, gr.State(value=False)],
            # Now ALSO outputs to audio_output initially
            outputs=[chatbot, status_output, sentence_index_to_process, processing_active, audio_output],
        ).then(fn=lambda: "", outputs=[query_input])

        enter_event.then(
            fn=sentence_generator_loop,
            inputs=[sentence_index_to_process, processing_active],
            # Generator still outputs its new segments to audio_output
            outputs=[status_output, sentence_index_to_process, processing_active, audio_output]
        )

        # --- Clear Button Logic --- 
        clear_btn.click(
            fn=storyteller.clear_session_for_ui, 
            outputs=[chatbot, status_output, sentence_index_to_process, processing_active],
        ).then(fn=lambda: None, outputs=[audio_output])
        
        # Provide usage instructions
        gr.Markdown("""
        ## How to Use
        1.  **Generate New:** Enter a prompt and click "Generate New" (or press Enter) to start a fresh story.
        2.  **Continue:** Enter a follow-up prompt and click "Continue" to add to the current story.
        3.  Listen as the story is narrated. Status updates show progress.
        4.  **Download:** Use the download button on the audio player to save the current story audio.
        5.  **Start New Story:** Click this button to clear everything and start over.
        """)
    
    # Launch the interface on all interfaces
    demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main() 