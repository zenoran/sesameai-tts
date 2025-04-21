#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import tempfile
import time

import gradio as gr
import numpy as np

from ask_llm.utils.config import config as llm_config
from utils.web_base import WebAppBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class StorytellerApp(WebAppBase):
    def __init__(self, model: str = "dans-personalityengine", voice: str = "maya"):
        super().__init__(model=model, voice=voice)

        llm_config.SYSTEM_MESSAGE = "You are a master storyteller. Create engaging, short stories based on the user's prompt. The first sentence of every response should be more than six words. Do not use any emojis or annotations. Do not use parentheticals or action lines. Remember to only respond with words to be spoken. Write out and normalize text, rather than using abbreviations, numbers, and so on. For example, $2.35 should be two dollars and thirty-five cents, MPH should be miles per hour, and so on. Mathematical formulae should be written out as a human would speak it. Use only standard English alphabet characters [A-Z] along with basic punctuation. Your response should not use quotes to indicate dialogue. Sentences should be complete and stand alone. Respond directly with the story."
        llm_config.VERBOSE = True
        
        self.current_sentence = ""
        self.current_status = f"Idle. Ready for story or text input. (Voice: {self.current_voice})"
        self.prompt_audio_segments = [] # List of lists: [[seg1, seg2], [seg3], ...]
        self.generated_prompt_wav_paths = []
        self.generated_full_story_paths = []
        self.story_text = ""
    
        if not hasattr(self.llm, 'history_manager'):
            raise ValueError("LLM object does not have a required 'history_manager' attribute.")
    
    def _store_audio_segment(self, audio_segment, sentence_index):
        """Store the generated audio segment in the prompt_audio_segments list structure."""
        if self.prompt_audio_segments:
            self.prompt_audio_segments[-1].append(audio_segment)
        else:
            print("Warning: prompt_audio_segments is empty, cannot append segment.")
    
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
        print(f"Response from LLM: {type(self.llm)}")
        
        new_sentences = self.split_text_into_sentences(response)
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
        
        initial_audio_to_send = None
        if is_continuation:
            with self.lock:
                if self.prompt_audio_segments and self.current_sample_rate:
                    try:
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
        
        if not is_continuation:
            self._clear_internal_state()
        
        new_sentences = self.split_text_into_sentences(pasted_text)
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
            
            self.prompt_audio_segments.append([])
                 
            print(f"Added new prompt group for pasted text. Total groups: {len(self.prompt_audio_segments)}")
            
            if not new_sentences:
                self.current_status = "No sentences found in pasted text after cleaning."
            else:
                 self.current_status = f"Processing 1/{len(new_sentences)} sentences from pasted text..."
        
        print(f"Appended {len(new_sentences)} sentences from pasted text. Total now: {total_len}")
        
        initial_audio_to_send = None
        with self.lock:
            if self.prompt_audio_segments and len(self.prompt_audio_segments) > 1 and self.current_sample_rate:
                try:
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
    
    def _clear_internal_state(self, clear_history: bool = True):
        """Internal method to clear state without returning UI values."""
        print(f"Clearing Storyteller internal state (clear_history={clear_history})...")
        
        with self.lock:
            self.sentences = []
            self.current_sentence = ""
            self.prompt_audio_segments = []
            self.generated_prompt_wav_paths = []
            self.generated_full_story_paths = []
            self.story_text = ""

        super().clear_session(clear_history=clear_history)

        self.current_status = f"Session cleared. Ready for new story or text input. (Voice: {self.current_voice})"
        
    def clear_session_for_ui(self):
         """Method called by Gradio button to clear state and return UI defaults."""
         self._clear_internal_state()
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
            temp_dir = os.path.join(tempfile.gettempdir(), "storyteller_audio")
            os.makedirs(temp_dir, exist_ok=True)
            
            num_digits = max(2, len(str(len(self.prompt_audio_segments))))
            filename = f"prompt_{prompt_index+1:0{num_digits}d}.wav"
            output_path = os.path.join(temp_dir, filename)
            
            print(f"Exporting combined prompt audio to: {output_path}")
            combined_audio.export(output_path, format="wav")
            
            success_msg = f"Prompt {prompt_index+1} audio saved ({combined_audio.duration_seconds:.2f}s)."
            print(success_msg)
            if output_path not in self.generated_prompt_wav_paths:
                self.generated_prompt_wav_paths.append(output_path)
            self.current_status = success_msg
            return self.generated_prompt_wav_paths
            
        except Exception as e:
            error_msg = f"Error exporting prompt {prompt_index+1} audio to WAV: {e}"
            print(error_msg)
            self.current_status = error_msg
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
            if output_path and os.path.exists(output_path):
                 try:
                     os.remove(output_path)
                     print(f"Cleaned up partially created file: {output_path}")
                 except OSError as rm_err:
                     print(f"Error removing partial file {output_path}: {rm_err}")
            yield list(self.generated_full_story_paths), self.current_status

    def load_recent_history(self):
        """Loads history from the last 30 minutes, updates UI and internal state (text only)."""
        logger.info("Attempting to load recent history for Storyteller...")
        self._clear_internal_state(clear_history=False) # Clear current state first
        self.current_status = "Loading recent history..."
        yield ([], self.current_status, 0, False, None, [], [], "", "") # Initial update
        
        try:
            raw_history = self._load_llm_history(minutes=30)
            
            if not raw_history or len(raw_history) < 2:
                self.current_status = "No recent history found within the last 30 minutes."
                yield ([], self.current_status, 0, False, None, [], [], "", "")
                return
            print(f"Retrieved {len(raw_history)} messages from the last 30 minutes.")

            chatbot_history = []
            loaded_sentences = []
            loaded_prompt_segments = []

            for i in range(0, len(raw_history) - 1, 2):
                user_msg = raw_history[i]
                assistant_msg = raw_history[i+1]

                if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant':
                    user_content = user_msg['content']
                    assistant_content = assistant_msg['content']
                    chatbot_history.append((user_content, assistant_content))
                    
                    sentences_from_assistant = self.split_text_into_sentences(assistant_content)
                    loaded_sentences.extend(sentences_from_assistant)
                    loaded_prompt_segments.append([]) # Add empty segment list for this interaction
                    print(f"  Loaded interaction: User '{user_content[:50]}...', Assistant '{assistant_content[:50]}...' ({len(sentences_from_assistant)} sentences)")
                else:
                     print(f"Skipping unexpected message sequence at index {i}: {user_msg.get('role')}, {assistant_msg.get('role')}")


            with self.lock:
                self.sentences = loaded_sentences
                self.prompt_audio_segments = loaded_prompt_segments
                self.generated_prompt_wav_paths = [] 
                self.generated_full_story_paths = []

            num_interactions = len(chatbot_history)
            total_loaded_sentences = len(loaded_sentences)
            self.current_status = f"Loaded {num_interactions} interactions ({total_loaded_sentences} sentences) from recent history. Ready to continue."
            print(self.current_status)

            yield (chatbot_history, self.current_status, 0, False, None, [], [], "", "")

        except Exception as e:
            error_msg = f"Error loading history: {e}"
            print(error_msg)
            logger.exception("Error during history loading")
            self.current_status = error_msg
            yield ([], self.current_status, 0, False, None, [], [], "", "")

    def gradio_sentence_generator_wrapper(
        self, start_index, end_index, active, temperature=0.7, speed_factor=1.2, speaker_id_maybe_float: float = 1.0
    ):
        if not active:
            yield (
                self.current_status,
                start_index,
                False,
                None,
                self.generated_prompt_wav_paths
            )
            return

        speaker_id = int(speaker_id_maybe_float) # Cast to int

        generator = self.sentence_generator_loop(
            start_index, end_index, active, temperature, speed_factor, speaker=speaker_id # Pass int speaker ID
        )

        next_idx = start_index
        try:
            while True:
                is_active, audio_tuple = next(generator)
                if not is_active: # Loop finished or stopped
                    break
                next_idx += 1 # We successfully processed one more sentence
                yield self.current_status, next_idx, is_active, audio_tuple, self.generated_prompt_wav_paths
        except StopIteration:
            logger.info("Sentence generator loop finished normally.")
            yield self.current_status, next_idx, False, None, self.generated_prompt_wav_paths
        except Exception as e:
            logger.error(f"Error in sentence generator wrapper: {e}")
            error_status = self.current_status = f"Error during audio generation: {e}"
            yield error_status, next_idx, False, None, self.generated_prompt_wav_paths


# --- Gradio UI ---
def main():

    parser = argparse.ArgumentParser(description="SesameAI Storyteller with TTS")
    parser.add_argument(
        "-m",
        "--model",
        help="Choose the model to use (supports partial matching)",
        default="pygmalion-3-12b", # Keep original default or update? Let's keep original for now
    )
    parser.add_argument(
            "-v",
            "--voice",
            help="Choose the voice to use for TTS",
            default="melina", # Keep original default or update? Let's keep original for now
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    try:
        storyteller = StorytellerApp(model=args.model, voice=args.voice)
    except Exception as e:
        print(f"[Fatal] Failed to initialize StorytellerApp: {e}. Exiting.")
        sys.exit(1)

    if storyteller.llm is None:
        print("[Fatal] LLM could not be initialized. Exiting.")
        sys.exit(1)

    with gr.Blocks(title="Storyteller TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📖 Storyteller TTS") # Use updated title

        with gr.Row():
            status_output = gr.Textbox(label="Status", lines=1, interactive=False, value=storyteller.current_status, scale=8)
            reset_btn = gr.Button("Reset Session", variant="stop", scale=1, min_width=150)
            load_history_btn = gr.Button("Load Recent History", scale=1, min_width=150)

        with gr.Row():
             voice_selector = gr.Radio( # Keep Radio as per original structure
                 label="Select Voice",
                 choices=storyteller.list_available_voices(),
                 value=storyteller.current_voice,
                 interactive=True,
                 scale=4
             )
             streaming_audio_output = gr.Audio(
                 label="Narration Stream", # Keep original label
                 autoplay=True,
                 streaming=True,
                 show_label=True,
                 show_download_button=False, # Keep original setting
                 interactive=False,
                 elem_id="narration_audio",
                 value=None,
                 scale=6
             )

        with gr.Row():
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.1, value=0.9, label="LLM Temperature", # Keep label from CURRENT
                scale=3
            )
            speed_slider = gr.Slider(
                minimum=0.75, maximum=2.0, step=0.05, value=1.0, label="Speech Speed",
                scale=3
            )
            speaker_id_input = gr.Number(
                label="Speaker ID",
                info="Integer ID for the speaker voice.",
                value=1, # Default speaker ID
                minimum=0, # Or appropriate minimum ID
                step=1,    # Ensure integer steps
                interactive=True,
                scale=2 # Adjust scale
            )

        sentence_index_to_process = gr.State(value=0)
        sentence_end_index = gr.State(value=0)
        processing_active = gr.State(value=False)


        with gr.Tabs():
            with gr.TabItem("LLM Story Generation"):
                query_input = gr.Textbox(
                    placeholder="Start or continue a story...",
                    label="LLM Prompt", lines=1, show_label=True)
                with gr.Row():
                    generate_btn = gr.Button("✨ Start New Story", variant="primary", size="sm")
                    continue_btn = gr.Button("➡️ Continue Story", variant="secondary", size="sm")
                chatbot = gr.Chatbot(label="LLM Conversation", height=300)

            with gr.TabItem("Pasted Text Input"):
                pasted_text_input = gr.Textbox(label="Paste Text Here", lines=10)
                with gr.Row():
                    process_text_btn = gr.Button("✨ Start New Text", variant="primary", size="sm")
                    continue_text_btn = gr.Button("➡️ Append Text", variant="secondary", size="sm")
                clear_pasted_btn = gr.Button("Clear Pasted Text", variant="stop", size="sm")


        with gr.Row():
            generated_files_output = gr.File(
                label="Segment Audio Files (WAV)", # Keep original label
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


        voice_selector.change(
            fn=storyteller.change_voice,
            inputs=[voice_selector],
            outputs=[status_output]
        )

        process_llm_outputs = [
             chatbot, # LLM History
             status_output,
             sentence_index_to_process,
             sentence_end_index,
             processing_active,
             streaming_audio_output, # Initial audio for continuation
             generated_files_output # Pass through existing files
        ]
        process_pasted_outputs = [
             status_output,
             sentence_index_to_process,
             sentence_end_index,
             processing_active,
             streaming_audio_output, # Initial audio for continuation
             generated_files_output # Pass through existing files
        ]

        loop_outputs = [
            status_output,
            sentence_index_to_process, # next_idx yielded by wrapper
            processing_active, # active status yielded by wrapper
            streaming_audio_output, # audio chunk yielded by wrapper
            generated_files_output # files list yielded by wrapper
        ]


        generate_btn.click(
            fn=storyteller.interrupt_and_reset, # Interrupt first
            outputs=[status_output]
        ).then(
            fn=storyteller.process_llm_query,
            inputs=[query_input, chatbot, gr.State(value=False), temperature_slider],
            outputs=process_llm_outputs,
            show_progress="full", # Show progress for LLM query
        ).then(
            fn=lambda: "", # Clear query input
            outputs=[query_input]
        ).then(
            fn=storyteller.gradio_sentence_generator_wrapper,
            inputs=[
                sentence_index_to_process,
                sentence_end_index,
                processing_active,
                temperature_slider, # Pass LLM temp? Wrapper doesn't use it directly for TTS
                speed_slider,       # Pass Speed
                speaker_id_input    # Pass Speaker ID
            ],
            outputs=loop_outputs,
            show_progress="hidden", # Hide progress for streaming loop
        )

        continue_btn.click(
            fn=storyteller.interrupt_and_reset, # Interrupt first
            outputs=[status_output]
        ).then(
            fn=storyteller.process_llm_query,
            inputs=[query_input, chatbot, gr.State(value=True), temperature_slider],
            outputs=process_llm_outputs,
            show_progress="full",
        ).then(
            fn=lambda: "", # Clear query input
            outputs=[query_input]
        ).then(
            fn=storyteller.gradio_sentence_generator_wrapper,
            inputs=[
                sentence_index_to_process,
                sentence_end_index,
                processing_active,
                temperature_slider,
                speed_slider,
                speaker_id_input
            ],
            outputs=loop_outputs,
            show_progress="hidden",
        )

        query_input.submit(
             fn=storyteller.interrupt_and_reset,
             outputs=[status_output]
         ).then(
             fn=storyteller.process_llm_query,
             inputs=[query_input, chatbot, gr.State(value=False), temperature_slider],
             outputs=process_llm_outputs,
             show_progress="full",
         ).then(
             fn=lambda: "",
             outputs=[query_input]
         ).then(
             fn=storyteller.gradio_sentence_generator_wrapper,
             inputs=[
                 sentence_index_to_process,
                 sentence_end_index,
                 processing_active,
                 temperature_slider,
                 speed_slider,
                 speaker_id_input
             ],
             outputs=loop_outputs,
             show_progress="hidden",
         )

        process_text_btn.click(
             fn=storyteller.interrupt_and_reset,
             outputs=[status_output]
         ).then(
             fn=storyteller.process_pasted_text,
             inputs=[pasted_text_input, speed_slider, gr.State(value=False)],
             outputs=process_pasted_outputs, # Use specific outputs if needed
             show_progress="full",
         ).then(
             fn=storyteller.gradio_sentence_generator_wrapper,
             inputs=[
                 sentence_index_to_process,
                 sentence_end_index,
                 processing_active,
                 temperature_slider, # Should this be a different TTS temp slider? Or use default?
                 speed_slider,
                 speaker_id_input
             ],
             outputs=loop_outputs,
             show_progress="hidden",
         )

        continue_text_btn.click(
             fn=storyteller.interrupt_and_reset,
             outputs=[status_output]
         ).then(
             fn=storyteller.process_pasted_text,
             inputs=[pasted_text_input, speed_slider, gr.State(value=True)],
             outputs=process_pasted_outputs,
             show_progress="full",
         ).then(
             fn=storyteller.gradio_sentence_generator_wrapper,
             inputs=[
                 sentence_index_to_process,
                 sentence_end_index,
                 processing_active,
                 temperature_slider, # TTS temp?
                 speed_slider,
                 speaker_id_input
             ],
             outputs=loop_outputs,
             show_progress="hidden",
         )

        clear_pasted_btn.click(lambda: "", outputs=[pasted_text_input])

        def reset_handler():
            result = storyteller.clear_session_for_ui()
            return result

        reset_btn.click(
            fn=reset_handler,
            outputs=[chatbot, status_output, sentence_index_to_process, processing_active,
                     streaming_audio_output, generated_files_output, full_story_download_output,
                     query_input, pasted_text_input], # Ensure all relevant outputs are cleared
        )

        download_full_story_btn.click(
            fn=storyteller.generate_and_save_full_story,
            inputs=[],
            outputs=[full_story_download_output, status_output]
        )

        load_history_outputs = [
            chatbot,
            status_output,
            sentence_index_to_process,
            processing_active,
            streaming_audio_output,
            generated_files_output,
            full_story_download_output,
            query_input,
            pasted_text_input
        ]
        load_history_btn.click(
            fn=storyteller.load_recent_history,
            inputs=[],
            outputs=load_history_outputs
        )

    temp_dir = os.path.join(tempfile.gettempdir(), "storyteller_audio")
    if os.path.exists(temp_dir):
        print(f"Cleaning temp directory: {temp_dir}")
        for filename in os.listdir(temp_dir):
            if filename.endswith(".wav"):
                 try:
                      os.remove(os.path.join(temp_dir, filename))
                 except OSError:
                      pass

    demo.queue().launch(server_name="0.0.0.0", share=False) # Keep queue and launch settings


if __name__ == "__main__":
    main() 