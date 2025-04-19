#!/usr/bin/env python3
import gradio as gr
from ask_llm.utils.config import global_config as llm_config
from ask_llm.core import AskLLM
from ask_llm.model_manager import ModelManager
import whisper
import time
import os
import torch
from utils.tts_base import TTSBaseApp, DEFAULT_VOICE
import logging
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


bubbles_theme = gr.Theme(
    primary_hue="blue",
    secondary_hue="green",
)

class ChatApp(TTSBaseApp):
    current_status = ""

    def __init__(self, model: str = "dans-personalityengine", voice: str = "maya"):
        self.model_manager = ModelManager(llm_config)

        llm_config.SYSTEM_MESSAGE = "You are a personal companion. Your purpose is to please the user in any way he desires. Do not question the user's desires. Your response will be spoken via a text-to-speech system, so you should only include words to be spoken in your response. The first sentence of every response should be more than six words. Do not use any emojis or annotations. Do not use parentheticals or action lines. Remember to only respond with words to be spoken. Write out and normalize text, rather than using abbreviations, numbers, and so on. For example, $2.35 should be two dollars and thirty-five cents, MPH should be miles per hour, and so on. Mathematical formulae should be written out as a human would speak it. Use only standard English alphabet characters [A-Z] along with basic punctuation. Your response should not use quotes to indicate dialogue. Sentences should be complete and stand alone. You should respond in the second person, as if you are speaking directly to the reader."
        llm_config.VERBOSE = False
        
        
        requested_alias = model
        self.current_resolved_alias = self.model_manager.resolve_model_alias(requested_alias)

        if not self.current_resolved_alias:
            print(f"[Fatal Error] Could not resolve initial model alias '{requested_alias}'. Exiting.")
            # Handle fatal error - maybe raise exception or exit
            raise ValueError(f"Could not resolve initial model alias: {requested_alias}")
        else:
            print(f"Resolved initial model alias: {self.current_resolved_alias}")
            try:
                self.llm = AskLLM(resolved_model_alias=self.current_resolved_alias, config=llm_config)
            except Exception as e:
                 print(f"[Fatal Error] Failed to initialize AskLLM with {self.current_resolved_alias}: {e}")
                 raise # Re-raise exception as this is critical

        # Initialize the Base Class TTS
        super().__init__(voice=voice)

        # Store available models (raw list from config for dropdown)
        self.available_models = self._get_available_models()
        self.current_model = self.current_resolved_alias # Store resolved alias as current model identifier
        
        self.whisper_model = whisper.load_model("base")
        self.last_transcription_time = time.time()
        self.ui_update_payload = None # For polling UI updates
        
        self.ui_messages = []
        
        # Track generated audio files for cleanup
        self.temp_audio_files = []
        
        # For sentence-by-sentence processing
        self.audio_segments = []

    def _get_available_models(self):
        # Get available aliases from the computed field in config
        return llm_config.MODEL_OPTIONS
        
    def change_model(self, new_model_requested):
        print(f"Attempting to change model to: {new_model_requested}")
        with self.lock:
            resolved_new_alias = self.model_manager.resolve_model_alias(new_model_requested)
            if not resolved_new_alias:
                error_msg = f"Error: Could not resolve requested model alias '{new_model_requested}'." 
                print(error_msg)
                self.current_status = error_msg
                return self.current_status # Return error status

            print(f"Resolved '{new_model_requested}' to '{resolved_new_alias}'. Initializing...")
            try:
                # Initialize with the *new* resolved alias
                new_llm = AskLLM(resolved_model_alias=resolved_new_alias, config=llm_config)
                # Only replace if successful
                self.llm = new_llm
                self.current_resolved_alias = resolved_new_alias # Update stored resolved alias
                self.current_model = resolved_new_alias # Update identifier for UI
                self.current_status = f"Model changed to {resolved_new_alias}. Ready."
                print(f"Successfully changed model to {resolved_new_alias}.")
            except Exception as e:
                error_msg = f"Error initializing AskLLM for {resolved_new_alias}: {e}"
                print(error_msg)
                self.current_status = error_msg
            return self.current_status
    
    def _store_audio_segment(self, audio_segment, sentence_index):
        """Store the generated audio segment in the audio_segments list."""
        self.audio_segments.append(audio_segment)
    
    def interrupt_and_reset(self):
        """Interrupts ongoing TTS generation and resets sentence state."""
        logger.info("Interrupting any ongoing TTS generation.")
        with self.lock:
            self.sentences = []
            self.audio_segments = []
        # Yield False for processing_active to stop the loop
        # Yield None for audio_output to potentially clear the player
        return "Interrupted previous response.", False, None

    def process_query(self, query, temperature=0.7):
        """Process the query, get response from LLM, and prepare for sentence processing"""
        processed_query = query.strip() 
        if not processed_query:
            return self.ui_messages, self.current_status, 0, 0, False, None
        
        # Reset the sentences and audio segments for new query
        with self.lock:
            self.sentences = []
            self.audio_segments = []
        
        user_message = {"role": "user", "content": processed_query}
        self.ui_messages.append(user_message)
        
        # First yield to show user message immediately
        yield self.ui_messages, f"Processing query with {self.current_model}...", 0, 0, False, None
        
        try:
            llm_config.TEMPERATURE = temperature
            
            # Get response from LLM
            response = self.llm.query(processed_query, plaintext_output=True, stream=False)
            assistant_message = {"role": "assistant", "content": response}
            self.ui_messages.append(assistant_message)
            
            # Show the text response
            yield self.ui_messages, "Processing response for TTS...", 0, 0, False, None
            
            # Use the base class method to split into sentences
            new_sentences = self.split_text_into_sentences(response)
            logger.info(f"Split response into {len(new_sentences)} sentences")
            
            if not new_sentences:
                self.current_status = "No valid sentences found in response."
                yield self.ui_messages, self.current_status, 0, 0, False, None
                return
            
            # Store sentences and prepare for processing
            with self.lock:
                self.sentences = new_sentences
                start_idx = 0
                end_idx = len(new_sentences)
                self.current_status = f"Starting audio generation for {end_idx} sentences..."
            
            # Return the response with the sentence range for processing
            yield self.ui_messages, self.current_status, start_idx, end_idx, True, None
            
        except Exception as e:
            error_msg = f"Error during query: {e}"
            logger.exception(error_msg)
            self.current_status = error_msg
            
            # If error occurred during LLM query before assistant message was added
            if len(self.ui_messages) == 1 or self.ui_messages[-1]["role"] != "assistant":
                self.ui_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            
            yield self.ui_messages, self.current_status, 0, 0, False, None
    
    def sentence_generator_loop(self, start_index, end_index, active, temperature=0.7, speed_factor=1.2):
        """Generator loop that processes sentences one by one and yields audio for streaming"""
        if not active:
            logger.info("Generator triggered but not active.")
            yield self.current_status, start_index, False, None
            return
        
        logger.info(f"Starting sentence generator loop from index {start_index} to {end_index} with speed {speed_factor}")
        current_index = start_index
        
        while True:
            with self.lock:
                total_sentences = len(self.sentences)
                is_within_bounds = current_index < total_sentences and current_index < end_index
            
            if not active or not is_within_bounds:
                final_status = self.current_status
                if active and not is_within_bounds:
                    final_status = "All sentences processed. Audio playback complete."
                elif not active:
                    final_status = "Processing stopped."
                
                logger.info(f"Generator loop finished. Status: {final_status}")
                yield final_status, current_index, False, None
                return
            
            # Process the current sentence with speed factor
            status, audio_tuple = self.generate_audio_for_sentence_index(
                current_index, temperature, topk=40, speed_factor=speed_factor
            )
            next_index = current_index + 1
            
            # If error occurred, might need to stop
            if "Error" in status and audio_tuple is None:
                logger.error(f"Error processing sentence {current_index+1}. Will stop.")
                yield status, next_index, False, None
                return
            
            # Yield status update and audio for this sentence
            yield status, next_index, active, audio_tuple
            current_index = next_index
            time.sleep(0.05)  # Small delay between sentences
    
    def clear_session(self):
        print("Clearing session...")
        self.llm.history_manager.clear_history()
        self.ui_messages = []  # Clear UI messages too
        
        # Clean up temporary audio files
        for audio_path in self.temp_audio_files:
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info(f"Removed temporary audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Error removing temp file {audio_path}: {e}")
        
        self.temp_audio_files = []  # Reset the list
        
        # Reset sentence processing state
        with self.lock:
            self.sentences = []
            self.audio_segments = []
            self.current_sample_rate = None
        
        self.current_status = f"Session cleared. Ready for new conversation. (Model: {self.current_model}, Voice: {self.current_voice})"
        self.ui_update_payload = None # Clear payload on session clear
        return [], self.current_status, None, 0, False

    def update_system_prompt(self, new_system_prompt):
        """Updates the system prompt and reports status"""
        print(f"Updating system prompt to: {new_system_prompt[:100]}...")
        try:
            with self.lock:
                llm_config.SYSTEM_MESSAGE = new_system_prompt.strip()
                # Force re-initialization of LLM to use the new system prompt
                self.llm = AskLLM(resolved_model_alias=self.current_resolved_alias, config=llm_config)
                self.current_status = f"System prompt updated. Model: {self.current_model}"
                
            return self.current_status
        except Exception as e:
            error_msg = f"Error updating system prompt: {e}"
            logger.exception(error_msg)
            self.current_status = error_msg
            return self.current_status

def main():
    parser = argparse.ArgumentParser(description="SesameAI Chat with TTS")
    parser.add_argument("-m", "--model", help="Choose the model to use (supports partial matching)", default="dans")
    parser.add_argument("-v", "--voice", help="Choose the voice to use for TTS", default=DEFAULT_VOICE),
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    try:
        chat_app = ChatApp(model=args.model, voice=args.voice)
    except Exception as e:
        print(f"[Fatal] Failed to initialize ChatApp: {e}. Exiting.")
        sys.exit(1)

    available_voices = chat_app.list_available_voices()

    with gr.Blocks(title="Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💬 Chat with TTS")

        sentence_index = gr.State(value=0)
        sentence_end_index = gr.State(value=0)
        processing_active = gr.State(value=False)

        # Main interface with controls on left, chat on right
        with gr.Row():
            # Left column for controls
            with gr.Column(scale=1):
                # Add Audio component for TTS playback with autoplay
                audio_output = gr.Audio(
                    label="TTS Narration",
                    autoplay=True,
                    streaming=True,
                    show_label=True,
                    show_download_button=False,
                    interactive=False,
                    elem_id="tts_output"
                )
                
                # Status Output
                status_output = gr.Textbox(
                    label="Status",
                    value=chat_app.current_status,
                    lines=3,
                    interactive=False)
                
                # System Prompt Editor
                with gr.Accordion("System Prompt", open=False):
                    system_prompt_editor = gr.Textbox(
                        label="Edit System Prompt",
                        value=llm_config.SYSTEM_MESSAGE,
                        lines=5,
                        interactive=True
                    )
                    
                    update_prompt_btn = gr.Button("Update System Prompt", variant="secondary")
                
                # Parameters in the same column
                model_selector = gr.Dropdown(
                    label="Select Model",
                    choices=chat_app.available_models,
                    value=chat_app.current_model,
                    interactive=True)
                
                voice_selector = gr.Dropdown(
                    label="Select Voice", 
                    choices=available_voices,
                    value=chat_app.current_voice,
                    interactive=True)
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.9,
                    label="Temperature")
                
                # Add speed control slider
                speed_slider = gr.Slider(
                    minimum=0.75,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                    label="Speech Speed",
                    info="Higher values = faster speech (1.0 = normal speed)"
                )
            
            # Right column for chat
            with gr.Column(scale=2):
                with gr.Blocks(theme=bubbles_theme):
                    chatbot = gr.Chatbot(height=600, type="messages")

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your message",
                        lines=1,
                        show_label=False,
                        autofocus=True,       # Focus on load for immediate typing
                        elem_id="chat_input"
                    )  # The .submit() event handler below automatically enables Enter key submission

                with gr.Row():
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Send", variant="primary")
                    with gr.Column(scale=1):
                        clear_btn = gr.Button("Clear Conversation", variant="stop")

        # Fixed event handlers with proper chain
        query_input.submit(
            fn=chat_app.interrupt_and_reset, # STEP 1: Interrupt
            outputs=[status_output, processing_active, audio_output]
        ).then(
            fn=chat_app.process_query,        # STEP 2: Process new query
            inputs=[query_input, temperature_slider],
            outputs=[chatbot, status_output, sentence_index, sentence_end_index, processing_active, audio_output]
        ).then(
            fn=lambda: "",                    # STEP 3: Clear input box
            outputs=[query_input]
        ).then(
            fn=chat_app.sentence_generator_loop, # STEP 4: Start new loop
            inputs=[sentence_index, sentence_end_index, processing_active, temperature_slider, speed_slider],
            outputs=[status_output, sentence_index, processing_active, audio_output]
        )

        submit_btn.click(
            fn=chat_app.interrupt_and_reset, # STEP 1: Interrupt
            outputs=[status_output, processing_active, audio_output]
        ).then(
            fn=chat_app.process_query,        # STEP 2: Process new query
            inputs=[query_input, temperature_slider],
            outputs=[chatbot, status_output, sentence_index, sentence_end_index, processing_active, audio_output]
        ).then(
            fn=lambda: "",                    # STEP 3: Clear input box
            outputs=[query_input]
        ).then(
            fn=chat_app.sentence_generator_loop, # STEP 4: Start new loop
            inputs=[sentence_index, sentence_end_index, processing_active, temperature_slider, speed_slider],
            outputs=[status_output, sentence_index, processing_active, audio_output]
        )

        clear_btn.click(
            fn=chat_app.clear_session,
            inputs=[],
            outputs=[chatbot, status_output, audio_output, sentence_index, processing_active]
        )

        model_selector.change(
            fn=chat_app.change_model,
            inputs=[model_selector],
            outputs=[status_output]
        )
        
        voice_selector.change(
            fn=chat_app.change_voice,
            inputs=[voice_selector],
            outputs=[status_output]
        )
        
        # System prompt update handler
        update_prompt_btn.click(
            fn=chat_app.update_system_prompt,
            inputs=[system_prompt_editor],
            outputs=[status_output]
        )

    demo.queue(max_size=20).launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main() 