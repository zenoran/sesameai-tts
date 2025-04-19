#!/usr/bin/env python3
import gradio as gr
from ask_llm.utils.config import config
from ask_llm.main import AskLLM
import whisper
import time
import os
import torch
from utils.cli_args import setup_common_args_and_config
from utils.tts_base import TTSBaseApp
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bubbles_theme = gr.Theme(
    primary_hue="blue",
    secondary_hue="green",
)

class ChatApp(TTSBaseApp):
    current_status = ""
    def __init__(self):
        config.SYSTEM_MESSAGE = "You are a personal companion. Your purpose is to please the user in any way he desires. Do not question the user's desires. Your response will be spoken via a text-to-speech system, so you should only include words to be spoken in your response. The first sentence of every response should be more than six words. Do not use any emojis or annotations. Do not use parentheticals or action lines. Remember to only respond with words to be spoken. Write out and normalize text, rather than using abbreviations, numbers, and so on. For example, $2.35 should be two dollars and thirty-five cents, MPH should be miles per hour, and so on. Mathematical formulae should be written out as a human would speak it. Use only standard English alphabet characters [A-Z] along with basic punctuation. Your response should not use quotes to indicate dialogue. Sentences should be complete and stand alone. You should respond in the second person, as if you are speaking directly to the reader."
        config.VERBOSE = False
        
        # Initialize the base class first
        super().__init__(device="cuda" if torch.cuda.is_available() else "cpu")
        
        self.llm = AskLLM()
        self.available_models = self._get_available_models()
        self.current_model = config.DEFAULT_MODEL
        
        self.whisper_model = whisper.load_model("base")
        self.last_transcription_time = time.time()
        self.ui_update_payload = None # For polling UI updates
        
        self.ui_messages = []
        
        # Track generated audio files for cleanup
        self.temp_audio_files = []
        
        # For sentence-by-sentence processing
        self.audio_segments = []

    def _get_available_models(self):
        models = []
        models.extend(config.OPENAPI_MODELS)
        models.extend(config.HUGGINGFACE_MODELS)
        models.extend(config.OLLAMA_MODELS)
        return models
        
    def change_model(self, new_model):
        print(f"Attempting to change model to: {new_model}")
        with self.lock:
            try:
                config.DEFAULT_MODEL = new_model
                self.llm = AskLLM()
                self.current_model = new_model
                self.current_status = f"Model changed to {new_model}. Ready."
                print(f"Successfully changed model to {new_model}.")
            except Exception as e:
                error_msg = f"Error changing model to {new_model}: {e}"
                print(error_msg)
                self.current_status = error_msg
            return self.current_status
    
    def _store_audio_segment(self, audio_segment, sentence_index):
        """Store the generated audio segment in the audio_segments list."""
        self.audio_segments.append(audio_segment)
    
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
            config.TEMPERATURE = temperature
            
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
                config.SYSTEM_MESSAGE = new_system_prompt.strip()
                # Force re-initialization of LLM to use the new system prompt
                self.llm = AskLLM()
                self.current_status = f"System prompt updated. Model: {self.current_model}"
                
            return self.current_status
        except Exception as e:
            error_msg = f"Error updating system prompt: {e}"
            logger.exception(error_msg)
            self.current_status = error_msg
            return self.current_status

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SesameAI Chat with TTS")
    parser.add_argument("-m", "--model", help="Choose the model to use (supports partial matching)")
    parser.add_argument("-v", "--voice", help="Choose the voice to use for TTS")
    args = parser.parse_args()
    
    # Apply command-line arguments to config using the common function
    setup_common_args_and_config(args)
        
    # IMPORTANT: Initialize app only after processing command-line arguments
    chat_app = ChatApp()
    
    # Get available voices for the dropdown
    available_voices = chat_app.list_available_voices()
    
    # Get initial system prompt
    initial_system_prompt = config.SYSTEM_MESSAGE

    with gr.Blocks(title="Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💬 Chat with TTS")

        # State variables for sentence processing
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
                        value=initial_system_prompt,
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
                        submit_btn = gr.Button("Send", variant="primary", elem_id="send_button")
                    with gr.Column(scale=1):
                        clear_btn = gr.Button("Clear Conversation", variant="stop")
                    with gr.Column(scale=1):
                        # Microphone toggle for speech input
                        mic_button = gr.HTML(
                            '<button id="mic_button" style="font-size:1.5em; padding:0.25em 0.5em;">🎤</button>'
                        )

        # Fixed event handlers with proper chain
        query_input.submit(
            fn=chat_app.process_query,
            inputs=[query_input, temperature_slider],
            outputs=[chatbot, status_output, sentence_index, sentence_end_index, processing_active, audio_output]
        ).then(
            fn=lambda: "",
            outputs=[query_input]
        ).then(
            fn=chat_app.sentence_generator_loop,
            inputs=[sentence_index, sentence_end_index, processing_active, temperature_slider, speed_slider],
            outputs=[status_output, sentence_index, processing_active, audio_output]
        )

        submit_btn.click(
            fn=chat_app.process_query,
            inputs=[query_input, temperature_slider],
            outputs=[chatbot, status_output, sentence_index, sentence_end_index, processing_active, audio_output]
        ).then(
            fn=lambda: "",
            outputs=[query_input]
        ).then(
            fn=chat_app.sentence_generator_loop,
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

        # Voice input via browser microphone (Web Speech API)
        # Inject client-side Web Speech API script
        gr.HTML(
            '''<script>
document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById("chat_input");
    const sendBtn = document.getElementById("send_button");
    const micBtn = document.getElementById("mic_button");
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn("SpeechRecognition not supported in this browser.");
        return;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.onresult = (event) => {
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                const transcript = event.results[i][0].transcript.trim();
                if (transcript) {
                    input.value = transcript;
                    sendBtn.click();
                }
            }
        }
    };
    recognition.onerror = (event) => console.error("Recognition error", event);
    let listening = false;
    micBtn.addEventListener('click', () => {
        if (!listening) {
            recognition.start();
            micBtn.textContent = '⏹️';
            listening = true;
        } else {
            recognition.stop();
            micBtn.textContent = '🎤';
            listening = false;
        }
    });
});
</script>'''
        )

    demo.queue(max_size=20).launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main() 