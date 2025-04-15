#!/usr/bin/env python3
import gradio as gr
import torch
from ask_llm.utils.config import config
from ask_llm.main import AskLLM
import threading
import numpy as np
import os
import whisper
import time
from fastrtc import Stream, ReplyOnPause
from fastrtc.utils import AdditionalOutputs

WHISPER_SAMPLE_RATE = 16000

class ChatApp:
    def __init__(self):
        config.SYSTEM_MESSAGE = """
        You are a helpful AI assistant. Provide clear, concise and accurate responses to user queries.
        """
        config.VERBOSE = False
        
        self.llm = AskLLM()
        self.lock = threading.Lock()
        self.available_models = self._get_available_models()
        self.current_model = config.DEFAULT_MODEL
        self.current_status = f"Ready. Using model: {self.current_model}"
        
        self.whisper_model = whisper.load_model("base")
        self.last_transcription_time = time.time()
        self.last_transcription = None

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
    
    def process_query_and_get_history(self, query, temperature=0.7):
        processed_query = query.strip()
        latest_transcription = ""
        
        if self.last_transcription:
            print(f"DEBUG: Found transcription in instance state: {self.last_transcription.get('text','')}")
            latest_transcription = self.last_transcription.get("text", "").strip()
            if latest_transcription:
                 processed_query = (processed_query + " " + latest_transcription).strip()
            self.last_transcription = None

        history_messages = self.llm.history_manager.get_messages_for_chatbot()
        
        if not processed_query:
             return history_messages, self.current_status
            
        history_messages.append({"role": "user", "content": processed_query})
        yield history_messages, "Processing..."

        history_messages.append({"role": "assistant", "content": "..."}) 
        yield history_messages, f"Processing query with {self.current_model}..."
        
        final_history = history_messages
        final_status = self.current_status
        
        try:
            config.TEMPERATURE = temperature
            response = self.llm.query(processed_query, plaintext_output=True) 
            
            if final_history and final_history[-1]["role"] == "assistant":
                final_history[-1]["content"] = response
            else:
                final_history.append({"role": "assistant", "content": response})

            final_status = f"Response complete. Ready for next query. (Model: {self.current_model})"
            if latest_transcription:
                 final_status += f" (Used transcription: '{latest_transcription}')"
            
        except Exception as e:
            error_msg = f"Error during query: {e}"
            print(error_msg)
            if final_history and final_history[-1]["role"] == "assistant":
                 final_history[-1]["content"] = f"ERROR: {error_msg}"
            else:
                 final_history.append({"role": "assistant", "content": f"ERROR: {error_msg}"})
            final_status = error_msg
            
        yield final_history, final_status
    
    def clear_session(self):
        print("Clearing session...")
        self.llm.history_manager.clear_history()
        self.current_status = f"Session cleared. Ready for new conversation. (Model: {self.current_model})"
        return [], self.current_status
    
    def transcribe_audio(self, user_aud):
        if user_aud is None:
            return "", "Audio input was null."

        true_start = time.process_time()
        
        orig_freq, audio_data = user_aud
        
        if audio_data is None or audio_data.size == 0:
             return "", "Received empty audio data."

        print(f"Received audio: {audio_data.shape} at {orig_freq} Hz")
        status = "Transcribing audio..."
        transcribed_text = ""
        
        try:
            audio_np = audio_data.astype(np.float32) / 32768.0
            audio_np = np.squeeze(audio_np)
            
            if orig_freq != WHISPER_SAMPLE_RATE:
                 print(f"Resampling required from {orig_freq} to {WHISPER_SAMPLE_RATE}")
                 pass 

            result = self.whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            transcribed_text = result['text'].strip()
            
            end = time.process_time()
            print(f"Transcription: {transcribed_text} (took {end-true_start:.2f}s)")
            
            if transcribed_text:
                status = f"Transcription complete: {transcribed_text}"
            else:
                status = "Transcription finished (no text)."
                
        except Exception as e:
            error_msg = f"Error during transcription: {e}"
            print(error_msg)
            status = error_msg

        self.last_transcription = {"text": transcribed_text, "status": status}

        yield (None, AdditionalOutputs({"text": transcribed_text, "status": status}))

def main():
    chat_app = ChatApp()
    
    with gr.Blocks(title="Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💬 Chat")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, type="messages")
                
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Type or dictate your message here...",
                        label="Your message",
                        lines=2,
                        show_label=False)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Send", variant="primary")
                    
                    with gr.Column(scale=1):
                        clear_btn = gr.Button("Clear Conversation", variant="stop")
            
            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="Status",
                    value=chat_app.current_status,
                    lines=3, 
                    interactive=False)
                
                model_selector = gr.Dropdown(
                    label="Select Model",
                    choices=chat_app.available_models,
                    value=chat_app.current_model,
                    interactive=True)
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.7,
                    label="Temperature")

                fastrtc_stream = Stream(
                    handler=ReplyOnPause(
                        chat_app.transcribe_audio, 
                        input_sample_rate=WHISPER_SAMPLE_RATE,
                        can_interrupt=False,
                    ),
                    modality="audio",
                    mode="send-receive",
                )
        
        submit_btn.click(
            fn=chat_app.process_query_and_get_history,
            inputs=[query_input, temperature_slider],
            outputs=[chatbot, status_output],
            show_progress="minimal"
        ).then(
            fn=lambda: "", 
            outputs=[query_input]
        )
        
        query_input.submit(
            fn=chat_app.process_query_and_get_history,
            inputs=[query_input, temperature_slider],
            outputs=[chatbot, status_output],
            show_progress="minimal"
        ).then(
            fn=lambda: "",
            outputs=[query_input]
        )
        
        clear_btn.click(
            fn=chat_app.clear_session,
            inputs=[],
            outputs=[chatbot, status_output] 
        )
        
        model_selector.change(
            fn=chat_app.change_model,
            inputs=[model_selector],
            outputs=[status_output]
        )
        
    demo.queue(max_size=20).launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main() 