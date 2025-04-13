#!/usr/bin/env python3
import gradio as gr
from ask_llm.utils.config import config
from ask_llm.main import AskLLM
import threading
import whisper
import time

WHISPER_SAMPLE_RATE = 16000

class ChatApp:
    current_status = ""
    def __init__(self):
        config.SYSTEM_MESSAGE = """
        You are a helpful AI assistant. Provide clear, concise and accurate responses to user queries.
        """
        config.VERBOSE = False
        
        self.llm = AskLLM()
        self.lock = threading.Lock()
        self.available_models = self._get_available_models()
        self.current_model = config.DEFAULT_MODEL
        
        self.whisper_model = whisper.load_model("base")
        self.last_transcription_time = time.time()
        self.ui_update_payload = None # For polling UI updates
        
        self.ui_messages = []

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
    
    def process_query(self, query, temperature=0.7):
        processed_query = query.strip() 
        if not processed_query:
            return self.ui_messages, self.current_status
        
        user_message = {"role": "user", "content": processed_query}
        self.ui_messages.append(user_message)
        
        yield self.ui_messages, f"Processing text query with {self.current_model}..."
        
        try:
            config.TEMPERATURE = temperature
            
            response = self.llm.query(processed_query, plaintext_output=True)
            assistant_message = {"role": "assistant", "content": ""}
            self.ui_messages.append(assistant_message)
            chunks = response.split('. ')
            current_response = ""
            
            for i, chunk in enumerate(chunks):
                if i < len(chunks) - 1:
                    chunk = chunk + '.'
                current_response += chunk + (' ' if i < len(chunks) - 1 else '')
                
                self.ui_messages[-1]["content"] = current_response
                
                yield self.ui_messages, f"Generating response ({i+1}/{len(chunks)})..."
                time.sleep(0.1)
            
        except Exception as e:
            error_msg = f"Error during query: {e}"
            print(error_msg)
            self.current_status = error_msg
        
        yield self.ui_messages, self.current_status
    
    def clear_session(self):
        print("Clearing session...")
        self.llm.history_manager.clear_history()
        self.ui_messages = []  # Clear UI messages too
        self.current_status = f"Session cleared. Ready for new conversation. (Model: {self.current_model})"
        self.ui_update_payload = None # Clear payload on session clear
        return [], self.current_status
    


def main():
    chat_app = ChatApp()


    with gr.Blocks(title="Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💬 Chat")

        # Chat Interface
        with gr.Row():
            with gr.Column(): # Use a single column to make this section full width
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

        # Status Output
        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                value=chat_app.current_status,
                lines=3,
                interactive=False)

        # Parameters
        with gr.Row():
            with gr.Column(scale=1):
                    model_selector = gr.Dropdown(
                    label="Select Model",
                    choices=chat_app.available_models,
                    value=chat_app.current_model,
                    interactive=True)
            with gr.Column(scale=1):
                    temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.7,
                    label="Temperature")



        # Event Handlers (remain the same, referencing the component variables)
        query_input.submit(
            fn=chat_app.process_query,
            inputs=[query_input, temperature_slider],
            outputs=[chatbot, status_output],
            show_progress="minimal"
        ).then(
            fn=lambda: "",
            outputs=[query_input]
        )

        submit_btn.click(
            fn=chat_app.process_query,
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