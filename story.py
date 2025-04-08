import sys
import os

# Add the ask_llm/src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ask_llm/src'))

from ask_llm.utils.config import config
from ask_llm.main import AskLLM
from runme import TTS
import re

def clean_text_for_tts(text):
    """
    Clean text to make it suitable for TTS processing by:
    1. Removing markdown formatting
    2. Removing special characters
    3. Converting to plain sentences
    """
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

def main():
    # Initialize the LLM and TTS engines
    config.SYSTEM_MESSAGE = """
    You are a storyteller.
    You will respond with clear and concise sentences without any formatting.
    Don't use any special characters or quotes, just alphabet characters and punctuation to designate pauses and flowing sentences.
    """
    
    # Set HuggingFace model as default
    huggingface_model = "PygmalionAI/pygmalion-3-12b"  # Or any other model you want to use
    config.DEFAULT_MODEL = huggingface_model
    
    # Add model to HuggingFace models list if not already there
    if huggingface_model not in config.HUGGINGFACE_MODELS:
        config.HUGGINGFACE_MODELS.append(huggingface_model)
    
    # Initialize with the HuggingFace model
    llm = AskLLM()
    
    # Use the original TTS implementation
    tts = TTS(device="cuda")
    tts.load_model()
    warmup = tts.generate_audio_segment("Lets get it started!")

    # If command line arguments were provided, use them as the first query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        process_query(llm, tts, query)
    
    # Continue in interactive loop
    print("\nEnter your prompts to continue the story or start a new one.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            print("\nWhat's next in your story? (or type 'exit' to quit)")
            query = input("> ")
            
            # Exit if requested
            if query.lower() in ["exit", "quit"]:
                print("Exiting storyteller...")
                break
                
            # Process the query
            process_query(llm, tts, query)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting storyteller...")
            break

def process_query(llm, tts, query):
    """Process a single query: get LLM response, clean it, and play audio"""
    # Get response from LLM (history is automatically maintained by AskLLM)
    print(f"Asking: {query}")
    response = llm.query(query)
    print(f"Response: {response}")
    
    # Clean the response text for TTS
    cleaned_response = clean_text_for_tts(response)
    
    # Convert the response to speech and play it
    tts.say(cleaned_response)

if __name__ == "__main__":
    main()


