import os
import time
import sys
from openai import OpenAI

# Ensure OPENAI_API_KEY is set in your environment variables
# client = OpenAI() # Automatically uses OPENAI_API_KEY env var
# If you need to explicitly set it (e.g., if env var isn't working):
# client = OpenAI(api_key="YOUR_API_KEY") 
client = OpenAI()

if len(sys.argv) != 2:
    print("Usage: python audio_to_text.py <audio_file_path>")
    sys.exit(1)

audio_file_path = sys.argv[1]

# Check if the file exists
if not os.path.exists(audio_file_path):
    print(f"Error: Audio file not found at {audio_file_path}")
    exit()

print(f"Starting transcription for {audio_file_path} using OpenAI API...")
start_time = time.time()

try:
    # Open the audio file in binary read mode
    with open(audio_file_path, "rb") as audio_file:
        # Call the OpenAI API to transcribe
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        
        # The response object has a 'text' attribute with the full transcription
        full_text = transcription.text

        end_time = time.time()
        print("-" * 20)
        print("Full transcribed text:")
        print(full_text.strip())
        print("-" * 20)
        print(f"Total processing time (including API call): {end_time - start_time:.2f} seconds")

except Exception as e:
    print(f"An error occurred during transcription: {e}")
