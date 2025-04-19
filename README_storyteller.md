# SesameAI Storyteller

An interactive storytelling application that uses SesameAI's Text-to-Speech model to narrate AI-generated stories.

## Features

- Interactive storytelling with AI-generated narratives
- High-quality text-to-speech narration
- Maintains context between prompts for coherent stories
- Available as both CLI and web interface

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster generation)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

Run the web interface with:

```bash
python web_storyteller.py
```

This will launch a Gradio web interface where you can:
1. Enter prompts to generate stories
2. Listen to the AI narration through streaming audio
3. See the text history of your conversation
4. Start new stories with the "Start New Story" button

The web interface will be available at http://localhost:7860 by default.

## Technical Details

- Uses the SesameAI TTS model for speech synthesis
- Leverages AskLLM for story generation and context maintenance
- Cleans text to ensure optimal TTS performance
- Web interface built with Gradio
- Audio streaming for faster playback (simulated by splitting sentences and adding to WAV in player)