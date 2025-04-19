# SesameAI Text-to-Speech Runner

This script provides a command-line interface for interacting with the SesameAI Text-to-Speech model, allowing users to generate high-quality speech from text input.

## Prerequisites

*   Python 3.12+
*   pip
*   `ffmpeg` (required by `pydub` for audio processing and playback)
    *   On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
    *   On macOS (using Homebrew): `brew install ffmpeg`  not sure if this works?
    *   I gave up on Windows so if you have windows do yourself a favor and install WSL or something.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:zenoran/sesame-tts.git
    cd sesame-tts
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    *Note: Ensure you have PyTorch installed according to your system's CUDA capabilities if using GPU. See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).*
    ```bash
    pip install torch torchaudio pydub
    ```
    *(If a `requirements.txt` file is provided, you can use `pip install -r requirements.txt` instead)*

4.  **Voice Samples:** Ensure you have a `samples.py` file in the same directory, containing dictionaries that define the available voices and their corresponding audio samples. The keys should be the string representation of the path to the audio file, and the values should be the corresponding transcriptions. Example structure based on the provided `samples.py`:
    ```python
    # samples.py
    from pathlib import Path

    # Base directory for audio files
    AUDIO_DIR = Path("wav")

    maya = {
        str(AUDIO_DIR / "crab-story" / "mono_1.wav"): "OK fresh start, how about this... Close your eyes for a second...",
        str(AUDIO_DIR / "crab-story" / "mono_2.wav"): "Alright, where was I?  Ah yes, the crimson sun...",
        # ... more samples for maya
    }

    melina = {
        str(AUDIO_DIR / "melina" / "melina-02.wav"): "As an ally by pact, in Marica's own words, I ask that you cease...",
        # ... more samples for melina
    }
    ```
    Make sure the audio files referenced in `samples.py` exist in the specified locations (e.g., a `wav` directory relative to the script).

## Usage

Run the script using `python tts_service.py`.

### Command-Line Arguments

*   `-d`, `--device`: Device to run inference on (`cuda` or `cpu`). Default: `cuda`.
*   `-v`, `--voice`: Voice to use (must match a dictionary name in `samples.py`). Defaults to the first voice found.
*   `text`: (Optional) Text to synthesize for a single utterance. If provided, the audio is saved to the file specified by `--output`.
*   `--output`: Output filename when providing `text`. Default: `output.wav`.
*   `--temp`, `--temperature`: Generation temperature (0.1-1.0). Lower values are more predictable, higher values are more creative. Default: `0.8`.
*   `--topk`: Top-K sampling value (10-100). Lower values are more focused, higher values are more varied. Default: `40`.

### Examples

1.  **Interactive Mode (using default voice 'alice' on CUDA):**
    ```bash
    python tts_service.py -v alice
    ```
    The script will load the model and voice, then wait for you to type text. Press Enter to synthesize and play. Type `exit` or `quit` to stop.

2.  **Interactive Mode (using voice 'bob' on CPU):**
    ```bash
    python tts_service.py -v bob -d cpu --temp 0.7 --topk 50
    ```

3.  **Single Utterance to File:**
    ```bash
    python tts_service.py -v alice "Hello, this is a test." --output test_audio.wav
    ```
    This will generate audio for the text "Hello, this is a test." using the 'alice' voice and save it as `test_audio.wav`.

4.  **Single Utterance to File (CPU, different parameters):**
    ```bash
    python tts_service.py -d cpu -v bob "Another example sentence." --output example.wav --temp 0.9 --topk 30
    ```
