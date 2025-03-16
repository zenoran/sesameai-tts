# SesameAI Text-to-Speech Model Runner

## Description
This script provides a user-friendly interface for interacting with the SesameAI Text-to-Speech model,
allowing users to generate high-quality speech from provided text input.

### Prerequisites
- Python 3.11
- `pip` (Python package installer)

### Setting Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies. You can create and activate a virtual environment using the following commands:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate

# On Unix or MacOS
source .venv/bin/activate
```

### Windows hacks - You can skipp this for Linux
On Windows, you need to install specific versions of `torch` and `torchaudio` that match your CUDA version. Also need this custom build of triton due to windows bugs.

```bash
pip install triton-windows
pip install torch==2.4.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp311-cp311-win_amd64.whl
```

### Download FFmpeg
Go to FFmpeg.org or directly to gyan.dev for a Windows build
Download the "FFmpeg Full" or "FFmpeg Essentials" build
- Add the bin folder to your path.


### Installing Requirements
Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Script
```bash
python runme.py
```

## Notes
- Input will be split based on sentences and play them as they are generated.
- A final output for every prompt is always available as `combined_output.wav` and it gets OVERWRITTEN for every prompt. (grab it if u wanna keep it)
