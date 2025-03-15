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

### Installing Requirements
Once the virtual environment is activated, you can install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Script
To run the `runme.py` script, use the following command:

```bash
python runme.py
```

## Notes
- Input will be split based on sentences and play them as they are generated.
- A final output for every prompt is always available as `combined_output.wav` and gets overwritten for every prompt.
