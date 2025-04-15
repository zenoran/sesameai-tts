"""
Command-line argument utilities for SesameAI applications.

This module provides common utilities for handling command-line arguments
across multiple applications to avoid code duplication.
"""
from ask_llm.utils.config import config
from ask_llm.utils.ollama_utils import find_matching_model


def setup_common_args_and_config(args):
    """
    Process command-line arguments and apply them to the configuration.
    
    Args:
        args: The parsed command-line arguments from argparse.
        
    Returns:
        None (modifies config in place)
    """
    # Handle model selection with partial matching
    if hasattr(args, 'model') and args.model:
        model_name = args.model
        all_models = config.HUGGINGFACE_MODELS + config.OPENAPI_MODELS + config.OLLAMA_MODELS
        match = find_matching_model(model_name, all_models)
        
        if match:
            config.DEFAULT_MODEL = match
            print(f"Using model: {match}")
        else:
            print(f"Warning: No models found matching '{model_name}'")
            print("Available models:")
            for m in all_models:
                print(f"  - {m}")
    
    # Handle legacy llm parameter (alias for model)
    if hasattr(args, 'llm') and args.llm:
        model_name = args.llm
        all_models = config.HUGGINGFACE_MODELS + config.OPENAPI_MODELS + config.OLLAMA_MODELS
        match = find_matching_model(model_name, all_models)
        
        if match:
            config.DEFAULT_MODEL = match
            print(f"Using model: {match}")
        else:
            print(f"Warning: No models found matching '{model_name}'")
            print("Available models:")
            for m in all_models:
                print(f"  - {m}")
    
    # Handle voice selection
    if hasattr(args, 'voice') and args.voice:
        config.DEFAULT_VOICE = args.voice
        print(f"Using voice: {args.voice}")
        # Note: The actual voice loading happens when the app is initialized 