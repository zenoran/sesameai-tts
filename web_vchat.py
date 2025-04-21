import os
import io
import numpy as np
import soundfile as sf 
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastrtc import (ReplyOnPause, Stream, get_stt_model) 
from ask_llm.core import AskLLM, Config
from openai import OpenAI 
from tts_service import TTS

client = OpenAI()

llm_config = Config()
llm = AskLLM(resolved_model_alias="dans-personalityengine", config=llm_config)

tts = TTS()

tts.load_model()
stt_model = get_stt_model()

def echo(audio):
    prompt = stt_model.stt(audio)
    response = llm.query(prompt, plaintext_output=True)
    for audio_chunk in tts.stream_audio_with_context(response):
        yield audio_chunk

# Stream initialization remains the same
stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":

    tts.load_model()
    tts.load_voice("maya")
    stream.ui.launch()