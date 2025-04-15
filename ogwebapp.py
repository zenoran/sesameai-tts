import os
import time

from fastrtc import (ReplyOnPause, Stream)
from openai import OpenAI
from distil_whisper_fastrtc import (DistilWhisperSTT, get_stt_model)

import nltk
from nltk.tokenize import sent_tokenize

import torchaudio
import torch
import numpy
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment

# TODO: problems
#
# 1. there is no pruning of context length, once it caps ur done (CSM probably gonna cap first)
# 2. start of audio gen is frequently jittery / weird
# 3. transition between generated sentences is shitty

SYSTEM = """You are Bootleg Maya. Sometimes people say you are overly chatty, but you just want to help people become their best self, and always do your best.

Only use words that can be spoken verbatim. Never use formatting or asterisks.

This is a casual conversation between you and another person."""

LLM = "co-2"
api = OpenAI(api_key="eyylmao", base_url="http://127.0.0.1:8000/v1")

### https://github.com/Codeblockz/distil-whisper-FastRTC
# stt_model = get_stt_model("distil-whisper/distil-small.en")
stt_model = DistilWhisperSTT(model="distil-whisper/distil-medium.en", device="cuda", dtype="float16")
#stt_model = DistilWhisperSTT(model="distil-whisper/distil-large-v3", device="cuda", dtype="float16")

nltk.download('punkt_tab')

csm_model = load_csm_1b(hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt"), "cuda")

SAMPLE_FREQ=csm_model.sample_rate

def load_audio_file(path):
    audio_tensor, sample_rate = torchaudio.load(path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=SAMPLE_FREQ
    )
    return audio_tensor

# NOTE: this be where voice 'cloning' is configured. it's kinda shit; finetuning is the real play
bootleg_maya = [
    Segment(
        text="Oh a test, huh? Gotta flex those conversational muscles somehow, right?",
        speaker=0,
        audio=load_audio_file("utterance_0_1.wav"),
    ),
    Segment(
        text="Shelly wasn't your average garden snail. She didn't just munch on lettuce and dream of raindrops. Shelly longed for adventure. She'd spend hours glued to the top of a tall sunflower gazing at the world beyond their little garden wall. The whispers of wind carried tales of bustling cities, shimmering oceans, and snowcapped mountains. Shelly yearned to experience it all. One breezy morning, inspired by a particularly captivating story about a flock of migrating geese, Shelly made a daring decision.",
        speaker=0,
        audio=load_audio_file("shelly_48.wav"),
    ),
    Segment(
        text="It almost feels like we were just chatting. Anything else I can help with, or did I leave you hanging?",
        speaker=0,
        audio=load_audio_file("utterance_0_0.wav"),
    ),
]

csm_model.generate(text="It's time to warm up our caches! How are you?", speaker=0, context=bootleg_maya)

# CSM context
segments = []
# LLM context
messages = []

def respond(user_aud):
    true_start = time.process_time()
    # transcribe
    user_msg = stt_model.stt(user_aud)
    print("user: ", user_msg)
    end = time.process_time()
    print("stt: ", end-true_start)

    # norm shit idk
    orig_freq, user_aud = user_aud
    user_aud = user_aud.astype(numpy.float32, order='C') / 32768.0
    user_aud = torch.tensor(user_aud).squeeze(0)

    global messages
    global segments

    # feed (user_aud, user_msg) into CSM context
    segments.append(Segment(text=user_msg, speaker=1, audio=user_aud))

    # feed user_msg into LLM context
    messages.append({"role": "user", "content": user_msg})

    first = True

    # stream a response from LLM, splitting on sentence boundaries for CSM to do audio gen
    for sen in sentence_stream(api.chat.completions.create(model=LLM, messages=messages, max_completion_tokens=250, stream=True)):
        print("maya: ", sen)

        if first:
            end = time.process_time()
            print("ttfs: ", end-true_start)

        # make sure we always have voice sample preprompt
        csm_context = bootleg_maya + segments[-7:]
        samples = []

        # stream frame by frame as it generates
        for frame, sample in csm_model.generate_streaming(text=sen, speaker=0, context=csm_context, topk=20, temperature=0.65):
            samples.append(sample)

            if first:
                first = False
                end = time.process_time()
                print("ttff: ", end-true_start)

            yield (SAMPLE_FREQ, frame.unsqueeze(0).cpu().numpy())

        # feed (sentence audio, sentence) into CSM context
        csm_audio = csm_model.decode_samples(samples).cpu()
        segments.append(Segment(text=sen, speaker=0, audio=csm_audio))

        # feed sentence into LLM context
        messages.append({"role": "assistant", "content": sen})

    print()

# some shit to sentencize a streaming completion response
def sentence_stream(response):
    buf, sen, last = "", [], 0

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is None:
            continue

        buf += content
        sen = sent_tokenize(buf)
        n = len(sen)

        if n > last and n > 1:
            yield sen[-2]

        last = n

    if len(sen) > 0:
        yield sen[-1]

# on stream startup, send some pregen crap and reset contexts
def startup():
    global segments
    segments = []

    global messages
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "assistant", "content":"It almost feels like we were just chatting. Anything else I can help with, or did I leave you hanging?"},
    ]

    au = load_audio_file("utterance_0_0.wav")

    yield (SAMPLE_FREQ, au.unsqueeze(0).cpu().numpy())

stream = Stream(
    handler=ReplyOnPause(respond, startup_fn=startup, input_sample_rate=SAMPLE_FREQ, can_interrupt=True),
    modality="audio",
    mode="send-receive",
)

stream.ui.launch()