#!/usr/bin/env python3
"""
SesameAI Text-to-Speech Model Runner

This script provides a user-friendly interface for interacting with the SesameAI Text-to-Speech model,
allowing users to generate high-quality speech from provided text input.
"""
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
import torch
import torchaudio
from pathlib import Path
from typing import Optional
from pydub import AudioSegment
from pydub.playback import play
from sesameai.generator import Segment, load_csm_1b
from sesameai.watermarking import CSM_1B_GH_WATERMARK, watermark
from samples import voice1
import numpy as np


# Suppress unnecessary warnings and configure environment
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism for better stability

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class TTS:
    """Wrapper class for text-to-speech functionality using SesameAI models."""
    
    def __init__(self, device: str = "cuda", model_repo: str = "sesame/csm-1b") -> None:
        """
        Initialize the Text-to-Speech engine.
        
        Args:
            device: Device to run inference on ("cuda" or "cpu")
            model_repo: HuggingFace repository ID for the model
        """
        self.device = device
        self.model_repo = model_repo
        self.generator = None
        self.cached_context_tokens = []
        self.cached_context_masks = []
        
        # Configure audio playback
        self._patch_audio_playback()
        
    def _patch_audio_playback(self) -> None:
        """Patch the audio playback functionality to avoid issues with ffplay."""
        from pydub import playback
        
        def patched_play_with_ffplay(seg: AudioSegment) -> None:
            """Enhanced playback function that properly cleans up temporary files."""
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            seg.export(path, format="wav")
            command = ["ffplay", path, "-nodisp", "-autoexit", "-loglevel", "quiet"]
            subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(path)  # Clean up temporary file
            
        playback._play_with_ffplay = patched_play_with_ffplay

    def load_model(self) -> None:
        """
        Load the CSM-1B TTS model and prepare it for inference.
        This method is called during initialization and doesn't need to be called manually.
        """
        print("Loading CSM-1B model...")
        try:
            self.generator = load_csm_1b(self.device)
            
            # Initialize watermarker
            from sesameai.watermarking import load_watermarker
            self.watermarker = load_watermarker(self.device)
            
            self.cached_context_tokens = []
            self.cached_context_masks = []
            self._prepare_context()
            print("Model loaded and ready for use.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _prepare_context(self) -> None:
        """Precompute context tokens for faster generation."""
        if not self.generator:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        print("Preparing reference audio context...")
        segments = [
            Segment(text=text, speaker=1, audio=self._load_audio(audio_path))
            for audio_path, text in voice1.items()
        ]
        
        # Cache tokenized representations for fixed context segments
        for segment in segments:
            tokens, masks = self.generator._tokenize_segment(segment)
            self.cached_context_tokens.append(tokens)
            self.cached_context_masks.append(masks)
        print("Reference audio context prepared")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file for model consumption.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio tensor
        """
        # Normalize path for cross-platform compatibility
        audio_path = Path(audio_path)
        logger.debug(f"Loading audio: {audio_path}")
        audio_tensor, sample_rate = torchaudio.load(str(audio_path))

        # Convert stereo to mono if necessary
        if audio_tensor.shape[0] > 1:
            logger.debug("Converting stereo to mono")
            audio_tensor = audio_tensor.mean(dim=0)

        # Resample if sample rates differ
        if sample_rate != self.generator.sample_rate:
            logger.debug(f"Resampling from {sample_rate}Hz to {self.generator.sample_rate}Hz")
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sample_rate, self.generator.sample_rate
            )

        return audio_tensor.squeeze()

    def generate_with_context(
        self, 
        prompt: str, 
        speaker: int = 1, 
        max_audio_length_ms: int = 30000, 
        temperature: float = 0.9, 
        topk: int = 50
    ) -> torch.Tensor:
        """
        Generate audio from text using cached context.
        
        Args:
            prompt: Text to synthesize
            speaker: Speaker ID
            max_audio_length_ms: Maximum duration in milliseconds
            temperature: Sampling temperature (higher = more random)
            topk: Top-k sampling parameter
            
        Returns:
            Audio tensor
        """
        self.generator._model.reset_caches()
        with torch.inference_mode():
            # Use mixed precision throughout the generation process
            with torch.autocast(self.device, dtype=torch.bfloat16):
                # Tokenize the new prompt
                gen_tokens, gen_masks = self.generator._tokenize_text_segment(prompt, speaker)
                # Combine cached tokens with new prompt tokens
                prompt_tokens = (
                    torch.cat(self.cached_context_tokens + [gen_tokens], dim=0)
                    .long()
                    .to(self.device)
                )
                prompt_tokens_mask = (
                    torch.cat(self.cached_context_masks + [gen_masks], dim=0)
                    .bool()
                    .to(self.device)
                )

                samples = []
                curr_tokens = prompt_tokens.unsqueeze(0)
                curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
                curr_pos = (
                    torch.arange(0, prompt_tokens.size(0))
                    .unsqueeze(0)
                    .long()
                    .to(self.device)
                )

                max_audio_frames = int(max_audio_length_ms / 80)
                max_seq_len = 2048 - max_audio_frames
                if curr_tokens.size(1) >= max_seq_len:
                    raise ValueError(f"Input too long ({curr_tokens.size(1)} tokens). Maximum is {max_seq_len} tokens.")

                for _ in range(max_audio_frames):
                    sample = self.generator._model.generate_frame(
                        curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                    )
                    if torch.all(sample == 0):
                        break
                    samples.append(sample)
                    curr_tokens = torch.cat(
                        [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
                    ).unsqueeze(1)
                    curr_tokens_mask = torch.cat(
                        [
                            torch.ones_like(sample).bool(),
                            torch.zeros(1, 1).bool().to(self.device),
                        ],
                        dim=1,
                    ).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1

            # Decode audio from tokens
            audio = (
                self.generator._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))
                .squeeze(0)
                .squeeze(0)
            )

            # Apply watermarking
            audio, wm_sample_rate = watermark(
                self.watermarker, audio, self.generator.sample_rate, CSM_1B_GH_WATERMARK
            )
            audio = torchaudio.functional.resample(
                audio, orig_freq=wm_sample_rate, new_freq=self.generator.sample_rate
            )

        return audio

    def generate_audio_segment(
        self, 
        prompt: str, 
        fade_duration: int = 50, 
        start_silence_duration: int = 500, 
        end_silence_duration: int = 100
    ) -> AudioSegment:
        """
        Generate an AudioSegment from text with proper silence padding and fading.
        
        Args:
            prompt: Text to synthesize
            fade_duration: Duration of fade-in and fade-out in milliseconds
            start_silence_duration: Duration of silence at the beginning in milliseconds
            end_silence_duration: Duration of silence at the end in milliseconds
            
        Returns:
            AudioSegment with the generated audio
        """
        
        # Generate raw audio
        audio = self.generate_with_context(prompt, speaker=1, max_audio_length_ms=30000)

        # Normalize audio
        audio = audio.to(torch.float32)
        if audio.dim() > 1:
            audio = audio.squeeze()
        audio = audio / max(audio.abs().max(), 1e-6)

        # Convert to 16-bit PCM
        audio_np = (audio.cpu().numpy() * 32767).astype("int16")
        audio_segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=self.generator.sample_rate,
            sample_width=2,
            channels=1,
        )

        # Add silence padding and fade-in/out
        start_silence = AudioSegment.silent(duration=start_silence_duration)
        end_silence = AudioSegment.silent(duration=end_silence_duration)
        audio_segment = start_silence + audio_segment + end_silence
        audio_segment = audio_segment.fade_in(fade_duration).fade_out(fade_duration)

        return audio_segment

    def _generate_audio_segment_wrapper(self, sentence, fade_duration, start_silence_duration, end_silence_duration):
        return self.generate_audio_segment(sentence, fade_duration, start_silence_duration, end_silence_duration)

    def say(
        self, 
        text: str, 
        output_filename: Optional[str] = "combined_output.wav", 
        fallback_duration: int = 1000, 
        fade_duration: int = 50, 
        start_silence_duration: int = 500, 
        end_silence_duration: int = 100
    ) -> None:
        """
        Generate and play audio for a given text, splitting into sentences for better quality.
        
        Args:
            text: Text to synthesize
            output_filename: Optional filename to save the combined audio
            fallback_duration: Duration of silence to use if generation fails
            fade_duration: Duration of fade-in and fade-out in milliseconds
            start_silence_duration: Duration of silence at the beginning in milliseconds
            end_silence_duration: Duration of silence at the end in milliseconds
        """
        # Normalize and split text into sentences
        text = textwrap.dedent(text).strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            print("No valid text to process")
            return

        segments = []
        
        # Import threading for parallel processing
        import threading
        import queue
        
        # Create a queue for segments to be played
        segment_queue = queue.Queue()
        
        # Flag to signal when to stop the player thread
        stop_event = threading.Event()
        
        # Player thread function
        def player_thread():
            while not stop_event.is_set() or not segment_queue.empty():
                try:
                    # Wait for a segment with a timeout
                    seg = segment_queue.get(timeout=0.5)
                    # Play the segment (blocking)
                    play(seg)
                    # Mark task as done
                    segment_queue.task_done()
                except queue.Empty:
                    continue
        
        # Start the player thread
        player = threading.Thread(target=player_thread)
        player.daemon = True
        player.start()
        
        # Generate and queue the first segment
        if sentences:
            try:
                start_time = time.time()
                first_sentence = sentences[0]
                print(f"> {first_sentence} ... ", end='', flush=True)
                
                # Generate first audio segment
                first_seg = self.generate_audio_segment(
                    first_sentence, 
                    fade_duration=fade_duration, 
                    start_silence_duration=start_silence_duration, 
                    end_silence_duration=end_silence_duration
                )
                end_time = time.time()
                
                # Compute metrics for first segment
                duration = first_seg.duration_seconds
                proc_time = end_time - start_time
                rtt_ratio = proc_time / duration
                rtf = 1 / rtt_ratio
                print(f"[Audio: {duration:.2f}s in {proc_time:.2f}s, RTF: {rtf:.2f}x]")
                
                segments.append(first_seg)
                # Queue the first segment for playback
                segment_queue.put(first_seg)
                
            except KeyboardInterrupt:
                print("\nExiting due to KeyboardInterrupt")
                stop_event.set()
                player.join(timeout=1.0)
                return
            except Exception as e:
                print(f"Error generating audio for sentence: {first_sentence}: {e}")
                first_seg = AudioSegment.silent(duration=fallback_duration)
                first_seg = first_seg.fade_in(fade_duration).fade_out(fade_duration)
                segments.append(first_seg)
                segment_queue.put(first_seg)
        
        # Process remaining sentences
        for sentence in sentences[1:]:
            try:
                start_time = time.time()
                print(f"> {sentence} ... ", end='', flush=True)
                
                # Generate audio segment
                seg = self.generate_audio_segment(
                    sentence, 
                    fade_duration=fade_duration, 
                    start_silence_duration=start_silence_duration, 
                    end_silence_duration=end_silence_duration
                )
                end_time = time.time()
                
                # Compute metrics
                duration = seg.duration_seconds
                proc_time = end_time - start_time
                rtt_ratio = proc_time / duration
                rtf = 1 / rtt_ratio
                print(f"[Audio: {duration:.2f}s in {proc_time:.2f}s, RTF: {rtf:.2f}x]")
                
                segments.append(seg)
                # Queue the segment for playback
                segment_queue.put(seg)
                
            except KeyboardInterrupt:
                print("\nExiting due to KeyboardInterrupt")
                break
            except Exception as e:
                print(f"Error generating audio for sentence: {sentence}: {e}")
                seg = AudioSegment.silent(duration=fallback_duration)
                seg = seg.fade_in(fade_duration).fade_out(fade_duration)
                segments.append(seg)
                segment_queue.put(seg)
        
        # Wait for all segments to be played
        segment_queue.join()
        # Signal player thread to exit
        stop_event.set()
        player.join(timeout=1.0)
        
        # Export combined audio if requested
        if output_filename and segments:
            combined = segments[0]
            for seg in segments[1:]:
                combined += seg
            output_path = Path(output_filename)
            logger.debug(f"\nExporting combined audio to {output_path.absolute()}...")
            combined.export(output_filename, format="wav")
            print(f"Export complete: {len(combined) / 1000:.2f} seconds of audio")

    def export_wav(
        self, 
        text: str, 
        output_filename: str, 
        fallback_duration: int = 1000, 
        max_retries: int = 2
    ) -> None:
        """
        Generate audio for a text and export it to a WAV file without playing.
        
        Args:
            text: Text to synthesize
            output_filename: Filename to save the combined audio
            fallback_duration: Duration of silence to use if generation fails
            max_retries: Maximum number of retries if generation fails
        """
        # Split text into sentences
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        segments = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue

            retries = 0
            seg = None
            while retries <= max_retries:
                try:
                    print(f"Export: Generating audio for sentence: {sentence} (Attempt {retries + 1})")
                    seg = self.generate_audio_segment(sentence)
                    break
                except Exception as e:
                    retries += 1
                    print(f"Export: Error for sentence: {sentence} (Attempt {retries}): {e}")
            
            if seg is None:
                print(f"Export: Using fallback for sentence: {sentence}")
                seg = AudioSegment.silent(duration=fallback_duration)
            segments.append(seg)

        if segments:
            # Concatenate segments
            combined = segments[0]
            for seg in segments[1:]:
                combined += seg
            print(f"Exporting to {output_filename}...")
            combined.export(output_filename, format="wav")
            print(f"Export complete: {len(combined) / 1000:.2f} seconds of audio")
        else:
            print("No audio segments to export")
            
    def generate_audio_for_gradio(
        self, 
        text: str,
        fade_duration: int = 50, 
        start_silence_duration: int = 500, 
        end_silence_duration: int = 100
    ) -> tuple:
        """
        Generate audio for a given text and return it in a format suitable for Gradio streaming.
        Returns a tuple containing:
        1. The sample rate
        2. The numpy array of audio samples
        
        Args:
            text: Text to synthesize
            fade_duration: Duration of fade-in and fade-out in milliseconds
            start_silence_duration: Duration of silence at the beginning in milliseconds
            end_silence_duration: Duration of silence at the end in milliseconds
        """
        # Normalize and split text into sentences
        text = textwrap.dedent(text).strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            print("No valid text to process")
            return None
            
        segments = []
        
        # Process all sentences
        for sentence in sentences:
            try:
                print(f"Generating audio for: {sentence}")
                seg = self.generate_audio_segment(
                    sentence, 
                    fade_duration=fade_duration, 
                    start_silence_duration=start_silence_duration, 
                    end_silence_duration=end_silence_duration
                )
                segments.append(seg)
            except Exception as e:
                print(f"Error generating audio for sentence: {sentence}: {e}")
                # Create a short silent segment as fallback
                seg = AudioSegment.silent(duration=1000)
                seg = seg.fade_in(fade_duration).fade_out(fade_duration)
                segments.append(seg)
        
        # Combine all segments
        if segments:
            combined = segments[0]
            for seg in segments[1:]:
                combined += seg
                
            # Convert to numpy array for Gradio
            sample_rate = combined.frame_rate
            audio_np = np.array(combined.get_array_of_samples())
            
            # Normalize to float between -1 and 1
            audio_np = audio_np / (2**15)
            
            return (sample_rate, audio_np)
        else:
            # Return empty audio if no segments were generated
            return (self.generator.sample_rate, np.zeros(0))
            
    def stream_audio_for_gradio(
        self, 
        text: str,
        fade_duration: int = 50, 
        start_silence_duration: int = 500, 
        end_silence_duration: int = 100
    ):
        """
        Generator function that yields audio segments as they are created.
        Each yield returns a tuple of (sample_rate, audio_numpy_array).
        
        Args:
            text: Text to synthesize
            fade_duration: Duration of fade-in and fade-out in milliseconds
            start_silence_duration: Duration of silence at the beginning in milliseconds
            end_silence_duration: Duration of silence at the end in milliseconds
        """
        # Normalize and split text into sentences
        text = textwrap.dedent(text).strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            print("No valid text to process")
            yield (self.generator.sample_rate, np.zeros(0))
            return
        
        # Process each sentence and yield its audio as it's generated
        for i, sentence in enumerate(sentences):
            try:
                print(f"Streaming audio for: {sentence}")
                seg = self.generate_audio_segment(
                    sentence, 
                    fade_duration=fade_duration, 
                    start_silence_duration=start_silence_duration if i == 0 else 100,  # Only add longer silence at the beginning
                    end_silence_duration=end_silence_duration if i == len(sentences)-1 else 100  # Only add longer silence at the end
                )
                
                # Convert to numpy array for Gradio
                sample_rate = seg.frame_rate
                audio_np = np.array(seg.get_array_of_samples())
                
                # Normalize to float between -1 and 1
                audio_np = audio_np / (2**15)
                
                yield (sample_rate, audio_np)
                
            except Exception as e:
                print(f"Error generating audio for sentence: {sentence}: {e}")
                # Create a short silent segment as fallback
                seg = AudioSegment.silent(duration=1000)
                seg = seg.fade_in(fade_duration).fade_out(fade_duration)
                
                sample_rate = seg.frame_rate
                audio_np = np.array(seg.get_array_of_samples())
                audio_np = audio_np / (2**15)
                
                yield (sample_rate, audio_np)


def main():
    """Main entry point for the script."""
    tts = TTS(device="cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tts.load_model()
        warmup = tts.generate_audio_segment("All warmed up baby!")
        play(warmup)
        
        print("\nSesameAI TTS System")
        print("====================")
        while True:
            try:
                user_input = input("\nEnter text (or press Ctrl+C to exit): ")
                if user_input.strip():
                    tts.say(user_input)
                else:
                    print("Please enter some text to generate audio.")
            except Exception as e:
                print(f"Error processing input: {e}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()