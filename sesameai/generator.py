from dataclasses import dataclass
from typing import List, Tuple, Generator as PyGenerator, Optional, Callable
import time
import queue
import threading

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from .models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device
        
        # Buffer size for streaming (number of frames to collect before decoding)
        self._stream_buffer_size = 10

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
    
    def _decode_frames(self, frames):
        """Decode a batch of frames into audio"""
        if not frames:
            return torch.tensor([])
            
        audio = self._audio_tokenizer.decode(torch.stack(frames).permute(1, 2, 0)).squeeze(0).squeeze(0)
        return audio

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> PyGenerator[torch.Tensor, None, None]:
        """
        Stream audio chunks as they're generated.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: Context segments
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            on_chunk_generated: Optional callback function to be called with each audio chunk
            
        Yields:
            Audio chunks as they're generated
        """
        # Clear the GPU memory cache to free up space
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        # Frame buffer for streaming
        frame_buffer = []
        
        # Process frames for streaming
        for i in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                # EOS token, finish generation
                break

            frame_buffer.append(sample)
            
            # When we have enough frames in the buffer, decode and yield
            if len(frame_buffer) >= self._stream_buffer_size:
                audio_chunk = self._decode_frames(frame_buffer)
                frame_buffer = []
                
                if on_chunk_generated:
                    on_chunk_generated(audio_chunk)
                    
                yield audio_chunk

            # Update for next token generation
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        
        # Decode and yield any remaining frames
        if frame_buffer:
            audio_chunk = self._decode_frames(frame_buffer)
            if on_chunk_generated:
                on_chunk_generated(audio_chunk)
            yield audio_chunk

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        stream: bool = False,
    ) -> torch.Tensor:
        """
        Generate audio for the given text.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: Context segments
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            stream: Whether to use streaming generation internally
            
        Returns:
            Generated audio tensor
        """
        if stream:
            # Use streaming implementation but combine all chunks at the end
            audio_chunks = []
            for chunk in self.generate_stream(text, speaker, context, max_audio_length_ms, temperature, topk):
                audio_chunks.append(chunk)
            
            if not audio_chunks:
                return torch.tensor([])
                
            # Concatenate all chunks
            return torch.cat(audio_chunks)
        
        # Non-streaming implementation
        # Clear the GPU memory cache to free up space
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for i in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            return torch.tensor([])
            
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        return audio


class AudioStreamWriter:
    """
    Helper class for writing streaming audio to a file.
    """
    def __init__(self, filename, sample_rate):
        self.filename = filename
        self.sample_rate = sample_rate
        self.audio_chunks = []
        self.lock = threading.Lock()
        
    def add_chunk(self, chunk):
        """Add an audio chunk to the buffer."""
        with self.lock:
            self.audio_chunks.append(chunk)
    
    def write_file(self):
        """Write all collected audio chunks to file."""
        with self.lock:
            if not self.audio_chunks:
                return
                
            # Concatenate all chunks
            audio = torch.cat(self.audio_chunks)
            # Save to file
            torchaudio.save(self.filename, audio.unsqueeze(0).cpu(), self.sample_rate)


def load_csm_1b(device: str = "cuda") -> Generator:
    # Enable cudnn benchmarking for optimal kernel selection
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # Enable flash attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
    
    model = Model.from_pretrained("sesame/csm-1b")
    model.decoder = torch.compile(model.decoder, mode='max-autotune', fullgraph=True, backend='inductor')
    # model.backbone = torch.compile(model.backbone,mode='max-autotune', fullgraph=True, backend='inductor')

    # Apply half-precision for faster inference
    model.to(device=device, dtype=torch.bfloat16)
    
    generator = Generator(model)
    return generator


def generate_streaming_audio(
    generator: Generator,
    text: str,
    speaker: int,
    context: List[Segment],
    output_file: str,
    max_audio_length_ms: float = 90_000,
    temperature: float = 0.7,
    topk: int = 30,
    play_audio: bool = False,
):
    """
    Generate audio with streaming output, optionally playing it in real-time.
    
    Args:
        generator: CSM generator
        text: Text to synthesize
        speaker: Speaker ID
        context: Context segments
        output_file: File to save the audio to
        max_audio_length_ms: Maximum audio length in milliseconds
        temperature: Sampling temperature
        topk: Top-k sampling parameter
        play_audio: Whether to play audio in real-time
    """
    # Create audio writer
    writer = AudioStreamWriter(output_file, generator.sample_rate)
    
    # Set up audio player if needed
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    
    if play_audio:
        try:
            import sounddevice as sd
            
            # Define player thread
            def audio_player():
                while not stop_event.is_set() or not audio_queue.empty():
                    try:
                        chunk = audio_queue.get(timeout=0.5)
                        sd.play(chunk.cpu().numpy(), generator.sample_rate)
                        sd.wait()
                    except queue.Empty:
                        continue
            
            # Start player thread
            player_thread = threading.Thread(target=audio_player)
            player_thread.start()
            
        except ImportError:
            print("sounddevice library not found. Install with 'pip install sounddevice' to enable real-time playback.")
            play_audio = False
    
    # Define callback for handling generated chunks
    def on_chunk_generated(chunk):
        writer.add_chunk(chunk)
        if play_audio:
            audio_queue.put(chunk)
    
    print("Generating audio in streaming mode...")
    start_time = time.time()
    
    # Generate audio chunks
    chunk_count = 0
    for _ in generator.generate_stream(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk,
        on_chunk_generated=on_chunk_generated
    ):
        chunk_count += 1
        print(f"Generated chunk {chunk_count}")
    
    # Write final file
    writer.write_file()
    
    # Clean up player if active
    if play_audio:
        stop_event.set()
        player_thread.join()
    
    print(f"Audio generation completed in {time.time() - start_time:.2f} seconds")