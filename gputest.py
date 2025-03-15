import torch, time
from generator import load_csm_1b
from huggingface_hub import hf_hub_download
import warnings


# Load generator
warnings.filterwarnings("ignore", category=FutureWarning)
generator = load_csm_1b(hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt"), "cuda")
torch.cuda.set_per_process_memory_fraction(0.9, generator.device)

# Set a fixed dummy input (you can adjust batch_shape as desired)
batch_shape = (1, 512, 33)  # fixed shape; you can experiment with different sizes

# Warm-up:
generator._model.reset_caches()
with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _ = generator._model.generate_frame(
                torch.zeros(*batch_shape, device=generator.device).long(),
                torch.ones(*batch_shape, device=generator.device).bool(),
                torch.arange(0, batch_shape[1], device=generator.device)
                    .unsqueeze(0)
                    .long(),
                temperature=0.9,
                topk=50,
            )
torch.cuda.empty_cache()

# Now measure a single forward pass using CUDA timing events.
iterations = 30
total_time = 0.0
for i in range(iterations):
    generator._model.reset_caches()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = generator._model.generate_frame(
                    torch.zeros(*batch_shape, device=generator.device).long(),
                    torch.ones(*batch_shape, device=generator.device).bool(),
                    torch.arange(0, batch_shape[1], device=generator.device)
                        .unsqueeze(0)
                        .long(),
                    temperature=0.9,
                    topk=50,
                )
    end_event.record()
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event)  # in milliseconds
    total_time += elapsed
    print(f"Iteration {i+1}: {elapsed/1000.0:.3f} sec")

avg_time = total_time/iterations/1000.0
print(f"\nAverage forward pass time: {avg_time:.3f} seconds ({1/avg_time:.2f} iterations per second)")
