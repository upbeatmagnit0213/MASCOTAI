import json
import sys
import traceback
from pathlib import Path

import torch

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


BASE_DIR = Path(r"C:\MascotAI")
MODEL_DIR = BASE_DIR / "models" / "Wan2.1-T2V-1.3B"
OUTPUT_DIR = BASE_DIR / "output"
JOB_DIR = BASE_DIR / "temp" / "jobs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)


def write_status(job_file, data):
    job_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


def main():

    if len(sys.argv) != 4:
        print(
            "Usage: wan_worker.py <job_id> <prompt_file> <output_file>",
            flush=True
        )
        sys.exit(1)

    job_id = sys.argv[1]
    prompt_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3])

    status_file = JOB_DIR / f"{job_id}.json"

    try:

        prompt = prompt_file.read_text(
            encoding="utf-8"
        ).strip()

        write_status(
            status_file,
            {
                "status": "loading",
                "job_id": job_id
            }
        )

        print("Loading DiffSynth Wan...", flush=True)

        vram_config = {
            "offload_device": "cpu",
            "offload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "preparing_dtype": torch.bfloat16,
            "computation_device": "cuda",
            "computation_dtype": torch.bfloat16,
        }

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(
                    path=str(
                        MODEL_DIR /
                        "diffusion_pytorch_model.safetensors"
                    ),
                    **vram_config
                ),
                ModelConfig(
                    path=str(
                        MODEL_DIR /
                        "models_t5_umt5-xxl-enc-bf16.pth"
                    ),
                    **vram_config
                ),
                ModelConfig(
                    path=str(
                        MODEL_DIR /
                        "Wan2.1_VAE.pth"
                    ),
                    **vram_config
                ),
            ],
            tokenizer_config=ModelConfig(
                path=str(
                    MODEL_DIR /
                    "google" /
                    "umt5-xxl"
                )
            ),
            vram_limit=(
                torch.cuda.mem_get_info("cuda")[1]
                / (1024 ** 3)
                - 2
            ),
        )

        print(
            "DIFFSYNTH WAN READY",
            flush=True
        )

        write_status(
            status_file,
            {
                "status": "generating",
                "job_id": job_id
            }
        )

        print(
            f"Generating: {prompt}",
            flush=True
        )

        video = pipe(
            prompt=prompt,

            negative_prompt=(
                "blurry, distorted, glitch, pixelated, "
                "noisy, low quality, deformed, "
                "static image, text, subtitles, "
                "watermark, oversaturated"
            ),

            seed=42,

            height=832,
            width=480,

            num_frames=81,

            cfg_scale=6.0,

            num_inference_steps=50,

            sigma_shift=8.0,

            tiled=True,

            output_type="quantized",
        )

        print(
            "VIDEO GENERATED",
            flush=True
        )

        from diffsynth.utils.data import save_video

        save_video(
            video,
            str(output_file),
            fps=15,
            quality=5
        )

        print(
            f"VIDEO SAVED: {output_file}",
            flush=True
        )

        write_status(
            status_file,
            {
                "status": "completed",
                "job_id": job_id,
                "filename": output_file.name
            }
        )

    except Exception as e:

        print(
            "GENERATION ERROR:",
            flush=True
        )

        traceback.print_exc()

        write_status(
            status_file,
            {
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

        sys.exit(1)


if __name__ == "__main__":
    main()

