import json
import sys
import time
import traceback
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import torch
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import (
    SymmetricPatchifier,
)
from ltx_video.models.transformers.transformer3d import (
    Transformer3DModel,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.inference import calculate_padding, prepare_conditioning


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\MascotAI")

MODEL_DIR = BASE_DIR / "models" / "LTX-Video"

LTX_CHECKPOINT = (
    MODEL_DIR
    / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
)

T5_DIR = MODEL_DIR / "T5-XXL-8bit"

TOKENIZER_DIR = T5_DIR / "tokenizer"

JOB_DIR = BASE_DIR / "temp" / "jobs"

CHATTERBOX_PYTHON = (
    BASE_DIR
    / "chatterbox_venv"
    / "Scripts"
    / "python.exe"
)

AUDIO_WORKER = (
    BASE_DIR
    / "app"
    / "audio_worker.py"
)

JOB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VIDEO SETTINGS
# ============================================================

DIMENSIONS = {
    "16:9": (576, 320),
    "9:16": (320, 576),
    "1:1": (512, 512),
}

STEPS = {
    "Fast": 4,
    "Standard": 8,
    "High": 8,
}

FPS = 24


# ============================================================
# STATUS
# ============================================================

def write_status(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# AUDIO WORKER
# ============================================================

def launch_audio_worker(
    job_id,
    silent_video,
    final_video,
    settings_file,
):
    if not CHATTERBOX_PYTHON.exists():
        raise FileNotFoundError(
            f"Chatterbox Python not found: "
            f"{CHATTERBOX_PYTHON}"
        )

    if not AUDIO_WORKER.exists():
        raise FileNotFoundError(
            f"Audio worker not found: "
            f"{AUDIO_WORKER}"
        )

    subprocess.Popen(
        [
            str(CHATTERBOX_PYTHON),
            str(AUDIO_WORKER),
            job_id,
            str(silent_video),
            str(final_video),
            str(settings_file),
        ],
        cwd=str(BASE_DIR),
        creationflags=getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        ),
    )


# ============================================================
# MOTION PROMPT
# ============================================================

def build_motion_prompt(
    prompt: str,
    motion_mode: str,
) -> str:

    if motion_mode != "High Motion":
        return prompt

    prefix = """
HIGH-MOTION GAMEPLAY DIRECTIVE:

This is a moving gameplay shot, not a posed scene.

Motion must begin immediately and continue throughout
the entire clip.

Use strong, readable temporal change between frames.

The player is actively moving through the environment.

The first-person camera moves with the player's body:
running bob, quick turns, natural head movement,
directional camera acceleration and deceleration.

The weapon and hands move continuously with believable
inertia.

Objects in the environment move through the frame because
the player is moving.

Do not hold a static composition.

Do not slowly zoom on a face.

Do not pose the character.

Do not create a still image with tiny animation.

Do not transition into a gamer-at-PC shot.

Prefer one clear action sequence that can visibly complete
during this short shot.
"""

    suffix = """
TEMPORAL QUALITY:

Continuous physical motion from first frame to last frame.

Clear beginning, middle and end of the requested action.

Strong camera/weapon motion, but keep geometry coherent.

Avoid frozen poses, slideshow-like movement, morphing limbs,
floating weapons, rubbery hands, teleporting enemies or
sudden scene changes.
"""

    return (
        prefix.strip()
        + "\n\n"
        + prompt.strip()
        + "\n\n"
        + suffix.strip()
    )


# ============================================================
# DURATION → LTX FRAME CALCULATION
# ============================================================

def calculate_num_frames(
    settings: dict,
) -> int:
    """
    LTX uses a frame structure compatible with:

        1 second  -> 25 frames
        2 seconds -> 49 frames
        3 seconds -> 73 frames
        4 seconds -> 97 frames
        5 seconds -> 121 frames
        10 seconds -> 241 frames

    This follows:

        frames = (duration * FPS) + 1

    The +1 keeps the frame count compatible with the
    temporal structure used by LTX.
    """

    duration_value = settings.get("duration")

    if duration_value is not None:
        try:
            duration = float(duration_value)

            if duration > 0:
                num_frames = int(
                    round(duration * FPS)
                ) + 1

                return max(
                    9,
                    num_frames,
                )

        except (
            TypeError,
            ValueError,
        ):
            pass

    # Backward compatibility:
    # If duration is not present, use the existing
    # num_frames value from the settings file.

    return max(
        9,
        int(
            settings.get(
                "num_frames",
                25,
            )
        ),
    )


# ============================================================
# ARGUMENT HANDLING
# ============================================================

def main():

    # --------------------------------------------------------
    # Production interface
    #
    # ltx_worker.py
    #   <job_id>
    #   <prompt_file>
    #   <output_file>
    #   <settings_file>
    #
    # Manual test interface
    #
    # ltx_worker.py
    #   generate
    #   <prompt_file>
    #   <output_file>
    #   <settings_file>
    # --------------------------------------------------------

    if (
        len(sys.argv) == 5
        and sys.argv[1].lower() == "generate"
    ):
        job_id = "manual_test"

        prompt_file = Path(
            sys.argv[2]
        )

        output_file = Path(
            sys.argv[3]
        )

        settings_file = Path(
            sys.argv[4]
        )

    elif len(sys.argv) == 5:

        job_id = sys.argv[1]

        prompt_file = Path(
            sys.argv[2]
        )

        output_file = Path(
            sys.argv[3]
        )

        settings_file = Path(
            sys.argv[4]
        )

    else:

        print(
            "Usage:",
            flush=True,
        )

        print(
            "  ltx_worker.py "
            "<job_id> "
            "<prompt_file> "
            "<output_file> "
            "<settings_file>",
            flush=True,
        )

        print(
            "or:",
            flush=True,
        )

        print(
            "  ltx_worker.py generate "
            "<prompt_file> "
            "<output_file> "
            "<settings_file>",
            flush=True,
        )

        sys.exit(1)

    status_file = (
        JOB_DIR
        / f"{job_id}.json"
    )

    try:

        # ====================================================
        # READ SETTINGS
        # ====================================================

        settings = json.loads(
            settings_file.read_text(
                encoding="utf-8-sig"
            )
        )

        raw_prompt = (
            prompt_file.read_text(
                encoding="utf-8-sig"
            )
            .strip()
        )

        if not raw_prompt:
            raise ValueError(
                "Prompt is empty."
            )

        motion_mode = str(
            settings.get(
                "motion_mode",
                "Normal",
            )
        ).strip()

        if motion_mode not in {
            "Normal",
            "High Motion",
        }:
            motion_mode = "Normal"

        prompt = build_motion_prompt(
            raw_prompt,
            motion_mode,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Calculate frames from the requested duration.
        # ----------------------------------------------------

        num_frames = calculate_num_frames(
            settings
        )

        duration_value = settings.get(
            "duration"
        )

        aspect_ratio = settings.get(
            "aspect_ratio",
            "16:9",
        )

        quality = settings.get(
            "quality",
            "Standard",
        )

        image_path_value = settings.get(
            "image_path"
        )

        audio_mode = settings.get(
            "audio_mode",
            "None",
        )

        if aspect_ratio not in DIMENSIONS:
            aspect_ratio = "16:9"

        if quality not in STEPS:
            quality = "Standard"

        width, height = DIMENSIONS[
            aspect_ratio
        ]

        steps = STEPS[
            quality
        ]

        image_path = (
            Path(
                str(image_path_value)
            )
            if image_path_value
            else None
        )

        if (
            image_path
            and not image_path.exists()
        ):
            raise FileNotFoundError(
                "Conditioning image not found: "
                f"{image_path}"
            )

        write_status(
            status_file,
            {
                "status": "loading",
                "job_id": job_id,
                "motion_mode": motion_mode,
            },
        )

        # ====================================================
        # SYSTEM INFORMATION
        # ====================================================

        print(
            "=" * 70,
            flush=True,
        )

        print(
            "LTX VIDEO WORKER",
            flush=True,
        )

        print(
            f"Python: {sys.executable}",
            flush=True,
        )

        print(
            f"PyTorch: {torch.__version__}",
            flush=True,
        )

        print(
            "CUDA available: "
            f"{torch.cuda.is_available()}",
            flush=True,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for LTX generation."
            )

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )

        print(
            "VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
            flush=True,
        )

        print(
            f"LTX checkpoint: "
            f"{LTX_CHECKPOINT}",
            flush=True,
        )

        print(
            f"T5 model: {T5_DIR}",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        # ====================================================
        # T5 TOKENIZER
        # ====================================================

        print(
            "Loading T5-XXL 8-bit on CPU...",
            flush=True,
        )

        tokenizer = (
            T5Tokenizer.from_pretrained(
                str(TOKENIZER_DIR)
            )
        )

        # ====================================================
        # T5 ENCODER
        # ====================================================

        text_encoder = (
            T5EncoderModel.from_pretrained(
                str(T5_DIR),
                device_map={
                    "": "cpu"
                },
            )
        )

        text_encoder.eval()

        print(
            "T5 READY",
            flush=True,
        )

        # ====================================================
        # PROMPT EMBEDDINGS
        # ====================================================

        print(
            "Encoding prompt...",
            flush=True,
        )

        with torch.no_grad():

            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )

            prompt_embeds = (
                text_encoder(
                    tokens.input_ids,
                    attention_mask=(
                        tokens.attention_mask
                    ),
                )[0]
            )

            prompt_attention_mask = (
                tokens.attention_mask
            )

        print(
            "PROMPT EMBEDDINGS: "
            f"{tuple(prompt_embeds.shape)}",
            flush=True,
        )

        # ====================================================
        # TRANSFORMER
        # ====================================================

        print(
            "Loading LTX transformer...",
            flush=True,
        )

        transformer = (
            Transformer3DModel
            .from_pretrained(
                str(LTX_CHECKPOINT),
                dtype=torch.bfloat16,
            )
            .to("cuda")
        )

        transformer.eval()

        print(
            "TRANSFORMER READY",
            flush=True,
        )

        # ====================================================
        # VAE
        # ====================================================

        print(
            "Loading VAE...",
            flush=True,
        )

        vae = (
            CausalVideoAutoencoder
            .from_pretrained(
                str(LTX_CHECKPOINT)
            )
            .to(
                torch.bfloat16
            )
            .to(
                "cuda"
            )
        )

        vae.eval()

        print(
            "VAE READY",
            flush=True,
        )

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler = (
            RectifiedFlowScheduler
            .from_pretrained(
                str(LTX_CHECKPOINT)
            )
        )

        # ====================================================
        # PATCHIFIER
        # ====================================================

        patchifier = (
            SymmetricPatchifier(
                patch_size=1
            )
        )

        # ====================================================
        # LTX PIPELINE
        # ====================================================

        print(
            "Creating LTX pipeline...",
            flush=True,
        )

        pipeline = LTXVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            patchifier=patchifier,
            prompt_enhancer_image_caption_model=None,
            prompt_enhancer_image_caption_processor=None,
            prompt_enhancer_llm_model=None,
            prompt_enhancer_llm_tokenizer=None,
            allowed_inference_steps=None,
        )

        # ====================================================
        # MOVE EMBEDDINGS TO GPU
        # ====================================================

        prompt_embeds = (
            prompt_embeds.to(
                device="cuda",
                dtype=torch.bfloat16,
            )
        )

        prompt_attention_mask = (
            prompt_attention_mask.to(
                "cuda"
            )
        )

        # ====================================================
        # IMPORTANT LTX FIX
        #
        # The T5 embeddings were already calculated.
        # The pipeline must therefore NOT try to run T5 again.
        #
        # We keep tokenizer attached during construction,
        # then replace encode_prompt with our precomputed
        # embeddings.
        # ====================================================

        pipeline.text_encoder = None

        pipeline = pipeline.to(
            "cuda"
        )

        def precomputed_encode_prompt(
            *args,
            **kwargs,
        ):
            return (
                prompt_embeds,
                prompt_attention_mask,
                None,
                None,
            )

        pipeline.encode_prompt = (
            precomputed_encode_prompt
        )

        print(
            "LTX PIPELINE READY",
            flush=True,
        )

        # ====================================================
        # IMAGE CONDITIONING
        # ====================================================

        conditioning_items = None

        if image_path:

            print(
                "Loading conditioning image: "
                f"{image_path}",
                flush=True,
            )

            height_padded = (
                (
                    (height - 1)
                    // 32
                    + 1
                )
                * 32
            )

            width_padded = (
                (
                    (width - 1)
                    // 32
                    + 1
                )
                * 32
            )

            padding = calculate_padding(
                height,
                width,
                height_padded,
                width_padded,
            )

            conditioning_items = (
                prepare_conditioning(
                    conditioning_media_paths=[
                        str(image_path)
                    ],
                    conditioning_strengths=[
                        1.0
                    ],
                    conditioning_start_frames=[
                        0
                    ],
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    padding=padding,
                    pipeline=pipeline,
                )
            )

            print(
                "IMAGE CONDITIONING READY",
                flush=True,
            )

        # ====================================================
        # GENERATION STATUS
        # ====================================================

        write_status(
            status_file,
            {
                "status": "generating",
                "job_id": job_id,
                "motion_mode": motion_mode,
            },
        )

        print(
            f"Motion mode: {motion_mode}",
            flush=True,
        )

        print(
            f"Generating: {prompt}",
            flush=True,
        )

        if duration_value is not None:
            print(
                f"Duration: {duration_value} seconds",
                flush=True,
            )

        print(
            f"Settings: "
            f"{num_frames} frames | "
            f"{width}x{height} | "
            f"{quality} | "
            f"{steps} steps | "
            f"{'I2V' if image_path else 'T2V'}",
            flush=True,
        )

        # ====================================================
        # IMAGE NOISE
        # ====================================================

        image_noise = (
            0.15
            if image_path
            else 0.0
        )

        # ====================================================
        # GENERATION
        # ====================================================

        start = time.perf_counter()

        torch.cuda.empty_cache()

        with torch.no_grad():

            result = pipeline(
                prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=(
                    prompt_attention_mask
                ),
                negative_prompt="",
                do_classifier_free_guidance=False,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=FPS,
                num_inference_steps=steps,
                timesteps=None,
                guidance_scale=1.0,
                stg_scale=0.0,
                rescaling_scale=1.0,
                skip_layer_strategy=None,
                skip_block_list=None,
                decode_timestep=0.05,
                decode_noise_scale=0.025,
                generator=torch.Generator(
                    device="cuda"
                ).manual_seed(
                    int(
                        settings.get(
                            "seed",
                            42,
                        )
                    )
                ),
                output_type="pt",
                return_dict=True,
                is_video=True,
                vae_per_channel_normalize=True,
                conditioning_items=(
                    conditioning_items
                ),
                image_cond_noise_scale=(
                    image_noise
                ),
                mixed_precision=False,
                offload_to_cpu=False,
                enhance_prompt=False,
                text_encoder_max_tokens=256,
                stochastic_sampling=True,
                tone_map_compression_ratio=0.0,
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        # ====================================================
        # VIDEO RESULT
        # ====================================================

        video = result.images

        if video.ndim != 5:
            raise RuntimeError(
                "Unexpected video tensor shape: "
                f"{tuple(video.shape)}"
            )

        frames = (
            video[0]
            .detach()
            .float()
            .clamp(0, 1)
            .permute(
                1,
                2,
                3,
                0,
            )
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # AUDIO
        # ====================================================

        audio_enabled = (
            audio_mode
            in {
                "AI Voice",
                "Voice Clone",
            }
            and bool(
                settings.get(
                    "narration_text",
                    "",
                ).strip()
            )
        )

        silent_video = (
            output_file.with_name(
                output_file.stem
                + "_silent.mp4"
            )
            if audio_enabled
            else output_file
        )

        # ====================================================
        # WRITE VIDEO
        # ====================================================

        imageio.mimsave(
            str(silent_video),
            frames,
            fps=FPS,
            codec="libx264",
            quality=8,
        )

        print(
            f"VIDEO SAVED: "
            f"{silent_video}",
            flush=True,
        )

        # ====================================================
        # AUDIO PIPELINE
        # ====================================================

        if audio_enabled:

            write_status(
                status_file,
                {
                    "status": "audio_generating",
                    "job_id": job_id,
                    "video_generation_time": round(
                        elapsed,
                        2,
                    ),
                    "motion_mode": motion_mode,
                },
            )

            launch_audio_worker(
                job_id,
                silent_video,
                output_file,
                settings_file,
            )

            return

        # ====================================================
        # COMPLETED
        # ====================================================

        write_status(
            status_file,
            {
                "status": "completed",
                "job_id": job_id,
                "filename": output_file.name,
                "generation_time": round(
                    elapsed,
                    2,
                ),
                "motion_mode": motion_mode,
                "mode": (
                    "Photo → Video"
                    if image_path
                    else "AI Video"
                ),
            },
        )

        print(
            f"COMPLETED in "
            f"{elapsed:.2f} seconds",
            flush=True,
        )

    except Exception as exc:

        print(
            "GENERATION ERROR:",
            flush=True,
        )

        traceback.print_exc()

        write_status(
            status_file,
            {
                "status": "failed",
                "job_id": job_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

        sys.exit(1)


if __name__ == "__main__":
    main()