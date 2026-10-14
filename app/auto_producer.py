import argparse
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


BASE_DIR = Path(r"C:\MascotAI")

MODEL_NAME = "Qwen/Qwen3-1.7B"

LOCAL_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "Qwen3-1.7B"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "temp"
    / "auto_plan.json"
)


# ============================================================
# DURATION → SHOT COUNT
# ============================================================

def choose_scene_count(seconds: int) -> int:

    if seconds <= 10:
        return 3

    if seconds <= 15:
        return 4

    if seconds <= 25:
        return 5

    if seconds <= 30:
        return 6

    if seconds <= 45:
        return 8

    if seconds <= 60:
        return 10

    if seconds <= 90:
        return 14

    return 18


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "Loading Qwen3-1.7B planner...",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(
                LOCAL_MODEL_DIR
            ),
        )
    )

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        dtype = torch.bfloat16

        print(
            "Planner GPU:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    else:

        device = torch.device(
            "cpu"
        )

        dtype = torch.float32

        print(
            "WARNING: Planner running on CPU.",
            flush=True,
        )


    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            cache_dir=str(
                LOCAL_MODEL_DIR
            ),
            torch_dtype=dtype,
        )
        .to(device)
    )

    model.eval()

    print(
        "QWEN PLANNER READY",
        flush=True,
    )

    return (
        tokenizer,
        model,
        device,
    )


# ============================================================
# PROMPT
# ============================================================

def build_planner_prompt(
    user_prompt: str,
    duration: int,
    aspect_ratio: str,
    video_format: str,
):

    scene_count = choose_scene_count(
        duration
    )

    return f"""
You are the director and production planner for a professional AI video generator.

Turn the user's request into a complete production plan.

USER REQUEST:
{user_prompt}

VIDEO SETTINGS:
Duration: {duration} seconds
Aspect ratio: {aspect_ratio}
Format: {video_format}
Target scenes: {scene_count}

IMPORTANT RULES:

1. The final result must tell a coherent story from beginning to end.

2. Break the video into multiple connected scenes.

3. Every scene must have:
   - start_time
   - end_time
   - duration
   - visual_prompt
   - camera
   - action
   - transition

4. The visual_prompt must be written specifically for a video generation model.

5. Maintain continuity:
   - same main characters
   - same clothing
   - same environment
   - same objects
   - consistent visual style

6. If the user requests gaming content, make it feel like authentic gameplay rather than a movie trailer.

7. If the request contains a timeline, preserve the requested events and ordering.

8. Automatically determine whether narration is useful.

9. Automatically create concise spoken narration for scenes that need it.

10. Automatically determine appropriate:
    - background music style
    - sound effects
    - ambience
    - important impact moments

11. Automatically determine captions/on-screen text when useful.

12. Do not invent copyrighted brands or recognizable characters.

13. Do not add logos or watermarks.

14. The narration must fit naturally inside the scene timing.

15. The final video should be entertaining and should have a clear opening hook and satisfying ending.

RETURN ONLY VALID JSON.

Use exactly this structure:

{{
  "project": {{
    "duration": {duration},
    "aspect_ratio": "{aspect_ratio}",
    "format": "{video_format}",
    "title": "",
    "genre": "",
    "style": "",
    "story_summary": ""
  }},

  "audio": {{
    "voice_required": true,
    "voice_style": "",
    "music_required": true,
    "music_style": "",
    "music_energy": "",
    "global_ambience": "",
    "sfx_required": true
  }},

  "scenes": [
    {{
      "scene_id": 1,
      "start_time": 0,
      "end_time": 0,
      "duration": 0,
      "purpose": "hook",
      "visual_prompt": "",
      "camera": "",
      "action": "",
      "continuity": "",
      "narration": "",
      "sfx": [],
      "on_screen_text": "",
      "transition": ""
    }}
  ],

  "final_edit": {{
    "pacing": "",
    "caption_style": "",
    "transition_style": "",
    "ending": ""
  }}
}}

Do not write explanations outside the JSON.
"""


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(text: str):

    text = text.strip()

    # Remove Qwen thinking blocks if returned.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()

    # Remove markdown fences.
    text = text.replace(
        "```json",
        "",
    )

    text = text.replace(
        "```",
        "",
    )

    start = text.find("{")

    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Planner did not return JSON."
        )

    candidate = text[
        start : end + 1
    ]

    return json.loads(
        candidate
    )


# ============================================================
# NORMALIZE PLAN
# ============================================================

def normalize_plan(
    plan: dict,
    duration: int,
    aspect_ratio: str,
    video_format: str,
):

    project = plan.setdefault(
        "project",
        {},
    )

    project["duration"] = duration
    project["aspect_ratio"] = (
        aspect_ratio
    )
    project["format"] = (
        video_format
    )


    audio = plan.setdefault(
        "audio",
        {},
    )

    audio.setdefault(
        "voice_required",
        True,
    )

    audio.setdefault(
        "music_required",
        True,
    )

    audio.setdefault(
        "sfx_required",
        True,
    )


    scenes = plan.get(
        "scenes",
        [],
    )

    if not scenes:
        raise ValueError(
            "Planner returned no scenes."
        )


    # Ensure scene timing is valid.
    current = 0.0

    for index, scene in enumerate(
        scenes,
        start=1,
    ):

        scene["scene_id"] = index

        scene_duration = float(
            scene.get(
                "duration",
                0,
            )
        )

        if scene_duration <= 0:

            remaining = (
                duration
                - current
            )

            scene_duration = max(
                1.0,
                remaining,
            )

        start = current

        end = min(
            duration,
            start + scene_duration,
        )

        scene["start_time"] = round(
            start,
            2,
        )

        scene["end_time"] = round(
            end,
            2,
        )

        scene["duration"] = round(
            end - start,
            2,
        )

        current = end

        if current >= duration:
            break


    # If the model created a gap,
    # extend the final scene to the end.
    if scenes:

        last = scenes[-1]

        if last["end_time"] < duration:

            last["end_time"] = duration

            last["duration"] = round(
                duration -
                last["start_time"],
                2,
            )


    return plan


# ============================================================
# GENERATE PLAN
# ============================================================

def generate_plan(
    tokenizer,
    model,
    device,
    user_prompt,
    duration,
    aspect_ratio,
    video_format,
):

    planner_prompt = (
        build_planner_prompt(
            user_prompt,
            duration,
            aspect_ratio,
            video_format,
        )
    )


    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional "
                "AI video production planner. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": planner_prompt,
        },
    ]


    prompt_text = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    print(
        "Planning scenes...",
        flush=True,
    )


    with torch.inference_mode():

        output = model.generate(
            **inputs,

            max_new_tokens=1800,

            do_sample=False,

            temperature=None,

            top_p=None,

            repetition_penalty=1.05,

        )


    generated = output[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]


    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


    plan = extract_json(
        text
    )


    plan = normalize_plan(
        plan,
        duration,
        aspect_ratio,
        video_format,
    )


    return plan


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        required=True,
        help="User video prompt",
    )

    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        help="Video duration in seconds",
    )

    parser.add_argument(
        "--aspect-ratio",
        default="9:16",
    )

    parser.add_argument(
        "--format",
        default="YouTube Short",
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    args = parser.parse_args()


    if args.duration <= 0:
        raise ValueError(
            "Duration must be positive."
        )


    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    tokenizer, model, device = (
        load_model()
    )


    plan = generate_plan(
        tokenizer,
        model,
        device,
        args.prompt,
        args.duration,
        args.aspect_ratio,
        args.format,
    )


    output_path.write_text(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print()
    print(
        "AUTO PRODUCER COMPLETE"
    )
    print(
        "PLAN SAVED:",
        output_path,
    )


    scenes = plan.get(
        "scenes",
        [],
    )

    print(
        "SCENES:",
        len(scenes),
    )

    print(
        "VOICE:",
        plan.get(
            "audio",
            {}
        ).get(
            "voice_required",
            True,
        ),
    )

    print(
        "MUSIC:",
        plan.get(
            "audio",
            {}
        ).get(
            "music_required",
            True,
        ),
    )

    print(
        "SFX:",
        plan.get(
            "audio",
            {}
        ).get(
            "sfx_required",
            True,
        ),
    )


if __name__ == "__main__":
    main()