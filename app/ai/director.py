from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re

import requests
from dotenv import load_dotenv


# ============================================================
# PROJECT / ENVIRONMENT
# ============================================================

BASE_DIR = Path(r"C:\MascotAI")
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# NVIDIA AI DIRECTOR CONFIGURATION
# ============================================================

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "openai/gpt-oss-20b"


# ============================================================
# LTX-SAFE DIRECTOR SYSTEM PROMPT
# ============================================================

DIRECTOR_SYSTEM_PROMPT = """
You are the AI Director for a real AI video generation system.

The generated plan will be sent to an LTX-Video generation engine.

Your job is NOT to create a complicated movie storyboard.

Your job is to create a SIMPLE, GENERATABLE, COHERENT video plan
that an image-to-video / text-to-video model can reliably generate.

LTX VIDEO LIMITATIONS:

- One main visible subject per scene.
- Keep the same main subject throughout the entire video.
- Prefer one subject performing one primary action.
- Use simple environments.
- Use simple camera movement.
- Avoid complex choreography.
- Avoid multiple simultaneous actions.
- Avoid crowds.
- Avoid multiple characters.
- Avoid vehicles.
- Avoid drones.
- Avoid holograms.
- Avoid readable text.
- Avoid signs and billboards.
- Avoid complicated UI.
- Avoid complicated visual effects.
- Avoid impossible physics.
- Avoid rapid camera rotations.
- Avoid 180-degree or 360-degree camera movements.
- Avoid complicated scene changes.
- Avoid multiple locations inside one scene.
- Avoid close-up shots if they can hide the full subject.
- Prefer medium or full-body framing.
- Keep the main subject centered and clearly visible.
- Use natural physical movement.
- Use one primary camera movement at most.
- The subject should remain visually consistent between scenes.

GOOD ACTION EXAMPLES:

- walks forward
- runs forward
- slowly turns
- looks around
- sits down
- stands up
- waves
- raises one arm
- walks toward the camera
- walks through a simple environment

BAD ACTION EXAMPLES:

- walks, turns, scans, jumps, activates an interface, and flies away
- complex fighting choreography
- multiple characters interacting
- complicated object manipulation
- multiple camera movements
- 360-degree camera rotation

GOOD CAMERA EXAMPLES:

- static camera
- slow push in
- slow pull back
- gentle tracking shot
- slow camera follow

BAD CAMERA EXAMPLES:

- 360-degree orbit
- 180-degree rotation
- rapid camera movement
- multiple camera transitions
- complex crane plus orbit plus zoom

VISUAL STYLE:

The scene should look cinematic and realistic, but the prompt must remain
simple enough for LTX to understand.

IMPORTANT:

Return ONLY valid JSON.

The JSON must contain no markdown and no explanation outside JSON.

Use a maximum of 4 scenes.

For short videos, 3 scenes is preferred.

Every scene must contain:

scene_number
title
duration_seconds
visual_prompt
camera
action
lighting
dialogue
voiceover
music
sound_effects
visual_effects

The total scene duration must equal the requested duration exactly.
"""


# ============================================================
# FORBIDDEN / COMPLEX CONTENT
# ============================================================

FORBIDDEN_PATTERNS = [
    r"\bcrowd(s)?\b",
    r"\bpedestrian(s)?\b",
    r"\bpeople\b",
    r"\bperson\b",
    r"\bhologram(s)?\b",
    r"\bholographic\b",
    r"\bdrone(s)?\b",
    r"\bhovercar(s)?\b",
    r"\bvehicle(s)?\b",
    r"\bbillboard(s)?\b",
    r"\badvertisement(s)?\b",
    r"\bdigital billboard(s)?\b",
    r"\binterface\b",
    r"\bUI\b",
    r"\bHUD\b",
    r"\b360[- ]?degree\b",
    r"\b360°\b",
    r"\b180[- ]?degree\b",
    r"\b180°\b",
    r"\borbit around\b",
    r"\bflies away\b",
    r"\bflight\b",
    r"\bcomplex choreography\b",
    r"\bmultiple characters\b",
    r"\bmultiple subjects\b",
    r"\bneon signs?\b",
    r"\bstreet signs?\b",
    r"\bdigital signs?\b",
    r"\breadable text\b",
    r"\btext appears\b",
    r"\btext displays\b",
]


FORBIDDEN_REPLACEMENTS = {
    "crowd": "empty environment",
    "crowds": "empty environment",
    "pedestrians": "empty environment",
    "people": "empty environment",
    "person": "empty environment",
    "hologram": "soft light",
    "holograms": "soft light",
    "holographic": "soft glowing",
    "drone": "soft light",
    "drones": "soft light",
    "hovercar": "empty street",
    "hovercars": "empty street",
    "vehicle": "empty street",
    "vehicles": "empty street",
    "billboard": "background light",
    "billboards": "background lights",
    "advertisement": "background light",
    "advertisements": "background lights",
    "digital billboard": "background lights",
    "interface": "soft light",
    "UI": "soft light",
    "HUD": "soft light",
    "orbit around": "slow camera follow",
    "360-degree": "slow camera movement",
    "360 degree": "slow camera movement",
    "360°": "slow camera movement",
    "180-degree": "slow camera movement",
    "180 degree": "slow camera movement",
    "180°": "slow camera movement",
    "neon signs": "soft neon lights",
    "neon sign": "soft neon light",
    "street signs": "soft background lights",
    "street sign": "soft background light",
    "digital signs": "soft background lights",
    "digital sign": "soft background light",
    "readable text": "simple background",
    "text appears": "simple background",
    "text displays": "simple background",
    "flies away": "walks forward",
    "flight": "walking movement",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _get_nvidia_api_key():
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. "
            "Check C:\\MascotAI\\.env."
        )

    return api_key


def _extract_json(text: str):
    text = text.strip()

    # Remove markdown code fences if the model adds them.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # First attempt: complete response is JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second attempt: extract first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "NVIDIA returned a response that does not contain valid JSON."
        )

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"NVIDIA returned invalid JSON: {exc}"
        ) from exc


# ============================================================
# LTX SAFETY CLEANING
# ============================================================

def _remove_forbidden_phrases(text: str):
    """
    Remove or simplify content that LTX commonly struggles with.
    """

    if not text:
        return ""

    cleaned = str(text)

    # Apply direct replacements first.
    for old, new in FORBIDDEN_REPLACEMENTS.items():
        cleaned = cleaned.replace(old, new)

    # Remove any remaining forbidden expressions.
    for pattern in FORBIDDEN_PATTERNS:
        cleaned = re.sub(
            pattern,
            "simple environment",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Remove duplicate spaces.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def _simplify_action(action: str):
    """
    Keep the first primary action instead of allowing a long
    chain of actions that LTX may fail to reproduce.
    """

    if not action:
        return "natural movement"

    action = _remove_forbidden_phrases(action)

    # Split common action chains.
    parts = re.split(
        r"\s*(?:,\s*then\s+|\s+then\s+|\s+and\s+then\s+|;\s*)",
        action,
        flags=re.IGNORECASE,
    )

    if parts:
        first = parts[0].strip()

        if first:
            return first

    return "natural movement"


def _simplify_camera(camera: str):
    """
    Keep camera direction simple.
    """

    if not camera:
        return "static camera"

    camera = _remove_forbidden_phrases(camera)

    lower = camera.lower()

    if "360" in lower or "orbit" in lower:
        return "slow camera follow"

    if "180" in lower:
        return "slow camera follow"

    if "tracking" in lower:
        return "gentle tracking shot"

    if "follow" in lower:
        return "slow camera follow"

    if "zoom out" in lower or "pull back" in lower:
        return "slow pull back"

    if "zoom in" in lower or "push in" in lower:
        return "slow push in"

    return "static camera"


def _simplify_scene(scene: dict):
    """
    Convert one Director scene into a safer LTX-oriented scene.
    """

    if not isinstance(scene, dict):
        scene = {}

    visual_prompt = _remove_forbidden_phrases(
        scene.get("visual_prompt", "")
    )

    action = _simplify_action(
        scene.get("action", "")
    )

    camera = _simplify_camera(
        scene.get("camera", "")
    )

    lighting = _remove_forbidden_phrases(
        scene.get("lighting", "")
    )

    # Limit visual prompt complexity.
    sentences = re.split(r"(?<=[.!?])\s+", visual_prompt)

    if len(sentences) > 4:
        visual_prompt = " ".join(sentences[:4])

    # Keep visual effects extremely simple.
    visual_effects = scene.get("visual_effects", [])

    if not isinstance(visual_effects, list):
        visual_effects = []

    safe_effects = []

    for effect in visual_effects:
        effect_text = _remove_forbidden_phrases(str(effect))

        if effect_text and len(safe_effects) < 2:
            safe_effects.append(effect_text)

    # Keep sound effects simple.
    sound_effects = scene.get("sound_effects", [])

    if not isinstance(sound_effects, list):
        sound_effects = []

    safe_sounds = []

    for sound in sound_effects:
        sound_text = _remove_forbidden_phrases(str(sound))

        if sound_text and len(safe_sounds) < 3:
            safe_sounds.append(sound_text)

    scene["visual_prompt"] = visual_prompt
    scene["action"] = action
    scene["camera"] = camera
    scene["lighting"] = lighting or "cinematic lighting"
    scene["visual_effects"] = safe_effects
    scene["sound_effects"] = safe_sounds

    # These fields are preserved for the production pipeline.
    scene.setdefault("dialogue", "")
    scene.setdefault("voiceover", "")
    scene.setdefault("music", "cinematic background music")

    return scene


# ============================================================
# SCENE DURATION NORMALISATION
# ============================================================

def _normalise_scene_durations(plan: dict, duration: int):
    scenes = plan.get("scenes", [])

    if not scenes:
        raise ValueError("AI Director returned no scenes.")

    # Maximum of 4 scenes.
    scenes = scenes[:4]

    # Prefer exactly 3 scenes for normal short videos.
    if duration >= 6 and len(scenes) > 3:
        scenes = scenes[:3]

    scene_count = len(scenes)

    if scene_count == 0:
        raise ValueError("AI Director returned no usable scenes.")

    base_duration = duration // scene_count
    remainder = duration % scene_count

    for index, scene in enumerate(scenes):
        scene_duration = base_duration

        if index < remainder:
            scene_duration += 1

        # Prevent zero-second scenes.
        if scene_duration < 1:
            scene_duration = 1

        scene["duration_seconds"] = scene_duration

    # Recalculate if extremely short duration created mismatch.
    total = sum(
        int(scene["duration_seconds"])
        for scene in scenes
    )

    if total != duration:
        difference = duration - total
        scenes[-1]["duration_seconds"] += difference

    plan["scenes"] = scenes

    return plan


# ============================================================
# PLAN VALIDATION
# ============================================================

def _validate_plan(plan: dict, duration: int, aspect_ratio: str):

    if not isinstance(plan, dict):
        raise ValueError(
            "AI Director response is not a JSON object."
        )

    scenes = plan.get("scenes")

    if not isinstance(scenes, list) or not scenes:
        raise ValueError(
            "AI Director response contains no scenes."
        )

    plan["duration"] = duration
    plan["aspect_ratio"] = aspect_ratio

    plan = _normalise_scene_durations(
        plan,
        duration,
    )

    cleaned_scenes = []

    for index, scene in enumerate(plan["scenes"]):

        scene = _simplify_scene(scene)

        scene["scene_number"] = index + 1
        scene["scene_id"] = index + 1

        scene.setdefault(
            "title",
            f"Scene {index + 1}",
        )

        if not scene.get("visual_prompt"):
            scene["visual_prompt"] = (
                "The main subject is clearly visible in a "
                "simple cinematic environment with realistic lighting."
            )

        if not scene.get("action"):
            scene["action"] = "natural movement"

        if not scene.get("camera"):
            scene["camera"] = "static camera"

        cleaned_scenes.append(scene)

    plan["scenes"] = cleaned_scenes

    total_duration = sum(
        int(scene["duration_seconds"])
        for scene in plan["scenes"]
    )

    if total_duration != duration:
        raise ValueError(
            f"Scene duration mismatch: expected {duration}, "
            f"got {total_duration}."
        )

    # Explicit metadata proving this plan has passed the LTX-safe layer.
    plan["version"] = "2.1"
    plan["generation_engine"] = "LTX-friendly"

    return plan


# ============================================================
# NVIDIA AI DIRECTOR
# ============================================================

def create_video_plan(
    user_prompt: str,
    duration: int = 10,
    aspect_ratio: str = "9:16",
):
    """
    Generate a production plan using NVIDIA GPT-OSS 20B,
    then convert it into an LTX-safe plan.
    """

    user_prompt = user_prompt.strip()

    if not user_prompt:
        raise ValueError(
            "Video prompt cannot be empty."
        )

    if duration <= 0:
        raise ValueError(
            "Video duration must be greater than zero."
        )

    if duration > 300:
        raise ValueError(
            "Maximum supported video duration is 300 seconds."
        )

    aspect_ratio = aspect_ratio.strip()

    if not aspect_ratio:
        aspect_ratio = "9:16"

    api_key = _get_nvidia_api_key()

    user_message = f"""
Create an LTX-friendly video plan for this request:

USER REQUEST:
{user_prompt}

REQUESTED DURATION:
{duration} seconds

REQUESTED ASPECT RATIO:
{aspect_ratio}

STRICT GENERATION REQUIREMENTS:

- Maximum 3 scenes for this video.
- One main subject.
- Keep the subject consistent.
- One primary action per scene.
- Simple environment.
- No crowds.
- No other characters.
- No vehicles.
- No drones.
- No holograms.
- No billboards.
- No readable signs.
- No readable text.
- No complex UI.
- No complicated choreography.
- No 180-degree camera movement.
- No 360-degree camera movement.
- No complicated camera movement.
- Use static, slow push, slow pull-back, or gentle tracking.
- Prefer medium or full-body framing.
- Keep the subject visible and centered.
- Use realistic physical movement.
- Make every scene independently generatable.
- The same main subject must appear in every scene.

The user's requested duration is authoritative.

Return ONLY JSON.

Use this structure:

{{
  "title": "short title",
  "description": "short description",
  "style": "simple visual style",
  "pacing": "simple pacing",
  "transition_style": "simple transition",
  "scenes": [
    {{
      "scene_number": 1,
      "title": "scene title",
      "duration_seconds": 4,
      "visual_prompt": "simple detailed visual generation prompt",
      "camera": "one simple camera movement",
      "action": "one primary subject action",
      "lighting": "simple lighting",
      "dialogue": "",
      "voiceover": "",
      "music": "music direction",
      "sound_effects": [],
      "visual_effects": []
    }}
  ],
  "audio": {{
    "voice": false,
    "music": true,
    "sound_effects": true,
    "captions": false
  }}
}}

The scene durations must add up exactly to {duration} seconds.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": DIRECTOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "temperature": 0.15,
        "max_tokens": 3000,
        "reasoning_effort": "low",
        "stream": False,
    }

    try:
        response = requests.post(
            NVIDIA_API_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to NVIDIA API: {exc}"
        ) from exc

    if response.status_code != 200:

        try:
            error_data = response.json()
        except Exception:
            error_data = response.text

        raise RuntimeError(
            f"NVIDIA API error {response.status_code}: "
            f"{error_data}"
        )

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "NVIDIA returned an invalid API response."
        ) from exc

    choices = response_data.get("choices", [])

    if not choices:
        raise RuntimeError(
            "NVIDIA API returned no choices."
        )

    message = choices[0].get("message", {})
    content = message.get("content")

    if not content:
        raise RuntimeError(
            "NVIDIA returned no visible content. "
            f"Finish reason: {choices[0].get('finish_reason')}"
        )

    plan = _extract_json(content)

    plan = _validate_plan(
        plan,
        duration=duration,
        aspect_ratio=aspect_ratio,
    )

    plan["original_prompt"] = user_prompt
    plan["generated_by"] = NVIDIA_MODEL
    plan["generated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    return plan


# ============================================================
# SAVE VIDEO PLAN
# ============================================================

def save_video_plan(plan: dict, output_path):
    """
    Save the generated video plan as JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            plan,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_prompt = (
        "A small red humanoid robot with glowing blue eyes "
        "walking through a neon city at night"
    )

    print("Creating LTX-safe Director test...")

    plan = create_video_plan(
        test_prompt,
        duration=12,
        aspect_ratio="9:16",
    )

    output_file = (
        BASE_DIR
        / "temp"
        / "director_ltx_test.json"
    )

    save_video_plan(
        plan,
        output_file,
    )

    print("LTX DIRECTOR TEST OK")
    print(f"Saved: {output_file}")
    print(
        f"Scenes: {len(plan['scenes'])} | "
        f"Duration: {plan['duration']} seconds | "
        f"Engine: {plan['generation_engine']}"
    )