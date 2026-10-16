import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(r"C:\MascotAI")
APP_DIR = BASE_DIR / "app"
TEMP_DIR = BASE_DIR / "temp"
JOBS_DIR = TEMP_DIR / "jobs"
OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_PLAN = TEMP_DIR / "auto_plan.json"
DEFAULT_PROMPT = TEMP_DIR / "auto_test_prompt.txt"

LTX_PYTHON = BASE_DIR / "ltx_venv" / "Scripts" / "python.exe"
CHATTERBOX_PYTHON = BASE_DIR / "chatterbox_venv" / "Scripts" / "python.exe"
LTX_WORKER = APP_DIR / "ltx_worker.py"
AUDIO_WORKER = APP_DIR / "audio_worker.py"
FFMPEG = "ffmpeg"

FPS = 24
AUDIO_SR = 48000

# Exact, practical gaming timing for the current 30 s test.
GAMING_DURATIONS = {
    10: [2, 2, 2, 2, 1, 1],
    25: [2, 5, 5, 6, 5, 2],
    30: [2, 6, 6, 7, 6, 3],
    45: [3, 9, 9, 10, 9, 5],
    60: [3, 12, 12, 14, 12, 7],
    120: [5, 25, 25, 30, 25, 10],
}


class StepError(RuntimeError):
    pass


def run_cmd(cmd, *, cwd=BASE_DIR, title=None):
    if title:
        print(f"\n=== {title} ===", flush=True)
    print("$ " + " ".join(_quote_for_log(x) for x in cmd), flush=True)
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise StepError(f"Command failed with exit code {proc.returncode}.")
    return proc.returncode


def _quote_for_log(value):
    value = str(value)
    if any(ch.isspace() for ch in value):
        return f'"{value}"'
    return value


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def repair_mojibake(text):
    if not isinstance(text, str):
        return text
    if "Ãƒ" not in text and "Ã°Å¸" not in text and "Ã‚" not in text:
        return text
    try:
        fixed = text.encode("latin1").decode("utf-8")
        return fixed
    except Exception:
        return text


def clean_text(value):
    return repair_mojibake(str(value or "")).strip()


def normalize_plan(plan):
    project = plan.setdefault("project", {})
    audio = plan.setdefault("audio", {})
    scenes = plan.setdefault("scenes", [])

    project["duration"] = int(float(project.get("duration", 30)))
    project["aspect_ratio"] = project.get("aspect_ratio", "9:16")
    project["format"] = clean_text(project.get("format", "YouTube Short"))
    project["title"] = clean_text(project.get("title", "AI Video"))
    project["genre"] = clean_text(project.get("genre", "General"))
    project["style"] = clean_text(project.get("style", "Cinematic"))

    audio["voice_required"] = bool(audio.get("voice_required", True))
    audio["voice_style"] = clean_text(audio.get("voice_style", "Natural, energetic"))
    audio["music_required"] = bool(audio.get("music_required", True))
    audio["music_style"] = clean_text(audio.get("music_style", "Energetic electronic"))
    audio["music_energy"] = clean_text(audio.get("music_energy", "High"))
    audio["global_ambience"] = clean_text(audio.get("global_ambience", "Environmental ambience"))
    audio["sfx_required"] = bool(audio.get("sfx_required", True))

    cleaned = []
    for idx, raw in enumerate(scenes, start=1):
        s = dict(raw or {})
        s["scene_id"] = idx
        s["purpose"] = clean_text(s.get("purpose", "scene"))
        s["visual_prompt"] = clean_text(s.get("visual_prompt", ""))
        s["camera"] = clean_text(s.get("camera", "natural camera movement"))
        s["action"] = clean_text(s.get("action", ""))
        s["continuity"] = clean_text(s.get("continuity", ""))
        s["narration"] = clean_text(s.get("narration", ""))
        s["on_screen_text"] = clean_text(s.get("on_screen_text", ""))
        raw_sfx = s.get("sfx", [])
        if isinstance(raw_sfx, str):
            raw_sfx = [raw_sfx]
        s["sfx"] = [clean_text(x) for x in raw_sfx if clean_text(x)]
        cleaned.append(s)
    plan["scenes"] = cleaned
    return plan


def is_gaming_plan(plan, source_prompt=""):
    text = " ".join(
        [
            clean_text(plan.get("project", {}).get("genre", "")),
            clean_text(plan.get("project", {}).get("style", "")),
            clean_text(source_prompt),
        ]
    ).lower()
    gaming_words = [
        "gaming",
        "gamer",
        "battle royale",
        "call of duty",
        "cod",
        "fps",
        "enemy",
        "gunfire",
        "reload",
        "weapon",
        "gameplay",
    ]
    return any(w in text for w in gaming_words)


def scaled_durations(total, count):
    count = max(1, count)
    weights = np.linspace(1.0, 1.0, count)
    raw = weights / weights.sum() * total
    vals = [max(1, int(round(x))) for x in raw]
    diff = total - sum(vals)
    i = count - 1
    step = 1 if diff > 0 else -1
    while diff != 0:
        if vals[i] + step >= 1:
            vals[i] += step
            diff -= step
        i = (i - 1) % count
    return vals


def unique_scenes(scenes):
    result = []
    seen = set()
    for s in scenes:
        key = (
            clean_text(s.get("purpose", "")).lower(),
            clean_text(s.get("on_screen_text", "")).lower(),
            re.sub(r"\W+", " ", clean_text(s.get("narration", "")).lower())[:100],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(s)
    return result


def make_gaming_scenes(plan, duration):
    source = unique_scenes(plan.get("scenes", []))
    by_purpose = {clean_text(s.get("purpose", "")).lower(): s for s in source}

    def first(*names):
        for name in names:
            if name in by_purpose:
                return by_purpose[name]
        return source[0] if source else {}

    s1 = first("hook", "intro")
    s2 = first("build-up", "development", "movement")
    s3 = first("climax", "engagement")
    s4 = first("survival", "counter")
    s5 = first("victory", "counterattack", "action")
    s6 = first("ending", "victory")

    shots = [
        {
            "purpose": "hook",
            "base": s1,
            "action_override": "Immediately hear enemy footsteps nearby and snap into a ready combat stance. Start in authentic battle-royale gameplay, already in danger.",
        },
        {
            "purpose": "movement",
            "base": s2,
            "action_override": "Sprint through a futuristic urban battle-royale map, move between cover, and spot an enemy ahead. Keep the action continuous and gameplay-like.",
        },
        {
            "purpose": "engagement",
            "base": s2 if s2 else s3,
            "action_override": "Take cover behind urban cover, lean out, aim directly at the enemy, and fire a short controlled burst. Show readable weapon motion and muzzle flash.",
        },
        {
            "purpose": "survival",
            "base": s4 if s4 else s3,
            "action_override": "The enemy suddenly attacks from the side. The player reacts, narrowly survives, ducks behind cover, and begins a frantic reload.",
        },
        {
            "purpose": "counterattack",
            "base": s5 if s5 else s3,
            "action_override": "Finish the reload, quickly re-aim, counterattack with a decisive burst, and eliminate the enemy. Keep the camera aggressive and close to the gameplay action.",
        },
        {
            "purpose": "victory",
            "base": s6 if s6 else s5,
            "action_override": "Hold on the player after the elimination, show the victorious combat moment, then settle into a confident end beat suitable for a gaming Short.",
        },
    ]

    durations = GAMING_DURATIONS.get(duration)
    if durations is None:
        durations = scaled_durations(duration, 6)

    out = []
    current = 0.0
    for i, shot in enumerate(shots):
        base = dict(shot["base"] or {})
        d = durations[i]
        start = current
        end = current + d
        current = end
        purpose = shot["purpose"]

        narration = clean_text(base.get("narration", ""))
        if purpose == "hook":
            narration = narration or "Wait... I hear someone right here."
        elif purpose == "movement":
            narration = narration or "I spotted him. Moving in."
        elif purpose == "engagement":
            narration = "Take cover... aim... fire."
        elif purpose == "survival":
            narration = "He pushed from the side! That was too close."
        elif purpose == "counterattack":
            narration = "Reloading... now we counter."
        elif purpose == "victory":
            narration = "Let's go! That's the win."

        text = clean_text(base.get("on_screen_text", ""))
        if purpose == "hook":
            text = "THAT WAS TOO CLOSE ðŸ˜³"
        elif purpose == "victory":
            text = "FOLLOW FOR MORE ðŸŽ®"
        else:
            text = ""

        out.append(
            {
                "scene_id": i + 1,
                "start_time": start,
                "end_time": end,
                "duration": d,
                "purpose": purpose,
                "visual_prompt": clean_text(base.get("visual_prompt", "")),
                "camera": clean_text(base.get("camera", "dynamic gameplay camera")),
                "action": shot["action_override"],
                "continuity": clean_text(base.get("continuity", "")),
                "narration_text": narration,
                "sfx": base.get("sfx", []),
                "on_screen_text": text,
                "transition": "hard cut",
            }
        )

    return out


def make_general_scenes(plan, duration):
    scenes = unique_scenes(plan.get("scenes", []))
    if not scenes:
        scenes = [{
            "purpose": "scene",
            "visual_prompt": "A compelling cinematic scene matching the user's story.",
            "camera": "dynamic camera movement",
            "action": "Characters perform the described action naturally.",
            "continuity": "Keep character appearance and environment consistent.",
            "narration": "",
            "sfx": [],
            "on_screen_text": "",
        }]

    durations = scaled_durations(duration, min(len(scenes), 6))
    scenes = scenes[: len(durations)]
    current = 0.0
    out = []
    for i, base in enumerate(scenes):
        d = durations[i]
        start = current
        end = current + d
        current = end
        item = dict(base)
        item.update(
            {
                "scene_id": i + 1,
                "start_time": start,
                "end_time": end,
                "duration": d,
            }
        )
        out.append(item)
    return out


def prepare_execution_plan(plan, duration, source_prompt):
    gaming = is_gaming_plan(plan, source_prompt)
    scenes = make_gaming_scenes(plan, duration) if gaming else make_general_scenes(plan, duration)
    plan = dict(plan)
    plan["project"] = dict(plan.get("project", {}))
    plan["project"]["duration"] = duration
    plan["execution_mode"] = "gaming-shot-pipeline" if gaming else "general-shot-pipeline"
    plan["execution_scenes"] = scenes
    return plan


def build_ltx_prompt(project, scene, previous_context=""):
    genre = clean_text(project.get("genre", ""))
    style = clean_text(project.get("style", ""))
    aspect = clean_text(project.get("aspect_ratio", "16:9"))
    continuity = clean_text(scene.get("continuity", ""))
    visual = clean_text(scene.get("visual_prompt", ""))
    camera = clean_text(scene.get("camera", ""))
    action = clean_text(scene.get("action", ""))
    purpose = clean_text(scene.get("purpose", "scene"))

    prompt = (
        f"High-quality AI video shot for a {genre} project, {style}. "
        f"Vertical {aspect} framing. Purpose: {purpose}. "
        "Authentic continuous gameplay/action, not a movie trailer, not a poster, not a still image. "
        "Characters must visibly move and the camera must visibly move. "
        f"Visual setup: {visual}. "
        f"Action: {action}. "
        f"Camera: {camera}. "
        f"Continuity: {continuity}. "
        "Natural human motion, physically plausible movement, coherent spatial layout, clear subject separation, "
        "detailed environment, realistic motion blur, consistent lighting, no logos, no watermark, no subtitles baked into the generated video."
    )

    if "gaming" in genre.lower() or "battle" in (genre + " " + visual).lower():
        prompt += (
            " First-person or over-the-shoulder battle-royale gameplay viewpoint. "
            "Show believable weapon handling, sprinting, aiming, recoil, cover, enemy movement and combat reactions. "
            "Do not show a gamer sitting at a PC or a streamer setup unless explicitly requested."
        )

    if previous_context:
        prompt += f" Previous shot ended with this visual state; preserve continuity: {previous_context}."

    return prompt


def frames_for_duration(seconds):
    seconds = max(1, int(round(float(seconds))))
    return 1 + 24 * seconds


def ffmpeg_path(path: Path):
    p = str(path).replace("\\", "/")
    p = p.replace(":", r"\:")
    p = p.replace("'", r"\'")
    return p


def choose_font():
    candidates = [
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def extract_last_frame(video: Path, image: Path):
    image.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-y",
        "-sseof",
        "-0.08",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(image),
    ]
    try:
        run_cmd(cmd, title=f"Extract continuity frame: {video.name}")
        return image.exists()
    except Exception:
        return False


def hash_seed(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def call_ltx(scene_dir, job_id, prompt, settings):
    prompt_file = scene_dir / "prompt.txt"
    settings_file = scene_dir / "settings.json"
    output_file = scene_dir / "silent.mp4"

    prompt_file.write_text(prompt, encoding="utf-8")
    write_json(settings_file, settings)

    cmd = [
        str(LTX_PYTHON),
        str(LTX_WORKER),
        job_id,
        str(prompt_file),
        str(output_file),
        str(settings_file),
    ]
    run_cmd(cmd, title=f"LTX scene {scene_dir.name}")
    if not output_file.exists():
        raise StepError(f"LTX did not create {output_file}")
    return output_file


def call_audio_worker(scene_dir, job_id, video_input, narration, voice_mode, voice_reference):
    """Use the audio_worker CLI actually present in this project:
       audio_worker.py <job_id> <silent_video> <final_video> <settings_file>

    The worker creates a video containing the narration audio. We extract
    that audio to a WAV so the executor can mix it with music/SFX.
    """
    settings_file = scene_dir / "audio_settings.json"
    narrated_video = scene_dir / "narrated.mp4"
    wav_output = scene_dir / "narration.wav"

    settings = {
        "audio_mode": voice_mode,
        "narration_text": narration,
    }
    if voice_reference:
        settings["voice_path"] = str(voice_reference)
    write_json(settings_file, settings)

    # IMPORTANT: the installed audio_worker.py expects exactly four
    # arguments after the script name: job_id, silent_video, final_video,
    # settings_file. Do not pass a WAV or status-file argument here.
    cmd = [
        str(CHATTERBOX_PYTHON),
        str(AUDIO_WORKER),
        job_id,
        str(video_input),
        str(narrated_video),
        str(settings_file),
    ]
    run_cmd(cmd, title=f"Chatterbox narration {scene_dir.name}")
    if not narrated_video.exists():
        raise StepError(f"Audio worker did not create {narrated_video}")

    # The audio worker writes the generated narration into narrated.mp4.
    # Extract the track for the scene-level mixer.
    run_cmd(
        [
            FFMPEG,
            "-y",
            "-i",
            str(narrated_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SR),
            "-c:a",
            "pcm_s16le",
            str(wav_output),
        ],
        title=f"Extract narration audio {scene_dir.name}",
    )
    if not wav_output.exists():
        raise StepError(f"Could not extract narration audio to {wav_output}")
    return wav_output

def envelope(n, attack, release):
    env = np.ones(n, dtype=np.float32)
    a = min(n, int(attack))
    r = min(n, int(release))
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r > 0:
        env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def osc(freq, t, phase=0.0):
    return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)


def procedural_music(duration, bpm=140, energy=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(duration * AUDIO_SR))
    t = np.arange(n, dtype=np.float32) / AUDIO_SR
    y = np.zeros(n, dtype=np.float32)

    beat = 60.0 / bpm
    step = int(round(beat * AUDIO_SR))
    bar = step * 4

    # Low synth bed.
    notes = np.array([55.0, 65.41, 73.42, 82.41], dtype=np.float32)
    note_len = int(round(beat * 2 * AUDIO_SR))
    for i in range(max(1, math.ceil(n / max(1, note_len)))):
        start = i * note_len
        end = min(n, start + note_len)
        if end <= start:
            continue
        f = float(notes[i % len(notes)])
        tt = np.arange(end - start, dtype=np.float32) / AUDIO_SR
        tone = 0.10 * osc(f, tt) + 0.045 * osc(f * 2.0, tt)
        y[start:end] += tone

    # Four-to-the-floor kick + snare + hats.
    for start in range(0, n, max(1, step)):
        end = min(n, start + int(0.16 * AUDIO_SR))
        m = end - start
        if m <= 0:
            continue
        tt = np.arange(m, dtype=np.float32) / AUDIO_SR
        if (start // max(1, step)) % 4 in (0, 2):
            kick = np.sin(2 * np.pi * (90.0 * np.exp(-12 * tt) + 42.0) * tt)
            kick *= np.exp(-18 * tt)
            y[start:end] += 0.40 * kick
        else:
            noise = rng.standard_normal(m).astype(np.float32)
            noise *= np.exp(-22 * tt)
            y[start:end] += 0.18 * noise

        hat_end = min(n, start + int(0.05 * AUDIO_SR))
        hm = hat_end - start
        if hm > 0:
            ht = np.arange(hm, dtype=np.float32) / AUDIO_SR
            h = rng.standard_normal(hm).astype(np.float32) * np.exp(-60 * ht)
            y[start:hat_end] += 0.07 * h

    # High electronic arpeggio.
    arp = [220.0, 277.18, 329.63, 369.99, 440.0, 369.99, 329.63, 277.18]
    arp_len = max(1, int(round(beat / 2 * AUDIO_SR)))
    for i in range(max(1, math.ceil(n / arp_len))):
        start = i * arp_len
        end = min(n, start + arp_len)
        if end <= start:
            continue
        tt = np.arange(end - start, dtype=np.float32) / AUDIO_SR
        a = 0.055 * osc(arp[i % len(arp)], tt)
        a *= envelope(len(a), 30, min(len(a), int(0.04 * AUDIO_SR)))
        y[start:end] += a

    y *= float(np.clip(energy, 0.25, 1.5))
    y = np.tanh(y * 1.35).astype(np.float32)
    stereo = np.column_stack([y, y]).astype(np.float32)
    return stereo


def effect_wave(kind, duration=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = max(1, int(round(duration * AUDIO_SR)))
    t = np.arange(n, dtype=np.float32) / AUDIO_SR

    if kind == "footstep":
        y = rng.standard_normal(n).astype(np.float32) * np.exp(-35 * t)
        y += 0.35 * osc(70, t) * np.exp(-20 * t)
    elif kind == "gunshot":
        y = rng.standard_normal(n).astype(np.float32) * np.exp(-55 * t)
        y += 0.8 * osc(55 + 35 * np.exp(-25 * t), t) * np.exp(-18 * t)
    elif kind == "reload":
        y = 0.3 * rng.standard_normal(n).astype(np.float32) * np.exp(-8 * t)
        clicks = max(1, n // 4)
        for i in range(3):
            c = min(n - 1, i * clicks + clicks // 2)
            e = min(n, c + int(0.015 * AUDIO_SR))
            y[c:e] += 0.5 * np.exp(-80 * (np.arange(e - c) / AUDIO_SR))
    elif kind == "impact":
        y = rng.standard_normal(n).astype(np.float32) * np.exp(-38 * t)
        y += 0.55 * osc(90, t) * np.exp(-15 * t)
    elif kind == "victory":
        y = np.zeros(n, dtype=np.float32)
        notes = [523.25, 659.25, 783.99]
        seg = max(1, n // len(notes))
        for i, f in enumerate(notes):
            s = i * seg
            e = min(n, s + seg)
            if e <= s:
                continue
            tt = np.arange(e - s, dtype=np.float32) / AUDIO_SR
            y[s:e] += 0.28 * osc(f, tt) * envelope(len(tt), 40, int(0.08 * AUDIO_SR))
    else:
        y = rng.standard_normal(n).astype(np.float32) * np.exp(-3 * t)

    y = np.tanh(y * 1.4).astype(np.float32)
    return y


def append_effect(track, effect, start_sec, gain=1.0):
    start = int(round(max(0.0, start_sec) * AUDIO_SR))
    end = min(track.shape[0], start + effect.shape[0])
    if end <= start:
        return
    track[start:end] += effect[: end - start] * gain


def classify_effects(scene):
    text = " ".join(
        [
            clean_text(scene.get("action", "")),
            clean_text(scene.get("narration_text", "")),
            " ".join(scene.get("sfx", []) or []),
        ]
    ).lower()
    kinds = []
    if any(x in text for x in ["footstep", "sprint", "move", "nearby", "approach"]):
        kinds.append("footstep")
    if any(x in text for x in ["shoot", "gunshot", "fire", "burst", "weapon"]):
        kinds.append("gunshot")
    if any(x in text for x in ["reload", "reloading"]):
        kinds.append("reload")
    if any(x in text for x in ["impact", "hit", "attack", "survive"]):
        kinds.append("impact")
    if any(x in text for x in ["victory", "win", "eliminate", "eliminates", "victorious"]):
        kinds.append("victory")
    return list(dict.fromkeys(kinds))


def load_mono(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)


def write_mono(path, data, sr=AUDIO_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), sr, subtype="PCM_16")


def convert_audio_with_ffmpeg(src, dst, duration):
    run_cmd(
        [
            FFMPEG,
            "-y",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-ar",
            str(AUDIO_SR),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        title=f"Normalize narration audio: {Path(src).name}",
    )


def make_scene_audio(scene_dir, scene, duration, narration_wav, music_required, sfx_required, seed):
    n = int(round(duration * AUDIO_SR))
    mix = np.zeros(n, dtype=np.float32)

    if music_required:
        music = procedural_music(duration, bpm=144, energy=1.0, seed=seed)
        music_mono = music.mean(axis=1) * 0.16
        mix += music_mono[:n]

    if sfx_required:
        effects = classify_effects(scene)
        if effects:
            positions = np.linspace(0.35, max(0.36, duration - 0.25), len(effects))
            for idx, (kind, pos) in enumerate(zip(effects, positions)):
                effect = effect_wave(kind, duration={"footstep": 0.22, "gunshot": 0.32, "reload": 0.65, "impact": 0.26, "victory": 1.1}.get(kind, 0.3), seed=seed + idx)
                gain = {"footstep": 0.22, "gunshot": 0.48, "reload": 0.30, "impact": 0.36, "victory": 0.38}.get(kind, 0.2)
                append_effect(mix, effect, float(pos), gain=gain)

    if narration_wav and narration_wav.exists():
        voice_norm = scene_dir / "narration_48k.wav"
        convert_audio_with_ffmpeg(narration_wav, voice_norm, duration)
        voice, sr = load_mono(voice_norm)
        if sr != AUDIO_SR:
            raise StepError("Unexpected narration sample rate after FFmpeg normalization.")
        voice = voice[:n]
        mix[: len(voice)] += voice * 1.0

    # Gentle limiter.
    mix = np.tanh(mix * 1.25).astype(np.float32)
    out = scene_dir / "mixed_audio.wav"
    write_mono(out, mix, AUDIO_SR)
    return out


def make_drawtext_filter(scene_dir, scene, captions_enabled):
    font = choose_font()
    filters = []
    if font is None:
        print("WARNING: No Windows font file found; captions may be skipped.", flush=True)
        return None

    font_q = ffmpeg_path(font)
    caption_text = clean_text(scene.get("narration_text", ""))
    onscreen = clean_text(scene.get("on_screen_text", ""))

    if captions_enabled and caption_text:
        cap_file = scene_dir / "caption.txt"
        cap_file.write_text(caption_text, encoding="utf-8")
        cap_path = ffmpeg_path(cap_file)
        filters.append(
            "drawtext="
            f"fontfile='{font_q}':textfile='{cap_path}':reload=0:"
            "fontcolor=white:fontsize=24:borderw=2:bordercolor=black:"
            "box=1:boxcolor=black@0.55:boxborderw=10:"
            "x=(w-text_w)/2:y=h-95:"
            "line_spacing=4"
        )

    if onscreen:
        text_file = scene_dir / "onscreen.txt"
        text_file.write_text(onscreen, encoding="utf-8")
        text_path = ffmpeg_path(text_file)
        filters.append(
            "drawtext="
            f"fontfile='{font_q}':textfile='{text_path}':reload=0:"
            "fontcolor=white:fontsize=30:borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=42:"
            "box=1:boxcolor=black@0.35:boxborderw=10"
        )

    return ",".join(filters) if filters else None


def build_scene_video(scene_dir, silent_video, mixed_audio, duration, scene, captions_enabled):
    final_video = scene_dir / "final.mp4"
    vf = make_drawtext_filter(scene_dir, scene, captions_enabled)
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(mixed_audio),
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += [
        "-t",
        f"{duration:.3f}",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        str(AUDIO_SR),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(final_video),
    ]
    run_cmd(cmd, title=f"Edit scene {scene_dir.name}")
    return final_video


def concat_scenes(scene_files, output_file):
    list_file = output_file.parent / "concat.txt"
    lines = []
    for f in scene_files:
        p = str(f).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{p}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_cmd(
        [
            FFMPEG,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_file),
        ],
        title="Stitch all shots into final video",
    )
    return output_file


def verify_duration(video):
    cmd = [
        FFMPEG,
        "-i",
        str(video),
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    text = proc.stderr + proc.stdout
    matches = re.findall(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if not matches:
        return None
    h, m, s = matches[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def validate_environment():
    missing = []
    for p in [LTX_PYTHON, CHATTERBOX_PYTHON, LTX_WORKER, AUDIO_WORKER]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    probe = subprocess.run(
        [FFMPEG, "-version"],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if probe.returncode != 0:
        raise FileNotFoundError("ffmpeg was not found on PATH.")


def main():
    parser = argparse.ArgumentParser(description="MascotAI automatic multi-shot video executor")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--prompt", default="", help="Original prompt, used for better execution routing")
    parser.add_argument("--duration", type=int, default=0, help="Override final duration in seconds")
    parser.add_argument("--output", default="", help="Final MP4 path")
    parser.add_argument("--voice-mode", choices=["No Audio", "AI Voice", "Voice Clone"], default="AI Voice")
    parser.add_argument("--voice-reference", default="", help="Reference WAV for Voice Clone")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--no-sfx", action="store_true")
    parser.add_argument("--no-captions", action="store_true")
    args = parser.parse_args()

    validate_environment()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan not found: {plan_path}")

    plan = normalize_plan(read_json(plan_path))

    source_prompt = args.prompt.strip()
    if not source_prompt and DEFAULT_PROMPT.exists():
        source_prompt = DEFAULT_PROMPT.read_text(encoding="utf-8-sig")

    duration = args.duration or int(plan["project"].get("duration", 30))
    if duration not in GAMING_DURATIONS and duration not in {10, 25, 30, 45, 60, 120}:
        raise ValueError("Duration must be one of: 10, 25, 30, 45, 60, 120 seconds.")

    execution_plan = prepare_execution_plan(plan, duration, source_prompt)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = TEMP_DIR / "auto_runs" / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    execution_json = run_dir / "execution_plan.json"
    write_json(execution_json, execution_plan)

    output_name = args.output.strip()
    if output_name:
        final_output = Path(output_name)
        if not final_output.is_absolute():
            final_output = OUTPUT_DIR / final_output
    else:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", clean_text(plan["project"].get("title", "auto_video")))
        final_output = OUTPUT_DIR / f"{slug}_{duration}s_{stamp}.mp4"

    project = execution_plan["project"]
    audio_plan = execution_plan.get("audio", {})
    voice_required = bool(audio_plan.get("voice_required", True)) and args.voice_mode != "No Audio"
    music_required = bool(audio_plan.get("music_required", True)) and not args.no_music
    sfx_required = bool(audio_plan.get("sfx_required", True)) and not args.no_sfx
    captions_enabled = not args.no_captions
    voice_reference = Path(args.voice_reference) if args.voice_reference else None

    if args.voice_mode == "Voice Clone":
        if voice_reference is None or not voice_reference.exists():
            raise FileNotFoundError("Voice Clone requires --voice-reference pointing to a real WAV file.")

    print("\nMascotAI AUTO PRODUCER", flush=True)
    print(f"Plan: {plan_path}", flush=True)
    print(f"Mode: {execution_plan['execution_mode']}", flush=True)
    print(f"Duration: {duration}s", flush=True)
    print(f"Final output: {final_output}", flush=True)
    print(f"Voice: {'ON' if voice_required else 'OFF'} ({args.voice_mode})", flush=True)
    print(f"Music: {'ON' if music_required else 'OFF'}", flush=True)
    print(f"SFX: {'ON' if sfx_required else 'OFF'}", flush=True)
    print(f"Captions: {'ON' if captions_enabled else 'OFF'}", flush=True)

    previous_frame = None
    previous_context = ""
    scene_files = []

    scenes = execution_plan["execution_scenes"]
    if not scenes:
        raise StepError("Execution plan contains no scenes.")

    total_check = sum(int(round(float(s["duration"]))) for s in scenes)
    if total_check != duration:
        raise StepError(f"Internal timing error: scenes total {total_check}s instead of {duration}s.")

    for index, scene in enumerate(scenes, start=1):
        scene_dir = run_dir / f"scene_{index:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        seed = hash_seed(f"{project.get('title','MascotAI')}|{duration}|{index}")
        settings = {
            "num_frames": frames_for_duration(scene["duration"]),
            "aspect_ratio": project.get("aspect_ratio", "9:16"),
            "quality": "Standard",
            "seed": seed,
            "audio_mode": "No Audio",
        }

        prompt = build_ltx_prompt(project, scene, previous_context)
        if previous_frame is not None and previous_frame.exists():
            settings["image_path"] = str(previous_frame)

        print(f"\n--- SHOT {index}/{len(scenes)} | {scene['duration']}s | {scene['purpose']} ---", flush=True)
        print(f"ACTION: {scene.get('action','')}", flush=True)
        print(f"NARRATION: {scene.get('narration','')}", flush=True)

        silent_video = call_ltx(
            scene_dir,
            f"auto_{stamp}_{index:02d}",
            prompt,
            settings,
        )

        if voice_required and clean_text(scene.get("narration_text", "")):
            narration_wav = call_audio_worker(
                scene_dir,
                f"audio_{stamp}_{index:02d}",
                silent_video,
                clean_text(scene.get("narration_text", "")),
                args.voice_mode,
                voice_reference,
            )
        else:
            narration_wav = None

        mixed_audio = make_scene_audio(
            scene_dir,
            scene,
            float(scene["duration"]),
            narration_wav,
            music_required,
            sfx_required,
            seed,
        )

        final_scene = build_scene_video(
            scene_dir,
            silent_video,
            mixed_audio,
            float(scene["duration"]),
            scene,
            captions_enabled,
        )
        scene_files.append(final_scene)

        last_frame = scene_dir / "last_frame.png"
        if index < len(scenes) and extract_last_frame(silent_video, last_frame):
            previous_frame = last_frame
            previous_context = f"continuation from shot {index}; preserve the same main subject, outfit, environment and lighting"
        else:
            previous_frame = None
            previous_context = ""

    final_output.parent.mkdir(parents=True, exist_ok=True)
    concat_scenes(scene_files, final_output)

    actual = verify_duration(final_output)
    print("\n=== COMPLETE ===", flush=True)
    print(f"VIDEO: {final_output}", flush=True)
    if actual is not None:
        print(f"FINAL DURATION: {actual:.2f}s", flush=True)
    print(f"RUN FILES: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print("\nAUTO PRODUCER ERROR:", flush=True)
        print(str(exc), flush=True)
        traceback.print_exc()
        raise SystemExit(1)

