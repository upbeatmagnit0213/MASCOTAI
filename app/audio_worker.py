import json
import subprocess
import sys
import traceback
import types
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from chatterbox.tts import ChatterboxTTS

BASE_DIR = Path(r"C:\MascotAI")
JOB_DIR = BASE_DIR / "temp" / "jobs"
FFMPEG = "ffmpeg"


def local_trim(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y, np.array([0, 0])
    amp = np.abs(y)
    peak = float(np.max(amp))
    if peak <= 0:
        return y, np.array([0, len(y)])
    threshold = peak * (10.0 ** (-top_db / 20.0))
    active = np.flatnonzero(amp > threshold)
    if active.size == 0:
        return y, np.array([0, len(y)])
    start = max(0, int(active[0]) - hop_length)
    end = min(len(y), int(active[-1]) + hop_length + 1)
    return y[start:end], np.array([start, end])

librosa.effects = types.SimpleNamespace(trim=local_trim)


def write_status(status_file, data):
    status_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_wav(path, audio, sample_rate):
    array = audio.detach().float().cpu().squeeze().numpy()
    sf.write(str(path), array, sample_rate, subtype="PCM_16")


def main():
    if len(sys.argv) != 5:
        print("Usage: audio_worker.py <job_id> <silent_video> <final_video> <settings_file>", flush=True)
        sys.exit(1)

    job_id = sys.argv[1]
    silent_video = Path(sys.argv[2])
    final_video = Path(sys.argv[3])
    settings_file = Path(sys.argv[4])
    status_file = JOB_DIR / f"{job_id}.json"

    try:
        settings = json.loads(settings_file.read_text(encoding="utf-8-sig"))
        audio_mode = settings.get("audio_mode", "None")
        narration = str(settings.get("narration_text", "")).strip()
        voice_path_value = settings.get("voice_path")

        if audio_mode not in {"AI Voice", "Voice Clone"}:
            raise ValueError("Audio worker received an invalid audio mode.")
        if not narration:
            raise ValueError("Narration text is empty.")
        if not silent_video.exists():
            raise FileNotFoundError(f"Silent video not found: {silent_video}")

        voice_path = None
        if audio_mode == "Voice Clone":
            if not voice_path_value:
                raise FileNotFoundError("Voice clone reference audio is missing.")
            voice_path = Path(str(voice_path_value))
            if not voice_path.exists():
                raise FileNotFoundError(f"Voice reference not found: {voice_path}")

        write_status(status_file, {"status": "audio_generating", "job_id": job_id})
        print(f"Loading Chatterbox for {audio_mode}...", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}", flush=True)
        model = ChatterboxTTS.from_pretrained(device=device)
        print("CHATTERBOX MODEL READY", flush=True)

        if audio_mode == "Voice Clone":
            print(f"Using reference voice: {voice_path}", flush=True)
            with torch.inference_mode():
                wav = model.generate(narration, audio_prompt_path=str(voice_path))
        else:
            with torch.inference_mode():
                wav = model.generate(narration)

        voice_wav = JOB_DIR / f"{job_id}_voice.wav"
        save_wav(voice_wav, wav, model.sr)
        print(f"VOICE SAVED: {voice_wav}", flush=True)

        temp_final = final_video.with_name(final_video.stem + "_muxing.mp4")
        cmd = [
            FFMPEG, "-y",
            "-i", str(silent_video),
            "-i", str(voice_wav),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(temp_final),
        ]
        print("Merging voice with video...", flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        temp_final.replace(final_video)

        try:
            voice_wav.unlink(missing_ok=True)
        except OSError:
            pass

        write_status(status_file, {
            "status": "completed",
            "job_id": job_id,
            "filename": final_video.name,
            "audio_mode": audio_mode,
        })
        print(f"FINAL VIDEO SAVED: {final_video}", flush=True)
    except Exception as exc:
        print("AUDIO ERROR:", flush=True)
        traceback.print_exc()
        write_status(status_file, {"status": "failed", "job_id": job_id, "error": str(exc), "traceback": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()


