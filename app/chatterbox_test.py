import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import librosa

from chatterbox.tts import ChatterboxTTS


BASE_DIR = Path(r"C:\MascotAI")

OUTPUT_DIR = BASE_DIR / "output"
INPUT_DIR = BASE_DIR / "input"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


AI_OUTPUT = (
    OUTPUT_DIR /
    "chatterbox_ai_voice.wav"
)

CLONE_OUTPUT = (
    OUTPUT_DIR /
    "chatterbox_cloned_voice.wav"
)

REFERENCE_AUDIO = (
    INPUT_DIR /
    "my_voice.wav"
)


# ============================================================
# LIBROSA TRIM REPLACEMENT
# ============================================================
# Chatterbox's voice encoder uses librosa.effects.trim().
# Your Windows Application Control blocks sklearn's native
# DLL, which librosa tries to load through that dependency.
#
# We only need simple silence trimming for the reference
# recording, so implement that locally instead.
# ============================================================

def local_trim(
    y,
    top_db=60,
    ref=np.max,
    frame_length=2048,
    hop_length=512,
):
    y = np.asarray(
        y,
        dtype=np.float32,
    )

    if y.size == 0:
        return (
            y,
            np.array([0, 0]),
        )

    amplitude = np.abs(y)

    peak = float(
        np.max(amplitude)
    )

    if peak <= 0:
        return (
            y,
            np.array(
                [0, len(y)]
            ),
        )

    threshold = (
        peak *
        (10.0 ** (-top_db / 20.0))
    )

    active = (
        amplitude >
        threshold
    )

    indices = np.flatnonzero(
        active
    )

    if indices.size == 0:
        return (
            y,
            np.array(
                [0, len(y)]
            ),
        )

    start = max(
        0,
        int(indices[0]) -
        hop_length,
    )

    end = min(
        len(y),
        int(indices[-1]) +
        hop_length +
        1,
    )

    return (
        y[start:end],
        np.array(
            [start, end]
        ),
    )


# Replace only librosa.effects.trim.
librosa.effects = types.SimpleNamespace(
    trim=local_trim
)


# ============================================================
# SAVE WAV
# ============================================================

def save_wav(
    path,
    audio,
    sample_rate,
):

    audio = (
        audio.detach()
        .float()
        .cpu()
        .squeeze()
        .numpy()
    )

    sf.write(
        str(path),
        audio,
        sample_rate,
        subtype="PCM_16",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "MASCOT AI - CHATTERBOX VOICE TEST"
    )
    print("=" * 60)


    if torch.cuda.is_available():

        device = "cuda"

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    else:

        device = "cpu"

        print(
            "WARNING: CUDA not available."
        )


    print(
        "Device:",
        device,
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "Loading Chatterbox model..."
    )


    model = (
        ChatterboxTTS
        .from_pretrained(
            device=device
        )
    )


    print(
        "CHATTERBOX MODEL READY"
    )


    # --------------------------------------------------------
    # AI VOICE
    # --------------------------------------------------------

    ai_text = (
        "Welcome to Mascot AI Studio. "
        "This voice was generated "
        "completely locally on my computer."
    )


    print()
    print(
        "Generating AI voice..."
    )


    with torch.inference_mode():

        ai_wav = model.generate(
            ai_text
        )


    save_wav(
        AI_OUTPUT,
        ai_wav,
        model.sr,
    )


    print(
        "AI VOICE SAVED:",
        AI_OUTPUT,
    )


    # --------------------------------------------------------
    # VOICE CLONE
    # --------------------------------------------------------

    if not REFERENCE_AUDIO.exists():

        print()
        print(
            "VOICE CLONE TEST SKIPPED."
        )

        print(
            "Reference file not found:"
        )

        print(
            REFERENCE_AUDIO
        )

        return


    print()
    print(
        "Reference voice found:"
    )

    print(
        REFERENCE_AUDIO
    )


    # Inspect the reference file.
    try:

        info = sf.info(
            str(REFERENCE_AUDIO)
        )

        print(
            "Reference format:",
            info.samplerate,
            "Hz",
            info.channels,
            "channel(s)",
            round(info.duration, 2),
            "seconds",
        )

    except Exception as exc:

        print(
            "Could not inspect reference audio:",
            exc,
        )


    clone_text = (
        "Hello, this is my cloned voice "
        "running locally inside Mascot AI Studio."
    )


    print()
    print(
        "Generating cloned voice..."
    )


    with torch.inference_mode():

        cloned_wav = model.generate(
            clone_text,
            audio_prompt_path=(
                str(REFERENCE_AUDIO)
            ),
        )


    save_wav(
        CLONE_OUTPUT,
        cloned_wav,
        model.sr,
    )


    print()
    print(
        "CLONED VOICE SAVED:",
        CLONE_OUTPUT,
    )


    print()
    print("=" * 60)
    print(
        "VOICE CLONE TEST COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()