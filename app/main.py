import librosa
import numpy as np
import os
import subprocess
import tempfile

INPUT_VIDEO = r"C:\MascotAI\input\speech_test.mp4"
OUTPUT_VIDEO = r"C:\MascotAI\output\speech_no_silence.mp4"
SILENCE_THRESHOLD_DB = -35
MIN_SILENCE_DURATION = 0.5

print("================================")
print("       MASCOT AI STUDIO")
print("================================")
print()
print("Analyzing audio...")
print()

if not os.path.exists(INPUT_VIDEO):
    print("ERROR: Input video not found.")
    exit()

temp_wav = os.path.join(
    tempfile.gettempdir(),
    "mascot_audio.wav"
)

print("Extracting audio...")

extract_command = [
    "ffmpeg",
    "-y",
    "-i",
    INPUT_VIDEO,
    "-vn",
    "-ac",
    "1",
    "-ar",
    "16000",
    temp_wav
]

result = subprocess.run(
    extract_command,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

if result.returncode != 0:
    print("ERROR: Could not extract audio.")
    exit()

audio, sample_rate = librosa.load(
    temp_wav,
    sr=16000,
    mono=True
)

frame_length = 1024
hop_length = 256

rms = librosa.feature.rms(
    y=audio,
    frame_length=frame_length,
    hop_length=hop_length
)[0]

db = librosa.amplitude_to_db(
    rms,
    ref=np.max
)

active = db > SILENCE_THRESHOLD_DB

times = librosa.frames_to_time(
    np.arange(len(active)),
    sr=sample_rate,
    hop_length=hop_length
)

segments = []
segment_start = None

for i, is_active in enumerate(active):

    if is_active and segment_start is None:
        segment_start = times[i]

    elif not is_active and segment_start is not None:

        segment_end = times[i]

        if segment_end - segment_start >= MIN_SILENCE_DURATION:
            segments.append(
                (segment_start, segment_end)
            )

        segment_start = None

if segment_start is not None:
    segments.append(
        (segment_start, times[-1])
    )

print("Active sections detected:")
print()

for start, end in segments:
    print(f"{start:.2f}s -> {end:.2f}s")

print()

if not segments:
    print("No active audio detected.")
    exit()

# Build video and audio trim filters
filter_parts = []

for index, (start, end) in enumerate(segments):

    filter_parts.append(
        f"[0:v]trim=start={start}:end={end},"
        f"setpts=PTS-STARTPTS[v{index}]"
    )

    filter_parts.append(
        f"[0:a]atrim=start={start}:end={end},"
        f"asetpts=PTS-STARTPTS[a{index}]"
    )

video_audio_inputs = ""

for index in range(len(segments)):
    video_audio_inputs += f"[v{index}][a{index}]"

filter_parts.append(
    f"{video_audio_inputs}"
    f"concat=n={len(segments)}:v=1:a=1[outv][outa]"
)

filter_complex = ";".join(filter_parts)

ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-i",
    INPUT_VIDEO,
    "-filter_complex",
    filter_complex,
    "-map",
    "[outv]",
    "-map",
    "[outa]",
    "-c:v",
    "libx264",
    "-preset",
    "fast",
    "-crf",
    "23",
    "-c:a",
    "aac",
    "-b:a",
    "192k",
    OUTPUT_VIDEO
]

print("Removing silent sections and preserving audio...")
print()

result = subprocess.run(ffmpeg_command)

if result.returncode != 0:
    print()
    print("ERROR: FFmpeg could not create the edited video.")
    exit()

print()
print("================================")
print("SILENCE REMOVAL COMPLETE")
print("================================")
print()
print(f"Output file : {OUTPUT_VIDEO}")
print()
print("Video and audio were successfully synchronized!")