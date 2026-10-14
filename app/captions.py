from faster_whisper import WhisperModel

INPUT_VIDEO = r"C:\MascotAI\input\speech_test.mp4"
OUTPUT_SRT = r"C:\MascotAI\output\speech_captions.srt"

print("================================")
print("       MASCOT AI STUDIO")
print("================================")
print()
print("Loading Whisper model...")
print()

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Transcribing with Voice Activity Detection...")
print()

segments, info = model.transcribe(
    INPUT_VIDEO,
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500
    ),
    condition_on_previous_text=False,
    word_timestamps=True
)

print(f"Detected language: {info.language}")
print()


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))

    if milliseconds >= 1000:
        secs += 1
        milliseconds -= 1000

    if secs >= 60:
        minutes += 1
        secs -= 60

    if minutes >= 60:
        hours += 1
        minutes -= 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


MAX_DURATION = 5.0
MAX_WORDS = 10
MAX_CHARACTERS = 55

caption_entries = []


for segment in segments:

    words = segment.words

    if not words:
        text = segment.text.strip()

        if text:
            start = segment.start
            end = min(segment.end, start + MAX_DURATION)

            caption_entries.append(
                (start, end, text)
            )

        continue

    current_words = []
    chunk_start = None
    previous_end = None

    for word in words:

        word_text = word.word.strip()

        if not word_text:
            continue

        if chunk_start is None:
            chunk_start = word.start

        current_words.append(word_text)

        current_text = " ".join(current_words)

        duration = word.end - chunk_start

        should_split = (
            duration >= MAX_DURATION
            or len(current_words) >= MAX_WORDS
            or len(current_text) >= MAX_CHARACTERS
        )

        if should_split:

            if len(current_words) > 1:

                caption_entries.append(
                    (
                        chunk_start,
                        word.end,
                        " ".join(current_words)
                    )
                )

                current_words = []
                chunk_start = None

            else:

                caption_entries.append(
                    (
                        chunk_start,
                        word.end,
                        " ".join(current_words)
                    )
                )

                current_words = []
                chunk_start = None

        previous_end = word.end

    if current_words and chunk_start is not None:

        caption_entries.append(
            (
                chunk_start,
                previous_end,
                " ".join(current_words)
            )
        )


count = 0

with open(OUTPUT_SRT, "w", encoding="utf-8") as subtitle_file:

    for start, end, text in caption_entries:

        text = text.strip()

        if not text:
            continue

        # Absolute safety limit:
        # never allow one subtitle to remain for more than 5 seconds.
        if end - start > MAX_DURATION:
            end = start + MAX_DURATION

        count += 1

        start_time = format_timestamp(start)
        end_time = format_timestamp(end)

        subtitle_file.write(f"{count}\n")
        subtitle_file.write(
            f"{start_time} --> {end_time}\n"
        )
        subtitle_file.write(
            f"{text}\n\n"
        )

        print(
            f"[{start_time} --> {end_time}] {text}"
        )


print()
print("================================")
print("CAPTION GENERATION COMPLETE")
print("================================")
print()
print(f"Captions generated: {count}")
print(f"Subtitle file : {OUTPUT_SRT}")