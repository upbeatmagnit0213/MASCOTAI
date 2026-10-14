from pathlib import Path
import sys
import json
import inspect
import traceback

import torch


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\MascotAI")

LTX_SOURCE = BASE_DIR / "temp" / "LTX-Video-main"

LTX_CHECKPOINT = (
    BASE_DIR
    / "models"
    / "LTX-Video"
    / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
)

T5_MODEL = (
    BASE_DIR
    / "models"
    / "LTX-Video"
    / "T5-XXL-8bit"
)

T5_TOKENIZER = T5_MODEL / "tokenizer"

EMBEDDINGS_FILE = (
    BASE_DIR
    / "temp"
    / "ltx_test"
    / "embeddings.pt"
)

SETTINGS_FILE = (
    BASE_DIR
    / "temp"
    / "ltx_test"
    / "settings.json"
)

PROMPT_FILE = (
    BASE_DIR
    / "temp"
    / "ltx_test"
    / "prompt.txt"
)


# ============================================================
# OUTPUT HELPERS
# ============================================================

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def show_path(label, path):
    path = Path(path)

    print(f"{label}:")
    print(f"  {path}")
    print(f"  EXISTS: {path.exists()}")

    if path.exists():
        try:
            print(
                f"  SIZE: "
                f"{path.stat().st_size / 1024 / 1024:.2f} MB"
            )
        except Exception:
            pass


# ============================================================
# BASIC SYSTEM
# ============================================================

section("1. PYTHON / PYTORCH / CUDA")

print("Python executable:")
print(sys.executable)

print("\nPython version:")
print(sys.version)

print("\nPyTorch:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():

    print("\nCUDA version:")
    print(torch.version.cuda)

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    props = torch.cuda.get_device_properties(0)

    print("\nGPU VRAM:")
    print(
        f"{props.total_memory / 1024**3:.2f} GB"
    )

    print("\nCompute capability:")
    print(
        f"{props.major}.{props.minor}"
    )


# ============================================================
# PATHS
# ============================================================

section("2. IMPORTANT FILES")

show_path("LTX SOURCE", LTX_SOURCE)
show_path("LTX CHECKPOINT", LTX_CHECKPOINT)
show_path("T5 MODEL", T5_MODEL)
show_path("T5 TOKENIZER", T5_TOKENIZER)
show_path("EMBEDDINGS", EMBEDDINGS_FILE)
show_path("SETTINGS", SETTINGS_FILE)
show_path("PROMPT", PROMPT_FILE)


# ============================================================
# LTX SOURCE
# ============================================================

section("3. LTX SOURCE STRUCTURE")

if LTX_SOURCE.exists():

    important_files = [
        LTX_SOURCE / "ltx_video" / "__init__.py",

        LTX_SOURCE
        / "ltx_video"
        / "models"
        / "transformers"
        / "transformer3d.py",

        LTX_SOURCE
        / "ltx_video"
        / "pipelines"
        / "pipeline_ltx_video.py",

        LTX_SOURCE
        / "ltx_video"
        / "schedulers"
        / "rf.py",

        LTX_SOURCE
        / "ltx_video"
        / "models"
        / "autoencoders"
        / "causal_video_autoencoder.py",
    ]

    for file in important_files:
        print()
        print(file)
        print("  EXISTS:", file.exists())

else:
    print("LTX source directory does not exist.")


# ============================================================
# IMPORT LTX
# ============================================================

section("4. LTX IMPORTS")

if str(LTX_SOURCE) not in sys.path:
    sys.path.insert(0, str(LTX_SOURCE))

try:

    from ltx_video.models.transformers.transformer3d import (
        Transformer3DModel,
    )

    print("Transformer3DModel import: OK")

    print("\nTransformer module:")
    print(
        inspect.getfile(Transformer3DModel)
    )

except Exception as e:

    print("Transformer3DModel import FAILED")
    print(type(e).__name__, e)
    traceback.print_exc()


try:

    from ltx_video.pipelines.pipeline_ltx_video import (
        LTXVideoPipeline,
    )

    print("\nLTXVideoPipeline import: OK")

    print("\nPipeline module:")
    print(
        inspect.getfile(LTXVideoPipeline)
    )

except Exception as e:

    print("LTXVideoPipeline import FAILED")
    print(type(e).__name__, e)
    traceback.print_exc()


try:

    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
    )

    print("\nCausalVideoAutoencoder import: OK")

    print("\nVAE module:")
    print(
        inspect.getfile(CausalVideoAutoencoder)
    )

except Exception as e:

    print("CausalVideoAutoencoder import FAILED")
    print(type(e).__name__, e)
    traceback.print_exc()


try:

    from ltx_video.schedulers.rf import (
        RectifiedFlowScheduler,
    )

    print("\nRectifiedFlowScheduler import: OK")

    print("\nScheduler module:")
    print(
        inspect.getfile(RectifiedFlowScheduler)
    )

except Exception as e:

    print("RectifiedFlowScheduler import FAILED")
    print(type(e).__name__, e)
    traceback.print_exc()


# ============================================================
# TRANSFORMER SIGNATURE
# ============================================================

section("5. EXACT TRANSFORMER CONSTRUCTOR")

try:

    print(
        inspect.signature(
            Transformer3DModel.__init__
        )
    )

except Exception as e:

    print(
        "Could not inspect Transformer3DModel:",
        e,
    )


# ============================================================
# PIPELINE SIGNATURE
# ============================================================

section("6. EXACT PIPELINE CONSTRUCTOR")

try:

    print(
        inspect.signature(
            LTXVideoPipeline.__init__
        )
    )

except Exception as e:

    print(
        "Could not inspect pipeline constructor:",
        e,
    )


section("7. EXACT PIPELINE CALL")

try:

    print(
        inspect.signature(
            LTXVideoPipeline.__call__
        )
    )

except Exception as e:

    print(
        "Could not inspect pipeline call:",
        e,
    )


# ============================================================
# TRANSFORMER CONFIG / DEFAULTS
# ============================================================

section("8. TRANSFORMER CLASS INFORMATION")

try:

    source = inspect.getsource(
        Transformer3DModel.__init__
    )

    print(
        source[:20000]
    )

except Exception as e:

    print(
        "Could not read Transformer3DModel source:",
        e,
    )


# ============================================================
# PIPELINE SOURCE - EMBEDDINGS
# ============================================================

section("9. PIPELINE EMBEDDING HANDLING")

try:

    pipeline_source = inspect.getsource(
        LTXVideoPipeline.__call__
    )

    lines = pipeline_source.splitlines()

    found = False

    for i, line in enumerate(lines):

        lower = line.lower()

        if (
            "prompt_embeds" in lower
            or "negative_prompt_embeds" in lower
            or "encode_prompt" in lower
        ):

            start = max(0, i - 5)
            end = min(len(lines), i + 8)

            print(
                f"\n--- pipeline source around line {i + 1} ---"
            )

            for source_line in lines[start:end]:
                print(source_line)

            found = True

    if not found:
        print(
            "No prompt embedding references found "
            "in inspected section."
        )

except Exception as e:

    print(
        "Could not inspect pipeline embedding handling:",
        e,
    )


# ============================================================
# EMBEDDINGS FILE
# ============================================================

section("10. EMBEDDINGS FILE CONTENT")

if EMBEDDINGS_FILE.exists():

    try:

        print(
            "Loading:",
            EMBEDDINGS_FILE
        )

        data = torch.load(
            EMBEDDINGS_FILE,
            map_location="cpu",
            weights_only=False,
        )

        print("\nPython type:")
        print(type(data))

        if isinstance(data, dict):

            print("\nDictionary keys:")

            for key in data.keys():
                print(
                    f"  {repr(key)}"
                )

            print("\nDictionary values:")

            for key, value in data.items():

                print(
                    f"\nKEY: {repr(key)}"
                )

                print(
                    f"TYPE: {type(value)}"
                )

                if isinstance(value, torch.Tensor):

                    print(
                        f"SHAPE: {tuple(value.shape)}"
                    )

                    print(
                        f"DTYPE: {value.dtype}"
                    )

                    print(
                        f"DEVICE: {value.device}"
                    )

                elif isinstance(value, dict):

                    print(
                        "Nested dictionary keys:",
                        list(value.keys())
                    )

                elif isinstance(value, (list, tuple)):

                    print(
                        "Length:",
                        len(value)
                    )

                else:

                    print(
                        "VALUE:",
                        repr(value)[:500]
                    )

        elif isinstance(data, torch.Tensor):

            print(
                "Tensor shape:",
                tuple(data.shape)
            )

            print(
                "Tensor dtype:",
                data.dtype
            )

            print(
                "Tensor device:",
                data.device
            )

        else:

            print(
                "\nVALUE:"
            )

            print(
                repr(data)[:5000]
            )

    except Exception as e:

        print(
            "\nFAILED TO LOAD EMBEDDINGS:"
        )

        print(
            type(e).__name__,
            e,
        )

        traceback.print_exc()

else:

    print(
        "Embeddings file does not exist."
    )


# ============================================================
# SETTINGS
# ============================================================

section("11. SETTINGS FILE")

if SETTINGS_FILE.exists():

    try:

        with open(
            SETTINGS_FILE,
            "r",
            encoding="utf-8-sig",
        ) as f:

            settings = json.load(f)

        print(
            json.dumps(
                settings,
                indent=4,
                ensure_ascii=False,
            )
        )

    except Exception as e:

        print(
            "SETTINGS ERROR:",
            type(e).__name__,
            e,
        )

else:

    print(
        "Settings file does not exist."
    )


# ============================================================
# PROMPT
# ============================================================

section("12. PROMPT FILE")

if PROMPT_FILE.exists():

    try:

        prompt = PROMPT_FILE.read_text(
            encoding="utf-8-sig"
        )

        print(
            repr(prompt)
        )

    except Exception as e:

        print(
            "Prompt read error:",
            e,
        )

else:

    print(
        "Prompt file does not exist."
    )


# ============================================================
# CHECKPOINT INFORMATION
# ============================================================

section("13. LTX CHECKPOINT INFORMATION")

if LTX_CHECKPOINT.exists():

    try:

        from safetensors import safe_open

        with safe_open(
            str(LTX_CHECKPOINT),
            framework="pt",
            device="cpu",
        ) as f:

            keys = list(f.keys())

            print(
                "Total checkpoint tensors:",
                len(keys)
            )

            print("\nFIRST 50 CHECKPOINT KEYS:")

            for key in keys[:50]:
                print(
                    "  ",
                    key
                )

            print("\nCHECKPOINT KEY PATTERNS:")

            patterns = [
                "transformer_blocks",
                "attn1",
                "attn2",
                "proj_in",
                "proj_out",
                "caption_projection",
                "time_embed",
                "norm",
            ]

            for pattern in patterns:

                matches = [
                    key
                    for key in keys
                    if pattern in key
                ]

                print(
                    f"{pattern}: {len(matches)}"
                )

                if matches:
                    for key in matches[:5]:
                        print(
                            "    ",
                            key
                        )

    except Exception as e:

        print(
            "Checkpoint inspection failed:",
            type(e).__name__,
            e,
        )

        traceback.print_exc()

else:

    print(
        "Checkpoint does not exist."
    )


# ============================================================
# T5
# ============================================================

section("14. T5 MODEL")

if T5_MODEL.exists():

    print(
        "T5 model directory exists."
    )

    print(
        "\nT5 files:"
    )

    for item in sorted(T5_MODEL.iterdir()):

        print(
            f"  {item.name}"
        )

else:

    print(
        "T5 model directory does not exist."
    )


# ============================================================
# DIFFUSERS / TRANSFORMERS
# ============================================================

section("15. INSTALLED LIBRARY VERSIONS")

for package_name in [
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "protobuf",
]:

    try:

        module = __import__(
            package_name
        )

        version = getattr(
            module,
            "__version__",
            "unknown",
        )

        print(
            f"{package_name}: {version}"
        )

    except Exception as e:

        print(
            f"{package_name}: NOT AVAILABLE ({e})"
        )


# ============================================================
# FINAL
# ============================================================

section("DIAGNOSTIC COMPLETE")

print(
    "NO PROJECT FILES WERE MODIFIED."
)

print(
    "NO VIDEO GENERATION WAS STARTED."
)

print(
    "This report is only collecting the actual "
    "state of the LTX installation."
)

print(
    "\nCopy the complete terminal output and send it to me."
)

print(
    "Do not change ltx_worker.py again yet."
)

print("=" * 80)