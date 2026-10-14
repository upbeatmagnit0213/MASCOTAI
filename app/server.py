from pathlib import Path
from datetime import datetime
import json
import subprocess
import uuid
import os

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\MascotAI")

# Load environment variables from C:\MascotAI\.env
load_dotenv(BASE_DIR / ".env")

# Supabase configuration
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_PUBLISHABLE_KEY = os.getenv(
    "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"
)

APP_DIR = BASE_DIR / "app"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
JOB_DIR = BASE_DIR / "temp" / "jobs"

LTX_PYTHON = BASE_DIR / "ltx_venv" / "Scripts" / "python.exe"
LTX_WORKER = APP_DIR / "ltx_worker.py"

for directory in (
    APP_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    JOB_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Mascot AI Studio",
    version="1.1.0",
)

app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output",
)


# ============================================================
# VALID SETTINGS
# ============================================================

VALID_DURATIONS = {
    1: 25,
    2: 49,
    3: 73,
    5: 121,
    10: 241,
}

VALID_ASPECT_RATIOS = {
    "16:9": (576, 320),
    "9:16": (320, 576),
    "1:1": (512, 512),
}

VALID_QUALITY = {
    "Fast": 4,
    "Standard": 8,
    "High": 8,
}

VALID_AUDIO_MODES = {
    "None",
    "AI Voice",
    "Voice Clone",
}


# ============================================================
# REQUEST MODEL
# ============================================================

import urllib.request
import urllib.error


def require_user(authorization: str | None = Header(None)):
    """Checks the Supabase login token sent by the browser."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Please log in first.")

    token = authorization[7:].strip()

    if not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
        raise HTTPException(500, "Supabase is not configured.")

    req = urllib.request.Request(
        SUPABASE_URL.rstrip("/") + "/auth/v1/user",
        headers={
            "apikey": SUPABASE_PUBLISHABLE_KEY,
            "Authorization": "Bearer " + token,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        raise HTTPException(401, "Your login expired. Please log in again.")
    except Exception:
        raise HTTPException(503, "Could not reach Supabase to check your login.")




class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "AI Video"
    duration: int = 1
    aspect_ratio: str = "16:9"
    quality: str = "Standard"
    audio_mode: str = "None"
    narration_text: str = ""


# ============================================================
# JOB FUNCTIONS
# ============================================================

def create_job_id():
    return (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )


def validate_settings(
    duration,
    aspect_ratio,
    quality,
    audio_mode,
):
    if duration not in VALID_DURATIONS:
        raise HTTPException(
            400,
            "Duration must be 1, 2, 3, 5, or 10 seconds.",
        )

    if aspect_ratio not in VALID_ASPECT_RATIOS:
        raise HTTPException(
            400,
            "Unsupported aspect ratio.",
        )

    if quality not in VALID_QUALITY:
        raise HTTPException(
            400,
            "Unsupported quality.",
        )

    if audio_mode not in VALID_AUDIO_MODES:
        raise HTTPException(
            400,
            "Unsupported audio mode.",
        )


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def start_ltx_worker(
    job_id,
    prompt_file,
    output_file,
    settings_file,
):
    if not LTX_PYTHON.exists():
        raise HTTPException(
            500,
            f"LTX Python not found: {LTX_PYTHON}",
        )

    if not LTX_WORKER.exists():
        raise HTTPException(
            500,
            f"LTX worker not found: {LTX_WORKER}",
        )

    subprocess.Popen(
        [
            str(LTX_PYTHON),
            str(LTX_WORKER),
            job_id,
            str(prompt_file),
            str(output_file),
            str(settings_file),
        ],
        cwd=str(BASE_DIR),
        creationflags=getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        ),
    )


def create_job(
    job_id,
    prompt,
    duration,
    aspect_ratio,
    quality,
    audio_mode="None",
    narration_text="",
    image_path=None,
    voice_path=None,
):
    prompt_file = JOB_DIR / f"{job_id}_prompt.txt"
    settings_file = JOB_DIR / f"{job_id}_settings.json"
    status_file = JOB_DIR / f"{job_id}.json"
    output_file = OUTPUT_DIR / f"mascot_{job_id}.mp4"

    prompt_file.write_text(
        prompt,
        encoding="utf-8",
    )

    settings = {
        "duration": duration,
        "num_frames": VALID_DURATIONS[duration],
        "aspect_ratio": aspect_ratio,
        "quality": quality,
        "audio_mode": audio_mode,
        "narration_text": narration_text,
    }

    if image_path:
        settings["image_path"] = image_path

    if voice_path:
        settings["voice_path"] = voice_path

    write_json(
        settings_file,
        settings,
    )

    write_json(
        status_file,
        {
            "status": "queued",
            "job_id": job_id,
        },
    )

    return (
        prompt_file,
        settings_file,
        output_file,
    )


# ============================================================
# FRONTEND
# ============================================================

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Mascot AI Studio</title>

<style>

:root{
    --bg:#07070c;
    --panel:#151522;
    --panel2:#0f0f18;
    --border:#303044;
    --muted:#9a9aad;
    --accent:#8b7cff;
    --accent2:#c69cff;
    --ok:#63e6be;
    --bad:#ff7b8e;
}

*{
    box-sizing:border-box;
}

body{
    margin:0;
    min-height:100vh;

    background:
        radial-gradient(
            circle at 20% 0%,
            rgba(113,93,255,.25),
            transparent 34%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(221,109,255,.16),
            transparent 28%
        ),
        linear-gradient(
            180deg,
            #10101b,
            #07070c 55%,
            #050507
        );

    color:#fff;
    font-family:Arial,Helvetica,sans-serif;
}

.container{
    width:min(1180px,calc(100% - 36px));
    margin:auto;
    padding:34px 0 60px;
}

.header{
    text-align:center;
    margin-bottom:28px;
}

.brand{
    font-weight:800;
    font-size:18px;
    display:inline-flex;
    gap:10px;
    align-items:center;
    margin-bottom:14px;
}

.brand-badge{
    width:38px;
    height:38px;
    border-radius:12px;
    display:grid;
    place-items:center;

    background:linear-gradient(
        135deg,
        var(--accent),
        var(--accent2)
    );

    color:#09090d;
    font-weight:900;
}

.header h1{
    margin:0;
    font-size:clamp(35px,6vw,58px);
}

.subtitle{
    margin-top:12px;
    color:var(--muted);
}

.modes{
    display:grid;
    grid-template-columns:repeat(
        2,
        minmax(0,1fr)
    );
    gap:18px;
}

.mode-card{
    padding:24px;
    border:1px solid var(--border);
    border-radius:22px;

    background:linear-gradient(
        180deg,
        #1e1e2d,
        #101018
    );

    cursor:pointer;
    transition:.18s;
}

.mode-card:hover{
    transform:translateY(-4px);
    border-color:#63637f;
}

.mode-card.active{
    border-color:var(--accent);

    box-shadow:
        0 0 0 1px
        rgba(139,124,255,.2);
}

.mode-card.disabled{
    opacity:.58;
}

.icon{
    font-size:40px;
    margin-bottom:14px;
}

.mode-title{
    font-weight:800;
    font-size:22px;
}

.mode-desc{
    color:var(--muted);
    line-height:1.5;
    margin-top:8px;
    font-size:14px;
}

.workspace{
    display:none;
    margin-top:22px;
    padding:28px;
    border:1px solid var(--border);
    border-radius:24px;

    background:rgba(22,22,34,.96);

    box-shadow:
        0 24px 70px
        rgba(0,0,0,.25);
}

.workspace-top{
    display:flex;
    justify-content:space-between;
    gap:16px;
    align-items:flex-start;
}

.workspace-title{
    margin:0;
    font-size:28px;
}

.workspace-desc{
    color:var(--muted);
    margin-top:7px;
    line-height:1.5;
}

.btn{
    border:0;
    border-radius:12px;
    padding:12px 16px;
    font-weight:800;
    cursor:pointer;
}

.back{
    background:#242431;
    color:#fff;
}

.label{
    display:block;
    margin-top:24px;
    font-weight:800;
}

.prompt{
    width:100%;
    min-height:145px;
    margin-top:10px;
    padding:16px;

    border-radius:16px;
    border:1px solid #414154;

    background:#0a0a10;
    color:#fff;

    font-size:16px;
    line-height:1.5;

    resize:vertical;
    outline:none;
}

.prompt:focus{
    border-color:var(--accent);
}

.panel{
    display:none;
    margin-top:18px;
    padding:18px;

    border-radius:18px;
    border:1px solid #38384b;

    background:var(--panel2);
}

.panel.show{
    display:block;
}

.file{
    width:100%;
    padding:12px;

    border:1px solid #44445a;
    border-radius:12px;

    background:#11111a;
    color:#fff;
}

.help{
    margin-top:8px;
    color:#88889a;
    font-size:13px;
}

.preview-wrap{
    display:none;
    margin-top:12px;
}

.preview-wrap.show{
    display:block;
}

.preview{
    max-width:100%;
    max-height:300px;

    border-radius:14px;
    border:1px solid #343446;
}

.controls{
    display:grid;

    grid-template-columns:repeat(
        3,
        minmax(0,1fr)
    );

    gap:14px;
    margin-top:18px;
}

.control{
    display:flex;
    flex-direction:column;
    gap:8px;
}

.control span{
    font-weight:700;
    font-size:13px;
    color:#d5d5df;
}

.select{
    padding:13px;

    border-radius:12px;
    border:1px solid #424254;

    background:#13131d;
    color:#fff;

    font-size:15px;
}

.audio-section{
    margin-top:22px;
    padding:18px;

    border-radius:18px;
    border:1px solid #38384b;

    background:#101018;
}

.audio-title{
    font-weight:900;
    margin-bottom:12px;
}

.audio-modes{
    display:flex;
    gap:10px;
    flex-wrap:wrap;
}

.audio-mode{
    padding:11px 14px;

    border-radius:12px;
    border:1px solid #414154;

    background:#171722;
    color:#fff;

    cursor:pointer;
}

.audio-mode.active{
    border-color:var(--accent);
    background:#201c39;
}

.audio-narration{
    display:none;
    margin-top:15px;
}

.audio-narration.show{
    display:block;
}

.audio-narration textarea{
    width:100%;
    min-height:100px;
    padding:14px;

    border-radius:13px;
    border:1px solid #414154;

    background:#0a0a10;
    color:#fff;

    resize:vertical;
}

.generate-row{
    margin-top:20px;
}

.generate{
    background:linear-gradient(
        135deg,
        var(--accent),
        var(--accent2)
    );

    color:#09090d;
    font-size:16px;
}

.generate:disabled{
    opacity:.5;
    cursor:wait;
}

.status{
    min-height:24px;
    margin-top:16px;
    color:#c8c8d4;
}

.status.ok{
    color:var(--ok);
}

.status.bad{
    color:var(--bad);
}

.result{
    margin-top:20px;
}

.result-card{
    padding:15px;

    border-radius:18px;
    border:1px solid #343448;

    background:#0d0d15;
}

.result video{
    display:block;
    width:100%;
    max-height:650px;

    border-radius:14px;
    background:#000;
}

.result a{
    display:inline-block;
    margin-top:12px;

    color:#c6baff;
    text-decoration:none;
}

.footer{
    text-align:center;
    color:#686878;
    font-size:12px;
    margin-top:30px;
}

@media(max-width:800px){

    .modes{
        grid-template-columns:1fr;
    }

    .controls{
        grid-template-columns:1fr;
    }

    .workspace{
        padding:20px;
    }

    .workspace-top{
        flex-direction:column;
    }
}


.authbar{display:flex;justify-content:flex-end;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:18px;color:var(--muted);font-size:14px}
.authbar .btn{padding:9px 14px}
.authcard{display:none;max-width:420px;margin:0 auto 22px;padding:22px;border:1px solid var(--border);border-radius:20px;background:rgba(22,22,34,.96)}
.authcard.show{display:block}
.authcard input{width:100%;padding:13px;margin-top:10px;border-radius:12px;border:1px solid #424254;background:#13131d;color:#fff;font-size:15px}
.authrow{display:flex;gap:10px;margin-top:14px}
.authrow .btn{flex:1}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#09090d}
.btn-ghost{background:#242431;color:#fff}
</style>
</head>

<body>

<div class="container">

<div class="authbar" id="authBar">
  <span id="authUser">Not signed in</span>
  <button type="button" class="btn btn-ghost" id="showAuthBtn" onclick="toggleAuthCard()">Log in / Sign up</button>
  <button type="button" class="btn btn-ghost" id="logoutBtn" style="display:none" onclick="signOutUser()">Log out</button>
</div>

<div class="authcard" id="authCard">
  <div style="font-weight:800;font-size:20px">Welcome to Mascot AI</div>
  <input type="email" id="authEmail" placeholder="Email" autocomplete="email">
  <input type="password" id="authPassword" placeholder="Password (min 6 characters)" autocomplete="current-password">
  <div class="authrow">
    <button type="button" class="btn btn-primary" onclick="signInUser()">Log in</button>
    <button type="button" class="btn btn-ghost" onclick="signUpUser()">Sign up</button>
  </div>
  <div class="status" id="authMsg"></div>
</div>


<div class="header">

<div class="brand">

<div class="brand-badge">M</div>

Mascot AI Studio

</div>

<h1>Create. Animate. Generate.</h1>

<div class="subtitle">
Local AI video creation with video, photo animation and voice.
</div>

</div>


<div class="modes">

<div
    class="mode-card"
    id="mode-ai"
    onclick="openMode('AI Video')"
>

<div class="icon">🤖</div>

<div class="mode-title">
AI Video
</div>

<div class="mode-desc">
Create a video from a text prompt.
</div>

</div>


<div
    class="mode-card"
    id="mode-photo"
    onclick="openMode('Photo → Video')"
>

<div class="icon">📸</div>

<div class="mode-title">
Photo → Video
</div>

<div class="mode-desc">
Animate your photo with local AI.
</div>

</div>


<div
    class="mode-card disabled"
    onclick="notReady('Gameplay Short')"
>

<div class="icon">🎮</div>

<div class="mode-title">
Gameplay Short
</div>

<div class="mode-desc">
Coming next.
</div>

</div>


<div
    class="mode-card disabled"
    onclick="notReady('Script → Video')"
>

<div class="icon">📝</div>

<div class="mode-title">
Script → Video
</div>

<div class="mode-desc">
Coming next.
</div>

</div>

</div>


<div class="workspace" id="workspace">

<div class="workspace-top">

<div>

<h2
    class="workspace-title"
    id="modeTitle"
></h2>

<div
    class="workspace-desc"
    id="modeDesc"
></div>

</div>

<button
    class="btn back"
    onclick="closeWorkspace()"
>
← Back
</button>

</div>


<label
    class="label"
    for="prompt"
>
Describe your video
</label>

<textarea
    class="prompt"
    id="prompt"
    placeholder="Describe the video you want..."
></textarea>


<div
    class="panel"
    id="imagePanel"
>

<div
    style="font-weight:800;margin-bottom:9px"
>
Source Image
</div>

<input
    class="file"
    type="file"
    id="imageInput"
    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
>

<div class="help">
JPG or PNG · maximum 20 MB
</div>

<div
    class="preview-wrap"
    id="previewWrap"
>

<img
    class="preview"
    id="preview"
    alt="Selected image"
>

</div>

</div>


<div class="controls">

<label class="control">

<span>Duration</span>

<select
    class="select"
    id="duration"
>

<option value="1">
1 second
</option>

<option value="2">
2 seconds
</option>

<option value="3">
3 seconds
</option>

<option
    value="5"
    selected
>
5 seconds
</option>

<option value="10">
10 seconds
</option>

</select>

</label>


<label class="control">

<span>Aspect Ratio</span>

<select
    class="select"
    id="aspectRatio"
>

<option
    value="16:9"
    selected
>
16:9 · Landscape
</option>

<option value="9:16">
9:16 · Vertical
</option>

<option value="1:1">
1:1 · Square
</option>

</select>

</label>


<label class="control">

<span>Quality</span>

<select
    class="select"
    id="quality"
>

<option value="Fast">
Fast
</option>

<option
    value="Standard"
    selected
>
Standard
</option>

<option value="High">
High
</option>

</select>

</label>

</div>


<div class="audio-section">

<div class="audio-title">
🔊 Audio
</div>

<div class="audio-modes">

<button
    type="button"
    class="audio-mode active"
    data-audio="None"
    onclick="setAudioMode('None')"
>
No Audio
</button>

<button
    type="button"
    class="audio-mode"
    data-audio="AI Voice"
    onclick="setAudioMode('AI Voice')"
>
🤖 AI Voice
</button>

<button
    type="button"
    class="audio-mode"
    data-audio="Voice Clone"
    onclick="setAudioMode('Voice Clone')"
>
🎙️ Voice Clone
</button>

</div>


<div
    class="audio-narration"
    id="audioNarration"
>

<label
    style="display:block;font-weight:800;margin-bottom:8px"
    for="narration"
>
Narration Text
</label>

<textarea
    id="narration"
    placeholder="Type exactly what you want Mascot AI to speak..."
></textarea>

<div class="help">
Generated locally with Chatterbox.
</div>

</div>


<div
    class="panel"
    id="voicePanel"
>

<div
    style="font-weight:800;margin-bottom:9px"
>
Reference Voice
</div>

<input
    class="file"
    type="file"
    id="voiceInput"
    accept=".wav,audio/wav"
>

<div class="help">
Use a clear WAV recording of a voice you have permission to clone.
</div>

<div
    id="voiceName"
    class="help"
></div>

</div>

</div>


<div class="generate-row">

<button
    type="button"
    class="btn generate"
    id="generateButton"
    onclick="startCreation()"
>
✨ Generate Video
</button>

</div>


<div
    class="status"
    id="status"
></div>

<div
    class="result"
    id="result"
></div>

</div>


<div class="footer">
Mascot AI Studio · Local video + voice pipeline
</div>

</div>


<script>

let currentMode = null;
let audioMode = "None";


function openMode(mode){

    currentMode = mode;

    document
        .querySelectorAll(".mode-card")
        .forEach(
            x => x.classList.remove("active")
        );

    const card =
        document.getElementById(
            mode === "AI Video"
                ? "mode-ai"
                : "mode-photo"
        );

    if(card){
        card.classList.add("active");
    }

    document.getElementById(
        "workspace"
    ).style.display = "block";

    document.getElementById(
        "modeTitle"
    ).innerText = mode;

    document.getElementById(
        "modeDesc"
    ).innerText =
        mode === "AI Video"
            ? "Describe the video you want Mascot AI to create."
            : "Upload an image and describe how you want it to move.";

    document.getElementById(
        "prompt"
    ).placeholder =
        mode === "AI Video"
            ? "Example: A futuristic gaming room with cinematic camera movement..."
            : "Example: The person slowly turns their head and the camera pushes in...";

    document.getElementById("prompt").value = "";

    document.getElementById("narration").value = "";

    document.getElementById("status").innerText = "";

    document.getElementById("status").className = "status";

    document.getElementById("result").innerHTML = "";

    document.getElementById(
        "imagePanel"
    ).classList.toggle(
        "show",
        mode === "Photo → Video"
    );

    document.getElementById(
        "imageInput"
    ).value = "";

    document.getElementById(
        "voiceInput"
    ).value = "";

    document.getElementById(
        "previewWrap"
    ).classList.remove("show");

    document.getElementById(
        "voiceName"
    ).innerText = "";

    setAudioMode("None");

    document.getElementById(
        "workspace"
    ).scrollIntoView({
        behavior:"smooth"
    });
}


function closeWorkspace(){

    document.getElementById(
        "workspace"
    ).style.display = "none";

    document
        .querySelectorAll(".mode-card")
        .forEach(
            x => x.classList.remove("active")
        );

    currentMode = null;
}


function notReady(name){

    alert(
        name + " is not connected yet."
    );
}


function setAudioMode(mode){

    audioMode = mode;

    document
        .querySelectorAll(".audio-mode")
        .forEach(
            x =>
                x.classList.toggle(
                    "active",
                    x.dataset.audio === mode
                )
        );

    document.getElementById(
        "audioNarration"
    ).classList.toggle(
        "show",
        mode !== "None"
    );

    document.getElementById(
        "voicePanel"
    ).classList.toggle(
        "show",
        mode === "Voice Clone"
    );
}


document
    .getElementById("imageInput")
    .addEventListener(
        "change",
        () => {

            const f =
                document.getElementById(
                    "imageInput"
                ).files[0];

            const w =
                document.getElementById(
                    "previewWrap"
                );

            if(!f){

                w.classList.remove("show");

                return;
            }

            if(
                ![
                    "image/jpeg",
                    "image/png"
                ].includes(f.type)
            ){

                alert(
                    "Please select a JPG or PNG image."
                );

                document.getElementById(
                    "imageInput"
                ).value = "";

                w.classList.remove("show");

                return;
            }

            if(
                f.size >
                20 * 1024 * 1024
            ){

                alert(
                    "Image must be smaller than 20 MB."
                );

                document.getElementById(
                    "imageInput"
                ).value = "";

                w.classList.remove("show");

                return;
            }

            document.getElementById(
                "preview"
            ).src =
                URL.createObjectURL(f);

            w.classList.add("show");
        }
    );


document
    .getElementById("voiceInput")
    .addEventListener(
        "change",
        () => {

            const f =
                document.getElementById(
                    "voiceInput"
                ).files[0];

            document.getElementById(
                "voiceName"
            ).innerText =
                f
                    ? "Selected: " + f.name
                    : "";
        }
    );


async function startCreation(){

    const prompt =
        document
            .getElementById("prompt")
            .value
            .trim();

    const duration =
        parseInt(
            document.getElementById(
                "duration"
            ).value,
            10
        );

    const aspectRatio =
        document.getElementById(
            "aspectRatio"
        ).value;

    const quality =
        document.getElementById(
            "quality"
        ).value;

    const button =
        document.getElementById(
            "generateButton"
        );

    const status =
        document.getElementById(
            "status"
        );

    const result =
        document.getElementById(
            "result"
        );


    if(!currentMode){

        alert(
            "Select a mode first."
        );

        return;
    }


    if(!prompt){

        status.innerText =
            "Please enter a video prompt.";

        status.className =
            "status bad";

        return;
    }


    if(
        audioMode !== "None" &&
        !document
            .getElementById("narration")
            .value
            .trim()
    ){

        status.innerText =
            "Enter narration text for the selected audio mode.";

        status.className =
            "status bad";

        return;
    }


    if(
        currentMode === "Photo → Video" &&
        !document
            .getElementById("imageInput")
            .files[0]
    ){

        status.innerText =
            "Please select a source image.";

        status.className =
            "status bad";

        return;
    }


    if(
        audioMode === "Voice Clone" &&
        !document
            .getElementById("voiceInput")
            .files[0]
    ){

        status.innerText =
            "Please select a reference WAV for Voice Clone.";

        status.className =
            "status bad";

        return;
    }


    button.disabled = true;

    result.innerHTML = "";

    status.innerText =
        "Starting generation...";

    status.className =
        "status";


    try{

        let response;


        if(
            currentMode === "Photo → Video"
        ){

            const fd =
                new FormData();

            fd.append(
                "prompt",
                prompt
            );

            fd.append(
                "duration",
                duration
            );

            fd.append(
                "aspect_ratio",
                aspectRatio
            );

            fd.append(
                "quality",
                quality
            );

            fd.append(
                "audio_mode",
                audioMode
            );

            fd.append(
                "narration_text",
                document
                    .getElementById("narration")
                    .value
                    .trim()
            );

            fd.append(
                "image",
                document
                    .getElementById("imageInput")
                    .files[0]
            );


            if(
                audioMode === "Voice Clone"
            ){

                fd.append(
                    "voice",
                    document
                        .getElementById("voiceInput")
                        .files[0]
                );
            }


            response =
                await fetch(
                    "/api/generate-i2v",
                    {
                        method:"POST",
                        body:fd
                    }
                );

        }else{

            if(
                audioMode === "Voice Clone"
            ){

                status.innerText =
                    "Voice Clone is currently enabled through Photo → Video. Select Photo → Video.";

                status.className =
                    "status bad";

                button.disabled = false;

                return;
            }


            response =
                await fetch(
                    "/api/generate",
                    {
                        method:"POST",

                        headers:{
                            "Content-Type":
                                "application/json"
                        },

                        body:JSON.stringify({

                            prompt:prompt,

                            mode:"AI Video",

                            duration:duration,

                            aspect_ratio:aspectRatio,

                            quality:quality,

                            audio_mode:audioMode,

                            narration_text:
                                document
                                    .getElementById(
                                        "narration"
                                    )
                                    .value
                                    .trim()
                        })
                    }
                );
        }


        const data =
            await response.json();


        if(!response.ok){

            throw new Error(
                data.detail ||
                "Could not start generation."
            );
        }


        pollJob(
            data.job_id,
            status,
            result,
            button
        );


    }catch(e){

        status.innerText =
            "Failed: " + e.message;

        status.className =
            "status bad";

        button.disabled = false;
    }
}


async function pollJob(
    jobId,
    status,
    result,
    button
){

    try{

        const response =
            await fetch(
                "/api/status/" + jobId
            );

        const data =
            await response.json();


        if(!response.ok){

            throw new Error(
                data.detail ||
                "Status failed."
            );
        }


        if(data.status === "queued"){

            status.innerText =
                "Waiting for the AI worker...";

            return setTimeout(
                () =>
                    pollJob(
                        jobId,
                        status,
                        result,
                        button
                    ),
                1000
            );
        }


        if(data.status === "loading"){

            status.innerText =
                "Loading local AI model...";

            return setTimeout(
                () =>
                    pollJob(
                        jobId,
                        status,
                        result,
                        button
                    ),
                1500
            );
        }


        if(data.status === "generating"){

            status.innerText =
                "Generating video...";

            return setTimeout(
                () =>
                    pollJob(
                        jobId,
                        status,
                        result,
                        button
                    ),
                1500
            );
        }


        if(
            data.status ===
            "audio_generating"
        ){

            status.innerText =
                "Generating and attaching voice...";

            return setTimeout(
                () =>
                    pollJob(
                        jobId,
                        status,
                        result,
                        button
                    ),
                1500
            );
        }


        if(
            data.status ===
            "completed"
        ){

            const url =
                "/output/" +
                data.filename;

            status.innerText =
                "Video generated successfully.";

            status.className =
                "status ok";


            result.innerHTML =
                '<div class="result-card">' +
                '<video controls autoplay loop playsinline src="' +
                url +
                '"></video>' +
                '<a href="' +
                url +
                '" target="_blank">' +
                'Open generated video' +
                '</a>' +
                '</div>';


            button.disabled = false;

            return;
        }


        if(
            data.status ===
            "failed"
        ){

            status.innerText =
                "Generation failed: " +
                (
                    data.error ||
                    "Unknown error."
                );

            status.className =
                "status bad";

            button.disabled = false;

            return;
        }


        throw new Error(
            "Unknown job status."
        );


    }catch(e){

        status.innerText =
            "Status error: " +
            e.message;

        status.className =
            "status bad";

        button.disabled = false;
    }
}

</script>

<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
<script>

let sb = null;
let currentUser = null;
let currentToken = null;

function setAuthMsg(text, kind){
    const el = document.getElementById("authMsg");
    el.innerText = text || "";
    el.className = "status" + (kind ? " " + kind : "");
}

function showAuthCard(show){
    document.getElementById("authCard").classList.toggle("show", show);
}

function toggleAuthCard(){
    const card = document.getElementById("authCard");
    showAuthCard(!card.classList.contains("show"));
}

function applySession(session){
    currentUser = session ? session.user : null;
    currentToken = session ? session.access_token : null;

    document.getElementById("authUser").innerText =
        currentUser ? "Signed in as " + currentUser.email : "Not signed in";

    document.getElementById("showAuthBtn").style.display =
        currentUser ? "none" : "inline-block";

    document.getElementById("logoutBtn").style.display =
        currentUser ? "inline-block" : "none";

    if(currentUser){
        showAuthCard(false);
        setAuthMsg("");
    }
}

async function initAuth(){
    try{
        const res = await fetch("/api/config");
        const cfg = await res.json();

        if(!cfg.supabase_url || !cfg.supabase_key){
            setAuthMsg("Supabase keys are missing in C:/MascotAI/.env", "bad");
            return;
        }

        sb = window.supabase.createClient(cfg.supabase_url, cfg.supabase_key);

        const { data } = await sb.auth.getSession();
        applySession(data.session);

        sb.auth.onAuthStateChange((_event, session) => applySession(session));
    }catch(e){
        setAuthMsg("Could not start login: " + e.message, "bad");
    }
}

function readAuthForm(){
    const email = document.getElementById("authEmail").value.trim();
    const password = document.getElementById("authPassword").value;

    if(!email || !password){
        setAuthMsg("Enter your email and password.", "bad");
        return null;
    }
    if(!sb){
        setAuthMsg("Login is not ready yet. Refresh the page.", "bad");
        return null;
    }
    return { email, password };
}

async function signUpUser(){
    const f = readAuthForm();
    if(!f) return;

    setAuthMsg("Creating account...");
    const { data, error } = await sb.auth.signUp(f);

    if(error){
        setAuthMsg(error.message, "bad");
        return;
    }

    if(data.session){
        setAuthMsg("Account created.", "ok");
    }else{
        setAuthMsg("Account created. Check your email and click the confirmation link, then log in.", "ok");
    }
}

async function signInUser(){
    const f = readAuthForm();
    if(!f) return;

    setAuthMsg("Logging in...");
    const { error } = await sb.auth.signInWithPassword(f);

    if(error){
        setAuthMsg(error.message, "bad");
    }
}

async function signOutUser(){
    if(sb){
        await sb.auth.signOut();
    }
}

// Require login before generating videos
const _startCreationOriginal = startCreation;

startCreation = function(){
    if(!currentUser){
        showAuthCard(true);
        setAuthMsg("Please log in to generate videos.", "bad");
        window.scrollTo({ top: 0, behavior: "smooth" });
        return;
    }
    return _startCreationOriginal();
};

// Send the login token with generation requests
const _fetchOriginal = window.fetch.bind(window);

window.fetch = function(url, options){
    options = options || {};

    if(
        typeof url === "string" &&
        url.indexOf("/api/generate") === 0 &&
        currentToken
    ){
        options.headers = Object.assign(
            {},
            options.headers || {},
            { Authorization: "Bearer " + currentToken }
        );
    }

    return _fetchOriginal(url, options);
};

initAuth();

</script>

</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():
    return PAGE


@app.post("/api/generate")
def generate(request: GenerateRequest, user: dict = Depends(require_user)):

    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(
            400,
            "Prompt cannot be empty.",
        )

    if request.mode != "AI Video":
        raise HTTPException(
            400,
            "Use Photo → Video for image-conditioned generation.",
        )

    validate_settings(
        request.duration,
        request.aspect_ratio,
        request.quality,
        request.audio_mode,
    )

    if request.audio_mode == "Voice Clone":
        raise HTTPException(
            400,
            "Voice Clone is currently available in Photo → Video mode.",
        )

    if (
        request.audio_mode != "None"
        and not request.narration_text.strip()
    ):
        raise HTTPException(
            400,
            "Narration text cannot be empty when audio is enabled.",
        )

    job_id = create_job_id()

    (
        prompt_file,
        settings_file,
        output_file,
    ) = create_job(
        job_id,
        prompt,
        request.duration,
        request.aspect_ratio,
        request.quality,
        audio_mode=request.audio_mode,
        narration_text=request.narration_text.strip(),
    )

    start_ltx_worker(
        job_id,
        prompt_file,
        output_file,
        settings_file,
    )

    return {
        "status": "started",
        "job_id": job_id,
        "mode": "AI Video",
    }


@app.post("/api/generate-i2v")
async def generate_i2v(
    prompt: str = Form(...),
    duration: int = Form(1),
    aspect_ratio: str = Form("16:9"),
    quality: str = Form("Standard"),
    audio_mode: str = Form("None"),
    narration_text: str = Form(""),
    image: UploadFile = File(...),
    voice: UploadFile | None = File(None),
    user: dict = Depends(require_user),
):

    prompt = prompt.strip()

    if not prompt:
        raise HTTPException(
            400,
            "Motion prompt cannot be empty.",
        )


    try:

        duration = int(duration)

    except (TypeError, ValueError) as exc:

        raise HTTPException(
            400,
            "Invalid duration.",
        ) from exc


    validate_settings(
        duration,
        aspect_ratio,
        quality,
        audio_mode,
    )


    narration_text = narration_text.strip()

    if (
        audio_mode != "None"
        and not narration_text
    ):

        raise HTTPException(
            400,
            "Narration text cannot be empty when audio is enabled.",
        )


    image_suffix = Path(
        image.filename or ""
    ).suffix.lower()


    if image_suffix not in {
        ".jpg",
        ".jpeg",
        ".png",
    }:

        raise HTTPException(
            400,
            "Only JPG and PNG images are supported.",
        )


    if image.content_type not in {
        "image/jpeg",
        "image/png",
    }:

        raise HTTPException(
            400,
            "Invalid image type.",
        )


    image_data = await image.read()


    if len(image_data) > 20 * 1024 * 1024:

        raise HTTPException(
            400,
            "Image must be smaller than 20 MB.",
        )


    voice_path = None
    voice_data = None


    if audio_mode == "Voice Clone":

        if voice is None:

            raise HTTPException(
                400,
                "Reference voice WAV is required for Voice Clone.",
            )


        voice_suffix = Path(
            voice.filename or ""
        ).suffix.lower()


        if (
            voice_suffix != ".wav"
            or voice.content_type not in {
                "audio/wav",
                "audio/x-wav",
                "audio/wave",
                "application/octet-stream",
            }
        ):

            raise HTTPException(
                400,
                "Voice Clone currently accepts a WAV reference recording.",
            )


        voice_data = await voice.read()


        if len(voice_data) > 50 * 1024 * 1024:

            raise HTTPException(
                400,
                "Reference voice must be smaller than 50 MB.",
            )


    job_id = create_job_id()


    image_file = (
        INPUT_DIR /
        f"i2v_{job_id}{image_suffix}"
    )


    image_file.write_bytes(
        image_data
    )


    if voice_data is not None:

        voice_path = (
            INPUT_DIR /
            f"voice_{job_id}.wav"
        )

        voice_path.write_bytes(
            voice_data
        )


    (
        prompt_file,
        settings_file,
        output_file,
    ) = create_job(
        job_id,
        prompt,
        duration,
        aspect_ratio,
        quality,
        audio_mode=audio_mode,
        narration_text=narration_text,
        image_path=str(
            image_file
        ),
        voice_path=(
            str(voice_path)
            if voice_path
            else None
        ),
    )


    start_ltx_worker(
        job_id,
        prompt_file,
        output_file,
        settings_file,
    )


    return {
        "status": "started",
        "job_id": job_id,
        "mode": "Photo → Video",
    }


@app.get("/api/status/{job_id}")
def status(job_id: str):

    status_file = (
        JOB_DIR /
        f"{job_id}.json"
    )


    if not status_file.exists():

        raise HTTPException(
            404,
            "Job not found.",
        )


    try:

        return json.loads(
            status_file.read_text(
                encoding="utf-8-sig"
            )
        )

    except Exception as exc:

        raise HTTPException(
            500,
            str(exc),
        ) from exc


@app.get("/api/config")
def public_config():
    # Publishable key is designed to be public (used in the browser).
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_key": SUPABASE_PUBLISHABLE_KEY,
    }


@app.get("/health")
def health():

    return {
        "status": "online",
        "studio": "Mascot AI Studio",
        "ai_video": "connected",
        "photo_to_video": "connected",
        "audio": "Chatterbox local",
        "generation_engine":
            "LTX-Video + Chatterbox",
        "supabase_configured":
            bool(
                SUPABASE_URL
                and SUPABASE_PUBLISHABLE_KEY
            ),
    }