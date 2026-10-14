"""
Step 2: makes the SERVER check the Supabase login before generating videos.

Run once from C:\\MascotAI (after patch_auth.py):
    .venv\\Scripts\\python.exe patch_auth2.py

Makes a backup (server.py.bak2) first.
"""

from pathlib import Path
import shutil
import sys

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\MascotAI\app\server.py")

HELPER = r'''import urllib.request
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


@app.get("/api/config")'''

FETCH_WRAPPER = r'''// Send the login token with generation requests
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

initAuth();'''


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        sys.exit(f"STOP: expected 1 match for {label}, found {count}. Nothing was changed.")
    return text.replace(old, new)


def main():
    if not TARGET.exists():
        sys.exit(f"STOP: file not found: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    if "def require_user" in text:
        sys.exit("Already patched. Nothing to do.")

    if "authBar" not in text:
        sys.exit("STOP: run patch_auth.py first.")

    text = replace_once(
        text,
        "from fastapi import FastAPI, HTTPException, UploadFile, File, Form",
        "from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends",
        "fastapi import line",
    )
    text = replace_once(text, '@app.get("/api/config")', HELPER, "/api/config route")
    text = replace_once(
        text,
        "def generate(request: GenerateRequest):",
        "def generate(request: GenerateRequest, user: dict = Depends(require_user)):",
        "generate()",
    )
    text = replace_once(
        text,
        "voice: UploadFile | None = File(None),",
        "voice: UploadFile | None = File(None),\n    user: dict = Depends(require_user),",
        "generate_i2v()",
    )
    text = replace_once(text, "initAuth();", FETCH_WRAPPER, "initAuth();")

    backup = TARGET.with_suffix(".py.bak2")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(text, encoding="utf-8")

    print("Done. Backup saved as:", backup)
    print("The server now requires a valid login to generate videos.")


main()
