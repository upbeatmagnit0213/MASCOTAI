"""
Fix for: NameError: name 'require_user' is not defined

Moves the login-check helper to the top of server.py, above the routes that use it.

Run once from C:\\MascotAI:
    .venv\\Scripts\\python.exe patch_auth3.py
"""

from pathlib import Path
import shutil
import sys

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\MascotAI\app\server.py")

START = "import urllib.request\nimport urllib.error\n"
END = '@app.get("/api/config")'
ANCHOR = "class GenerateRequest(BaseModel):"


def main():
    if not TARGET.exists():
        sys.exit(f"STOP: file not found: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    if text.count(START) != 1 or text.count(END) != 1 or text.count(ANCHOR) != 1:
        sys.exit("STOP: could not find the expected places. Nothing was changed.")

    if text.index("def require_user") < text.index(ANCHOR):
        sys.exit("Already fixed. Nothing to do.")

    start = text.index(START)
    end = text.index(END)

    if start > end:
        sys.exit("STOP: unexpected order. Nothing was changed.")

    block = text[start:end]

    if "def require_user" not in block:
        sys.exit("STOP: helper not found. Nothing was changed.")

    text = text[:start] + text[end:]

    anchor_pos = text.index(ANCHOR)

    text = text[:anchor_pos] + block + "\n\n" + text[anchor_pos:]

    backup = TARGET.with_suffix(".py.bak3")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(text, encoding="utf-8")

    print("Done. Backup saved as:", backup)
    print("require_user now sits above the routes.")


main()
