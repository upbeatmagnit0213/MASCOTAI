"""
Adds Supabase Login / Sign Up / Logout to C:\\MascotAI\\app\\server.py

Run once from C:\\MascotAI:
    .venv\\Scripts\\python.exe patch_auth.py

It makes a backup (server.py.bak) first and changes nothing else.
"""

from pathlib import Path
import shutil
import sys

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\MascotAI\app\server.py")

CSS = r'''
.authbar{display:flex;justify-content:flex-end;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:18px;color:var(--muted);font-size:14px}
.authbar .btn{padding:9px 14px}
.authcard{display:none;max-width:420px;margin:0 auto 22px;padding:22px;border:1px solid var(--border);border-radius:20px;background:rgba(22,22,34,.96)}
.authcard.show{display:block}
.authcard input{width:100%;padding:13px;margin-top:10px;border-radius:12px;border:1px solid #424254;background:#13131d;color:#fff;font-size:15px}
.authrow{display:flex;gap:10px;margin-top:14px}
.authrow .btn{flex:1}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#09090d}
.btn-ghost{background:#242431;color:#fff}
</style>'''

HTML = r'''<div class="container">

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
'''

JS = r'''<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
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

initAuth();

</script>

</body>'''

CONFIG_ROUTE = r'''@app.get("/api/config")
def public_config():
    # Publishable key is designed to be public (used in the browser).
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_key": SUPABASE_PUBLISHABLE_KEY,
    }


@app.get("/health")'''


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        sys.exit(f"STOP: expected 1 match for {label}, found {count}. Nothing was changed.")
    return text.replace(old, new)


def main():
    if not TARGET.exists():
        sys.exit(f"STOP: file not found: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    if "authBar" in text:
        sys.exit("Already patched. Nothing to do.")

    text = replace_once(text, "</style>", CSS, "</style>")
    text = replace_once(text, '<div class="container">', HTML, '<div class="container">')
    text = replace_once(text, "</body>", JS, "</body>")
    text = replace_once(text, '@app.get("/health")', CONFIG_ROUTE, '@app.get("/health")')

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(text, encoding="utf-8")

    print("Done. Backup saved as:", backup)
    print("Login / Sign up / Logout added to:", TARGET)


main()
