import streamlit as st
st.set_page_config(page_title="DeepGuard AI", page_icon="[DG]", layout="wide", initial_sidebar_state="expanded")

import hashlib, os, tempfile, time
import streamlit.components.v1 as components
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from auth import (init_db, authenticate, register_user, get_all_users,
                  toggle_user_status, change_user_role, delete_user,
                  save_analysis, get_user_analyses, get_all_analyses, get_stats)
from predict import predict as predict_current
from predict_dfd import predict as predict_dfd
from report import generate_report
from forensics import compute_phash, extract_exif

init_db()

for k, v in {"user": None, "page": "analyze", "last_result": None,
             "last_file": None, "last_sha": None, "auth_tab": "login",
             "model_profile": "DFD High-Accuracy (Recommended)"}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def human_size(n):
    for u in ("B","KB","MB","GB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def device_label():
    try:
        import torch
        return f"CUDA / {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    except: return "CPU"

def verdict_info(result):
    s = result["score"]
    if 0.45 <= s <= 0.55: return "#f5a623", "INCONCLUSIVE", "#1a1200"
    if result["verdict"] == "FAKE": return "#ff3b3b", "SYNTHETIC MEDIA DETECTED", "#1a0000"
    return "#00e676", "AUTHENTIC — NO MANIPULATION DETECTED", "#001a0a"

def bgr2rgb(arr): return arr[..., ::-1]

def verdict_explanation(result):
    s   = result["score"]
    n   = result.get("meta", {}).get("n_faces_detected", 0)
    pfs = result.get("meta", {}).get("per_face_scores", [])
    suspicious = [i for i, sc in enumerate(pfs) if sc >= 0.5]
    lines = []
    if result["verdict"] == "FAKE":
        lines.append(f"Model confidence: {result['confidence']*100:.1f}% that this media is synthetic or manipulated.")
        if suspicious:
            lines.append(f"Face(s) {', '.join(str(i+1) for i in suspicious)} showed significant frequency-domain and boundary inconsistencies.")
        lines.append("GAN-generated textures or facial boundary artifacts were detected in the Xception neural pass.")
        if s > 0.85: lines.append("High-confidence indicator: pixel-level entropy deviations suggest generative model origin.")
    elif result["verdict"] == "REAL":
        lines.append(f"Model confidence: {result['confidence']*100:.1f}% that this media is authentic.")
        lines.append("No significant GAN-fingerprint, facial boundary inconsistencies, or frequency-domain artifacts detected.")
        if n > 0: lines.append(f"{n} face(s) detected and individually verified — none flagged.")
    else:
        lines.append("Score falls in the inconclusive range (45-55%). The model cannot make a high-confidence determination.")
        lines.append("Recommend additional forensic analysis: ELA, noise analysis, or manual expert review.")
    return lines

_PARTICLE_JS = ""

def inject_canvas():
    components.html('''
    <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -999; background: #080c0a; overflow: hidden;">
        <canvas id="c"></canvas>
    </div>
    <script>
    const c = document.getElementById('c');
    const ctx = c.getContext('2d');
    let w = c.width = window.innerWidth;
    let h = c.height = window.innerHeight;
    
    let scanY = -100;
    
    function draw() {
        // Fade out previous frames
        ctx.fillStyle = 'rgba(8,12,10,0.15)';
        ctx.fillRect(0,0,w,h);
        
        // Draw background grid
        ctx.strokeStyle = "rgba(0, 230, 118, 0.015)";
        ctx.lineWidth = 1;
        let t = (Date.now() / 100) % 40;
        ctx.beginPath();
        for(let x = t; x < w; x+=40) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
        for(let y = t; y < h; y+=40) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
        ctx.stroke();

        // Draw scanning laser
        scanY += 3.5;
        if (scanY > h + 150) scanY = -100;
        
        // Laser glow effect
        const grad = ctx.createLinearGradient(0, scanY - 40, 0, scanY + 2);
        grad.addColorStop(0, 'rgba(0, 230, 118, 0)');
        grad.addColorStop(0.8, 'rgba(0, 230, 118, 0.03)');
        grad.addColorStop(1, 'rgba(0, 230, 118, 0.3)');
        
        ctx.fillStyle = grad;
        ctx.fillRect(0, scanY - 40, w, 42);
        
        // Bright laser line
        ctx.fillStyle = 'rgba(0, 255, 150, 0.4)';
        ctx.fillRect(0, scanY + 1, w, 1);

        requestAnimationFrame(draw);
    }
    draw();
    
    window.addEventListener('resize', () => {
        w = c.width = window.innerWidth;
        h = c.height = window.innerHeight;
    });
    </script>
    ''', height=0, width=0)
    # The components.html injects an iframe. We'll use CSS to force the iframe to fill the screen in the background.
    st.markdown('''<style>
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: -9999 !important;
        border: none !important;
    }
    /* Hide the container wrapper that Streamlit creates for components */
    [data-testid="stHtml"] {
        height: 0px !important;
        width: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>''', unsafe_allow_html=True)


# ── Global CSS ─────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600;700&display=swap');

/* Base Font */
p, h1, h2, h3, h4, h5, h6, li, label, .stTextInput input, .stSelectbox div {
    font-family: 'Exo 2', sans-serif !important;
}

/* Ensure Streamlit Icons work */
.material-symbols-rounded, .material-icons, .stIcon, [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

/* Code Font */
code, pre, .stCode, [data-testid="stCode"], [data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
}

/* Ensure sidebar toggle is visible always */
[data-testid="collapsedControl"] {
    display: flex !important;
    z-index: 100000 !important;
    color: #00e676 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

.stApp {
    background-color: transparent !important;
}

/* Remove default top padding to push content to the very top */
.stApp .main .block-container {
    padding-top: 2rem !important;
}

@keyframes spin { to { transform: rotate(360deg); } }

[data-testid="stMetricLabel"] { color: #00e67655 !important; font-size: 10px !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; }

.stInfo { background: rgba(0,230,118,.03) !important; border: 1px solid #00e67628 !important; color: #a0c8a8 !important; border-radius: 3px !important; }
.stSuccess { background: rgba(0,230,118,.04) !important; border: 1px solid #00e67630 !important; color: #00e676 !important; border-radius: 3px !important; }
.stError { background: rgba(255,59,59,.03) !important; border: 1px solid rgba(255,59,59,.22) !important; color: #ff8080 !important; border-radius: 3px !important; }
.stWarning { background: rgba(245,166,35,.03) !important; border: 1px solid rgba(245,166,35,.22) !important; color: #f5a623 !important; border-radius: 3px !important; }

.stSpinner>div { border-top-color: #00e676 !important; }

hr { border-color: #00e67614 !important; }
.stCaption { color: #00e67655 !important; }
[data-testid="stImage"] img { border: 1px solid #00e67618; border-radius: 3px; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.anim-in { animation: fadeIn .4s ease both; }
</style>
"""


# ── AUTH PAGE ─────────────────────────────────────────────────────────────────
def show_auth():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    inject_canvas()
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("""
        <div style="margin-top:60px;text-align:center;">
          <div style="font-size:11px;color:rgba(0,230,118,.35);letter-spacing:3px;margin-bottom:8px;font-family:'Share Tech Mono',monospace;">FORENSIC ANALYSIS SYSTEM v3.0</div>
          <div style="font-size:42px;font-weight:700;color:#d0e8c0;letter-spacing:4px;">DEEPGUARD</div>
          <div style="width:50%;height:1px;background:linear-gradient(90deg,transparent,#00e67640,transparent);margin:12px auto;"></div>
          <div style="font-size:12px;color:rgba(0,230,118,.45);letter-spacing:2px;margin-bottom:32px;font-family:'Share Tech Mono',monospace;">SYNTHETIC MEDIA DETECTION ENGINE</div>
        </div>
        
        </div>
        <style>@keyframes bgScan {0%{transform:translateY(-100%);} 100%{transform:translateY(100%);}}</style>
        <div style="background:rgba(8,12,10,.88);border:1px solid rgba(0,230,118,.2);border-radius:4px;padding:26px 28px;backdrop-filter:blur(16px);">
        """, unsafe_allow_html=True)
        tab_login, tab_reg = st.tabs(["AUTHENTICATE", "REGISTER"])
        with tab_login:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            uname = st.text_input("Username", key="li_user", placeholder="enter username")
            pwd   = st.text_input("Password", type="password", key="li_pwd", placeholder="password")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            login_placeholder = st.empty()
            if login_placeholder.button("Sign In", type="primary", use_container_width=True, key="btn_login"):
                if not uname or not pwd:
                    st.error("Username and password required.")
                else:
                    login_placeholder.markdown('''
                    <div style="display:flex;justify-content:center;align-items:center;background:#00e676;color:#000;padding:10px;border-radius:4px;font-weight:600;font-size:16px;">
                        AUTHENTICATING... <div style="margin-left:12px;width:18px;height:18px;border:3px solid rgba(0,0,0,0.3);border-top-color:#000;border-radius:50%;animation:spin 0.8s linear infinite;"></div>
                    </div>
                    ''', unsafe_allow_html=True)
                    time.sleep(1.0)
                    user = authenticate(uname.strip(), pwd)
                    if user:
                        st.session_state.user = user
                        st.session_state.page = "analyze"
                        st.rerun()
                    else:
                        st.error("Invalid credentials or account disabled.")
            st.markdown('<div style="text-align:center;margin-top:12px;font-size:11px;color:rgba(0,230,118,.25);font-family:Share Tech Mono,monospace;">default: admin / Admin@123</div>', unsafe_allow_html=True)
        with tab_reg:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            r_user  = st.text_input("Username",         key="rg_user",  placeholder="choose a username")
            r_email = st.text_input("Email",            key="rg_email", placeholder="your@email.com")
            r_pwd   = st.text_input("Password",         type="password", key="rg_pwd",  placeholder="min 8 chars, uppercase, digit")
            r_pwd2  = st.text_input("Confirm Password", type="password", key="rg_pwd2", placeholder="repeat password")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True, key="btn_reg"):
                if not all([r_user, r_email, r_pwd, r_pwd2]):
                    st.error("All fields are required.")
                elif r_pwd != r_pwd2:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("CREATING ACCOUNT..."):
                        time.sleep(0.8)
                        ok, msg = register_user(r_user, r_email, r_pwd)
                        (st.success if ok else st.error)(msg)
        st.markdown("</div><div style='height:40px'></div>", unsafe_allow_html=True)

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
def show_sidebar():
    u = st.session_state.user
    with st.sidebar:
        st.markdown("""
        <div style="padding:18px 8px 6px;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#d0e8c0;letter-spacing:3px;">DEEPGUARD</div>
          <div style="font-size:10px;color:rgba(0,230,118,.35);letter-spacing:2px;margin-top:2px;font-family:'Share Tech Mono',monospace;">FORENSIC ENGINE v3.0</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#00e67614!important;margin:6px 0;">', unsafe_allow_html=True)
        nav_items = [("ANALYZE",   "analyze"),
                     ("DASHBOARD", "dashboard"),
                     ("HISTORY",   "history")]
        if u["role"] == "admin":
            nav_items.append(("ADMIN", "admin"))
        for label, key in nav_items:
            active = st.session_state.page == key
            prefix = "> " if active else "  "
            if st.button(f"{prefix}{label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
        st.markdown('<hr style="border-color:#00e67614!important;margin:6px 0;">', unsafe_allow_html=True)
        st.selectbox(
            "VIDEO MODEL PROFILE",
            ["DFD High-Accuracy (Recommended)", "Current / Rollback"],
            key="model_profile",
            help="DFD profile passed the locked actor-disjoint test. Current restores the previous pipeline instantly.",
        )
        rc = "#f5a623" if u["role"] == "admin" else "#00e676"
        st.markdown(f"""
        <div style="background:rgba(0,230,118,.02);border:1px solid #00e67618;border-radius:3px;padding:10px 12px;font-size:12px;">
          <div style="color:#d0e8c0;font-weight:600;">{u['username']}</div>
          <div style="color:{rc};font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-top:2px;">{u['role']}</div>
          <div style="color:rgba(0,230,118,.35);font-size:11px;margin-top:2px;">{u['email']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True, key="btn_logout"):
            st.session_state.user = None
            st.session_state.last_result = None
            st.rerun()

# ── SECTION HEADER helper ─────────────────────────────────────────────────────
def sec_header(title, subtitle=None):
    sub = f'<div style="font-size:12px;color:rgba(0,230,118,.4);letter-spacing:2px;font-family:\'Share Tech Mono\',monospace;margin-bottom:4px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:20px 0 12px;">
      {sub}
      <div style="font-size:18px;font-weight:700;color:#d0e8c0;letter-spacing:1px;border-bottom:1px solid #00e67618;padding-bottom:8px;">{title}</div>
    </div>""", unsafe_allow_html=True)

def row_divider():
    st.markdown('<hr style="border-color:#00e67614!important;margin:20px 0 16px;">', unsafe_allow_html=True)

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def show_dashboard():
    sec_header("Dashboard", "SYSTEM OVERVIEW")
    stats = get_stats()
    c1,c2,c3,c4 = st.columns(4)
    def stat_card(col, label, value, color):
        col.markdown(f"""
        <div style="background:#0d1410;border:1px solid {color}28;border-radius:4px;padding:18px 16px;text-align:center;">
          <div style="font-size:28px;font-weight:700;color:{color};font-family:'Share Tech Mono',monospace;">{value}</div>
          <div style="font-size:11px;color:rgba(0,230,118,.4);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px;">{label}</div>
        </div>""", unsafe_allow_html=True)
    stat_card(c1, "Total Analyses", stats["total_analyses"], "#00e676")
    stat_card(c2, "Fakes Detected",  stats["total_fakes"],   "#ff3b3b")
    stat_card(c3, "Verified Real",   stats["total_real"],    "#00e676")
    stat_card(c4, "Detection Rate",  f"{stats['detection_rate']:.1f}%", "#f5a623")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    sec_header("Recent Analyses")
    uid = st.session_state.user["id"]
    rows = get_user_analyses(uid, limit=10)
    if not rows:
        st.info("No analyses yet. Go to Analyze to get started.")
        return
    st.markdown("""
    <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr 1fr;gap:8px;padding:8px 14px;
      background:#0d1410;border-bottom:1px solid #00e67622;font-size:10px;color:rgba(0,230,118,.38);
      text-transform:uppercase;letter-spacing:1px;font-family:'Share Tech Mono',monospace;">
      <span>Filename</span><span>Type</span><span>Verdict</span><span>Score</span><span>Confidence</span><span>Date</span>
    </div>""", unsafe_allow_html=True)
    for i, r in enumerate(rows):
        vc  = "#ff3b3b" if r["verdict"]=="FAKE" else "#00e676"
        bg  = "#0d1410" if i%2==0 else "#0a0f0c"
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr 1fr;gap:8px;padding:10px 14px;
          background:{bg};border-bottom:1px solid #00e67610;align-items:center;font-size:12px;">
          <span style="color:#c0d8b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{r['filename']}</span>
          <span style="color:rgba(0,230,118,.45);text-transform:uppercase;font-size:10px;">{r['file_type']}</span>
          <span style="color:{vc};font-weight:700;font-size:11px;">{r['verdict']}</span>
          <span style="color:#c0d8b8;">{r['score']*100:.0f}%</span>
          <span style="color:#c0d8b8;">{r['confidence']*100:.0f}%</span>
          <span style="color:rgba(0,230,118,.3);font-size:10px;">{r['analyzed_at'][:16].replace('T',' ')}</span>
        </div>""", unsafe_allow_html=True)

# ── VERDICT CARD ──────────────────────────────────────────────────────────────
def render_verdict(result):
    color, label, bg = verdict_info(result)
    score = result["score"] * 100
    conf  = result["confidence"] * 100
    lines = verdict_explanation(result)
    st.markdown(f"""
    <div class="anim-in" style="border:1px solid {color}60;border-radius:4px;padding:24px 28px;
      margin:16px 0;background:{bg};box-shadow:0 0 40px {color}14;">
      <div style="font-size:10px;color:{color}88;letter-spacing:3px;margin-bottom:6px;font-family:'Share Tech Mono',monospace;">ANALYSIS VERDICT</div>
      <div style="font-size:28px;font-weight:700;color:{color};margin-bottom:18px;">{label}</div>
      <div style="display:flex;gap:32px;margin-bottom:20px;">
        <div>
          <div style="font-size:30px;font-weight:700;color:{color};font-family:'Share Tech Mono',monospace;">{score:.1f}%</div>
          <div style="font-size:10px;color:rgba(0,230,118,.4);letter-spacing:1.5px;text-transform:uppercase;margin-top:2px;">Fake Probability</div>
        </div>
        <div style="width:1px;background:#00e67618;"></div>
        <div>
          <div style="font-size:30px;font-weight:700;color:{color};font-family:'Share Tech Mono',monospace;">{conf:.1f}%</div>
          <div style="font-size:10px;color:rgba(0,230,118,.4);letter-spacing:1.5px;text-transform:uppercase;margin-top:2px;">Model Confidence</div>
        </div>
        <div style="width:1px;background:#00e67618;"></div>
        <div>
          <div style="font-size:30px;font-weight:700;color:rgba(0,230,118,.6);font-family:'Share Tech Mono',monospace;">{result['elapsed_ms']:.0f} ms</div>
          <div style="font-size:10px;color:rgba(0,230,118,.4);letter-spacing:1.5px;text-transform:uppercase;margin-top:2px;">Inference Time</div>
        </div>
      </div>
      <div style="border-top:1px solid {color}25;padding-top:14px;">
        <div style="font-size:10px;color:{color}88;letter-spacing:2px;margin-bottom:8px;font-family:'Share Tech Mono',monospace;">ANALYSIS EXPLANATION</div>
        {''.join(f'<div style="font-size:13px;color:#a0b8a0;line-height:1.7;margin-bottom:4px;">&rsaquo; {l}</div>' for l in lines)}
      </div>
    </div>""", unsafe_allow_html=True)
    st.progress(float(result["confidence"]))

# ── TIMELINE CHART ────────────────────────────────────────────────────────────
def render_timeline(result):
    sec_header("Per-Frame Analysis Timeline", "VIDEO FORENSICS")
    xs = np.array(result["frame_indices"])
    ys = np.array(result["per_frame"])
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#080c0a")
    ax.set_facecolor("#0d1410")
    ax.fill_between(xs, ys, .5, where=(ys>.5),  interpolate=True, color="#ff3b3b", alpha=.25)
    ax.fill_between(xs, ys, .5, where=(ys<=.5), interpolate=True, color="#00e676", alpha=.18)
    ax.plot(xs, ys, color="#00e676", lw=1.8, zorder=3)
    ax.scatter(xs, ys, s=16, color="#00bcd4", zorder=4, linewidths=0)
    ax.axhline(.5, color="#f5a623", lw=1.2, ls="--", alpha=.7)
    ax.text(xs[-1], .53, "  threshold", color="#f5a623", fontsize=8, ha="right", family="monospace")
    if result.get("top_frames"):
        for tf in result["top_frames"]:
            ax.axvline(tf["frame_index"], color="#ff3b3b", lw=.9, alpha=.35, ls=":")
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(-.05, 1.08)
    ax.set_xlabel("Frame Index", color="#00e67640", fontsize=8, family="monospace")
    ax.set_ylabel("Fake Probability", color="#00e67640", fontsize=8, family="monospace")
    ax.tick_params(colors="#00e67640", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#00e67618")
    fp = mpatches.Patch(color="#ff3b3b", alpha=.6, label="Suspicious")
    rp = mpatches.Patch(color="#00e676", alpha=.6, label="Clean")
    ax.legend(handles=[fp, rp], framealpha=.1, labelcolor="#a0b8a0",
              fontsize=8, facecolor="#0d1410", edgecolor="#00e67618", loc="upper right",
              prop={"family": "monospace"})
    ax.set_title(f"Sampled @ {result['fps_sampled']:.1f} fps  /  {len(xs)} frames  /  median fake probability: {result['score']*100:.1f}%",
                 color="#00e67640", fontsize=8, pad=6, family="monospace")
    fig.tight_layout(pad=1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── SUSPICIOUS TIMESTAMPS (VIDEO) ─────────────────────────────────────────────
def render_suspicious_timestamps(result):
    frames = result.get("top_frames", [])
    if not frames: return
    fps = result.get("fps_sampled", 1) or 1
    sec_header("Suspicious Frames & Timestamps", "WHERE MANIPULATION IS DETECTED")
    n = len(frames)
    cols = st.columns(min(n, 4))
    for idx, (col, tf) in enumerate(zip(cols, frames[:4])):
        with col:
            # NumPy image arrays cannot be combined with boolean or; select by presence.
            img = tf.get("heatmap_bgr")
            if img is None:
                img = tf.get("face_bgr")
            if img is not None: st.image(bgr2rgb(img), use_container_width=True)
            ts_sec = tf["frame_index"] / fps
            vc = "#ff3b3b" if tf["score"] >= .5 else "#00e676"
            st.markdown(f"""
            <div style="text-align:center;margin-top:6px;font-family:'Share Tech Mono',monospace;font-size:12px;">
              <div style="color:{vc};font-weight:700;">{tf['score']*100:.0f}% fake</div>
              <div style="color:rgba(0,230,118,.4);font-size:10px;margin-top:2px;">Frame {tf['frame_index']}  /  {ts_sec:.1f}s</div>
            </div>""", unsafe_allow_html=True)

# ── GRAD-CAM + FACE PANELS ────────────────────────────────────────────────────
def render_image_panels(result):
    sec_header("Grad-CAM Heatmap — Suspicious Regions", "WHY THE MODEL SUSPECTS MANIPULATION")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p style="font-size:11px;color:rgba(0,230,118,.4);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Primary Face (extracted)</p>', unsafe_allow_html=True)
        face = result.get("face_crop_bgr")
        if face is not None: st.image(bgr2rgb(face), use_container_width=True, caption="224x224 face crop")
        else: st.warning("No face detected in the image.")
    with c2:
        st.markdown('<p style="font-size:11px;color:rgba(0,230,118,.4);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Grad-CAM Attention Heatmap</p>', unsafe_allow_html=True)
        hmap = result.get("heatmap_bgr")
        if hmap is not None: st.image(bgr2rgb(hmap), use_container_width=True, caption="Red = regions model suspects most")
        else: st.info("Heatmap unavailable for this analysis.")

    all_faces = result.get("all_faces") or []
    if len(all_faces) >= 2:
        row_divider()
        sec_header(f"Multiple Faces Detected ({len(all_faces)})", "PER-FACE ANALYSIS")
        st.caption("Each face is scored independently. The highest-scoring face drives the overall verdict.")
        cols_per_row = 4
        for row_start in range(0, len(all_faces), cols_per_row):
            row  = all_faces[row_start:row_start + cols_per_row]
            cols = st.columns(len(row))
            for col, f in zip(cols, row):
                with col:
                    fs     = float(f.get("score", 0))
                    bl     = float(f.get("blur",  0))
                    is_pri = f.get("is_primary", False)
                    color  = "#ff3b3b" if fs >= 0.5 else "#00e676"
                    bl_lbl = "blurry" if bl < 30 else ("soft" if bl < 60 else "sharp")
                    primary_tag = " (primary)" if is_pri else ""
                    sub1, sub2 = st.columns(2)
                    crop = f.get("crop_bgr"); heat = f.get("heatmap_bgr")
                    if crop is not None: sub1.image(bgr2rgb(crop), use_container_width=True)
                    if heat is not None: sub2.image(bgr2rgb(heat), use_container_width=True)
                    st.markdown(f"""
                    <div style="text-align:center;font-size:12px;margin-top:4px;">
                      <span style="color:{color};font-weight:700;">{fs*100:.0f}% fake</span>
                      <span style="color:rgba(0,230,118,.4);font-size:10px;"> / {bl_lbl}{primary_tag}</span>
                    </div>""", unsafe_allow_html=True)

# ── EXIF / METADATA PANEL ─────────────────────────────────────────────────────
def render_exif(tmp_path, is_video):
    try:
        exif = extract_exif(tmp_path)
    except Exception as e:
        st.warning(f"Metadata extraction failed: {e}")
        return
    if not exif:
        st.info("No metadata found in this file.")
        return
    row_divider()
    sec_header("File Metadata & EXIF", "DIGITAL EVIDENCE AUTHENTICATION")
    items = list(exif.items())
    half  = (len(items)+1)//2
    c1, c2 = st.columns(2)
    for col, chunk in [(c1, items[:half]), (c2, items[half:])]:
        with col:
            for k, v in chunk:
                if v is None or str(v).strip() == "": continue
                st.markdown(f"""
                <div style="display:flex;gap:10px;padding:7px 0;border-bottom:1px solid #00e67610;font-size:12px;">
                  <span style="color:rgba(0,230,118,.45);min-width:160px;font-family:'Share Tech Mono',monospace;font-size:11px;">{k}</span>
                  <span style="color:#c0d8b8;">{v}</span>
                </div>""", unsafe_allow_html=True)

# ── SCAN OVERLAY (SHARED) ───────────────────────────────────────────────────
def render_scan_overlay_image(tmp_path, filename, is_video=False):
    if is_video:
        st.video(tmp_path)
    else:
        st.image(tmp_path, use_container_width=True, caption=filename)

# ── ANALYZE PAGE ──────────────────────────────────────────────────────────────
def show_analyze():
    # Removed static base64 background
    pass
    
    uploaded_files = st.file_uploader(
        "Upload evidence files",
        type=["jpg","jpeg","png","webp","mp4","mov","avi","mkv"],
        accept_multiple_files=True,
        help="Click Browse files or drag and drop. Max 200 MB.",
    )
    
    if not uploaded_files:
        st.markdown('<div style="text-align:center;padding:18px;color:rgba(0,230,118,.22);font-family:Share Tech Mono,monospace;font-size:11px;letter-spacing:1.5px;">Awaiting file(s) — use Browse Files or drag and drop</div>', unsafe_allow_html=True)
        return

    for uploaded in uploaded_files:
        st.markdown(f"### File: {uploaded.name}")
        ext      = os.path.splitext(uploaded.name)[1].lower()
        is_video = ext in {".mp4",".mov",".avi",".mkv"}
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        sha = sha256_file(tmp_path)
        try:    phash = compute_phash(tmp_path)
        except: phash = "unavailable"

        # ── Preview with scan overlay (image) or plain video player ──────────────
        sec_header(f"Evidence Preview: {uploaded.name}")
        placeholder = st.empty()
        with placeholder.container():
            render_scan_overlay_image(tmp_path, uploaded.name, is_video=is_video)
        with st.spinner(f"Running forensic neural analysis on {uploaded.name}..."):
            predictor = predict_dfd if st.session_state.model_profile.startswith("DFD") else predict_current
        placeholder.empty()
        # Show clean after analysis
        if is_video:
            st.video(tmp_path)
        else:
            st.image(tmp_path, use_container_width=True, caption=uploaded.name)

        save_analysis(st.session_state.user["id"], uploaded.name,
                      "video" if is_video else "image", result, sha)
        st.session_state.last_result = result
        st.session_state.last_sha    = sha

        # ── Verdict ───────────────────────────────────────────────────────────────
        render_verdict(result)

        # ── Visual forensics ──────────────────────────────────────────────────────
        if result["kind"] == "video":
            render_timeline(result)
            render_suspicious_timestamps(result)
        else:
            row_divider()
            render_image_panels(result)

        # ── EXIF / Metadata ───────────────────────────────────────────────────────
        render_exif(tmp_path, is_video)

        # ── File integrity hashes ─────────────────────────────────────────────────
        row_divider()
        sec_header("File Integrity — Chain of Custody", "EVIDENCE AUTHENTICATION")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown('<div style="font-size:10px;color:rgba(0,230,118,.4);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;font-family:Share Tech Mono,monospace;">SHA-256 Hash</div>', unsafe_allow_html=True)
            st.code(sha, language=None)
        with fc2:
            st.markdown('<div style="font-size:10px;color:rgba(0,230,118,.4);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;font-family:Share Tech Mono,monospace;">Perceptual Hash</div>', unsafe_allow_html=True)
            st.code(phash, language=None)

        # ── Report download ───────────────────────────────────────────────────────
        row_divider()
        sec_header("Forensic Report")
        col_dl, col_info = st.columns([2, 3])
        with col_dl:
            if st.button(f"Generate PDF Report for {uploaded.name}", type="primary", use_container_width=True, key=f"btn_report_{sha}"):
                with st.spinner("Generating forensic report..."):
                    pdf = generate_report(input_path=tmp_path, verdict=result, sha256=sha, phash=phash)
                st.download_button("Download PDF", data=pdf,
                                   file_name=f"{uploaded.name}.report.pdf",
                                   mime="application/pdf", use_container_width=True, key=f"btn_dl_{sha}")
        with col_info:
            color, label, _ = verdict_info(result)
            st.markdown(f"""
            <div style="font-size:12px;color:#80a080;line-height:2.1;padding-top:6px;">
              Verdict: <span style="color:{color};font-weight:700;">{label}</span><br>
              File: <span style="color:#c0d8b8;">{uploaded.name}</span> ({human_size(os.path.getsize(tmp_path))})<br>
              Inference: <span style="color:#c0d8b8;">{result['elapsed_ms']:.0f} ms</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br><hr style='border: 2px dashed #00e67650'><br>", unsafe_allow_html=True)

# ── HISTORY PAGE ──────────────────────────────────────────────────────────────
def show_history():
    sec_header("Analysis History", "PAST ANALYSES")
    uid  = st.session_state.user["id"]
    rows = get_user_analyses(uid, limit=50)
    if not rows:
        st.info("No analyses yet. Go to Analyze to get started.")
        return
    filter_col, _ = st.columns([1, 3])
    with filter_col:
        f = st.selectbox("Filter by verdict", ["All", "FAKE", "REAL"], key="hist_filter")
    filtered = rows if f == "All" else [r for r in rows if r["verdict"] == f]
    st.markdown("""
    <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr 1fr;gap:8px;padding:8px 14px;
      background:#0d1410;border-bottom:1px solid #00e67622;font-size:10px;color:rgba(0,230,118,.38);
      text-transform:uppercase;letter-spacing:1px;font-family:'Share Tech Mono',monospace;margin-top:12px;">
      <span>Filename</span><span>Type</span><span>Verdict</span><span>Score</span><span>Confidence</span><span>Date</span>
    </div>""", unsafe_allow_html=True)
    for i, r in enumerate(filtered):
        vc = "#ff3b3b" if r["verdict"]=="FAKE" else "#00e676"
        bg = "#0d1410" if i%2==0 else "#0a0f0c"
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr 1fr;gap:8px;padding:10px 14px;
          background:{bg};border-bottom:1px solid #00e67610;align-items:center;font-size:12px;">
          <span style="color:#c0d8b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{r['filename']}">{r['filename']}</span>
          <span style="color:rgba(0,230,118,.45);text-transform:uppercase;font-size:10px;">{r['file_type']}</span>
          <span style="color:{vc};font-weight:700;font-size:11px;">{r['verdict']}</span>
          <span style="color:#c0d8b8;">{r['score']*100:.0f}%</span>
          <span style="color:#c0d8b8;">{r['confidence']*100:.0f}%</span>
          <span style="color:rgba(0,230,118,.3);font-size:10px;">{r['analyzed_at'][:16].replace('T',' ')}</span>
        </div>""", unsafe_allow_html=True)

# ── ADMIN PAGE ────────────────────────────────────────────────────────────────
def show_admin():
    if st.session_state.user["role"] != "admin":
        st.error("Access denied. Admin role required.")
        return
    sec_header("Admin Panel", "SYSTEM ADMINISTRATION")
    stats = get_stats()
    c1,c2,c3,c4 = st.columns(4)
    def s(col, label, val, color):
        col.markdown(f"""
        <div style="background:#0d1410;border:1px solid {color}28;border-radius:4px;padding:16px;text-align:center;">
          <div style="font-size:24px;font-weight:700;color:{color};font-family:'Share Tech Mono',monospace;">{val}</div>
          <div style="font-size:10px;color:rgba(0,230,118,.4);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px;">{label}</div>
        </div>""", unsafe_allow_html=True)
    s(c1, "Active Users",   stats["total_users"],    "#00e676")
    s(c2, "Total Analyses", stats["total_analyses"], "#00bcd4")
    s(c3, "Fakes Found",    stats["total_fakes"],    "#ff3b3b")
    s(c4, "Detection Rate", f"{stats['detection_rate']:.1f}%", "#f5a623")

    row_divider()
    sec_header("User Management")
    users = get_all_users()
    for u in users:
        is_admin_acc = u["username"] == "admin"
        sc = "#00e676" if u["is_active"] else "#ff3b3b"
        uc = "#f5a623" if u["role"]=="admin" else "#00e676"
        with st.container():
            col_info, col_role, col_act = st.columns([4, 2, 2])
            with col_info:
                st.markdown(f"""
                <div style="background:#0d1410;border:1px solid #00e67618;border-radius:4px;padding:12px 14px;margin:3px 0;">
                  <span style="font-weight:700;color:#d0e8c0;">{u['username']}</span>
                  <span style="font-size:10px;color:{uc};margin-left:8px;text-transform:uppercase;
                    letter-spacing:1px;background:{uc}14;padding:2px 8px;border-radius:3px;">{u['role']}</span>
                  <span style="font-size:11px;color:{sc};margin-left:8px;">{'Active' if u['is_active'] else 'Inactive'}</span><br>
                  <span style="font-size:12px;color:rgba(0,230,118,.4);">{u['email']}</span>
                  <span style="font-size:11px;color:rgba(0,230,118,.25);margin-left:12px;">
                    Last login: {u['last_login'][:16].replace('T',' ') if u['last_login'] else 'Never'}</span>
                </div>""", unsafe_allow_html=True)
            with col_role:
                if not is_admin_acc:
                    new_role = st.selectbox("Role", ["analyst","admin"],
                                            index=0 if u["role"]=="analyst" else 1,
                                            key=f"role_{u['id']}")
                    if new_role != u["role"]:
                        change_user_role(u["id"], new_role); st.rerun()
            with col_act:
                if not is_admin_acc:
                    lbl = "Disable" if u["is_active"] else "Enable"
                    if st.button(lbl, key=f"tog_{u['id']}", use_container_width=True):
                        toggle_user_status(u["id"]); st.rerun()

    row_divider()
    sec_header("All Analyses Log")
    all_rows = get_all_analyses(limit=30)
    for i, r in enumerate(all_rows):
        vc = "#ff3b3b" if r["verdict"]=="FAKE" else "#00e676"
        bg = "#0d1410" if i%2==0 else "#0a0f0c"
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 2fr 1fr 1fr 1fr;gap:8px;padding:9px 14px;
          background:{bg};border-bottom:1px solid #00e67610;font-size:12px;">
          <span style="color:#80b0a0;">{r['username']}</span>
          <span style="color:#c0d8b8;overflow:hidden;text-overflow:ellipsis;" title="{r['filename']}">{r['filename']}</span>
          <span style="color:{vc};font-weight:700;font-size:11px;">{r['verdict']}</span>
          <span style="color:#c0d8b8;">{r['score']*100:.0f}%</span>
          <span style="color:rgba(0,230,118,.3);font-size:10px;">{r['analyzed_at'][:16].replace('T',' ')}</span>
        </div>""", unsafe_allow_html=True)

# ── MAIN ENTRY ────────────────────────────────────────────────────────────────
def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    if not st.session_state.user:
        show_auth()
        return
    inject_canvas()
    show_sidebar()
    page = st.session_state.page
    if   page == "dashboard": show_dashboard()
    elif page == "analyze":   show_analyze()
    elif page == "history":   show_history()
    elif page == "admin":     show_admin()

if __name__ == "__main__":
    main()
