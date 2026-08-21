from __future__ import annotations

import base64
import hashlib
import html
import os
import tempfile
import time

import streamlit as st

from predict import predict as predict_current
from predict_dfd import predict as predict_dfd


st.set_page_config(page_title="DeepGuard Scanner", page_icon="◉", layout="centered")

GREEN = "#61f7b1"
MUTED = "#7d9388"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 18% -8%, rgba(72,255,169,.10), transparent 34%),
    linear-gradient(180deg, #020604 0%, #050a07 100%);
  color: #eafff3;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container { max-width: 900px; padding-top: 2.3rem; padding-bottom: 4rem; }
.brand { display:flex; align-items:center; gap:11px; margin-bottom:26px; }
.brand-dot { width:13px; height:13px; border-radius:50%; background:#61f7b1; box-shadow:0 0 24px #61f7b1; }
.brand-name { font:700 15px 'JetBrains Mono'; letter-spacing:.18em; color:#dffff0; }
.live-pill { margin-left:auto; border:1px solid #174e34; border-radius:999px; padding:6px 11px; color:#61f7b1; font:600 10px 'JetBrains Mono'; letter-spacing:.13em; }
.hero h1 { font-size:clamp(35px,6vw,64px); line-height:1.02; letter-spacing:-.055em; margin:0; max-width:760px; }
.hero h1 span { color:#61f7b1; text-shadow:0 0 30px rgba(97,247,177,.2); }
.hero p { color:#7d9388; max-width:650px; font-size:15px; line-height:1.7; margin:18px 0 30px; }
.eyebrow { color:#61f7b1; font:600 11px 'JetBrains Mono'; letter-spacing:.18em; text-transform:uppercase; margin-bottom:12px; }
[data-testid="stFileUploader"] { background:#07100b; border:1px solid #183c2a; border-radius:14px; padding:8px; }
[data-testid="stFileUploaderDropzone"] { background:#050b08; border:1px dashed #2a704d; }
[data-testid="stFileUploaderDropzone"] button { background:#0d2418; color:#61f7b1; border:1px solid #2a704d; }
.stButton > button { width:100%; border-radius:10px; border:1px solid #61f7b1; background:#61f7b1; color:#031008; font-weight:800; letter-spacing:.04em; min-height:48px; }
.stButton > button:hover { background:#85ffc4; color:#031008; border-color:#85ffc4; box-shadow:0 0 30px rgba(97,247,177,.20); }
.scan-shell { position:relative; overflow:hidden; border:1px solid #2b8d5d; border-radius:14px; background:#020604; min-height:250px; box-shadow:0 0 40px rgba(72,255,169,.10); }
.scan-shell img { width:100%; max-height:520px; object-fit:contain; display:block; opacity:.78; }
.scan-grid { position:absolute; inset:0; background-image:linear-gradient(rgba(97,247,177,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(97,247,177,.08) 1px,transparent 1px); background-size:26px 26px; }
.scan-line { position:absolute; left:0; right:0; height:3px; top:0; background:#61f7b1; box-shadow:0 0 12px #61f7b1,0 0 35px #61f7b1; animation:scan 1.65s ease-in-out infinite alternate; }
.scan-glow { position:absolute; left:0; right:0; height:80px; top:0; transform:translateY(-50%); background:linear-gradient(180deg,transparent,rgba(97,247,177,.14),transparent); animation:scan 1.65s ease-in-out infinite alternate; }
.scan-status { position:absolute; left:15px; bottom:14px; border:1px solid #2b8d5d; background:rgba(2,8,5,.88); border-radius:7px; padding:8px 11px; color:#61f7b1; font:600 10px 'JetBrains Mono'; letter-spacing:.12em; animation:pulse 1s infinite; }
@keyframes scan { from { top:2%; } to { top:98%; } }
@keyframes pulse { 50% { opacity:.48; } }
.result { border:1px solid var(--result-color); border-radius:14px; padding:22px; margin-top:18px; background:linear-gradient(135deg,color-mix(in srgb,var(--result-color) 10%,#050a07),#050a07); }
.result-label { color:var(--result-color); font:600 10px 'JetBrains Mono'; letter-spacing:.18em; }
.result-title { color:var(--result-color); font-size:30px; font-weight:800; margin:7px 0 15px; }
.metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:9px; }
.metric { background:#07100b; border:1px solid #173826; border-radius:9px; padding:12px; }
.metric-value { color:#eafff3; font:700 18px 'JetBrains Mono'; }
.metric-name { color:#60786b; font:600 9px 'JetBrains Mono'; letter-spacing:.1em; margin-top:4px; }
.notice { margin-top:18px; color:#71887c; font-size:12px; line-height:1.65; border-left:2px solid #28583f; padding-left:12px; }
.hash { word-break:break-all; color:#719180; font:10px 'JetBrains Mono'; background:#050b08; border:1px solid #173826; padding:10px; border-radius:8px; }
@media(max-width:600px){ .metrics{grid-template-columns:1fr;} .block-container{padding-left:1rem;padding-right:1rem;} }
</style>
""",
    unsafe_allow_html=True,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verdict_style(verdict: str):
    return {
        "FAKE": ("#ff5f6d", "MANIPULATION SUSPECTED"),
        "REAL": (GREEN, "LIKELY AUTHENTIC"),
        "INCONCLUSIVE": ("#ffc857", "INCONCLUSIVE"),
        "NO_FACE": ("#69a9ff", "NO SUITABLE FACE"),
    }.get(verdict, ("#ffc857", "INCONCLUSIVE"))


def scanner_html(data: bytes, mime: str, is_video: bool) -> str:
    if is_video:
        preview = '<div style="height:340px;background:radial-gradient(circle,#0c2116,#020604);"></div>'
    else:
        encoded = base64.b64encode(data).decode("ascii")
        preview = f'<img src="data:{html.escape(mime)};base64,{encoded}" alt="Uploaded evidence">'
    return f"""
      <div class="scan-shell">
        {preview}<div class="scan-grid"></div><div class="scan-glow"></div>
        <div class="scan-line"></div><div class="scan-status">SCANNING EVIDENCE · MODEL ACTIVE</div>
      </div>"""


st.markdown(
    '<div class="brand"><div class="brand-dot"></div><div class="brand-name">DEEPGUARD</div><div class="live-pill">MODEL ONLINE</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><div class="eyebrow">Digital evidence screening</div><h1>Scan media for <span>manipulation.</span></h1><p>Upload an image or video. DeepGuard checks detected faces, tracks people across frames, and preserves uncertainty when the evidence is too blurry or incomplete.</p></div>',
    unsafe_allow_html=True,
)

profile = st.radio(
    "VIDEO MODEL PROFILE",
    ["DFD High-Accuracy (Recommended)", "Current / Rollback"],
    horizontal=True,
    help="DFD profile: 89.9% balanced accuracy on the locked actor-disjoint DFD test. Current instantly restores the prior pipeline.",
)

uploaded = st.file_uploader(
    "Upload evidence",
    type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv", "webm"],
)

if uploaded:
    payload = uploaded.getvalue()
    ext = os.path.splitext(uploaded.name)[1].lower()
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if is_video:
        st.video(payload)
    else:
        st.image(payload, caption=uploaded.name, width="stretch")

    if st.button("RUN FORENSIC SCAN", type="primary"):
        scan_slot = st.empty()
        result_slot = st.empty()
        scan_slot.markdown(scanner_html(payload, uploaded.type or "image/jpeg", is_video), unsafe_allow_html=True)
        time.sleep(0.15)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(payload)
                tmp_path = tmp.name
            result = predict_dfd(tmp_path) if profile.startswith("DFD") else predict_current(tmp_path)
            color, label = verdict_style(result.get("verdict", "INCONCLUSIVE"))
            meta = result.get("meta", {})
            count = meta.get("n_tracked_ids", meta.get("n_faces_kept", 0))
            unit = "TRACKS" if result.get("kind") == "video" else "FACES"
            result_slot.markdown(
                f"""
                <div class="result" style="--result-color:{color}">
                  <div class="result-label">ANALYSIS COMPLETE</div>
                  <div class="result-title">{label}</div>
                  <div class="metrics">
                    <div class="metric"><div class="metric-value">{result['score']*100:.1f}%</div><div class="metric-name">SCREENING SCORE</div></div>
                    <div class="metric"><div class="metric-value">{result['confidence']*100:.1f}%</div><div class="metric-name">CONFIDENCE</div></div>
                    <div class="metric"><div class="metric-value">{count}</div><div class="metric-name">{unit}</div></div>
                  </div>
                  <div class="notice">Automated screening aid only. Preserve the original file and obtain qualified forensic review before legal or judicial reliance.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            scan_slot.empty()

            faces = result.get("all_faces") or []
            if faces:
                st.markdown("### Detected faces")
                cols = st.columns(min(4, len(faces)))
                for i, face in enumerate(faces):
                    with cols[i % len(cols)]:
                        st.image(face["crop_bgr"][..., ::-1], width="stretch")
                        st.caption(f"Face {i + 1} · {face.get('verdict', 'INCONCLUSIVE')} · {face.get('score', 0)*100:.1f}%")

            tracks = result.get("tracks") or []
            if tracks:
                st.markdown("### Person tracks")
                for track in tracks:
                    st.write(f"Track {track['face_id']} — {track['verdict']} · {track['score']*100:.1f}% · {track['observations']} clear frames")

            st.markdown("#### Evidence SHA-256")
            st.markdown(f'<div class="hash">{sha256_bytes(payload)}</div>', unsafe_allow_html=True)
        except Exception as exc:
            scan_slot.empty()
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
else:
    st.markdown('<div class="notice">Accepted formats: JPG, PNG, WEBP, MP4, MOV, AVI, MKV and WEBM. For facial analysis, the face must be sufficiently visible.</div>', unsafe_allow_html=True)
