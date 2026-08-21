from __future__ import annotations

import base64
import hashlib
import html
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from auth import authenticate, init_db, register_user, save_analysis
from evidence import inspect_evidence, matching_hash_records, suspicious_moments
from forensics import ela_image
from predict import predict as predict_current
from predict_dfd import predict as predict_dfd
from report import generate_report


st.set_page_config(
    page_title="DeepGuard Forensic Scanner",
    page_icon="DG",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_db()

for key, value in {
    "user": None,
    "auth_notice": None,
    "model_profile": "DFD High-Accuracy (Recommended)",
}.items():
    st.session_state.setdefault(key, value)

GREEN = "#61f7b1"
MUTED = "#7d9388"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { color:#eafff3;background-color:#020604;background-image:linear-gradient(to bottom,transparent,rgba(97,247,177,.03) 55%,rgba(97,247,177,.32) 98%,#61f7b1),linear-gradient(rgba(97,247,177,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(97,247,177,.055) 1px,transparent 1px),radial-gradient(circle at 15% -8%,rgba(72,255,169,.10),transparent 32%),linear-gradient(180deg,#020604,#050a07);background-repeat:no-repeat,repeat,repeat,no-repeat,no-repeat;background-size:100% 110px,38px 38px,38px 38px,100% 100%,100% 100%;background-position:0 -110px,0 0,0 0,0 0,0 0;animation:authLaser 4.2s linear infinite; }
@keyframes authLaser { from { background-position:0 -110px,0 0,0 0,0 0,0 0; } to { background-position:0 calc(100vh + 110px),0 0,0 0,0 0,0 0; } }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"], #MainMenu, footer { visibility:hidden; }
.block-container { max-width:1100px; padding-top:2.2rem; padding-bottom:4rem; }
.brand { display:flex;align-items:center;gap:11px;margin-bottom:26px; }
.brand-dot { width:13px;height:13px;border-radius:50%;background:#61f7b1;box-shadow:0 0 24px #61f7b1; }
.brand-name { font:700 15px 'JetBrains Mono';letter-spacing:.18em;color:#dffff0; }
.pill { margin-left:auto;border:1px solid #174e34;border-radius:999px;padding:6px 11px;color:#61f7b1;font:600 10px 'JetBrains Mono';letter-spacing:.13em; }
.eyebrow { color:#61f7b1;font:600 11px 'JetBrains Mono';letter-spacing:.18em;text-transform:uppercase;margin-bottom:12px; }
.hero h1 { font-size:clamp(35px,6vw,62px);line-height:1.02;letter-spacing:-.055em;margin:0;max-width:780px; }
.hero h1 span { color:#61f7b1;text-shadow:0 0 30px rgba(97,247,177,.2); }
.hero p { color:#7d9388;max-width:720px;font-size:15px;line-height:1.7;margin:18px 0 28px; }
.auth-shell { position:relative;z-index:3;max-width:560px;margin:5vh auto 0;background:rgba(5,14,9,.90);border:1px solid #214d35;border-radius:14px;padding:28px;box-shadow:0 30px 80px rgba(0,0,0,.42),0 0 45px rgba(97,247,177,.06);backdrop-filter:blur(8px); }
.steps { display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:18px 0 24px; }
.step { border:1px solid #173826;border-radius:8px;padding:11px;color:#779083;font:600 10px 'JetBrains Mono';letter-spacing:.08em; }
.step strong { color:#61f7b1;margin-right:7px; }
[data-testid="stFileUploader"], [data-testid="stCameraInput"] { background:#07100b;border:1px solid #183c2a;border-radius:14px;padding:8px; }
[data-testid="stFileUploaderDropzone"] { background:#050b08;border:1px dashed #2a704d; }
.stButton>button { width:100%;border-radius:9px;border:1px solid #61f7b1;background:#61f7b1;color:#031008;font-weight:800;min-height:46px; }
.stButton>button:hover { background:#85ffc4;color:#031008;border-color:#85ffc4; }
.preview-frame { max-width:760px;margin:18px auto;border:1px solid #2b8d5d;border-radius:14px;background:#020604;overflow:hidden;position:relative;box-shadow:0 0 40px rgba(72,255,169,.10); }
.preview-frame img { display:block;width:100%;height:min(52vh,460px);object-fit:contain;background:#010402;opacity:.83; }
.scan-grid { position:absolute;inset:0;background-image:linear-gradient(rgba(97,247,177,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(97,247,177,.07) 1px,transparent 1px);background-size:26px 26px; }
.scan-line { position:absolute;left:0;right:0;height:3px;top:0;background:#61f7b1;box-shadow:0 0 12px #61f7b1,0 0 35px #61f7b1;animation:scan 1.6s ease-in-out infinite alternate; }
.scan-status { position:absolute;left:14px;bottom:14px;border:1px solid #2b8d5d;background:rgba(2,8,5,.9);border-radius:7px;padding:8px 11px;color:#61f7b1;font:600 10px 'JetBrains Mono';letter-spacing:.12em; }
@keyframes scan { from { top:2%; } to { top:98%; } }
.result { border:1px solid var(--result);border-radius:14px;padding:24px;margin-top:20px;background:linear-gradient(135deg,color-mix(in srgb,var(--result) 9%,#050a07),#050a07); }
.result-label { color:var(--result);font:600 10px 'JetBrains Mono';letter-spacing:.18em; }
.result-title { color:var(--result);font-size:clamp(27px,4vw,42px);font-weight:800;margin:8px 0 18px; }
.metrics { display:grid;grid-template-columns:repeat(3,1fr);gap:10px; }
.metric { background:#07100b;border:1px solid #173826;border-radius:9px;padding:14px; }
.metric-value { color:#eafff3;font:700 21px 'JetBrains Mono'; }
.metric-name { color:#60786b;font:600 9px 'JetBrains Mono';letter-spacing:.1em;margin-top:5px; }
.notice { margin-top:18px;color:#71887c;font-size:12px;line-height:1.65;border-left:2px solid #28583f;padding-left:12px; }
.hash { word-break:break-all;color:#719180;font:10px 'JetBrains Mono';background:#050b08;border:1px solid #173826;padding:11px;border-radius:8px; }
.source-note { color:#6d8276;font:11px/1.6 'JetBrains Mono';margin:10px 0 18px; }
@media(max-width:650px){.metrics{grid-template-columns:1fr}.block-container{padding-left:1rem;padding-right:1rem}.auth-shell{padding:18px}.steps{grid-template-columns:1fr}}
</style>
""",
    unsafe_allow_html=True,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def brand(status: str) -> None:
    st.markdown(
        f'<div class="brand"><div class="brand-dot"></div><div class="brand-name">DEEPGUARD</div><div class="pill">{html.escape(status)}</div></div>',
        unsafe_allow_html=True,
    )


def show_auth() -> None:
    brand("SECURE ACCESS")
    st.markdown(
        '<div class="eyebrow">Digital evidence authentication</div><div class="hero"><h1>Register. Log in.<br><span>Analyze securely.</span></h1><p>Analysis tools are available only after authentication.</p></div><div class="steps"><div class="step"><strong>01</strong>CREATE ACCOUNT</div><div class="step"><strong>02</strong>AUTHENTICATE & ENTER</div></div>',
        unsafe_allow_html=True,
    )
    tab_register, tab_login = st.tabs(["1 · REGISTER", "2 · LOGIN"])
    with tab_register:
        username = st.text_input("Choose username", key="register_username")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Create password", type="password", key="register_password", help="At least 8 characters, one uppercase letter and one number.")
        confirm = st.text_input("Confirm password", type="password", key="register_confirm")
        if st.button("CREATE ACCOUNT", key="register_button"):
            if not all((username, email, password, confirm)):
                st.error("Complete every registration field.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = register_user(username, email, password)
                if ok:
                    st.session_state.auth_notice = "Account authenticated. Open step 2 and log in."
                    st.success(st.session_state.auth_notice)
                else:
                    st.error(message)
    with tab_login:
        if st.session_state.auth_notice:
            st.success(st.session_state.auth_notice)
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("AUTHENTICATE & LOGIN", key="login_button"):
            user = authenticate(username.strip(), password) if username and password else None
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials. Register first or check your username and password.")


def verdict_style(verdict: str) -> tuple[str, str]:
    return {
        "FAKE": ("#ff6572", "MANIPULATION SUSPECTED"),
        "REAL": (GREEN, "LIKELY AUTHENTIC"),
        "INCONCLUSIVE": ("#ffc857", "INCONCLUSIVE"),
        "NO_FACE": ("#69a9ff", "NO SUITABLE FACE"),
    }.get(verdict, ("#ffc857", "INCONCLUSIVE"))


def scan_preview(payload: bytes, mime: str, is_video: bool) -> None:
    if is_video:
        media = '<div style="height:300px;background:radial-gradient(circle,#0c2116,#020604)"></div>'
    else:
        encoded = base64.b64encode(payload).decode("ascii")
        media = f'<img src="data:{html.escape(mime)};base64,{encoded}" alt="Evidence preview">'
    st.markdown(
        f'<div class="preview-frame">{media}<div class="scan-grid"></div><div class="scan-line"></div><div class="scan-status">SCANNING EVIDENCE · MODEL ACTIVE</div></div>',
        unsafe_allow_html=True,
    )


def render_evidence_authentication(evidence: dict, result: dict, tmp_path: str, repeat_count: int) -> None:
    st.subheader("Digital evidence authentication")
    metadata_state = "AVAILABLE" if evidence["metadata_available"] else "LIMITED / ABSENT"
    repeat_label = "NEW RECORD" if repeat_count <= 1 else f"MATCHED {repeat_count} RECORDS"
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "FILE INTEGRITY", "SHA-256 RECORDED"),
        (c2, "METADATA", metadata_state),
        (c3, "MODIFICATION TRACE", evidence["modification_status"]),
        (c4, "REPRODUCIBILITY", repeat_label),
    ]
    for column, label, value in cards:
        column.markdown(
            f'<div class="metric"><div class="metric-name">{html.escape(label)}</div><div style="color:#eafff3;font:700 12px JetBrains Mono;margin-top:9px">{html.escape(value)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="notice"><b>Evidence record:</b> {html.escape(evidence["evidence_id"])}<br><b>Authenticated analyst:</b> {html.escape(evidence["authenticated_user"])}<br><b>Analyzed at:</b> {html.escape(evidence["analyzed_at_utc"])} UTC<br><b>Credential status:</b> {html.escape(evidence["c2pa_status"])}<br><b>Interpretation:</b> {html.escape(evidence["modification_detail"])}<br>{html.escape(evidence["scope_note"])}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Metadata, EXIF, codec and consistency details", expanded=True):
        checks = evidence.get("consistency_checks") or []
        if checks:
            for check in checks:
                icon = "PASS" if check.get("ok") else "UNAVAILABLE"
                st.write(f'{icon} - {check.get("label")}: {check.get("value")}')
        metadata = evidence.get("metadata") or {}
        if metadata:
            st.json(metadata)
        else:
            st.info("No embedded EXIF or container metadata was available. Metadata absence is not proof of manipulation.")

    moments = suspicious_moments(result)
    if moments:
        st.subheader("Suspicious frames and timestamps")
        columns = st.columns(min(3, len(moments)))
        for index, moment in enumerate(moments):
            with columns[index % len(columns)]:
                image = moment.get("heatmap_bgr")
                if image is None:
                    image = moment.get("face_bgr")
                if image is not None:
                    st.image(image[..., ::-1], use_container_width=True)
                st.caption(f'Time {moment["timestamp"]} - frame {moment["frame_index"]} - score {moment["score"]*100:.1f}%')

    if result.get("kind") == "image":
        ela = ela_image(tmp_path)
        if ela is not None:
            with st.expander("Error-level analysis (supporting indicator)"):
                st.image(ela, caption="Brighter differences indicate unequal JPEG recompression. This is not proof of deepfake manipulation.", use_container_width=True)

    st.subheader("Integrity identifiers")
    h1, h2 = st.columns(2)
    with h1:
        st.caption("SHA-256 - exact file identity")
        st.code(evidence["sha256"], language=None)
    with h2:
        st.caption("Perceptual hash - similarity aid")
        st.code(evidence["perceptual_hash"], language=None)

    report_bytes = generate_report(
        input_path=tmp_path,
        verdict=result,
        sha256=evidence["sha256"],
        phash=evidence["perceptual_hash"],
    )
    st.download_button(
        "DOWNLOAD FORENSIC PDF REPORT",
        data=report_bytes,
        file_name=f'{Path(tmp_path).stem}-deepguard-report.pdf',
        mime="application/pdf",
        use_container_width=True,
    )

def render_analysis(source, source_label: str) -> None:
    payload = source.getvalue()
    suffix = Path(source.name).suffix.lower() or ".jpg"
    is_video = suffix in VIDEO_EXTENSIONS
    mime = getattr(source, "type", None) or ("video/mp4" if is_video else "image/jpeg")
    evidence_hash = sha256_bytes(payload)

    if is_video:
        st.video(payload)
    else:
        center = st.columns([1, 3, 1])[1]
        with center:
            st.image(payload, caption=source.name, use_container_width=True)

    if not st.button("RUN FORENSIC ANALYSIS", type="primary", key=f"scan_{source_label}_{evidence_hash[:12]}"):
        return

    scan_slot = st.empty()
    with scan_slot.container():
        scan_preview(payload, mime, is_video)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        predictor = predict_dfd if st.session_state.model_profile.startswith("DFD") else predict_current
        with st.spinner("Running face detection and forensic neural analysis…"):
            started = time.perf_counter()
            result = predictor(tmp_path)
            elapsed = (time.perf_counter() - started) * 1000
        evidence = inspect_evidence(tmp_path, payload, result, st.session_state.user["username"])
        result["evidence_auth"] = evidence
        scan_slot.empty()

        color, title = verdict_style(result.get("verdict", "INCONCLUSIVE"))
        metadata = result.get("meta", {})
        count = metadata.get("n_tracked_ids", metadata.get("n_faces_kept", metadata.get("n_faces_detected", 0)))
        unit = "PERSON TRACKS" if result.get("kind") == "video" else "FACES"
        st.markdown(
            f'<div class="result" style="--result:{color}"><div class="result-label">ANALYSIS COMPLETE</div><div class="result-title">{title}</div><div class="metrics"><div class="metric"><div class="metric-value">{result.get("score",0)*100:.1f}%</div><div class="metric-name">SCREENING SCORE</div></div><div class="metric"><div class="metric-value">{result.get("confidence",0)*100:.1f}%</div><div class="metric-name">CONFIDENCE</div></div><div class="metric"><div class="metric-value">{count}</div><div class="metric-name">{unit}</div></div></div><div class="notice">Automated screening aid only. Preserve the original file and obtain qualified forensic review before legal or judicial reliance.</div></div>',
            unsafe_allow_html=True,
        )

        faces = result.get("all_faces") or []
        if faces:
            st.subheader("Detected faces")
            columns = st.columns(min(4, len(faces)))
            for index, face in enumerate(faces):
                with columns[index % len(columns)]:
                    crop = face.get("crop_bgr")
                    if crop is not None:
                        st.image(crop[..., ::-1], use_container_width=True)
                    st.caption(f'Face {index+1} · {face.get("verdict","INCONCLUSIVE")} · {face.get("score",0)*100:.1f}%')

        tracks = result.get("tracks") or []
        if tracks:
            st.subheader("Person tracks")
            for track in tracks:
                st.write(f'Track {track["face_id"]} — {track["verdict"]} · {track["score"]*100:.1f}% · {track["observations"]} clear frames')

        st.caption(f'Model: {result.get("model_name","DeepGuard")} · Runtime: {elapsed:.0f} ms')
        try:
            save_analysis(st.session_state.user["id"], source.name, "video" if is_video else "image", result, evidence_hash)
        except Exception as exc:
            st.warning(f"Result displayed, but history could not be saved: {exc}")
        repeat_count = matching_hash_records(evidence_hash)
        render_evidence_authentication(evidence, result, tmp_path, repeat_count)
    except Exception as exc:
        scan_slot.empty()
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def show_scanner() -> None:
    brand("MODEL ONLINE")
    st.markdown(
        '<div class="eyebrow">Authenticated forensic workspace</div><div class="hero"><h1>Scan media for <span>manipulation.</span></h1><p>Upload evidence or capture a live camera frame. DeepGuard analyzes detected faces and records an evidence hash.</p></div>',
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown(f'**Signed in:** `{st.session_state.user["username"]}`')
        st.selectbox("MODEL PROFILE", ["DFD High-Accuracy (Recommended)", "Current / Rollback"], key="model_profile")
        if st.button("SIGN OUT"):
            st.session_state.user = None
            st.rerun()

    upload_tab, camera_tab = st.tabs(["UPLOAD EVIDENCE", "LIVE CAMERA"])
    with upload_tab:
        uploaded = st.file_uploader("Choose image or video", type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv", "webm"])
        if uploaded:
            render_analysis(uploaded, "upload")
        else:
            st.markdown('<div class="source-note">Accepted: JPG, PNG, WEBP, MP4, MOV, AVI, MKV and WEBM.</div>', unsafe_allow_html=True)
    with camera_tab:
        st.markdown('<div class="source-note">Allow camera access, frame the subject, then capture. The captured evidence is analyzed by the same model pipeline.</div>', unsafe_allow_html=True)
        capture = st.camera_input("Live camera capture", key="camera_capture")
        if capture:
            render_analysis(capture, "camera")


if st.session_state.user is None:
    show_auth()
else:
    show_scanner()
