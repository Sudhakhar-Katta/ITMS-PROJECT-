import cgi
import csv
import html
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "analysis_results"
MPL_CONFIG_DIR = PROJECT_DIR / ".cache" / "matplotlib"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
JOBS = {}
JOBS_LOCK = threading.Lock()
DETECTION_COLUMNS = [
    ("eyes_closed", "Eyes Closed"),
    ("tired", "Tired"),
    ("asleep", "Asleep"),
    ("sleep_pose_signal", "Sleep Pose"),
    ("away_signal", "Away Signal"),
    ("looking_away", "Looking Away"),
    ("distracted", "Distracted"),
    ("drowsiness_active", "Drowsiness"),
    ("yawning", "Yawning"),
    ("yawn_alert", "Yawn Pattern"),
]


def ensure_directories():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name):
    name = Path(name or "uploaded_video").name
    cleaned = []

    for char in name:
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")

    result = "".join(cleaned).strip("._")
    return result or "uploaded_video"


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {"total_frames": 0, "fps": 0.0, "duration_seconds": 0.0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    duration_seconds = total_frames / fps if total_frames > 0 and fps > 0 else 0.0

    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": duration_seconds,
    }


def read_events(events_path):
    if not events_path.exists() or events_path.stat().st_size == 0:
        return [], {}

    rows = []
    counts = {}

    try:
        with events_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                rows.append(row)
                event_name = row.get("event") or "unknown"
                counts[event_name] = counts.get(event_name, 0) + 1
    except OSError:
        return [], {}

    return rows, counts


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_detection_instances(predictions_path, limit=200):
    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        return [], {}

    rows = []
    counts = {label: 0 for _, label in DETECTION_COLUMNS}

    try:
        with predictions_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for frame_number, row in enumerate(reader, start=1):
                active = [
                    label for column, label in DETECTION_COLUMNS
                    if truthy(row.get(column, ""))
                ]

                if not active:
                    continue

                for label in active:
                    counts[label] += 1

                rows.append(
                    {
                        "frame": frame_number,
                        "time_sec": row.get("time_sec", ""),
                        "detections": active,
                        "ear": row.get("ear", ""),
                        "mar": row.get("mar", ""),
                        "perclos": row.get("perclos", ""),
                        "gaze": row.get("gaze", ""),
                        "mode": row.get("attention_mode", ""),
                        "reason": first_present(
                            row,
                            [
                                "tired_reason",
                                "sleep_reason",
                                "distraction_reason",
                                "gaze_reason",
                            ],
                        ),
                    }
                )
    except OSError:
        return [], counts

    return rows[-limit:], counts


def build_detection_instances_csv(predictions_path):
    output = [
        [
            "frame",
            "time_sec",
            "detections",
            "ear",
            "mar",
            "perclos",
            "gaze",
            "attention_mode",
            "reason",
        ]
    ]

    rows, _ = read_detection_instances(predictions_path, limit=10_000_000)

    for row in rows:
        output.append(
            [
                row["frame"],
                row["time_sec"],
                ", ".join(row["detections"]),
                row["ear"],
                row["mar"],
                row["perclos"],
                row["gaze"],
                row["mode"],
                row["reason"],
            ]
        )

    lines = []
    for row in output:
        escaped = []
        for value in row:
            text = str(value)
            if "," in text or '"' in text or "\n" in text:
                text = '"' + text.replace('"', '""') + '"'
            escaped.append(text)
        lines.append(",".join(escaped))

    return ("\n".join(lines) + "\n").encode("utf-8")


def first_present(row, columns):
    for column in columns:
        value = row.get(column, "")
        if value:
            return value
    return ""


def count_prediction_rows(predictions_path):
    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        return 0

    try:
        with predictions_path.open("r", newline="", encoding="utf-8") as file:
            return max(sum(1 for _ in file) - 1, 0)
    except OSError:
        return 0


def get_job(run_id):
    with JOBS_LOCK:
        return dict(JOBS.get(run_id, {}))


def update_job(run_id, **updates):
    with JOBS_LOCK:
        if run_id in JOBS:
            JOBS[run_id].update(updates)


def job_status(run_id):
    job = get_job(run_id)

    if not job:
        return None

    predictions_path = Path(job["predictions_path"])
    events_path = Path(job["events_path"])
    stdout_path = Path(job["stdout_path"])
    stderr_path = Path(job["stderr_path"])

    processed_frames = count_prediction_rows(predictions_path)
    total_frames = int(job.get("total_frames") or 0)
    progress = processed_frames / total_frames if total_frames > 0 else 0.0
    progress = max(0.0, min(progress, 1.0))
    events, counts = read_events(events_path)
    detections, detection_counts = read_detection_instances(predictions_path)
    elapsed = time.perf_counter() - float(job["started_at"])

    stderr_tail = ""
    if stderr_path.exists():
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]

    return {
        "run_id": run_id,
        "video_name": job["video_name"],
        "state": job["state"],
        "returncode": job.get("returncode"),
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "progress": progress,
        "progress_percent": round(progress * 100, 1),
        "events": events[-50:],
        "event_counts": counts,
        "event_count": len(events),
        "detections": detections,
        "detection_counts": detection_counts,
        "detection_instance_count": sum(detection_counts.values()),
        "elapsed_seconds": round(elapsed, 1),
        "fps": round(float(job.get("fps") or 0.0), 2),
        "duration_seconds": round(float(job.get("duration_seconds") or 0.0), 1),
        "stderr_tail": stderr_tail,
        "downloads_ready": predictions_path.exists(),
        "links": {
            "video": f"/video/{run_id}",
            "predictions": f"/results/{run_id}/predictions.csv",
            "events": f"/results/{run_id}/events.csv",
            "detection_instances": f"/results/{run_id}/detection_instances.csv",
            "stdout": f"/results/{run_id}/stdout.log",
            "stderr": f"/results/{run_id}/stderr.log",
        },
    }


def run_analysis_job(run_id):
    job = get_job(run_id)
    predictions_path = Path(job["predictions_path"])
    stdout_path = Path(job["stdout_path"])
    stderr_path = Path(job["stderr_path"])

    command = [
        sys.executable,
        "main.py",
        "--source",
        job["video_path"],
        "--video_name",
        job["video_name"],
        "--save_csv",
        str(predictions_path),
        "--no_display",
        "--video_time",
    ]

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
    update_job(run_id, state="running")

    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file:
            with stderr_path.open("w", encoding="utf-8") as stderr_file:
                completed = subprocess.run(
                    command,
                    cwd=str(BASE_DIR),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env,
                    text=True,
                )

        update_job(
            run_id,
            state="completed" if completed.returncode == 0 else "failed",
            returncode=completed.returncode,
            completed_at=time.perf_counter(),
        )
    except Exception as exc:
        stderr_path.write_text(str(exc), encoding="utf-8")
        update_job(run_id, state="failed", returncode=-1, completed_at=time.perf_counter())


def start_job(video_path, result_dir, video_name):
    metadata = get_video_metadata(video_path)
    run_id = result_dir.name

    with JOBS_LOCK:
        JOBS[run_id] = {
            "run_id": run_id,
            "video_name": video_name,
            "video_path": str(video_path),
            "result_dir": str(result_dir),
            "predictions_path": str(result_dir / "predictions.csv"),
            "events_path": str(result_dir / "events.csv"),
            "stdout_path": str(result_dir / "stdout.log"),
            "stderr_path": str(result_dir / "stderr.log"),
            "state": "queued",
            "returncode": None,
            "started_at": time.perf_counter(),
            **metadata,
        }

    thread = threading.Thread(target=run_analysis_job, args=(run_id,), daemon=True)
    thread.start()
    return run_id


def page(title, body):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f6f8;
      color: #17202c;
    }}
    body {{
      margin: 0;
      padding: 28px;
    }}
    main {{
      max-width: 1480px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 18px;
    }}
    h2 {{
      font-size: 18px;
      margin: 26px 0 12px;
    }}
    form, .panel, table {{
      background: #ffffff;
      border: 1px solid #d7dde6;
      border-radius: 8px;
    }}
    form, .panel {{
      padding: 18px;
    }}
    label {{
      display: block;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    input[type="file"] {{
      display: block;
      margin-bottom: 16px;
      width: 100%;
    }}
    button, .button {{
      appearance: none;
      border: 0;
      border-radius: 6px;
      background: #1559a8;
      color: white;
      cursor: pointer;
      display: inline-block;
      font-weight: 700;
      margin: 4px 6px 4px 0;
      padding: 10px 14px;
      text-decoration: none;
    }}
    button:hover, .button:hover {{
      background: #0e407c;
    }}
    .muted {{
      color: #5c6876;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }}
    .workspace {{
      align-items: start;
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(340px, 0.95fr) minmax(420px, 1.25fr);
    }}
    .video-panel {{
      position: sticky;
      top: 18px;
    }}
    video {{
      aspect-ratio: 16 / 9;
      background: #0b1220;
      border-radius: 8px;
      display: block;
      object-fit: contain;
      width: 100%;
    }}
    .scroll-table {{
      border: 1px solid #d7dde6;
      border-radius: 8px;
      max-height: 440px;
      overflow: auto;
    }}
    .scroll-table table {{
      border: 0;
      border-radius: 0;
    }}
    .clickable-row {{
      cursor: pointer;
    }}
    .clickable-row:hover {{
      background: #eef6ff;
    }}
    .metric {{
      background: #ffffff;
      border: 1px solid #d7dde6;
      border-radius: 8px;
      padding: 14px;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
      margin-bottom: 4px;
    }}
    .progress {{
      background: #dfe5ed;
      border-radius: 999px;
      height: 14px;
      overflow: hidden;
    }}
    .bar {{
      background: #1c7c54;
      height: 100%;
      transition: width 250ms linear;
      width: 0%;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid #e4e8ef;
      padding: 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
      font-size: 13px;
    }}
    pre {{
      background: #111827;
      border-radius: 8px;
      color: #f9fafb;
      max-height: 240px;
      overflow: auto;
      padding: 12px;
      white-space: pre-wrap;
    }}
    @media (max-width: 920px) {{
      body {{
        padding: 16px;
      }}
      .workspace {{
        grid-template-columns: 1fr;
      }}
      .video-panel {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>"""


def upload_page():
    body = """
<h1>ITMS Live Video Analysis</h1>
<form action="/analyze" method="post" enctype="multipart/form-data">
  <label for="video">Upload a cabin-facing driver video</label>
  <input id="video" name="video" type="file" accept=".mp4,.avi,.mov,.mkv,.webm,video/*" required>
  <button type="submit">Start Live Analysis</button>
  <p class="muted">The browser switches to a live job page immediately after upload. Progress, frame count, and alert events update while the detector is still processing.</p>
</form>
"""
    return page("ITMS Live Video Analysis", body)


def live_job_page(run_id):
    body = f"""
<h1>Live Analysis</h1>
<p class="muted" id="videoName">Starting...</p>
<div class="workspace">
  <section class="video-panel">
    <h2>Video</h2>
    <video id="sourceVideo" controls preload="metadata">
      <source src="/video/{html.escape(run_id)}">
    </video>
    <div class="panel">
      <div class="progress"><div class="bar" id="progressBar"></div></div>
      <p class="muted" id="progressText">Waiting for the detector...</p>
    </div>
    <h2>Status</h2>
    <div class="grid">
      <div class="metric"><strong id="state">queued</strong><span>State</span></div>
      <div class="metric"><strong id="frames">0</strong><span>Frames</span></div>
      <div class="metric"><strong id="events">0</strong><span>Events</span></div>
      <div class="metric"><strong id="detections">0</strong><span>Detection Instances</span></div>
      <div class="metric"><strong id="elapsed">0.0s</strong><span>Elapsed</span></div>
    </div>
  </section>
  <section>
    <h2>Detection Instances</h2>
    <div class="panel">
      <p class="muted" id="detectionCounts">No active detector states yet.</p>
    </div>
    <div class="scroll-table">
      <table>
        <thead>
          <tr><th>Frame</th><th>Time Sec</th><th>Detections</th><th>EAR</th><th>MAR</th><th>PERCLOS</th><th>Gaze</th><th>Mode</th><th>Reason</th></tr>
        </thead>
        <tbody id="detectionRows"><tr><td colspan="9">No detection instances yet.</td></tr></tbody>
      </table>
    </div>
    <h2>Detected Events</h2>
    <div class="scroll-table">
      <table>
        <thead>
          <tr><th>Event</th><th>Time Sec</th><th>Duration Sec</th><th>Reason</th><th>Mode</th></tr>
        </thead>
        <tbody id="eventRows"><tr><td colspan="5">No alert events yet.</td></tr></tbody>
      </table>
    </div>
    <h2>Outputs</h2>
    <div class="panel" id="downloads">
      <span class="muted">Downloads become useful as soon as rows start writing, and final when the job completes.</span>
    </div>
    <h2>Runtime Log</h2>
    <pre id="log">Waiting for detector output...</pre>
  </section>
</div>
<p><a href="/">Analyze another video</a></p>
<script>
const runId = {json.dumps(run_id)};
let stopped = false;

function text(id, value) {{
  document.getElementById(id).textContent = value;
}}

function renderEvents(rows) {{
  const target = document.getElementById("eventRows");
  if (!rows.length) {{
    target.innerHTML = '<tr><td colspan="5">No alert events yet.</td></tr>';
    return;
  }}
  target.innerHTML = rows.map((row) => `
    <tr class="clickable-row" data-time="${{escapeHtml(row.time_sec || "")}}">
      <td>${{escapeHtml(row.event || "")}}</td>
      <td>${{escapeHtml(row.time_sec || "")}}</td>
      <td>${{escapeHtml(row.duration_sec || "")}}</td>
      <td>${{escapeHtml(row.reason || "")}}</td>
      <td>${{escapeHtml(row.mode || "")}}</td>
    </tr>
  `).join("");
}}

function renderDetections(rows) {{
  const target = document.getElementById("detectionRows");
  if (!rows.length) {{
    target.innerHTML = '<tr><td colspan="9">No detection instances yet.</td></tr>';
    return;
  }}
  target.innerHTML = rows.map((row) => `
    <tr class="clickable-row" data-time="${{escapeHtml(row.time_sec || "")}}">
      <td>${{escapeHtml(row.frame || "")}}</td>
      <td>${{escapeHtml(row.time_sec || "")}}</td>
      <td>${{escapeHtml((row.detections || []).join(", "))}}</td>
      <td>${{escapeHtml(row.ear || "")}}</td>
      <td>${{escapeHtml(row.mar || "")}}</td>
      <td>${{escapeHtml(row.perclos || "")}}</td>
      <td>${{escapeHtml(row.gaze || "")}}</td>
      <td>${{escapeHtml(row.mode || "")}}</td>
      <td>${{escapeHtml(row.reason || "")}}</td>
    </tr>
  `).join("");
  bindSeekRows();
}}

function bindSeekRows() {{
  const video = document.getElementById("sourceVideo");
  document.querySelectorAll(".clickable-row").forEach((row) => {{
    row.onclick = () => {{
      const seconds = Number(row.dataset.time);
      if (Number.isFinite(seconds)) {{
        video.currentTime = seconds;
        video.play();
      }}
    }};
  }});
}}

function renderDetectionCounts(counts) {{
  const entries = Object.entries(counts || {{}}).filter((entry) => entry[1] > 0);
  document.getElementById("detectionCounts").textContent = entries.length
    ? entries.map(([name, count]) => `${{name}}: ${{count}}`).join(" | ")
    : "No active detector states yet.";
}}

function renderDownloads(data) {{
  const links = data.links;
  document.getElementById("downloads").innerHTML = `
    <a class="button" href="${{links.predictions}}">Frame CSV</a>
    <a class="button" href="${{links.events}}">Event CSV</a>
    <a class="button" href="${{links.detection_instances}}">Detection Instances CSV</a>
    <a class="button" href="${{links.stdout}}">Stdout</a>
    <a class="button" href="${{links.stderr}}">Stderr</a>
  `;
}}

function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, (char) => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }}[char]));
}}

async function refresh() {{
  if (stopped) return;
  const response = await fetch(`/api/jobs/${{runId}}`, {{ cache: "no-store" }});
  const data = await response.json();
  text("videoName", `Video: ${{data.video_name}}`);
  text("state", data.state);
  text("frames", `${{data.processed_frames}} / ${{data.total_frames || "?"}}`);
  text("events", data.event_count);
  text("detections", data.detection_instance_count);
  text("elapsed", `${{data.elapsed_seconds}}s`);
  text("progressText", `${{data.progress_percent}}% complete`);
  document.getElementById("progressBar").style.width = `${{data.progress_percent}}%`;
  document.getElementById("log").textContent = data.stderr_tail || "No runtime warnings.";
  renderDownloads(data);
  renderEvents(data.events || []);
  renderDetectionCounts(data.detection_counts || {{}});
  renderDetections(data.detections || []);

  if (data.state === "completed" || data.state === "failed") {{
    stopped = true;
    return;
  }}
  setTimeout(refresh, 1000);
}}

refresh();
</script>
"""
    return page("Live Analysis", body)


class UploadHandler(BaseHTTPRequestHandler):
    server_version = "ITMSUploadServer/2.0"

    def send_html(self, content, status=200):
        data = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location):
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_html(upload_page())
            return

        if parsed.path.startswith("/jobs/"):
            run_id = parsed.path.strip("/").split("/", 1)[1]
            if not get_job(run_id):
                self.send_html(page("Not Found", "<h1>Job Not Found</h1>"), status=404)
                return
            self.send_html(live_job_page(run_id))
            return

        if parsed.path.startswith("/api/jobs/"):
            run_id = parsed.path.strip("/").split("/", 2)[2]
            status = job_status(run_id)
            if status is None:
                self.send_json({"error": "job not found"}, status=404)
                return
            self.send_json(status)
            return

        if parsed.path.startswith("/video/"):
            self.serve_video(parsed.path)
            return

        if parsed.path.startswith("/results/"):
            self.serve_result_file(parsed.path)
            return

        self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)

    def serve_video(self, request_path):
        parts = [unquote(part) for part in request_path.strip("/").split("/")]

        if len(parts) != 2:
            self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
            return

        _, run_id = parts
        job = get_job(run_id)

        if not job:
            self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
            return

        video_path = Path(job["video_path"])
        try:
            video_path.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
            return

        if not video_path.exists():
            self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
            return

        content_types = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
        }
        content_type = content_types.get(video_path.suffix.lower(), "application/octet-stream")
        data = video_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if urlparse(self.path).path != "/analyze":
            self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )

        if "video" not in form or not getattr(form["video"], "file", None):
            self.send_html(page("Upload Error", "<h1>No video file uploaded.</h1>"), status=400)
            return

        uploaded = form["video"]
        original_name = safe_filename(uploaded.filename)
        extension = Path(original_name).suffix.lower()

        if extension not in VIDEO_EXTENSIONS:
            body = "<h1>Unsupported File</h1><p>Upload MP4, AVI, MOV, MKV, or WEBM.</p>"
            self.send_html(page("Unsupported File", body), status=400)
            return

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        upload_path = UPLOAD_DIR / f"{run_id}-{original_name}"
        result_dir = RESULTS_DIR / run_id
        result_dir.mkdir(parents=True, exist_ok=True)

        with upload_path.open("wb") as file:
            shutil.copyfileobj(uploaded.file, file)

        start_job(upload_path, result_dir, original_name)
        self.redirect(f"/jobs/{run_id}")

    def serve_result_file(self, request_path):
        parts = [unquote(part) for part in request_path.strip("/").split("/")]

        if len(parts) != 3:
            self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
            return

        _, run_id, filename = parts
        if "/" in run_id or "\\" in run_id or filename not in {
            "predictions.csv",
            "events.csv",
            "detection_instances.csv",
            "stdout.log",
            "stderr.log",
        }:
            self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
            return

        file_path = RESULTS_DIR / run_id / filename
        if filename == "detection_instances.csv":
            predictions_path = RESULTS_DIR / run_id / "predictions.csv"
            if not predictions_path.exists():
                self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
                return

            data = build_detection_instances_csv(predictions_path)
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f"attachment; filename={filename}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if not file_path.exists():
            self.send_html(page("Not Found", "<h1>Not Found</h1>"), status=404)
            return

        content_type = "text/csv" if filename.endswith(".csv") else "text/plain"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Disposition", f"attachment; filename={filename}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ensure_directories()
    host = "127.0.0.1"
    port = int(os.environ.get("ITMS_UPLOAD_PORT", "8090"))
    server = ThreadingHTTPServer((host, port), UploadHandler)
    print(f"Upload server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
