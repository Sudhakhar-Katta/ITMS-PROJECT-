# ITMS Driver State Detection Module

Real-time driver fatigue and distraction monitoring using Python, OpenCV, MediaPipe FaceMesh, EAR, MAR, gaze scoring, head pose estimation, adaptive gaze reliability, and temporal alert logic.

This module is part of the larger **ITMS — Intelligent Traffic Management System** project. It focuses on the **driver-facing camera** side of the system: drowsiness, micro-sleep, tiredness, yawning, looking away, and driver distraction.

## Latest Update

The latest tired-filter change removes `sustained_perclos` as a direct visible `TIRED!` trigger. `TIRED!` now appears only when there is a current eye-based tiredness signal: `slow_eye_closure`, `heavy_eyelids`, or `critical_perclos_with_eye_signal`. High PERCLOS alone is still logged as `tired_raw`, but it does not show the live tired alert if the eyes are open/normal.

---

## 1. Project Goal

The goal of this module is to detect unsafe driver states from a webcam or cabin-facing video stream.

The system should detect:

- Driver drowsiness / prolonged eye closure
- Micro-sleep patterns
- Tiredness using rolling PERCLOS
- Yawning and repeated-yawn patterns
- Looking away
- Driver distraction
- Gaze reliability problems caused by sunglasses, glare, or occlusion

The system is designed to work in two attention modes:

```text
1. gaze_plus_head_pose
   Used when eyes/iris are visible and gaze is reliable.

2. head_pose_only
   Used when sunglasses, glare, darkness, or occlusion make gaze unreliable.
```

This matters because when the driver wears dark sunglasses, gaze detection cannot be trusted. In that case, the system falls back to head pose / face direction.

---

## 2. Current System Summary

The current driver-state module detects driver fatigue and distraction from webcam/video using:

- **MediaPipe FaceMesh** for 478 facial landmarks
- **EAR**: Eye Aspect Ratio
- **MAR**: Mouth Aspect Ratio
- **Gaze score**: iris/eye-center based eye direction signal
- **Head pose**: roll, pitch, yaw
- **PERCLOS**: rolling percentage of eye closure
- **Adaptive gaze reliability gate**
- **Temporal event monitors** for multi-second alert logic
- **Frame-level CSV logging**
- **Event-level alert logging**

In short:

```text
Visible eyes       -> use gaze + head pose
Hidden eyes/glare  -> use head pose only
Eyes closed 3 sec  -> drowsiness alert
3 long closures    -> micro-sleep alert
MAR high           -> yawn visual alert
3 yawns / 5 min    -> yawning pattern event
Slow closure/heavy eyelids -> filtered TIRED alert
Look away 5 sec    -> distraction alert
```

---

## 3. Qualification Criteria

| Module | Functional Trigger / Requirement | Qualification Target |
|---|---|---|
| Driver Drowsiness | Trigger when eyes remain closed continuously for >= 3 seconds | Detection Accuracy >= 95%, Precision >= 95%, Recall >= 95%, False Alarm < 2%, Detection Delay < 500 ms after trigger condition is satisfied |
| Micro Sleep | Trigger after 3 eye closures of >= 2 seconds each within 1 minute | Detection Accuracy >= 95%, False Positives < 2%, Delay < 500 ms after pattern condition is satisfied |
| Yawning | Trigger after 3 yawns within 5 minutes | Precision >= 95%, Recall >= 95%, False Positives < 3% |
| Driver Distraction | Estimate head pose and gaze. Trigger if the driver is looking away for >= 5 seconds | Head Pose Accuracy >= 95%, Gaze Classification >= 95% when eyes are visible, Precision >= 95%, Recall >= 95%, False Alerts < 2% |

### Important clarification about detection delay

For rules that require time, detection delay is measured **after the requirement becomes true**, not from the start of the behavior.

Example for drowsiness:

```text
Eyes close at 10.0 sec
Requirement becomes true at 13.0 sec because 10.0 + 3.0 = 13.0
System alerts at 13.2 sec
Detection delay = 0.2 sec
```

This satisfies `Detection Delay < 500 ms`.

---

## 4. High-Level Architecture

```text
Camera / Video
    |
    v
OpenCV Frame Capture
    |
    v
MediaPipe FaceMesh
    |
    +--> Eye Detector
    |       +--> EAR
    |       +--> Gaze Score
    |
    +--> Mouth Detector
    |       +--> MAR
    |
    +--> Head Pose Estimator
    |       +--> Roll / Pitch / Yaw
    |
    +--> Gaze Reliability Gate
    |       +--> gaze_plus_head_pose OR head_pose_only
    |
    +--> Driver State Monitor
    |       +--> drowsiness
    |       +--> micro_sleep
    |       +--> yawning
    |       +--> yawning_pattern
    |
    +--> Tired Filter
    |       +--> tired_raw
    |       +--> tired filtered live alert
    |
    +--> Distraction Monitor
            +--> looking_away
            +--> distracted
            +--> away_duration
            +--> distraction_reason
```

---

## 5. Main Detection Features

### 5.1 Driver Drowsiness / Eyes Closed

Uses **EAR — Eye Aspect Ratio**.

Logic:

```text
if EAR <= ear_thresh continuously for >= ear_time_thresh:
    trigger driver_drowsiness
```

Default values:

```text
ear_thresh = 0.22
ear_time_thresh = 3.0 sec
```

The module also allows a small open-frame gap so that one bad MediaPipe frame does not reset the timer.

Recommended behavior:

```text
Close eyes for 1 sec -> no alert
Close eyes for 3 sec -> DROWSINESS / ASLEEP alert
```

---

### 5.2 Micro Sleep

Micro-sleep is not just one long eye closure. It is repeated long closures.

Requirement:

```text
Trigger after 3 eye closures of >= 2 seconds each within 60 seconds.
```

Logic:

```text
When eyes reopen:
    measure closure duration

if closure duration >= 2 sec:
    save closure timestamp

if 3 closure timestamps exist within 60 sec:
    trigger micro_sleep
```

Default values:

```text
micro_closure_seconds = 2.0
micro_closure_count = 3
micro_window_seconds = 60.0
```

---

### 5.3 Tired / PERCLOS

The tired module uses **PERCLOS as a raw fatigue signal**, but it does **not** use sustained PERCLOS alone as the final live `TIRED!` trigger.

PERCLOS means:

```text
Percentage of time eyes are closed over a rolling time window.
```

The system now separates raw fatigue estimation from the visible alert:

```text
tired_raw = raw rolling PERCLOS output
tired     = cleaned live alert shown on screen
tired_reason = reason why the live tired state changed
```

This separation is important because raw PERCLOS has memory. After the driver opens their eyes, PERCLOS can stay high for a few seconds. If PERCLOS directly controls the UI, normal blinking or recently closed eyes can incorrectly keep `TIRED!` on.

Final live `TIRED!` now turns on only for these reasons:

```text
slow_eye_closure
heavy_eyelids
critical_perclos_with_eye_signal
```

High PERCLOS alone with normal/open eyes stays:

```text
not_tired
```

Expected behavior:

```text
Normal blink                         -> no TIRED
Repeated normal blinks               -> no TIRED
Slow blink / short sleep / quick wake -> TIRED
Heavy eyelids for about 1 sec         -> TIRED
Critical PERCLOS + bad eye signal     -> TIRED
Eyes open again                       -> TIRED clears quickly
```

Recommended live filter settings:

```text
tired_on_perclos = 0.14
tired_off_perclos = 0.07
critical_perclos = 0.30
slow_closure_seconds = 0.55
blink_ignore_seconds = 0.35
heavy_ear_thresh = ear_thresh + 0.10
heavy_ear_seconds = 1.0
clear_after_open_seconds = 1.0
min_active_seconds = 0.8
```

---

### 5.4 Yawning

Uses **MAR — Mouth Aspect Ratio**.

Logic for one yawn:

```text
if MAR >= yawn_mar_thresh for >= yawn_time_thresh:
    detect one yawn
```

Default values:

```text
yawn_mar_thresh = 0.55
yawn_time_thresh = 1.2 sec
```

The UI can show:

```text
YAWNING!
```

while the yawn is occurring.

---

### 5.5 Yawning Pattern

The official stronger yawning requirement is:

```text
Trigger after 3 yawns within 5 minutes.
```

Correct configuration:

```text
yawn_pattern_count = 3
yawn_window_seconds = 300.0
```

This is better than alerting from one yawn because one mouth opening may be:

- Talking
- Laughing
- Breathing
- Mouth movement
- False MAR spike

The stronger `yawning_pattern` event is triggered only after repeated yawns.

---

### 5.6 Gaze Reliability

This module decides whether gaze can be trusted.

Possible modes:

```text
gaze_plus_head_pose
head_pose_only
```

Gaze is trusted when:

```text
Eyes are open
Iris landmarks are stable
Eye region is visible
No severe glare / dark sunglasses / occlusion
```

Gaze is not trusted when:

```text
Dark sunglasses hide the iris
Glare/reflection blocks the eye
Eyes are closed
Iris landmarks are unstable
Eye region is unusable
```

Important design decision:

```text
Do not simply detect “glasses”.
Instead, detect whether gaze is reliable.
```

Because:

```text
Transparent glasses may still allow accurate gaze detection.
Dark sunglasses should force head_pose_only mode.
```

Expected behavior:

```text
Normal visible eyes:
    Mode: gaze_plus_head_pose
    Gaze reliable: True

Transparent glasses:
    Mode: gaze_plus_head_pose
    Gaze reliable: True

Dark sunglasses:
    Mode: head_pose_only
    Gaze reliable: False
```

---

### 5.7 Looking Away / Driver Distraction

Driver distraction uses adaptive gaze-head fusion.

When gaze is reliable:

```text
looking_away = gaze_away OR head_away
```

When gaze is not reliable:

```text
looking_away = head_away only
```

Head pose signals:

```text
yaw   = left/right head turn
pitch = up/down face direction
roll  = sideways head tilt
```

The system calibrates a neutral forward pose for the first 2 seconds:

```text
base_roll
base_pitch
base_yaw
```

Then it uses relative angles:

```text
rel_roll = current_roll - base_roll
rel_pitch = current_pitch - base_pitch
rel_yaw = current_yaw - base_yaw
```

Recommended starting thresholds:

```text
lookaway_yaw_thresh = 20-22 degrees
lookaway_pitch_thresh = 18-20 degrees
severe_yaw_thresh = 30-32 degrees
severe_pitch_thresh = 25-27 degrees
severe_roll_thresh = 25-28 degrees
distraction_seconds = 5.0 sec
```

Trigger:

```text
if away_signal remains true for >= 5 seconds:
    trigger driver_distraction
```

This prevents instant false alerts.

---

## 6. Important Files

```text
driver_state_detection/
    main.py
    parser.py
    state_monitor.py
    mouth_detector.py
    gaze_reliability.py
    distraction_monitor.py
    tired_filter.py
    run_dataset_eval.py
    combine_completed_predictions.py
    evaluate_completed_metrics.py
    check_eval_inputs.py
    inspect_json_labels.py
    find_label_files.py
```

### main.py

Main runtime script. Handles:

- Webcam/video capture
- MediaPipe FaceMesh
- EAR/MAR/gaze/head-pose computation
- State monitor updates
- UI display
- CSV logging

### parser.py

Defines command-line arguments:

- `--source`
- `--video_name`
- `--save_csv`
- `--no_display`
- `--ear_thresh`
- `--ear_time_thresh`
- `--yawn_mar_thresh`
- `--yawn_time_thresh`
- `--micro_closure_seconds`
- `--micro_window_seconds`
- `--gaze_*` reliability settings

### state_monitor.py

Handles:

- Continuous eye closure
- Drowsiness event
- Micro-sleep pattern
- Yawn event
- Yawning pattern event

### mouth_detector.py

Computes MAR from MediaPipe mouth landmarks.

### gaze_reliability.py

Decides whether to use gaze or fall back to head pose.

### distraction_monitor.py

Handles:

- Forward-pose calibration
- Relative yaw/pitch/roll
- Looking away timer
- Distraction event trigger after 5 seconds

### tired_filter.py

Filters raw PERCLOS into a clean live `TIRED!` alert. It ignores normal blinks, does not allow sustained PERCLOS alone to trigger `TIRED!`, triggers on slow eye closure/heavy eyelids/critical PERCLOS with eye signal, and clears quickly after the eyes reopen.

### run_dataset_eval.py

Runs all videos in a dataset folder and saves one prediction CSV per video.

### evaluate_completed_metrics.py

Evaluates predictions against `labels.csv`. This requires manually labeled ground truth.

---

## 7. Installation

From the project folder:

```powershell
cd "C:\Users\valla\ITMS-SYSTEM1\Driver-State-Detection\driver_state_detection"
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the venv is in the parent folder:

```powershell
..\.venv\Scripts\Activate.ps1
```

Install required packages:

```powershell
python -m pip install opencv-python mediapipe numpy
```

Test OpenCV:

```powershell
python -c "import cv2; print(cv2.__version__)"
```

Test MediaPipe:

```powershell
python -c "import cv2, mediapipe, numpy; print('ok')"
```

---

## 8. Running the System

### 8.0 Run Upload-Based Video Analysis

Use this when you want to upload a video from a browser and watch the full
OpenCV + MediaPipe driver-state pipeline process it live without opening an
OpenCV preview window.

```powershell
cd driver_state_detection
C:\Users\satya\AppData\Local\Programs\Python\Python312\python.exe upload_server.py
```

Then open:

```text
http://127.0.0.1:8090
```

Upload an MP4, AVI, MOV, MKV, or WEBM file. The server saves the video, starts
`main.py --no_display` in the background, and immediately opens a live job page
that updates frame progress, runtime status, and detected alert events while
the analysis is still running. It also shows live frame-level detection
instances for active states such as eyes closed, tired, asleep, looking away,
distracted, drowsiness, and yawning side by side with the uploaded video.
Click a detection or event row to jump the video to that timestamp. The page
also includes links for:

- `predictions.csv` frame-level analysis
- `events.csv` event-level alerts
- `detection_instances.csv` filtered rows where at least one detector state is active
- `stdout.log`
- `stderr.log`

Generated uploads and analysis outputs are written under:

```text
driver_state_detection/uploads/
driver_state_detection/analysis_results/
```

### 8.1 Run Webcam

```powershell
python main.py --source 0 --video_name "webcam" --save_csv "C:\Users\valla\ITMS-SYSTEM1\datasets\webcam_test\predictions\webcam_predictions.csv" --ear_thresh 0.22 --ear_time_thresh 3 --yawn_mar_thresh 0.55 --yawn_time_thresh 1.2
```

Press:

```text
q
```

to quit.

If webcam 0 does not open, try:

```powershell
python main.py --source 1 --video_name "webcam" --save_csv "C:\Users\valla\ITMS-SYSTEM1\datasets\webcam_test\predictions\webcam_predictions.csv" --ear_thresh 0.22 --ear_time_thresh 3 --yawn_mar_thresh 0.55 --yawn_time_thresh 1.2
```

---

### 8.2 Run One Video

```powershell
python main.py --source "C:\path\to\video.mp4" --video_name "video.mp4" --save_csv "C:\Users\valla\ITMS-SYSTEM1\datasets\test_eval\predictions\video_predictions.csv" --ear_thresh 0.22 --ear_time_thresh 3 --yawn_mar_thresh 0.55 --yawn_time_thresh 1.2
```

Run without display:

```powershell
python main.py --source "C:\path\to\video.mp4" --video_name "video.mp4" --save_csv "C:\Users\valla\ITMS-SYSTEM1\datasets\test_eval\predictions\video_predictions.csv" --ear_thresh 0.22 --ear_time_thresh 3 --yawn_mar_thresh 0.55 --yawn_time_thresh 1.2 --no_display
```

---

### 8.3 Run Dash Female Glasses Video

```powershell
python main.py --source "C:\Users\valla\Downloads\archive_3_extracted\Dash\Dash\Female\3-FemaleGlasses.avi" --video_name "3-FemaleGlasses.avi" --save_csv "C:\Users\valla\ITMS-SYSTEM1\datasets\dash_female_eval\predictions\3-FemaleGlasses_predictions.csv" --ear_thresh 0.22 --ear_time_thresh 3 --yawn_mar_thresh 0.55 --yawn_time_thresh 1.2
```

If OpenCV cannot open AVI, convert to MP4:

```powershell
ffmpeg -i "C:\Users\valla\Downloads\archive_3_extracted\Dash\Dash\Female\3-FemaleGlasses.avi" -c:v libx264 -crf 23 -preset fast -c:a aac "C:\Users\valla\Downloads\archive_3_extracted\Dash\Dash\Female\3-FemaleGlasses.mp4"
```

Then run:

```powershell
python main.py --source "C:\Users\valla\Downloads\archive_3_extracted\Dash\Dash\Female\3-FemaleGlasses.mp4" --video_name "3-FemaleGlasses.mp4" --save_csv "C:\Users\valla\ITMS-SYSTEM1\datasets\dash_female_eval\predictions\3-FemaleGlasses_predictions.csv" --ear_thresh 0.22 --ear_time_thresh 3 --yawn_mar_thresh 0.55 --yawn_time_thresh 1.2
```

---

### 8.4 Run a Folder of Videos

```powershell
python run_dataset_eval.py --videos "C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\videos" --out_dir "C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\predictions" --combined_csv "C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\all_predictions.csv" --ear_thresh 0.22 --ear_time_thresh 3 --yawn_mar_thresh 0.55 --yawn_time_thresh 1.2
```

The resumable runner will skip videos that already have prediction CSVs.

---

## 9. CSV Outputs

### 9.1 Frame-Level Prediction CSV

Example columns:

```csv
video,time_sec,ear,mar,perclos,gaze,gaze_used,gaze_reliable,gaze_reason,attention_mode,roll,pitch,yaw,eyes_closed,closed_duration,tired_raw,tired,tired_reason,asleep,looking_away,distracted,drowsiness_active,yawning,micro_closure_count,away_signal,away_duration,distraction_reason,events
```

This is useful for:

- Debugging thresholds
- Plotting signals over time
- Finding false alerts
- Evaluating frame-level behavior

---

### 9.2 Event-Level CSV

Recommended event CSV format:

```csv
video,event,start_sec,end_sec,duration_sec,reason,mode,detection_delay
```

Example:

```csv
webcam,driver_drowsiness,12.0,15.3,3.3,eyes_closed,ear,0.12
webcam,driver_distraction,40.0,46.2,6.2,head_away,head_pose_only,0.09
webcam,yawning_pattern,100.0,240.0,140.0,three_yawns,mar,0.20
```

Event-level logs are better for final reports because they show actual alerts instead of thousands of frame rows.

---

## 10. Evaluation and Metrics

### 10.1 Current Evaluation Status

Prediction generation works.

However, the dataset JSON that was inspected contained only camera calibration:

```text
extrinsics
intrinsics
focallength
principal_point
distortion
```

It did not contain behavior labels.

So real precision/recall is not valid until a proper `labels.csv` exists.

---

### 10.2 Required Ground Truth Format

Create:

```text
C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\labels.csv
```

Format:

```csv
video,start_sec,end_sec,label
3-FemaleGlasses.avi,12.0,18.0,distracted
3-FemaleGlasses.avi,35.0,39.0,drowsiness
3-FemaleGlasses.avi,70.0,72.0,yawning
```

Allowed labels:

```text
drowsiness
tired
distracted
yawning
micro_sleep
```

Minimum validation set:

```text
5 manually labeled videos
```

Better validation set:

```text
10+ manually labeled videos
```

---

### 10.3 Run Evaluation

```powershell
python evaluate_completed_metrics.py --predictions "C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\completed_predictions.csv" --labels "C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\labels.csv"
```

Metrics calculated:

```text
TP
FP
FN
TN
accuracy
precision
recall
f1
false_alarm_rate
```

---

## 11. Threshold Tuning Guide

### Drowsiness

If sleep is missed:

```text
Increase EAR threshold: 0.22 -> 0.24
```

If false sleep alerts occur:

```text
Decrease EAR threshold: 0.22 -> 0.20
```

---

### Tired / PERCLOS

Current rule:

```text
PERCLOS is logged as tired_raw.
PERCLOS alone should not directly show TIRED!
The visible TIRED! alert should require an eye signal.
```

Allowed live `TIRED!` reasons:

```text
slow_eye_closure
heavy_eyelids
critical_perclos_with_eye_signal
```

If normal blinks trigger TIRED:

```text
Increase blink_ignore_seconds: 0.35 -> 0.45
Increase slow_closure_seconds: 0.55 -> 0.70
Increase heavy_ear_seconds: 1.0 -> 1.3
```

If TIRED does not show for slow sleep / heavy eyelids:

```text
Decrease slow_closure_seconds: 0.55 -> 0.45
Decrease heavy_ear_seconds: 1.0 -> 0.8
Increase heavy_ear_thresh: ear_thresh + 0.10 -> ear_thresh + 0.12
```

If TIRED stays on too long after waking:

```text
Decrease clear_after_open_seconds: 1.0 -> 0.8
Increase tired_off_perclos slightly only if needed
```

---

### Yawning

If yawns are missed:

```text
Lower MAR threshold: 0.55 -> 0.50
```

If talking triggers yawn:

```text
Increase MAR threshold: 0.55 -> 0.60
Increase yawn_time_thresh: 1.2 -> 1.8
```

---

### Distraction

If looking away is missed:

```text
Lower yaw threshold: 22 -> 20 -> 18
Lower pitch threshold: 20 -> 18 -> 15
```

If false alerts occur:

```text
Raise yaw threshold: 20 -> 22 -> 25
Raise distraction duration: 5 -> 6 sec
```

Recommended starting values:

```text
lookaway_yaw_thresh = 22
lookaway_pitch_thresh = 20
distraction_seconds = 5
```

---

### Gaze Reliability

If normal eyes are marked unreliable:

```text
Relax eye visibility thresholds
Use relative eye-to-face brightness instead of absolute brightness
```

If sunglasses still show gaze reliable:

```text
Tighten dark-ratio and eye-to-face brightness thresholds
```

---

## 12. Known Limitations

### 12.1 No true gaze through dark sunglasses

If dark sunglasses hide the eyes, the system cannot know the real eyeball direction.

In that case, it can only use:

```text
head pose
face direction
yaw/pitch/roll
```

So this case is impossible with only a normal RGB camera:

```text
Head facing forward but eyes looking sideways behind dark sunglasses.
```

The system should report:

```text
head_pose_only
```

not fake gaze.

---

### 12.2 Head-pose calibration can be wrong

The first 2 seconds are used as forward pose calibration.

If the driver starts while already looking sideways, the baseline can be wrong.

Best test protocol:

```text
For the first 2 seconds, face forward.
Then perform test actions.
```

---

### 12.3 Precision/recall requires labels

The system cannot honestly report precision/recall without ground-truth labels.

Prediction CSVs alone only show what the system predicted, not whether it was correct.

---

## 13. Current Completed Work

Completed:

```text
Webcam support
Single-video support
Dataset folder runner
Resume processing for dataset videos
Frame-level CSV logging
Drowsiness detection
Micro-sleep detection
MAR-based yawn detection
Yawning pattern requirement added: 3 yawns / 5 min
PERCLOS raw logging as tired_raw
Tired live filter integrated
Sustained PERCLOS removed as direct TIRED trigger
TIRED now triggers only from slow_eye_closure, heavy_eyelids, or critical_perclos_with_eye_signal
Adaptive gaze reliability concept
Sunglasses fallback concept
Head-pose-only distraction mode
Forward pose calibration
Looking-away timer
Distraction alert after 5 sec
Evaluation scripts created
Debug scripts created
```

Still required:

```text
Verify final gaze_reliability.py relative-brightness version on normal eyes, clear glasses, and dark sunglasses
Verify tired_reason is written to CSV
Verify event-level CSV logging is working
Create real labels.csv manually
Run actual precision/recall
Tune thresholds from metrics
```

---

## 14. Recommended Next Steps

Do these in order:

```text
1. Test gaze reliability using normal eyes, clear glasses, and dark sunglasses.
2. Confirm `tired_reason` appears in the UI and CSV.
3. Confirm yawn pattern uses 300 sec, not 5 sec.
4. Confirm event-level CSV is created.
5. Manually label 5 short videos.
6. Run precision/recall evaluation.
7. Tune thresholds from actual false positives/false negatives.
8. Lock the driver-state module.
9. Move to next ITMS models: seatbelt, smoking, drinking, phone usage.
```

---

## 15. Future Integration With ITMS

This module should eventually output event JSON to the ITMS backend/frontend.

Example event:

```json
{
  "camera": "cabin",
  "event": "driver_distraction",
  "severity": "warning",
  "duration_sec": 6.2,
  "reason": "head_away",
  "mode": "head_pose_only",
  "timestamp": "2026-07-07T10:30:00",
  "detection_delay": 0.12
}
```

Full ITMS driver monitoring architecture:

```text
Cabin Camera
    |
    +--> Driver State Module
    |       +--> drowsiness
    |       +--> micro_sleep
    |       +--> tired
    |       +--> yawning
    |       +--> distraction
    |
    +--> YOLO Cabin Behavior Module
    |       +--> seatbelt
    |       +--> phone usage
    |       +--> smoking
    |       +--> drinking
    |
    +--> Alert Fusion Layer
            +--> event JSON
            +--> frontend dashboard
            +--> backend logging
```

---

## 16. Final Project Description

The driver-state module performs real-time driver fatigue and distraction monitoring using OpenCV and MediaPipe FaceMesh. It extracts facial landmarks to compute EAR for eye closure, MAR for yawning, gaze score for eye direction, and head pose angles for face direction.

The system is adaptive. When the eyes and iris are visible, it uses both gaze and head pose to estimate attention. When gaze is unreliable due to sunglasses, glare, or eye occlusion, it automatically switches to head-pose-only mode. This prevents false gaze interpretation when MediaPipe estimates iris points on glasses instead of real eyes.

Drowsiness is triggered when the eyes remain closed continuously for at least 3 seconds. Micro-sleep is triggered when three eye-closure events of at least 2 seconds occur within one minute. Tiredness is handled with a two-layer design: raw PERCLOS is logged as `tired_raw`, while the visible `TIRED!` alert is triggered only by slow eye closure, heavy eyelids, or critical PERCLOS combined with a bad eye signal. High PERCLOS alone with normal/open eyes does not directly trigger `TIRED!`. Yawning is detected using MAR, with a stronger yawning-pattern event triggered after three yawns within five minutes. Distraction is triggered when the driver looks away continuously for at least 5 seconds using adaptive gaze-head fusion.

The module logs both frame-level predictions and event-level alerts for evaluation. Frame logs include EAR, MAR, PERCLOS, `tired_raw`, filtered `tired`, `tired_reason`, gaze reliability, head pose, away signal, away duration, and detected events. Event logs store event type, reason, mode, duration, and detection delay.

