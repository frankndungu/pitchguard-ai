# PitchGuard AI

PitchGuard AI is a local hackathon prototype for reviewing observable player
movement changes in short football clips. It looks for one pattern: a tracked
person's movement speed falling sharply and staying lower for several sampled
measurements. A flagged change is only a prompt for human review. It is not an
injury prediction or diagnosis.

## How it works

The Streamlit dashboard accepts a local video upload and displays the clip.
When analysis is requested, the app temporarily writes the upload to a local
temporary file and removes that file after processing. No video is sent to a
service or retained by the app.

OpenCV's built-in HOG people detector finds people in two sampled frames per
second. A nearest-centre tracker associates detections between adjacent
sampled frames. The app estimates image-plane movement speed as bounding-box
heights travelled per second. The event detector compares the median of three
movement measurements with the next three. It flags a candidate when the
baseline median is at least 0.4 body lengths per second and the later median
is at least 60 percent lower.

The dashboard can also show an event-linked inspection of up to three sampled
frames near a candidate timestamp. It always draws the existing HOG boxes.
When the optional local pose models are enabled, it overlays visible pose
landmarks and connections only when a pose detection matches a HOG detection
unambiguously. A tracked player is highlighted only after repeated unique
tracker matches. Uncertain associations are not highlighted. The uploaded
video remains available separately in its original form.

Video processing is limited to the first 30 seconds, sampled at up to two
frames per second. Frame previews are held in memory for the current analysis
and are not written as generated videos.

The displayed confidence is a heuristic evidence score based on the size of
the speed decrease and how consistently the later measurements are low. It
is not a calibrated probability or a measure of medical certainty. The
explanation reports the timestamp and the measured before-and-after speeds.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then
install the declared packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependencies are Streamlit and OpenCV's headless Python package. They run
locally; no account, paid API, or cloud service is used. Pose estimation uses
OpenCV DNN on the CPU and does not add a Python dependency.

## Run the app

From the project root, start Streamlit:

```powershell
python -m streamlit run app.py
```

Choose a short supported video file in the dashboard and select **Analyze
video**. The app displays the uploaded video and the candidate events found
during local processing. The dashboard reports both the full video duration
and the analyzed interval. If the detector cannot open or decode a clip, the
dashboard displays an error instead of treating the analysis as successful.

## Optional pose models

Pose skeletons require two OpenCV Zoo ONNX model files. The app does not
download them automatically. From PowerShell, create the local model folder
and download the quantized files from the public OpenCV Hugging Face model
repositories:

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest `
  -Uri "https://huggingface.co/opencv/person_detection_mediapipe/resolve/main/person_detection_mediapipe_2023mar_int8bq.onnx" `
  -OutFile "models\person_detection_mediapipe_2023mar_int8bq.onnx"
Invoke-WebRequest `
  -Uri "https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar_int8bq.onnx" `
  -OutFile "models\pose_estimation_mediapipe_2023mar_int8bq.onnx"
```

Both model repositories describe their files as Apache License 2.0. Review
the [person detector license](https://github.com/opencv/opencv_zoo/blob/main/models/person_detection_mediapipe/LICENSE)
and [pose model license](https://github.com/opencv/opencv_zoo/blob/main/models/pose_estimation_mediapipe/LICENSE)
before redistribution. The pose adapter follows the OpenCV Zoo model input
and output conventions. Model files are optional, ignored by Git, and are
never copied into the project by setup. If a model is missing or cannot be
loaded, the app explains that pose is unavailable and continues with HOG
detection and movement analysis.

Pose inference runs on the OpenCV CPU backend. It adds work for each sampled
frame and may be slow on longer clips or less capable computers. No GPU is
required. Mocked tests exercise the model interface without downloading
weights; they do not validate inference quality on real football footage.

## Reproduce the demo

The dashboard includes a synthetic sustained slowdown example. It loads
`data/synthetic_movement.json`, runs those measurements through the same event
detector and explanation used for video analysis, and displays the resulting
timestamp, evidence, and heuristic score. This demonstrates the event
pipeline without requiring a video or model weights.

Run the event tests from the project root with Python's standard test runner:

```powershell
python -m unittest discover -s tests -v
```

## Video permissions

Only use videos you have permission to process. Uploaded clips are processed
locally by this prototype, briefly stored in a temporary file for decoding,
and removed after analysis. Sampled frames are retained in memory only for the
current analysis. Videos and model weights are excluded from Git.

## Limitations and safety

This prototype detects people rather than identifying players. HOG detection
and nearest-centre tracking can fail with occlusion, small or overlapping
players, camera movement, cuts, low-quality footage, and rapid movement.
Pose estimation can also miss or misplace landmarks, particularly with
overlapping players, unusual camera angles, or partial bodies. Frame-level
pose-to-box association and cross-frame track identity are heuristic. The app
withholds the event highlight when the tracker association is not sufficiently
clear, but this does not make the remaining associations error-free.

Movement is measured in image coordinates, so perspective, zoom, and camera
motion can affect the result. Missed detections can break a track. Thresholds
are simple heuristics and are not validated for performance or clinical use.
The models and mocked tests have not been validated on real football footage.
The system cannot identify a specific injury and does not diagnose injuries.
Every candidate event requires human review in the context of the full video.
