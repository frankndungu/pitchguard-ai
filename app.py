import json
import tempfile
from pathlib import Path

import streamlit as st

from src.event_engine import (
    MovementSample,
    classify_analysis_outcome,
    detect_slowdowns,
)
from src.event_explainer import explain_event
from src.pose_estimation import (
    OpenCVZooPoseEstimator,
    draw_annotated_frame,
    select_reliable_highlight,
)
from src.video_analysis import (
    MAX_VIDEO_SECONDS,
    analyze_video,
    select_nearby_frames,
)


ROOT = Path(__file__).parent
SYNTHETIC_MOVEMENT_PATH = ROOT / "data" / "synthetic_movement.json"


def load_synthetic_events():
    with SYNTHETIC_MOVEMENT_PATH.open(encoding="utf-8") as sample_file:
        sample = json.load(sample_file)
    movements = [
        MovementSample(
            timestamp_seconds=item["timestamp_seconds"],
            speed_body_lengths_per_second=item[
                "speed_body_lengths_per_second"
            ],
        )
        for item in sample["samples"]
    ]
    return sample["title"], detect_slowdowns(movements)


def render_events(events):
    for event in events:
        st.subheader(f"Candidate movement change at {event['timestamp']}")
        st.write(event["observation"])
        st.write(
            f"Detector confidence: {event['confidence']:.0%} "
            "(heuristic evidence score, not a probability)"
        )


st.set_page_config(page_title="PitchGuard AI", page_icon=None)
st.title("PitchGuard AI")
st.write(
    "A local prototype that highlights observable changes in player movement. "
    "It does not diagnose injuries."
)
st.warning(
    "Movement anomalies are not injury diagnoses. Review all candidate events "
    "in the video context; this prototype is not medical advice."
)

uploaded_video = st.file_uploader(
    "Choose a short football video from your computer",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is None:
    st.info("No video selected. You can still try the synthetic example below.")
else:
    video_bytes = uploaded_video.getvalue()
    st.video(video_bytes)
    st.info(
        f"Analysis is limited to the first {MAX_VIDEO_SECONDS} seconds. "
        "The full uploaded video is displayed; the analyzed interval will be "
        "shown after analysis."
    )
    pose_enabled = st.checkbox(
        "Enable pose skeletons in frame inspection",
        value=True,
        help=(
            "Uses local OpenCV Zoo ONNX models from the models folder. "
            "The existing movement analysis continues if the models are absent."
        ),
    )
    if st.button("Analyze video"):
        temp_path = None
        try:
            pose_estimator = None
            if pose_enabled:
                try:
                    pose_estimator = OpenCVZooPoseEstimator(
                        ROOT
                        / "models"
                        / "person_detection_mediapipe_2023mar_int8bq.onnx",
                        ROOT
                        / "models"
                        / "pose_estimation_mediapipe_2023mar_int8bq.onnx",
                    )
                except (FileNotFoundError, RuntimeError) as error:
                    st.warning(
                        "Pose skeletons are unavailable. Video and movement "
                        f"analysis will continue without them. Details: {error}"
                    )
            suffix = Path(uploaded_video.name).suffix or ".video"
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as temp_video:
                temp_video.write(video_bytes)
                temp_path = Path(temp_video.name)
            with st.spinner("Analyzing locally. This may take a little while."):
                analysis = analyze_video(
                    temp_path, pose_estimator=pose_estimator
                )
            movement_by_track = analysis.movement_by_track
            events = []
            for track_id, track_samples in movement_by_track.items():
                events.extend(
                    {
                        **event,
                        "track_id": track_id,
                    }
                    for event in detect_slowdowns(track_samples)
                )
            events.sort(key=lambda event: event["timestamp_seconds"])
            explained_events = [explain_event(event) for event in events]
            st.subheader("Video analysis")
            st.caption(
                f"Full video duration: "
                f"{analysis.video_duration_seconds:.1f} seconds. "
                f"Analyzed interval: 00:00 to "
                f"{analysis.analyzed_duration_seconds:.1f} seconds."
            )
            if analysis.pose_error:
                st.warning(
                    "Pose estimation stopped for this clip. Movement analysis "
                    f"continued without further skeletons. Details: {analysis.pose_error}"
                )
            outcome = classify_analysis_outcome(
                analysis.people_detected, movement_by_track, events
            )
            if outcome == "no_people":
                st.info(
                    "No people were detected in the analyzed interval. Try a "
                    "clip with clearly visible players."
                )
            elif outcome == "insufficient_samples":
                st.info(
                    "People were detected, but there were not enough "
                    "continuous movement samples to assess a change."
                )
            elif outcome == "no_events":
                st.info(
                    "Analysis complete. No candidate movement changes were "
                    "detected in the analyzed interval."
                )
            else:
                render_events(explained_events)
                event_options = list(range(len(explained_events)))
                selected_event_index = st.selectbox(
                    "Inspect frames near a candidate event",
                    event_options,
                    format_func=lambda index: (
                        f"{explained_events[index]['timestamp']} "
                        f"(track {explained_events[index]['track_id']})"
                    ),
                )
                selected_event = explained_events[selected_event_index]
                st.subheader(
                    f"Frame inspection near {selected_event['timestamp']}"
                )
                st.write(selected_event["observation"])
                st.write(
                    f"Detector confidence: "
                    f"{selected_event['confidence']:.0%} "
                    "(heuristic evidence score, not a probability)"
                )
                nearby_frames = select_nearby_frames(
                    analysis.frames,
                    selected_event["timestamp_seconds"],
                )
                if not nearby_frames:
                    st.info("No sampled frames are available for inspection.")
                else:
                    columns = st.columns(len(nearby_frames))
                    for column, sampled_frame in zip(
                        columns, nearby_frames
                    ):
                        with column:
                            st.caption(
                                f"Frame at {sampled_frame.timestamp_seconds:.1f} seconds"
                            )
                            if sampled_frame.image_bytes is None:
                                st.info(
                                    sampled_frame.preview_error
                                    or "This sampled frame is unavailable."
                                )
                                continue
                            highlight = select_reliable_highlight(
                                sampled_frame.people,
                                selected_event["track_id"],
                            )
                            try:
                                annotated_frame = draw_annotated_frame(
                                    sampled_frame.image_bytes,
                                    sampled_frame.people,
                                    highlight,
                                )
                            except ValueError as error:
                                st.info(
                                    "This sampled frame could not be rendered: "
                                    f"{error}"
                                )
                                continue
                            st.image(annotated_frame, channels="BGR")
                            selected_track = next(
                                (
                                    person
                                    for person in sampled_frame.people
                                    if person.track_id
                                    == selected_event["track_id"]
                                ),
                                None,
                            )
                            if (
                                selected_track is not None
                                and not selected_track.association_reliable
                            ):
                                st.caption(
                                    "Tracked-player association is uncertain "
                                    "in this frame; no player is highlighted."
                                )
                            elif selected_track is None:
                                st.caption(
                                    "The event track was not detected in this "
                                    "frame; no player is highlighted."
                                )
                            elif (
                                pose_enabled
                                and selected_track.pose is None
                            ):
                                st.caption(
                                    "No usable pose landmarks were available "
                                    "for this tracked player in this frame."
                                )
        except (OSError, ValueError, RuntimeError) as error:
            st.error(
                "Video analysis failed. Check that the file is a readable "
                f"video and try a short clip. Details: {error}"
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

st.divider()
st.header("Try a synthetic example")
st.write(
    "This example uses synthetic movement measurements to exercise the same "
    "event detector and explanation shown for video analysis."
)
try:
    example_title, synthetic_events = load_synthetic_events()
    st.caption(example_title)
    render_events(
        [explain_event(event) for event in synthetic_events]
    )
except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
    st.error(f"The synthetic example could not be loaded: {error}")
