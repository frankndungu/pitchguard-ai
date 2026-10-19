from dataclasses import dataclass
from statistics import median
from typing import Literal, Sequence


EVENT_WINDOW_SIZE = 3
MINIMUM_EVENT_SAMPLES = 2 * EVENT_WINDOW_SIZE
AnalysisOutcome = Literal[
    "no_people", "insufficient_samples", "no_events", "events"
]


@dataclass(frozen=True)
class MovementSample:
    timestamp_seconds: float
    speed_body_lengths_per_second: float


def detect_slowdowns(
    samples: Sequence[MovementSample],
    *,
    window_size: int = EVENT_WINDOW_SIZE,
    minimum_baseline_speed: float = 0.4,
    minimum_drop_ratio: float = 0.6,
) -> list[dict]:
    """Find sustained drops from a prior median movement speed."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if not 0 <= minimum_drop_ratio <= 1:
        raise ValueError("minimum_drop_ratio must be between 0 and 1")
    for sample in samples:
        if sample.timestamp_seconds < 0:
            raise ValueError("timestamps must not be negative")
        if sample.speed_body_lengths_per_second < 0:
            raise ValueError("movement speeds must not be negative")

    events = []
    next_start = window_size
    while next_start + window_size <= len(samples):
        baseline = [
            sample.speed_body_lengths_per_second
            for sample in samples[next_start - window_size : next_start]
        ]
        current = [
            sample.speed_body_lengths_per_second
            for sample in samples[next_start : next_start + window_size]
        ]
        baseline_speed = median(baseline)
        current_speed = median(current)

        if baseline_speed >= minimum_baseline_speed:
            drop_ratio = (baseline_speed - current_speed) / baseline_speed
            first_current_is_low = (
                current[0] <= baseline_speed * (1.0 - minimum_drop_ratio)
            )
            if drop_ratio >= minimum_drop_ratio and first_current_is_low:
                consistent_samples = sum(
                    speed <= baseline_speed * 0.5 for speed in current
                )
                drop_evidence = min(
                    1.0,
                    (drop_ratio - minimum_drop_ratio)
                    / (1.0 - minimum_drop_ratio)
                    if minimum_drop_ratio < 1
                    else 1.0,
                )
                confidence = 0.5 * drop_evidence + 0.5 * (
                    consistent_samples / window_size
                )
                events.append(
                    {
                        "timestamp_seconds": samples[
                            next_start
                        ].timestamp_seconds,
                        "baseline_speed": baseline_speed,
                        "observed_speed": current_speed,
                        "drop_ratio": drop_ratio,
                        "confidence": confidence,
                    }
                )
                next_start += window_size
            else:
                next_start += 1
        else:
            next_start += 1

    return events


def classify_analysis_outcome(
    people_detected: bool,
    movement_by_track: dict[int, list[MovementSample]],
    events: Sequence[dict],
) -> AnalysisOutcome:
    if not people_detected:
        return "no_people"
    if not any(
        len(samples) >= MINIMUM_EVENT_SAMPLES
        for samples in movement_by_track.values()
    ):
        return "insufficient_samples"
    return "events" if events else "no_events"
