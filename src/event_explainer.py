def format_timestamp(timestamp_seconds: float) -> str:
    total_seconds = max(0, int(timestamp_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def explain_event(event: dict) -> dict:
    drop_percent = round(event["drop_ratio"] * 100)
    return {
        **event,
        "timestamp": format_timestamp(event["timestamp_seconds"]),
        "observation": (
            "Tracked movement speed fell from "
            f"{event['baseline_speed']:.2f} to "
            f"{event['observed_speed']:.2f} body lengths per second "
            f"({drop_percent}% decrease, comparing three sampled movement "
            "measurements before and after the change). This is an observable "
            "movement change, not evidence of a specific injury."
        ),
    }
