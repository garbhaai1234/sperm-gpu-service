"""Fast linker-only replay of recorded detector observations, not a model replay.

Missing observation indices are reported explicitly. Full-video replay remains
the acceptance check because historical logs may omit blocked detections.
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from sperm_pipeline.pipeline import SpermAnalysisPipeline


def replay(source, destination):
    meta = source / "meta"
    original = json.loads((meta / "identity_assignment_diagnostics.json").read_text())
    quality = json.loads((meta / "tracking_quality_summary.json").read_text())
    observations = defaultdict(dict)
    for row in original:
        if row.get("current_bbox") is None:
            continue
        observations[row["frame"]][row["detection_index"]] = {
            "tracker_id": row.get("byte_track_id"), "confidence": row.get("detector_score", 0),
            "center": tuple(row["current_center"]), "bbox": tuple(row["current_bbox"]),
        }
    pipeline = object.__new__(SpermAnalysisPipeline)
    width, height = quality["resolution"]
    state = pipeline._initialize_application_identity_state(width, height, quality["fps"])
    rows, missing = [], []
    snapshots = []
    for frame in range(quality["frames"]):
        observed = observations[frame]
        if observed and len(observed) != max(observed) + 1:
            missing.append({"frame": frame, "missing": sorted(set(range(max(observed) + 1)) - set(observed))})
        items = [observed[i] for i in sorted(observed)]
        links = pipeline._link_inference_application_ids(items, state, frame)
        for item, link in zip(items, links):
            if link is not None:
                rows.append({"frame_index": frame, "application_id": link[0], "tracker_id": item["tracker_id"]})
        snapshots.append({"frame": frame, "groups": [dict(g) for g in state["overlap_groups"].values()]})
    result = pipeline._build_tracking_quality_summary(quality["frames"], quality["fps"], width, height, rows, state)
    destination.mkdir(parents=True, exist_ok=True)
    result["missing_observation_indices"] = missing
    (destination / "summary.json").write_text(json.dumps(result, indent=2))
    for name, data in (("diagnostics", state["identity_assignment_diagnostics"]),
                       ("lifecycle", state["lifecycle_events"]), ("quality_events", state["quality_events"]),
                       ("rows", rows), ("group_snapshots", snapshots)):
        (destination / f"{name}.json").write_text(json.dumps(data, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "missing_observation_indices"}))
    print(f"Frames with missing logged observation indices: {len(missing)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    replay(args.source, args.destination)
