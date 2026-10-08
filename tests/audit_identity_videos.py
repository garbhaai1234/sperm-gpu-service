"""Offline evidence audit; run explicitly, never collected by pytest.

Retains per-event evidence rather than treating proximity flags as ground truth.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def audit(directory):
    meta = directory / "meta"
    quality = load(meta / "tracking_quality_summary.json")
    registry = load(meta / "application_identity_registry.json")
    diagnostics = load(meta / "identity_assignment_diagnostics.json")
    lifecycle = load(meta / "application_id_lifecycle.json")
    events = load(meta / "application_identity_quality_events.json")
    fps = quality["fps"]
    by_detection, accepted, by_id = defaultdict(list), defaultdict(list), defaultdict(list)
    observations = {}
    for row in diagnostics:
        key = (row["frame"], row.get("detection_index"))
        by_detection[key].append(row)
        if row.get("current_bbox") is not None:
            observations[key] = row
        if row.get("accepted") and row.get("candidate_application_id") is not None:
            accepted[row["candidate_application_id"]].append(row)
    for row in lifecycle:
        by_id[row.get("application_id")].append(row)

    def state_at(app, frame):
        states = {"ID_CONFIRMED": "CONFIRMED", "ID_LOST": "LOST", "ID_OCCLUDED": "OCCLUDED",
                  "ID_TERMINATED": "TERMINATED", "ID_UNRESOLVED": "UNRESOLVED",
                  "ID_LOW_SUPPORT": "LOW_SUPPORT", "ID_CREATED": "CONFIRMED",
                  "ID_REIDENTIFIED": "CONFIRMED"}
        rows = [r for r in by_id[app] if r["frame"] <= frame and r["event"] in states]
        return states[rows[-1]["event"]] if rows else "UNKNOWN"

    new_events = []
    for row in diagnostics:
        if row.get("assignment_type") != "NEW_HIGH_ID" or row["frame"] == 0:
            continue
        rivals = [dict(r, prior_state=state_at(r.get("candidate_application_id"), row["frame"] - 1))
                  for r in by_detection[(row["frame"], row["detection_index"]) ]
                  if r.get("candidate_application_id") != row["candidate_application_id"]
                  and r.get("distance_to_high_anchor") is not None]
        rivals.sort(key=lambda r: (r.get("predicted_center_distance") or 0)
                    + (r.get("distance_to_high_anchor") or 0))
        plausible = [r for r in rivals if r.get("frame_gap", 9999) <= fps * 3
                     and r.get("bbox_size_ratio", 9999) <= 2
                     and r.get("iou_to_high_anchor", 0) >= .2
                     and r["prior_state"] != "TERMINATED"]
        strong = [r for r in plausible if r.get("byte_track_id") == r.get("previous_byte_track_id")
                  and r.get("frame_gap") == 1
                  and r.get("predicted_center_distance", 9999) <= 80]
        classification = "LIKELY_FRAGMENTATION" if strong else "AMBIGUOUS" if plausible else "INSUFFICIENT_EVIDENCE"
        new_events.append({"frame": row["frame"], "application_id": row["candidate_application_id"],
                           "reason": row["rejection_reason"], "classification": classification,
                           "basis": "Diagnostic geometry only; visibility and biological identity require image review.",
                           "detection": row, "plausible_candidates": plausible,
                           "rejected_candidates": rivals})

    expired = []
    for group in registry["overlap_groups"]:
        if group["state"] != "EXPIRED":
            continue
        members = group.get("all_identity_ids", group["identity_ids"])
        start, last = group["start_frame"], group["last_frame"]
        end = min(quality["frames"] - 1, last + int(fps * 3) + 1)
        related = [r for r in diagnostics if start <= r["frame"] <= end
                   and r.get("candidate_application_id") in members]
        merged = [observations.get((r["frame"], r["detection_index"]))
                  for r in group.get("merged_detection_indices", [])]
        trajectories = {str(app): [r for r in accepted[app] if start - fps <= r["frame"] <= end]
                        for app in members}
        replacements = [e for e in new_events if start <= e["frame"] <= end
                        and any(r.get("candidate_application_id") in members for r in e["rejected_candidates"])]
        expired.append({"group": group, "member_states_at_start": {str(i): state_at(i, start) for i in members},
                        "member_states_after_overlap": {str(i): state_at(i, last + 1) for i in members},
                        "member_lifecycle": {str(i): by_id[i] for i in members},
                        "member_detections": trajectories, "merged_detections": merged,
                        "post_overlap_candidate_evidence": related,
                        "rejection_counts": dict(Counter(r["rejection_reason"] for r in related if not r.get("accepted"))),
                        "possible_replacement_events": [{k: e[k] for k in ("frame", "application_id", "reason", "classification")}
                                                        for e in replacements],
                        "classification": "INSUFFICIENT_EVIDENCE"})
        expired[-1]["predicted_positions_first_post_overlap"] = {
            str(i): [record["center"][axis] + record["velocity"][axis] * (last + 1 - record["frame"])
                     for axis in range(2)]
            for i in members for record in [group["pre_overlap_state"][str(i)]]}
        expired[-1]["nearby_subsequent_creations"] = [
            {"frame": e["frame"], "application_id": e["application_id"],
             "candidate_evidence": [r for r in e["rejected_candidates"]
                                    if r.get("candidate_application_id") in members
                                    and r.get("distance_to_high_anchor", 9999) <= 100]}
            for e in replacements if any(r.get("candidate_application_id") in members
                                         and r.get("distance_to_high_anchor", 9999) <= 100
                                         for r in e["rejected_candidates"])]

    candidate_frames = defaultdict(set)
    for row in events:
        if row.get("event") == "UNRESOLVED_CANDIDATE":
            candidate_frames[row["candidate_id"]].add(row["frame"])
    episodes = []
    for cid, frames in candidate_frames.items():
        frames = sorted(frames)
        rows = [r for r in events if r.get("candidate_id") == cid]
        possible = set(i for r in rows for i in r.get("possible_identities", []))
        following = [r for i in possible for r in accepted[i]
                     if frames[-1] < r["frame"] <= frames[-1] + fps]
        episodes.append({"candidate_id": cid, "start": frames[0], "end": frames[-1],
                         "observed_frames": len(frames), "duration_frames": frames[-1] - frames[0] + 1,
                         "duration_seconds": (frames[-1] - frames[0] + 1) / fps,
                         "possible_identities": sorted(possible),
                         "expired": any(r["event"] == "UNRESOLVED_CANDIDATE_EXPIRED" for r in rows),
                         "resolved": any(r["event"] == "UNRESOLVED_CANDIDATE_RESOLVED" for r in rows),
                         "original_identity_seen_within_one_second": sorted({r["candidate_application_id"] for r in following}),
                         "note": "Later visibility alone does not establish correct resolution."})
        observation_rows = [observations.get((r["frame"], r.get("detection_index")))
                            for r in rows if r["event"] == "UNRESOLVED_CANDIDATE"]
        observation_rows = [r for r in observation_rows if r is not None]
        latest = [r for r in observation_rows if r["frame"] == frames[-1]]
        last_bytes = {r["byte_track_id"] for r in latest if r.get("byte_track_id") is not None}
        follow = sorted([r for rs in accepted.values() for r in rs
                         if frames[-1] < r["frame"] <= frames[-1] + fps
                         and r.get("byte_track_id") in last_bytes], key=lambda r: r["frame"])
        exact_resolutions = [r for r in rows if r["event"] == "UNRESOLVED_CANDIDATE_RESOLVED"]
        outcome = "PENDING_AT_END" if not episodes[-1]["expired"] else "EXPIRED_WITHOUT_LINKED_RECOVERY"
        if follow:
            outcome = ("ORIGINAL_ID_RECOVERY" if follow[0]["candidate_application_id"] in possible
                       else "NEW_ID_AFTER_UNRESOLVED" if follow[0]["assignment_type"] == "NEW_HIGH_ID"
                       else "OTHER_EXISTING_ID")
        if exact_resolutions:
            outcome = "ORIGINAL_ID_RECOVERY"
        episodes[-1].update(observation_count=len(observation_rows),
                            repeated_same_frame_observations=len(observation_rows) - len(frames),
                            byte_track_ids=sorted({r["byte_track_id"] for r in observation_rows
                                                   if r.get("byte_track_id") is not None}),
                            outcome=outcome, following_same_byte_assignment=follow[:1],
                            biological_correctness="NOT_ESTABLISHED_WITHOUT_GROUND_TRUTH")
    ambiguity_frames = defaultdict(set)
    for row in diagnostics:
        if row.get("assignment_type") == "AMBIGUOUS_REJECT":
            ambiguity_frames[(row.get("byte_track_id"), row.get("candidate_application_id"))].add(row["frame"])
    ambiguity_episodes = []
    for key, frames in ambiguity_frames.items():
        runs = []
        for frame in sorted(frames):
            if not runs or frame > runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(frame)
        for run in runs:
            ambiguity_episodes.append({"byte_id": key[0], "application_id": key[1], "start": run[0], "end": run[-1],
                                       "frames": len(run), "seconds": len(run) / fps})
    stale = []
    identities = {r["application_id"]: r for r in registry["identities"]}
    for group in registry["overlap_groups"]:
        if group["state"] in {"RESOLVED", "EXPIRED"}:
            continue
        viable = [identities[i] for i in group["identity_ids"] if identities[i]["state"] != "TERMINATED"]
        if not viable or (len(viable) == 1 and viable[0]["state"] == "CONFIRMED"
                          and viable[0]["last_high_conf_frame"] > group["last_frame"]):
            stale.append(group["overlap_group_id"])
    csv_rows = list(csv.DictReader((meta / "inference_tracking.csv").open()))
    video_ids = {int(r["application_id"]) for r in csv_rows}
    summary = load(meta / "summary.json")
    api_ids = {int(r["sperm_id"]) for r in summary}
    trajectory_ids = {int(i) for i in load(meta / "trajectories.json")["trajectories"]}
    casa_ids = {int(r["sperm_id"]) for r in summary if r.get("motility_features")}
    morphology_ids = {int(r["sperm_id"]) for r in summary if r.get("morphology_features")}
    contract = {"video_equals_api": video_ids == api_ids, "api_equals_trajectory": api_ids == trajectory_ids,
                "api_equals_casa": api_ids == casa_ids, "morphology_subset_of_canonical": morphology_ids <= api_ids,
                "video_local_namespace_starts_at_one": min(video_ids, default=1) == 1}
    return {"quality": quality, "canonical_contract": contract,
            "unresolved_outcomes": dict(Counter(e["outcome"] for e in episodes)),
            "group_states": dict(Counter(g["state"] for g in registry["overlap_groups"])),
            "stale_groups": stale, "expired_groups": expired, "new_id_events": new_events,
            "new_id_classifications": dict(Counter(r["classification"] for r in new_events)),
            "unresolved_episodes": episodes, "ambiguity_episodes": ambiguity_episodes,
            "split_history": [r for r in diagnostics if r.get("byte_track_id") == 1014
                              or r.get("candidate_application_id") in (66, 68)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("quality", "group_states", "stale_groups", "new_id_classifications")}, indent=2))
