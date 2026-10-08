"""Render saved real-video evidence without rerunning detection models."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from sperm_pipeline.pipeline import SpermAnalysisPipeline


def render(source, directory, destination):
    destination.mkdir(parents=True, exist_ok=True)
    meta = directory / "meta"
    rows = list(csv.DictReader((meta / "inference_tracking.csv").open()))
    quality = json.loads((meta / "tracking_quality_summary.json").read_text())
    lifecycle = json.loads((meta / "application_id_lifecycle.json").read_text())
    pipeline = object.__new__(SpermAnalysisPipeline)
    review = pipeline._write_overlap_identity_review(
        str(source), str(destination), quality["fps"], rows, lifecycle, quality["frames"])
    print(review, flush=True)
    registry = json.loads((meta / "application_identity_registry.json").read_text())
    diagnostics = json.loads((meta / "identity_assignment_diagnostics.json").read_text())
    observations = {(r["frame"], r.get("detection_index")): r for r in diagnostics if r.get("current_bbox")}
    accepted = [r for r in diagnostics if r.get("accepted") and r.get("current_bbox")]
    cap = cv2.VideoCapture(str(source))

    def sheet(name, frames, ids, boxes):
        lo = np.min(np.array(boxes)[:, :2], axis=0) - 60
        hi = np.max(np.array(boxes)[:, 2:], axis=0) + 60
        panels = []
        for f in frames:
            f = max(0, min(quality["frames"] - 1, int(f)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            for r in accepted:
                if r["frame"] != f:
                    continue
                box = tuple(int(v) for v in r["current_bbox"])
                app = r["candidate_application_id"]
                color = (0, 255, 0) if app in ids else (180, 180, 180)
                cv2.rectangle(frame, box[:2], box[2:], color, 1)
                cv2.putText(frame, f'A{app} B{r.get("byte_track_id")}',
                            (box[0], max(15, box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            x1, y1 = max(0, int(lo[0])), max(0, int(lo[1]))
            x2, y2 = min(frame.shape[1], int(hi[0])), min(frame.shape[0], int(hi[1]))
            frame = frame[y1:y2, x1:x2]
            if not frame.size:
                continue
            frame = cv2.resize(frame, (480, 360))
            cv2.rectangle(frame, (0, 0), (480, 28), (0, 0, 0), -1)
            cv2.putText(frame, f'{name} f={f} IDs={ids}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            panels.append(frame)
        while len(panels) % 3:
            panels.append(np.zeros_like(panels[0]))
        image = np.concatenate([np.concatenate(panels[i:i+3], axis=1) for i in range(0, len(panels), 3)])
        cv2.imwrite(str(destination / f"{name}.jpg"), image)

    for group in registry["overlap_groups"]:
        if group["state"] != "EXPIRED":
            continue
        ids = group.get("all_identity_ids", group["identity_ids"])
        start, last = group["start_frame"], group["last_frame"]
        boxes = [r["bbox"] for r in group["pre_overlap_state"].values()]
        step = max(1, round(quality["fps"] * .5))
        sheet(f'group_{group["overlap_group_id"]}',
              [start-1, start, last, last+1, last+step, last+2*step], ids, boxes)
    split = [r for r in accepted if r.get("candidate_application_id") in (66, 68)]
    if split:
        sheet("BT1014", [460, 465, 469, 470, 471, 475], [66, 68], [r["current_bbox"] for r in split])
    creations = [r for r in accepted if r["assignment_type"] == "NEW_HIGH_ID" and r["frame"] > 0]
    for row in creations:
        app, frame = row["candidate_application_id"], row["frame"]
        sheet(f"new_APP{app:03d}", [frame-4, frame, frame+4], [app], [row["current_bbox"]])
    for start in range(0, len(creations), 6):
        cards = []
        for row in creations[start:start+6]:
            im = cv2.imread(str(destination / f'new_APP{row["candidate_application_id"]:03d}.jpg'))
            cards.append(cv2.resize(im, (1080, 270)))
        cv2.imwrite(str(destination / f"new_event_page_{start//6+1:02d}.jpg"), np.concatenate(cards))
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(args.source, args.directory, args.destination)
