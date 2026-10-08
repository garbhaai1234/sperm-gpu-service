"""Explicit full-video validation runner; not collected by pytest.

Run from the repository root with PYTHONPATH=sperm_pipeline.
Original artifacts are never overwritten.
"""
import json
import logging
import argparse
import gc
import hashlib
import shutil
import cv2
import torch
from pathlib import Path

from sperm_pipeline.pipeline import SpermAnalysisPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="verified")
    parser.add_argument("--identity-pass", action="store_true",
                        help="Rerun every source frame through Mask R-CNN/ByteTrack, identity, morphology and CASA; reuse the independent processed-mask video.")
    args = parser.parse_args()
    root = Path("outputs/identity_registry_validation")
    sources = json.loads((root / "current_validation_metrics.json").read_text())["videos"]
    for name, baseline in sources.items():
        output = root / name.replace("current_", args.prefix + "_")
        if (output / "replay_complete.json").exists():
            print(f"Already complete: {output}", flush=True)
            continue
        print(f"Starting {name}: {baseline['source']}", flush=True)
        pipeline = SpermAnalysisPipeline(
            detection_model_path="detection_best.pt",
            maskrcnn_model_path="best_resnet50_transfer_from_101.pth",
            hnk_model_path="HNK_best.pt", device="cuda",
            frame_skip=1, maskrcnn_sperm_class_ids="2",
        )
        if args.identity_pass:
            output.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(baseline["source"])
            assert cap.isOpened(), baseline["source"]
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            pipeline._reset_run_state()
            pipeline.source_fps = fps
            pipeline._generate_inference_video(baseline["source"], str(output), fps, width, height)
            analysis = pipeline._compute_final_analysis(fps)
            processed = root / name.replace("current_", "validated_") / "output_processed_video.mp4"
            assert processed.exists(), processed
            shutil.copy2(processed, output / processed.name)
            result = {"output_paths": pipeline._save_results(str(output), analysis),
                      "video_properties": {"total_frames": frames, "fps": fps, "width": width, "height": height}}
        else:
            result = pipeline.process_video(
                baseline["source"], str(output),
                progress_callback=lambda message, percent: print(
                    f"{name}: {percent:.1f}% {message}", flush=True),
            )
        (output / "replay_complete.json").write_text(json.dumps({
            "source": baseline["source"],
            "video_properties": result["video_properties"],
            "output_paths": result["output_paths"],
            "replay_mode": "all_frame_identity_morphology_casa" if args.identity_pass else "full_pipeline",
            "processed_video_reused_from": str(processed) if args.identity_pass else None,
            "pipeline_sha256": hashlib.sha256(Path("sperm_pipeline/sperm_pipeline/pipeline.py").read_bytes()).hexdigest(),
        }, indent=2))
        print(f"Completed {name}", flush=True)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
