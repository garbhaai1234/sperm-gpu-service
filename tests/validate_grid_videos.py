"""Explicit GPU validation on three original uploads; no baseline artifacts overwritten.
PYTHONPATH=sperm_pipeline python tests/validate_grid_videos.py
"""
import csv
import hashlib
import json
import logging
from pathlib import Path

import cv2
from sperm_pipeline.pipeline import SpermAnalysisPipeline
from sperm_pipeline.grid_tracking import GridConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    logging.basicConfig(level=logging.ERROR)
    source_map=json.loads(Path('outputs/identity_registry_validation/current_validation_metrics.json').read_text())['videos']
    root=Path('outputs/grid_tracking_validation')
    root.mkdir(exist_ok=True)
    pipeline=SpermAnalysisPipeline('detection_best.pt','best_resnet50_transfer_from_101.pth','HNK_best.pt',device='cuda',maskrcnn_sperm_class_ids='2')
    results=[]
    for name,source_info in source_map.items():
        name=name.replace('current_','')
        source=source_info['source']
        cap=cv2.VideoCapture(source)
        width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=cap.get(cv2.CAP_PROP_FPS); frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        summaries=[]; tracks=[]; video_hashes=[]
        for enabled in [False,True]:
            destination=root/name/('enabled' if enabled else 'disabled')
            destination.mkdir(parents=True,exist_ok=True)
            print(f'START {name} grid={enabled}: {frames} frames {width}x{height} {fps}fps',flush=True)
            pipeline._reset_run_state()
            pipeline.grid_config=GridConfig(show_tracking_grid=enabled)
            pipeline.source_fps=fps
            pipeline._generate_inference_video(source,str(destination),fps,width,height)
            summary=pipeline._compute_final_analysis(fps)
            summaries.append(summary)
            tracks.append(pipeline.application_trajectories)
            (destination/'meta/validation_summary.json').write_text(json.dumps(summary))
            video_hashes.append(digest(destination/'output_inference_video.mp4'))
            print(f'DONE {name} grid={enabled}',flush=True)
        d=root/name/'enabled'
        payload=json.loads((d/'meta/grid_tracking_coordinates.json').read_text())
        observed={(r['frame_index'],r['application_id']):r for r in payload['records'] if r['application_id'] is not None and r['coordinate_kind']=='observed'}
        with (d/'meta/inference_tracking.csv').open() as f: rows=list(csv.DictReader(f))
        exact=all((r:=observed[(int(t['frame_index']),int(t['application_id']))])['x']==(float(t['x1'])+float(t['x2']))/2 and r['y']==(float(t['y1'])+float(t['y2']))/2 for t in rows)
        cells={}; crossings=0
        for row in payload['records']:
            app=row['application_id']
            if app is None or row['coordinate_kind']!='observed': continue
            if app in cells and cells[app]!=row['grid_cell']: crossings+=1
            cells[app]=row['grid_cell']
        cap=cv2.VideoCapture(str(d/'output_grid_tracking_video.mp4'))
        video_props=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        cap.release()
        lifecycle=json.loads((d/'meta/application_id_lifecycle.json').read_text())
        predictions=[r for r in payload['records'] if r['coordinate_kind']=='predicted']
        result=dict(name=name,source=source,source_sha256=digest(source),frames=frames,fps=fps,resolution=[width,height],
                    grid_video_properties=video_props,coordinates_match_inference_csv=exact,
                    assigned_grid_rows=len(observed),inference_csv_rows=len(rows),canonical_grid_crossings=crossings,
                    predictions=len(predictions),unresolved_observations=sum(r['application_id'] is None and r['coordinate_kind']=='observed' for r in payload['records']),
                    occlusion_recoveries=sum(e['label']=='OCCLUSION_RECOVERY' for e in payload['events']),
                    expected_occlusion_recoveries=sum(e['event']=='ID_REIDENTIFIED' and bool(e.get('overlap_partners')) for e in lifecycle),
                    predictions_labeled=sum(r['identity_state'] in ['OCCLUDED','LOST','UNRESOLVED'] and r['detection_confidence'] is None for r in predictions)==len(predictions),
                    all_frames_logged={r['frame_index'] for r in payload['records']}==set(range(frames)),
                    summary_morphology_casa_exactly_equal=summaries[0]==summaries[1],trajectories_exactly_equal=tracks[0]==tracks[1],
                    standard_video_sha256_equal=video_hashes[0]==video_hashes[1],standard_video_sha256=video_hashes,
                    diagnostic_video=str(d/'output_grid_tracking_video.mp4'),coordinates_csv=str(d/'meta/grid_tracking_coordinates.csv'))
        assert exact and len(observed)==len(rows)
        assert result['all_frames_logged'] and result['predictions_labeled']
        assert result['occlusion_recoveries']==result['expected_occlusion_recoveries']
        assert result['summary_morphology_casa_exactly_equal'] and result['trajectories_exactly_equal']
        assert result['standard_video_sha256_equal']
        assert video_props[:2]==[width,height] and video_props[3]==frames and abs(video_props[2]-fps)<.001
        results.append(result)
        (root/'validation_results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__': main()
