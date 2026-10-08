"""Diagnostic coordinates must never become identity/tracking decisions."""
import copy
import csv
import json
import inspect
from collections import defaultdict, deque

import cv2
import numpy as np
import pytest

from sperm_pipeline.grid_tracking import GridConfig, GridDiagnostics, bbox_center, grid_cell, coordinate_history
from sperm_pipeline.pipeline import SpermAnalysisPipeline


def observation(x, y=20, byte_id=11):
    return dict(bbox=(x-2, y-2, x+2, y+2), center=(x, y), confidence=.95, tracker_id=byte_id)


def test_bbox_coordinates_are_casa_center_not_centroid():
    assert bbox_center([10, 20, 31, 43]) == (20.5, 31.5)
    row = dict(frame_index=0, application_id=7, x1=10, y1=20, x2=31, y2=43, morphology_valid=True)
    assert SpermAnalysisPipeline._trajectories_from_inference_rows([row])[7][0][1:] == bbox_center([10,20,31,43])


@pytest.mark.parametrize('width,height', [(1280,1024),(698,528),(640,480),(101,77)])
def test_grid_resolution_and_boundaries(width,height):
    cfg=GridConfig()
    assert grid_cell(0,0,width,height,cfg)==(0,0,'A1')
    assert grid_cell(width/8,height/8,width,height,cfg)==(1,1,'B2')
    assert grid_cell(width-0.001,height-0.001,width,height,cfg)==(7,7,'H8')
    for x,y in [(width,0),(-1,0),(0,height),(0,-1),(float('nan'),0)]:
        assert grid_cell(x,y,width,height,cfg)==(None,None,None)
    assert grid_cell(26.5,0,28,10,GridConfig(columns=28)) == (0,26,'AA1')


def test_crossing_cells_and_same_cell_never_reassign_ids():
    grid=GridDiagnostics('job',80,80,30)
    state={}
    for frame,x in enumerate([5,15,25]):
        grid.capture(frame,[observation(x),observation(x+1,byte_id=12)],[(7,'KEEP',''),(8,'KEEP','')],state)
    assert [(r['application_id'],r['grid_cell']) for r in grid.records]==[(7,'A3'),(8,'A3'),(7,'B3'),(8,'B3'),(7,'C3'),(8,'C3')]
    assert state=={}


def test_occluded_predictions_unresolved_and_recovery_are_distinct():
    pipeline=object.__new__(SpermAnalysisPipeline)
    grid=GridDiagnostics('job',100,100,30)
    items=[observation(20),observation(40,byte_id=12)]
    state=pipeline._initialize_application_identity_state(100,100,30)
    links=pipeline._link_inference_application_ids(items,state,0)
    grid.capture(0,items,links,state,pipeline._predicted_inference_center)
    merged=dict(bbox=(10,10,50,30),center=(30,20),confidence=.99,tracker_id=99)
    links=pipeline._link_inference_application_ids([merged],state,1)
    before=copy.deepcopy(state)
    rows=grid.capture(1,[merged],links,state,pipeline._predicted_inference_center)
    assert state==before
    assert rows[0]['application_id'] is None and rows[0]['identity_state']=='UNRESOLVED'
    predictions=[r for r in rows if r['coordinate_kind']=='predicted']
    assert {r['application_id'] for r in predictions}=={1,2}
    assert all(r['identity_state']=='OCCLUDED' and r['detection_confidence'] is None for r in predictions)
    assert all(r['overlap_group_ids'] for r in predictions)
    state['lifecycle_events'].append(dict(frame=2,event='ID_REIDENTIFIED',application_id=1,overlap_partners=[2]))
    grid.capture(2,[items[0]],[(1,'REIDENTIFIED','')],state,pipeline._predicted_inference_center)
    assert any(e['label']=='OCCLUSION_RECOVERY' for e in grid.events)


def test_outside_prediction_not_clamped_or_added_to_observed_history():
    g=GridDiagnostics('job',80,80,30)
    state={'identities':{7:{'state':'LOST','last_center':(20,20),'last_frame':0,'velocity':(100,0)}}}
    g.capture(0,[observation(20)],[(7,'KEEP','')],{})
    rows=g.capture(1,[],[],state,SpermAnalysisPipeline._predicted_inference_center)
    assert rows[0]['x']==120 and rows[0]['grid_cell'] is None
    assert g.last[7]['x']==20


def test_csv_and_history_match_observations_and_empty_frames(tmp_path):
    g=GridDiagnostics('job',100,100,49)
    g.capture(0,[observation(20.125)],[(7,'KEEP','')],{})
    g.capture(1,[],[],{})
    paths=g.write(tmp_path)
    with open(paths['grid_tracking_csv_path']) as f:
        rows=list(csv.DictReader(f))
    assert float(rows[0]['x'])==20.125
    assert json.loads(rows[0]['bbox'])==list(observation(20.125)['bbox'])
    assert rows[1]['application_id']=='' and rows[1]['identity_state']=='NO_OBSERVATIONS'
    assert float(rows[1]['timestamp_seconds'])==1/49
    history=coordinate_history(paths['grid_tracking_json_path'],7)
    assert history['trajectory']==[[0,20.125,20.0]]
    assert history['first_observed_frame']==history['last_observed_frame']==0
    with pytest.raises(KeyError): coordinate_history(paths['grid_tracking_json_path'],999)


def test_disabled_default_writes_logs_without_video(tmp_path):
    assert not GridConfig().show_tracking_grid
    assert inspect.signature(SpermAnalysisPipeline).parameters['show_tracking_grid'].default is False
    g=GridDiagnostics('job',80,80,30)
    g.capture(0,[],[],{})
    p=object.__new__(SpermAnalysisPipeline)
    p._finish_grid_diagnostics(g,'nonexistent.mp4',str(tmp_path))
    assert 'grid_tracking_video_path' not in p.grid_paths
    assert (tmp_path/'meta/grid_tracking_coordinates.csv').exists()


def test_advisory_flags_do_not_change_assignments():
    g=GridDiagnostics('job',200,100,30)
    g.capture(0,[observation(20)],[(7,'KEEP','')],{})
    g.capture(1,[observation(100,byte_id=55)],[(7,'KEEP','')],{})
    g.capture(2,[observation(101,byte_id=56)],[(8,'NEW','')],{})
    assert {e['label'] for e in g.events} >= {'POSSIBLE_ID_SWITCH','POSSIBLE_FRAGMENTATION'}
    assert [r['application_id'] for r in g.records]==[7,7,8]


def test_direction_exchange_diagnostic_requires_shared_overlap():
    g=GridDiagnostics('job',200,100,30)
    state={'overlap_groups':{1:{'state':'CONTACT','identity_ids':[7,8]}}}
    for frame,positions in enumerate([(20,60),(30,50),(20,60)]):
        g.capture(frame,[observation(positions[0]),observation(positions[1],byte_id=12)],[(7,'KEEP',''),(8,'KEEP','')],state)
    assert any(e['application_ids']==[7,8] and 'reversed' in e['reason'] for e in g.events)


def test_renderer_uses_canonical_ids_and_bounds_history(tmp_path,monkeypatch):
    source=tmp_path/'source.avi'
    w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),30,(160,120))
    for _ in range(3): w.write(np.zeros((120,160,3),dtype=np.uint8))
    w.release()
    g=GridDiagnostics('job',160,120,30,GridConfig(show_tracking_grid=True,history_length=2))
    for frame in range(3): g.capture(frame,[observation(20+frame)],[(42,'KEEP','')],{})
    labels=[]; original=cv2.putText
    def record(img,text,*args,**kwargs):
        labels.append(text); return original(img,text,*args,**kwargs)
    monkeypatch.setattr(cv2,'putText',record)
    g.render(source,tmp_path/'grid.avi')
    assert len([s for s in labels if s.startswith('ID 42 |')])==6
    assert not any(s.startswith('ID 11 |') for s in labels)
    trails=defaultdict(deque)
    for frame in range(3): g.draw(np.zeros((120,160,3),dtype=np.uint8),frame,[g.records[frame]],trails)
    assert [v[0] for v in trails[42]]==[1,2]


def test_csrt_records_have_no_fake_bytetrack_id():
    g=GridDiagnostics('job',100,100,30)
    item=dict(observation(20),coordinate_kind='tracked')
    row=g.capture(0,[item],[(7,'CSRT_UPDATE','')],{},tracker_kind='csrt')[0]
    assert row['byte_track_id'] is None
    assert row['underlying_tracker_id']==11 and row['coordinate_kind']=='tracked'


@pytest.mark.parametrize('kwargs',[{'rows':0},{'columns':-1},{'history_length':0},{'rows':1.5},{'jump_speed_px_s':float('nan')}])
def test_invalid_grid_config(kwargs):
    with pytest.raises(ValueError): GridConfig(**kwargs)


def test_prediction_marker_uses_dashed_ellipse(monkeypatch):
    g=GridDiagnostics('job',100,100,30)
    row=g._row(0,7,point=(20,20),state='OCCLUDED',kind='predicted')
    arcs=[]
    original=cv2.ellipse
    def draw(*args,**kwargs):
        arcs.append(args[4:6]); return original(*args,**kwargs)
    monkeypatch.setattr(cv2,'ellipse',draw)
    g.draw(np.zeros((100,100,3),np.uint8),0,[row],defaultdict(deque))
    assert arcs==[(0,45),(90,135),(180,225),(270,315)]


def test_expired_and_old_resolved_groups_not_current_overlap():
    g=GridDiagnostics('job',100,100,30)
    state={'overlap_groups':{1:{'state':'EXPIRED','identity_ids':[7]},2:{'state':'RESOLVED','resolved_frame':-1,'identity_ids':[7]}}}
    row=g.capture(0,[observation(20)],[(7,'KEEP','')],state)[0]
    assert row['overlap_group_ids']==[]


def test_history_api_auth_lookup_and_traversal(tmp_path,monkeypatch):
    import main
    from fastapi.testclient import TestClient
    monkeypatch.setattr(main,'OUTPUT_DIR',str(tmp_path))
    monkeypatch.setattr(main,'API_KEY','test-key')
    g=GridDiagnostics('job',100,100,30)
    g.capture(0,[observation(20)],[(7,'KEEP','')],{})
    g.write(tmp_path/'job')
    client=TestClient(main.app)
    url='/diagnostics/job/sperm/7'
    assert client.get(url).status_code==401
    result=client.get(url,headers={'X-API-Key':'test-key'})
    assert result.status_code==200 and result.json()['trajectory']==[[0,20,20]]
    assert client.get('/diagnostics/job/sperm/8',headers={'X-API-Key':'test-key'}).status_code==404
    assert client.get('/diagnostics/job../sperm/7',headers={'X-API-Key':'test-key'}).status_code==400
    (tmp_path/'link').symlink_to('/tmp')
    assert client.get('/diagnostics/link/sperm/7',headers={'X-API-Key':'test-key'}).status_code==403


def test_csrt_snapshot_uses_tracker_state_and_does_not_confirm_predictions():
    from types import SimpleNamespace
    detection=dict(observation(20),application_id=7,track_id=11)
    tracker=SimpleNamespace(_tracks={11:{'status':'predicted_during_detector_miss'}})
    original=copy.deepcopy(detection)
    items,state=SpermAnalysisPipeline._csrt_grid_snapshot([detection],tracker,{},False)
    assert detection==original
    g=GridDiagnostics('job',100,100,30)
    row=g.capture(0,items,[(7,'CSRT_UPDATE','')],state,tracker_kind='csrt')[0]
    assert row['coordinate_kind']=='predicted' and row['detection_confidence'] is None
    assert row['identity_state']=='PREDICTED_DURING_DETECTOR_MISS'
    assert row['byte_track_id'] is None and 7 not in g.last
