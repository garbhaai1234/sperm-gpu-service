"""Read-only lifecycle investigation of the accepted recorded video runs.

Explicit runner, not a pytest test. No detector or production thresholds change.
All case links reproduce audit_identity_videos.py, deduplicated by creation.
Unknown biological predecessors remain null; candidate evidence stays separate.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/identity_registry_validation"
DEST = BASE / "unresolved_lifecycle_investigation"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def distance(a, b):
    return math.dist(a, b) if a is not None and b is not None else None


def iou(a, b):
    if not a or not b:
        return None
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0


def velocity(rows):
    rows = sorted({r['frame']: r for r in rows}.values(), key=lambda r:r['frame'])[-4:]
    if len(rows) < 2:
        return None
    a, b = rows[0], rows[-1]
    return [(b['current_center'][i]-a['current_center'][i])/(b['frame']-a['frame']) for i in range(2)]


def run(name):
    source = BASE / ('accepted_' + name)
    meta = source / 'meta'
    audit = read(BASE / 'targeted_audit' / ('final_' + name + '.json'))
    diag = read(meta / 'identity_assignment_diagnostics.json')
    life = read(meta / 'application_id_lifecycle.json')
    events = read(meta / 'application_identity_quality_events.json')
    registry = read(meta / 'application_identity_registry.json')
    quality = read(meta / 'tracking_quality_summary.json')
    by_detection, by_app, by_byte, life_app = (defaultdict(list) for _ in range(4))
    for r in diag:
        by_detection[r['frame'], r.get('detection_index')].append(r)
        if r.get('accepted') and r.get('candidate_application_id') is not None:
            by_app[r['candidate_application_id']].append(r)
            if r.get('byte_track_id') is not None:
                by_byte[r['byte_track_id']].append(r)
    for rows in list(by_app.values()) + list(by_byte.values()):
        rows.sort(key=lambda r:r['frame'])
    for r in life:
        life_app[r['application_id']].append(r)
    chains = {r['candidate_id']: r for r in registry['unresolved_candidate_archive'] + registry['unresolved_candidates']}
    state_events = {'ID_CONFIRMED':'CONFIRMED', 'ID_CREATED':'CONFIRMED', 'ID_REIDENTIFIED':'CONFIRMED',
                    'ID_TERMINATED':'TERMINATED', 'ID_LOST':'LOST', 'ID_OCCLUDED':'OCCLUDED',
                    'ID_LOW_SUPPORT':'LOW_SUPPORT', 'ID_UNRESOLVED':'UNRESOLVED'}

    def state_at(app, frame):
        rows = [r for r in life_app[app] if r['frame'] <= frame and r['event'] in state_events]
        return state_events[rows[-1]['event']] if rows else None

    def groups_at(app, frame):
        result = []
        for g in registry['overlap_groups']:
            if app not in g.get('all_identity_ids', g['identity_ids']) or g['start_frame'] > frame:
                continue
            terminal = g.get('resolved_frame')
            state = g['state'] if terminal is not None and terminal <= frame else 'ACTIVE_HISTORICAL_STATE_NOT_LOGGED'
            result.append({'group_id':g['overlap_group_id'], 'state_at_creation':state,
                           'start_frame':g['start_frame'], 'last_frame':g['last_frame'],
                           'terminal_frame':terminal, 'final_state':g['state']})
        return result

    grouped = defaultdict(list)
    for ep in audit['unresolved_episodes']:
        if ep['outcome'] == 'NEW_ID_AFTER_UNRESOLVED':
            new = ep['following_same_byte_assignment'][0]
            grouped[new['frame'], new['candidate_application_id']].append(ep)
    cases = []
    for num, ((frame, new_id), episodes) in enumerate(sorted(grouped.items()), 1):
        new = episodes[0]['following_same_byte_assignment'][0]
        start = min(e['start'] for e in episodes)
        ids = sorted({i for e in episodes for i in e['possible_identities']})
        candidate_evidence = []
        for old in ids:
            prior = [r for r in by_app[old] if r['frame'] < start and r.get('current_bbox')]
            last = [r for r in by_app[old] if r['frame'] < frame and r.get('current_bbox')]
            anchor = prior[-1] if prior else None
            recent = last[-1] if last else None
            rejections = [r for r in by_detection[frame, new['detection_index']]
                          if r.get('candidate_application_id') == old and not r.get('accepted')]
            attempts = [r for r in diag if start <= r['frame'] <= frame
                        and r.get('candidate_application_id') == old
                        and r.get('assignment_type') == 'OCCLUDED_REIDENTIFICATION'
                        and any(r['frame'] == ob['frame'] and r.get('detection_index') == ob['detection_index']
                                for ep in episodes for ob in chains[ep['candidate_id']]['observations'])]
            simultaneous = [r for r in by_app[old] if frame <= r['frame'] <= frame+5]
            v = velocity(prior)
            new_v = velocity([r for r in by_app[new_id] if frame <= r['frame'] <= frame+5])
            dot = sum(a*b for a,b in zip(v,new_v)) if v is not None and new_v is not None else None
            norm = math.hypot(*v)*math.hypot(*new_v) if dot is not None else 0
            last_state = state_at(old, frame)
            last_groups = groups_at(old, frame)
            candidate_evidence.append({
                'old_app_id':old, 'old_state_at_creation_end_of_frame':last_state,
                'old_retired':last_state == 'TERMINATED',
                'old_last_before_unresolved_frame':anchor['frame'] if anchor else None,
                'frames_since_last_old_identity':frame-recent['frame'] if recent else None,
                'old_bbox_before_unresolved':anchor['current_bbox'] if anchor else None,
                'old_position_before_unresolved':anchor['current_center'] if anchor else None,
                'old_velocity_before_unresolved':v, 'new_velocity_after_unresolved':new_v,
                'velocity_provenance':'finite difference over up to four accepted observations; not exact internal velocity',
                'old_byte_before_unresolved':anchor.get('byte_track_id') if anchor else None,
                'same_byte_before_unresolved':anchor.get('byte_track_id') == new['byte_track_id'] if anchor else None,
                'distance_gap':distance(anchor['current_center'],new['current_center']) if anchor else None,
                'IoU_if_available':iou(anchor['current_bbox'],new['current_bbox']) if anchor else None,
                'direction_cosine':dot/norm if norm else None,
                'bbox_long_side_ratio':max(max(anchor['current_bbox'][2]-anchor['current_bbox'][0],anchor['current_bbox'][3]-anchor['current_bbox'][1])/max(new['current_bbox'][2]-new['current_bbox'][0],new['current_bbox'][3]-new['current_bbox'][1]),max(new['current_bbox'][2]-new['current_bbox'][0],new['current_bbox'][3]-new['current_bbox'][1])/max(anchor['current_bbox'][2]-anchor['current_bbox'][0],anchor['current_bbox'][3]-anchor['current_bbox'][1])) if anchor else None,
                'motion_consistency':[r.get('motion_consistent') for r in rejections],
                'overlap_groups':last_groups,
                'reidentification_attempted_during_episode':bool(attempts),
                'reidentification_attempted_at_creation':any(r['assignment_type']=='OCCLUDED_REIDENTIFICATION' for r in rejections),
                'reidentification_result': 'INELIGIBLE_TERMINATED' if last_state == 'TERMINATED' else 'NOT_ASSIGNED',
                'candidate_scores':[{'type':r['assignment_type'],'score':r.get('best_candidate_score'),'reason':r['rejection_reason']} for r in rejections],
                'creation_rejections':rejections,
                'episode_reidentification_rejection_counts':dict(Counter(r['rejection_reason'] for r in attempts)),
                'old_observations_during_episode_and_later':[r for r in by_app[old] if start <= r['frame'] <= frame+int(quality['fps'])],
                'old_visible_at_or_soon_after_creation_frames':[r['frame'] for r in simultaneous],
                'termination_events':[r for r in life_app[old] if r['event']=='ID_TERMINATED'],
            })
        chain_records = [chains[e['candidate_id']] for e in episodes]
        after = [r for r in by_app[new_id] if frame <= r['frame'] <= frame+int(quality['fps'])]
        all_rejections = [r for r in by_detection[frame,new['detection_index']] if not r.get('accepted')]
        case = {
            'case_id':f'{name.upper()}-{num:02}', 'old_app_id':None,
            'old_app_id_note':'No biological predecessor established. See candidate_ids and per_candidate_evidence; even a singleton plausible candidate is not proof.',
            'new_app_id':new_id, 'byte_track_id':new['byte_track_id'], 'start_frame':start,
            'unresolved_start_frame':start, 'unresolved_end_frame':max(e['end'] for e in episodes),
            'new_id_creation_frame':frame,
            'frames_since_last_old_identity':{str(r['old_app_id']):r['frames_since_last_old_identity'] for r in candidate_evidence},
            'old_bbox_before_unresolved':{str(r['old_app_id']):r['old_bbox_before_unresolved'] for r in candidate_evidence},
            'new_bbox_after_unresolved':new['current_bbox'],
            'old_position_before_unresolved':{str(r['old_app_id']):r['old_position_before_unresolved'] for r in candidate_evidence},
            'new_position_after_unresolved':new['current_center'],
            'old_velocity_before_unresolved':{str(r['old_app_id']):r['old_velocity_before_unresolved'] for r in candidate_evidence},
            'new_velocity_after_unresolved':velocity(after[:4]),
            'distance_gap':{str(r['old_app_id']):r['distance_gap'] for r in candidate_evidence},
            'IoU_if_available':{str(r['old_app_id']):r['IoU_if_available'] for r in candidate_evidence},
            'motion_consistency':{str(r['old_app_id']):r['motion_consistency'] for r in candidate_evidence},
            'detection_confidence':new['detector_score'],
            'overlap_group_id':sorted({g['group_id'] for r in candidate_evidence for g in r['overlap_groups']}),
            'overlap_group_state':{str(r['old_app_id']):r['overlap_groups'] for r in candidate_evidence},
            'number_of_competing_candidates':len(ids), 'candidate_ids':ids,
            'candidate_scores':{str(r['old_app_id']):r['candidate_scores'] for r in candidate_evidence},
            'reidentification_attempted':any(r['reidentification_attempted_during_episode'] or r['reidentification_attempted_at_creation'] for r in candidate_evidence),
            'reidentification_result':'NO_OLD_CANDIDATE_ASSIGNED_TO_THIS_DETECTION',
            'reason_new_id_created':new['rejection_reason'],
            'classification':'UNCERTAIN',
            'root_cause':'ALL_EPISODE_CANDIDATES_RETIRED' if all(r['old_retired'] for r in candidate_evidence) else 'NO_ELIGIBLE_PLAUSIBLE_CANDIDATE_AT_CREATION',
            'classification_basis':'ByteTrack links the unresolved detection to the creation, not to an established old biological identity. Geometry alone does not prove sameness or difference.',
            'episode_ids':[e['candidate_id'] for e in episodes], 'episodes':episodes,
            'candidate_chains':chain_records, 'per_candidate_evidence':candidate_evidence,
            'all_creation_rejections':all_rejections,
            'later_new_id_observations':after,
            'visual_review':'NOT_YET_REVIEWED',
            'enough_evidence_to_defer':'Deferral returned false at creation; exact eligible-set calculation in replay snapshots when available.',
            'same_sperm_could_reappear':'Biologically possible; not established by these logs.',
        }
        cases.append(case)
    write(DEST / (name + '_cases.json'), cases)
    with (DEST / (name + '_cases.csv')).open('w') as f:
        fields = ['case_id','old_app_id','candidate_ids','new_app_id','byte_track_id','new_id_creation_frame','frames_since_last_old_identity','reason_new_id_created','classification','root_cause']
        w=csv.DictWriter(f,fields);w.writeheader()
        for c in cases:
            w.writerow({k:json.dumps(c[k]) if isinstance(c[k],(list,dict)) else c[k] for k in fields})

    # Partition exactly the quality metric's unit: unique ambiguous (frame, detection).
    ambiguous = []
    for key, rows in sorted(by_detection.items()):
        amb = [r for r in rows if r.get('ambiguous') and r['assignment_type']!='UNRESOLVED_CANDIDATE']
        if not amb:
            continue
        byte = amb[0].get('byte_track_id')
        ids = sorted({r['candidate_application_id'] for r in amb if r.get('candidate_application_id') is not None})
        follow = [r for r in by_byte.get(byte,[]) if r['frame']>key[0]] if byte is not None else []
        first = follow[0] if follow else None
        outcome = ('NO_LATER_SAME_BYTE_ASSIGNMENT' if first is None else
                   'CANDIDATE_ID_ASSIGNED' if first['candidate_application_id'] in ids else
                   'NEW_ID_CREATED' if first['assignment_type']=='NEW_HIGH_ID' else 'OTHER_EXISTING_ID_ASSIGNED')
        ambiguous.append({'frame':key[0],'detection_index':key[1],'byte_track_id':byte,
                          'candidate_ids':ids,'outcome':outcome,'first_later_same_byte_assignment':first,
                          'classification':'UNCERTAIN','reason':'Computational assignment outcome; not biological ground truth.',
                          'diagnostics':amb})
    assert len(ambiguous)==quality['ambiguous_events']
    write(DEST/(name+'_ambiguity_events.json'),ambiguous)

    comparisons=[]
    for ep in audit['unresolved_episodes']:
        if ep['outcome'] not in {'ORIGINAL_ID_RECOVERY','NEW_ID_AFTER_UNRESOLVED'}:
            continue
        exact=[r for r in events if r['event']=='UNRESOLVED_CANDIDATE_RESOLVED' and r['candidate_id']==ep['candidate_id']]
        accepted = ep.get('following_same_byte_assignment',[])
        if exact:
            accepted=[r for r in by_app[exact[0]['application_id']] if r['frame']==exact[0]['frame']]
        row=accepted[0] if accepted else None
        comparisons.append({'candidate_id':ep['candidate_id'],'outcome':ep['outcome'],
                            'duration_frames':ep['duration_frames'],'duration_seconds':ep['duration_seconds'],
                            'number_of_candidates':len(ep['possible_identities']),
                            'candidate_ids':ep['possible_identities'],
                            'accepted_assignment':row,
                            'candidate_states_at_outcome':{str(i):state_at(i,row['frame']) for i in ep['possible_identities']} if row else {},
                            'note':'Recovered is assignment outcome, not visually proven identity.'})
    write(DEST/(name+'_recovery_comparison.json'),comparisons)
    summary = {'quality':quality,'audit_episode_outcomes':audit['unresolved_outcomes'],
               'distinct_new_id_creations':len(cases),'linked_episodes':sum(len(c['episodes']) for c in cases),
               'ambiguity_assignment_outcomes':dict(Counter(r['outcome'] for r in ambiguous)),
               'ambiguity_biological_classifications':dict(Counter(r['classification'] for r in ambiguous)),
               'case_classifications':dict(Counter(c['classification'] for c in cases)),
               'case_root_causes':dict(Counter(c['root_cause'] for c in cases)),
               'provenance':read(source/'replay_complete.json'),
               'current_pipeline_sha256':hashlib.sha256((ROOT/'sperm_pipeline/sperm_pipeline/pipeline.py').read_bytes()).hexdigest(),
               'comparison':{}}
    for outcome in ['ORIGINAL_ID_RECOVERY','NEW_ID_AFTER_UNRESOLVED']:
        rows=[r for r in comparisons if r['outcome']==outcome]
        stats={}
        for field in ['duration_frames','duration_seconds','number_of_candidates']:
            vals=[r[field] for r in rows];stats[field]={'min':min(vals),'median':statistics.median(vals),'max':max(vals)} if vals else None
        for field in ['frame_gap','distance_to_high_anchor','predicted_center_distance','bbox_size_ratio','detector_score','best_candidate_score']:
            vals=[r['accepted_assignment'].get(field) for r in rows if r['accepted_assignment'] and r['accepted_assignment'].get(field) is not None]
            stats[field]={'n':len(vals),'min':min(vals),'median':statistics.median(vals),'max':max(vals)} if vals else None
        summary['comparison'][outcome]=stats
    write(DEST/(name+'_summary.json'),summary)
    print(name, json.dumps({k:summary[k] for k in ['distinct_new_id_creations','linked_episodes','ambiguity_assignment_outcomes','case_root_causes']}),flush=True)


if __name__ == '__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    for name in ['primary','third']:
        run(name)
