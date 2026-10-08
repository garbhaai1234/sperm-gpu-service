"""Replay saved observations and capture pre-allocation lifecycle evidence."""
import copy
import json
from collections import defaultdict, Counter
from investigate_unresolved_identity import BASE, DEST, read, write
from sperm_pipeline.pipeline import SpermAnalysisPipeline


class Probe(SpermAnalysisPipeline):
    def _assign_existing_application_id(self, events, state, item, index, application_id, frame_index,
                                        decision, reason, metrics, update_velocity):
        if (frame_index, application_id) in self.success_targets:
            identity=state['identities'][application_id]
            self.success_snapshots.append({'frame':frame_index,'application_id':application_id,
                'observation':copy.deepcopy(item),'identity_before_assignment':copy.deepcopy(identity),
                'exact_reid_velocity':self._recent_inference_velocity(identity),
                'metrics':copy.deepcopy(metrics),'decision':decision,'reason':reason,
                'groups':[copy.deepcopy(g) for g in state['overlap_groups'].values()
                          if application_id in g.get('all_identity_ids',g['identity_ids'])]})
        return super()._assign_existing_application_id(events,state,item,index,application_id,frame_index,
                                                        decision,reason,metrics,update_velocity)

    def _defer_unresolved_new_detection(self, item, detection_index, state, frame_index):
        result = super()._defer_unresolved_new_detection(item, detection_index, state, frame_index)
        if not result and (frame_index, detection_index) in self.targets:
            ids = self.targets[frame_index, detection_index]
            self.snapshots[frame_index, detection_index] = {
                'frame':frame_index,'detection_index':detection_index,
                'deferral_returned':result,
                'recovery_window_frames':self._identity_recovery_window_frames(state.get('source_fps')),
                'candidate_states':{},
                'groups':copy.deepcopy(list(state['overlap_groups'].values())),
                'pending_chains':[{'candidate_id':c['candidate_id'],'first_frame':c['first_frame'],
                                  'last_frame':c['last_frame'],'possible_identities':c['possible_identities'],
                                  'byte_track_ids':c.get('byte_track_ids',[]),
                                  'observations':copy.deepcopy(c['observations'])}
                                 for c in state['unresolved_candidates'].values()
                                 if set(ids).intersection(c['possible_identities'])],
            }
            for app in ids:
                identity = state['identities'].get(app)
                if identity:
                    self.snapshots[frame_index,detection_index]['candidate_states'][app] = dict(
                        copy.deepcopy(identity),
                        exact_reid_velocity=self._recent_inference_velocity(identity),
                        deferred_trajectory_evidence=self._unresolved_candidate_trajectory_evidence(item,app,state,frame_index))
        return result


def serializable(value):
    if isinstance(value,dict): return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)): return [serializable(v) for v in value]
    return value


for name in ['primary','third']:
    meta=BASE/('accepted_'+name)/'meta'
    original=read(meta/'identity_assignment_diagnostics.json')
    q=read(meta/'tracking_quality_summary.json')
    cases=read(DEST/(name+'_cases.json'))
    observations=defaultdict(dict)
    for row in original:
        if row.get('current_bbox') is not None:
            observations[row['frame']][row['detection_index']]={'tracker_id':row.get('byte_track_id'),
                'confidence':row['detector_score'],'center':tuple(row['current_center']),'bbox':tuple(row['current_bbox'])}
    pipeline=object.__new__(Probe)
    comparisons=read(DEST/(name+'_recovery_comparison.json'))
    pipeline.success_targets={(r['accepted_assignment']['frame'],r['accepted_assignment']['candidate_application_id'])
                              for r in comparisons if r['outcome']=='ORIGINAL_ID_RECOVERY' and r['accepted_assignment']}
    pipeline.success_snapshots=[]
    pipeline.targets={(c['new_id_creation_frame'],c['episodes'][0]['following_same_byte_assignment'][0]['detection_index']):c['candidate_ids'] for c in cases}
    pipeline.snapshots={}
    state=pipeline._initialize_application_identity_state(*q['resolution'],q['fps'])
    missing=[]
    for frame in range(q['frames']):
        obs=observations[frame]
        if obs and set(obs)!=set(range(max(obs)+1)):
            missing.append(frame)
        pipeline._link_inference_application_ids([obs[i] for i in sorted(obs)],state,frame)
    def signature(rows):
        return [(r['frame'],r['detection_index'],r['candidate_application_id'],r['assignment_type']) for r in rows if r.get('accepted')]
    expected=signature(original);actual=signature(state['identity_assignment_diagnostics'])
    def identity_signature(rows):
        return sorted((r['frame'], -1 if r.get('byte_track_id') is None else r['byte_track_id'],
                       r['candidate_application_id'],r['assignment_type'],tuple(r['current_bbox']))
                      for r in rows if r.get('accepted'))
    equivalence={'accepted_assignments_exact_match':expected==actual,'expected_count':len(expected),
                 'accepted_identity_geometry_match_ignoring_detection_index':identity_signature(original)==identity_signature(state['identity_assignment_diagnostics']),
                 'actual_count':len(actual),'missing_internal_detection_index_frames':missing,
                 'original_ambiguous_events':q['ambiguous_events'],
                 'replay_ambiguous_events':len({(r['frame'],r['detection_index']) for r in state['identity_assignment_diagnostics'] if r.get('ambiguous') and r['assignment_type']!='UNRESOLVED_CANDIDATE'}),
                 'limitation':'Saved-observation replay, not a detector rerun; undetected trailing omissions cannot be excluded. Exact accepted assignments checked.'}
    write(DEST/(name+'_replay_equivalence.json'),equivalence)
    write(DEST/(name+'_creation_snapshots.json'),serializable(list(pipeline.snapshots.values())))
    write(DEST/(name+'_successful_recovery_snapshots.json'),serializable(pipeline.success_snapshots))
    print(name,equivalence,flush=True)
