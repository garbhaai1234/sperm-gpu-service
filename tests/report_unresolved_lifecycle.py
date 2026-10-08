"""Assemble evidence-qualified investigation report; never edits tracker code."""
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from investigate_unresolved_identity import BASE, DEST, read, write, distance, iou

VISUAL_NOTES = {
 'PRIMARY-01':'The unresolved chain and new box continue locally; neither old candidate is visually connected through the full interval.',
 'PRIMARY-02':'A1 and A25 have different ByteTrack IDs separated by 50 trusted frames. The one-frame U37 to A25 continuity does not bridge that gap.',
 'PRIMARY-03':'Three episode records point to A26. Multiple former identities and changing candidate membership prevent a unique predecessor.',
 'PRIMARY-04':'A27 follows U39, but U39 has two retired possible predecessors and only one observed unresolved frame.',
 'PRIMARY-05':'Long predecessor gaps, multiple sperm and changing detector boxes; the sampled frames do not establish which previous sperm this is.',
 'PRIMARY-06':'The chain is locally visible, but several remote old identities are plausible by the permissive deferral envelope. No old-ID continuity is proven.',
 'PRIMARY-07':'A17 is absent as a trusted identity for 50 frames. The new sperm is visible, but the sampled interval does not establish whether it is A17.',
 'PRIMARY-08':'A30 remains separately visible at creation and afterward. A17 has retired, so A30 can be excluded but A17 cannot be ruled in or out.',
 'PRIMARY-09':'A dark instrument crosses the field during the interval; one early unresolved box covers the instrument region. Candidate-chain biological continuity is not established.',
 'PRIMARY-10':'The last low-support A32 box overlaps the instrument. Two distinct new sperm A45/A46 follow the same sole old candidate; proximity cannot decide which, if either, is A32.',
 'PRIMARY-11':'A46 is visibly distinct from A45. Both share old candidate A32 in the audit; recovering both as A32 would be incorrect.',
 'PRIMARY-12':'A43 retired; A45 is still lost but fails the deferral geometry. Sparse samples and candidate changes do not prove either biological association.',
 'PRIMARY-13':'A44 and A50 are separated by a long trusted-observation gap at the field edge. Local continuity starts too late to identify A44.',
 'PRIMARY-14':'The new sperm is visible after deferral, but all three historical candidates retired; no unique predecessor through the interval.',
 'PRIMARY-15':'Four candidate chains lead to this creation. Several old identities retired and the live A42 does not pass deferral. No unique biological link.',
 'PRIMARY-16':'A55 and A56 are visibly distinct nearby sperm at creation; both candidate pools include A42/A43. Sampled evidence cannot assign the earlier identities safely.',
 'PRIMARY-17':'A56 is visibly distinct from A55, and one linked chain previously observed A55’s ByteTrack. This is not proof of a coherent biological chain.',
 'PRIMARY-18':'The chain and new detection contain a large round structure and a nearby sperm/tail; detector support and a 56-frame gap do not establish old A47.',
 'PRIMARY-19':'A51/B932 and A60/B947 are separate, nonoverlapping sperm at frame 440 and remain separate at frame 448. A51 was not replaced by A60.',
 'PRIMARY-20':'Multiple retired candidates plus live A48; the narrow new box and interrupted candidate history do not identify a unique old sperm.',
 'PRIMARY-21':'A52/B923 is separately recovered at frame 454 while A62/B952 is assigned to a different location; A52 also recovers later at frame 475. This is not replacement of A52.',
 'PRIMARY-22':'The new track changes ByteTrack again shortly afterward. Several retired candidates and the long interval prevent a defensible predecessor assignment.',
 'THIRD-01':'A26/B47 and A35/B46 are visibly separate at frames 6 and 14; A26 recovered independently. Creating A35 avoids merging them.',
 'THIRD-02':'A5/B37 and A22/B53 both remain on separate visible sperm when A36/B39 is created, and afterward.',
 'THIRD-03':'A34/B38 and A37/B43 have distinct visible heads at frames 11 and 19; the original candidate remains independently tracked.',
 'THIRD-04':'A5 retires before this creation. Its last high observation is much older than its last low-support observation; these samples do not bridge it to A42.',
 'THIRD-05':'A20 remains eligible by age, but the new sperm is far from its trusted anchor and opposed to its prediction. A continuous biological connection is not visible in these samples.',
 'THIRD-06':'U64 starts on A20/B157 but later jumps among six ByteTrack IDs, several owned by other App IDs. The final B139 previously belonged to A15. Locally smooth final motion cannot establish A20 to A45.',
 'THIRD-07':'A6, A13 and A40 are separately visible, but older retired candidates and prior B169 ownership by A13 leave the complete identity history uncertain.',
}
DIFFERENT={'PRIMARY-19','PRIMARY-21','THIRD-01','THIRD-02','THIRD-03'}


def stats(vals):
    vals=[v for v in vals if v is not None]
    return {'n':len(vals),'min':min(vals),'median':statistics.median(vals),'max':max(vals)} if vals else {'n':0,'min':None,'median':None,'max':None}


def enrich(name):
    cases=read(DEST/(name+'_cases.json'))
    snapshots=read(DEST/(name+'_creation_snapshots.json'))
    q=read(DEST/(name+'_summary.json'))
    diag=read(BASE/('accepted_'+name)/'meta/identity_assignment_diagnostics.json')
    by_byte=defaultdict(list)
    by_detection=defaultdict(list)
    for r in diag:
        by_detection[r['frame'],r.get('detection_index')].append(r)
        if r.get('accepted') and r.get('byte_track_id') is not None: by_byte[r['byte_track_id']].append(r)
    pair_features=[]
    for c in cases:
        c['visual_review']='SIX_SOURCE_FRAMES_REVIEWED; NOT_CONTINUOUS_GROUND_TRUTH'
        c['visual_review_note']=VISUAL_NOTES[c['case_id']]
        if c['case_id'] in DIFFERENT:
            c['classification']='DIFFERENT_SPERM_CORRECT_NEW_ID'
            c['classification_basis']=VISUAL_NOTES[c['case_id']]
            c['root_cause']='OLD_CANDIDATE_RECOVERED_ON_SEPARATE_SPERM'
        c['classification_scope']='Relative to the investigated prior candidate identity/identities; does not certify every historic assignment in the video.'
        c['prior_same_byte_assignments']=[r for r in by_byte[c['byte_track_id']] if r['frame']<c['new_id_creation_frame']]
        observation_keys={(o['frame'],o['detection_index']) for ch in c['candidate_chains'] for o in ch['observations']}
        episode_rows=[r for key in observation_keys for r in by_detection[key]]
        explicit_ambiguity=sorted({(r['frame'],r['detection_index']) for r in episode_rows if r.get('ambiguous') and r['assignment_type']!='UNRESOLVED_CANDIDATE'})
        c['episode_explicit_ambiguity_detections']=explicit_ambiguity
        c['episode_candidate_rejection_counts']=dict(Counter(r['rejection_reason'] for r in episode_rows if not r.get('accepted') and r.get('candidate_application_id') in c['candidate_ids']))
        c['candidate_ids_semantics']='Union of possible identities across linked episodes, not proven predecessors or simultaneous eligible contenders.'
        c['competing_candidates_considered_at_creation']=sorted({r['candidate_application_id'] for r in c['all_creation_rejections'] if r.get('candidate_application_id') is not None})
        s=next(s for s in snapshots if s['frame']==c['new_id_creation_frame'] and s['detection_index']==c['episodes'][0]['following_same_byte_assignment'][0]['detection_index'])
        c['replay_snapshot_reference']=name+'_creation_snapshots.json'
        c['trusted_identity_gaps_at_creation']={}
        c['exact_candidate_state_before_creation']={}
        c['old_velocity_before_creation_exact']={}
        c['enough_evidence_to_defer']='No candidate passed the current deferral gates at this frame. This is a code-path result, not biological proof.'
        c['legitimate_ambiguity']='A possible-identity list is a proximity hypothesis, not proof of physical overlap or a genuine identity tie.'
        for r in c['per_candidate_evidence']:
            app=str(r['old_app_id']); ident=s['candidate_states'][app]
            frame=c['new_id_creation_frame'];gap=frame-ident['last_high_conf_frame']
            v=ident['exact_reid_velocity'];anchor=ident['trusted_center'];box=ident['trusted_bbox'];new=c['new_position_after_unresolved']
            pred=[anchor[i]+v[i]*gap for i in range(2)]
            size=max(box[2]-box[0],box[3]-box[1],1);newbox=c['new_bbox_after_unresolved'];newsize=max(newbox[2]-newbox[0],newbox[3]-newbox[1],1)
            ratio=max(size/newsize,newsize/size);lastdist=distance(anchor,new);preddist=distance(pred,new)
            envelope=max(60,3*size)
            status=ident['state']
            gate=('TERMINATED_INELIGIBLE' if status=='TERMINATED' else 'ALREADY_CONFIRMED_OTHER_DETECTION' if status=='CONFIRMED' else
                  'RECOVERY_WINDOW_EXCEEDED' if gap>s['recovery_window_frames'] else
                  'BBOX_RATIO_EXCEEDS_DEFERRAL_GATE' if ratio>3 else
                  'OUTSIDE_DEFERRAL_ENVELOPE' if min(lastdist,preddist)>envelope else 'OTHER_ELIGIBILITY_STATE; SEE_SNAPSHOT')
            c['trusted_identity_gaps_at_creation'][app]=gap
            c['exact_candidate_state_before_creation'][app]=status
            c['old_velocity_before_creation_exact'][app]=v
            groups=[{'group_id':g['overlap_group_id'],'state':g['state'],'all_identity_ids':g.get('all_identity_ids',g['identity_ids'])}
                    for g in s['groups'] if r['old_app_id'] in g.get('all_identity_ids',g['identity_ids'])]
            r.update(trusted_gap_before_creation=gap,exact_state_before_creation=status,
                     exact_velocity_before_creation=v, trusted_bbox_before_creation=box,
                     trusted_position_before_creation=anchor,exact_overlap_groups_before_creation=groups,
                     deferral_rejection_condition=gate, trusted_distance_at_creation=lastdist,
                     trusted_prediction_distance_at_creation=preddist,trusted_iou_at_creation=iou(box,newbox),
                     deferral_envelope_px=envelope, trusted_bbox_ratio_at_creation=ratio,
                     deferred_trajectory_evidence=ident['deferred_trajectory_evidence'])
            # Keep pair-level measurements; never silently select a biological predecessor by distance.
            pair_features.append({'case_id':c['case_id'],'old_candidate':int(app),'outcome':'NEW_ID_CREATED',
                'gap':gap,'distance':lastdist,'prediction_distance':preddist,'velocity':v,'speed':math.hypot(*v),
                'direction_dot':sum((new[i]-anchor[i])*v[i] for i in range(2)),
                'bbox_ratio':ratio,'confidence':c['detection_confidence'],'state':status,'groups':groups,
                'candidate_count':len(c['candidate_ids']),'candidate_scores':r['candidate_scores'],
                'recovery_limit':min(100,20+7*gap),'elapsed_seconds':gap/q['quality']['fps']})
        c['questions_and_answers']={
          '1_why_previous_id_not_recovered':{str(r['old_app_id']):r['deferral_rejection_condition'] for r in c['per_candidate_evidence']},
          '2_was_reidentification_attempted':{'during_linked_episode':c['reidentification_attempted'],
            'at_creation_by_candidate':{str(r['old_app_id']):r['reidentification_attempted_at_creation'] for r in c['per_candidate_evidence']}},
          '3_candidates_considered':{'historical_episode_candidates':c['candidate_ids'],'all_logged_at_creation':c['competing_candidates_considered_at_creation']},
          '4_rejection_reasons':{str(r['old_app_id']):{'eligibility':r['deferral_rejection_condition'],'logged':r['candidate_scores'],'during_episode':r['episode_reidentification_rejection_counts']} for r in c['per_candidate_evidence']},
          '5_old_identity_retired':{str(r['old_app_id']):r['exact_state_before_creation']=='TERMINATED' for r in c['per_candidate_evidence']},
          '6_overlap_group_expired':{str(r['old_app_id']):r['exact_overlap_groups_before_creation'] for r in c['per_candidate_evidence']},
          '7_legitimate_ambiguity':{'explicit_ambiguous_detections_in_linked_chains':explicit_ambiguity,
            'candidate_rejection_counts':c['episode_candidate_rejection_counts'],
            'interpretation':c['legitimate_ambiguity']},
          '8_enough_evidence_to_defer':c['enough_evidence_to_defer'],
          '9_could_same_sperm_reappear':'Possible for unobserved candidates; not established. Co-visible old candidates cannot be assigned simultaneously to this different detection.'}
        c['overlap_group_state']={str(r['old_app_id']):r['exact_overlap_groups_before_creation'] for r in c['per_candidate_evidence']}
        c['overlap_group_state_provenance']='Pre-creation replay snapshot; stress replay qualification in replay_equivalence.json.'
    success=read(DEST/(name+'_successful_recovery_snapshots.json'))
    for s in success:
        ident=s['identity_before_assignment'];item=s['observation'];v=s['exact_reid_velocity'];gap=s['frame']-ident['last_high_conf_frame'];a=ident['trusted_center'];b=ident['trusted_bbox'];nb=item['bbox']
        size=max(b[2]-b[0],b[3]-b[1],1);ns=max(nb[2]-nb[0],nb[3]-nb[1],1)
        pair_features.append({'frame':s['frame'],'old_candidate':s['application_id'],'outcome':'ORIGINAL_ID_ASSIGNED',
            'gap':gap,'distance':distance(a,item['center']),
            'prediction_distance':distance([a[i]+v[i]*gap for i in range(2)],item['center']),
            'velocity':v,'speed':math.hypot(*v),'direction_dot':sum((item['center'][i]-a[i])*v[i] for i in range(2)),
            'bbox_ratio':max(size/ns,ns/size),'confidence':item['confidence'],'state':ident['state'],
            'groups':[{'group_id':g['overlap_group_id'],'state':g['state']} for g in s['groups']],
            'candidate_scores':s['metrics'],'recovery_limit':min(100,20+7*gap),'elapsed_seconds':gap/q['quality']['fps']})
    write(DEST/(name+'_feature_comparison.json'),pair_features)
    feature_stats={outcome:{field:stats([r[field] for r in pair_features if r['outcome']==outcome])
                            for field in ['gap','distance','prediction_distance','speed','direction_dot','bbox_ratio','confidence']}
                   for outcome in ['ORIGINAL_ID_ASSIGNED','NEW_ID_CREATED']}
    q['feature_comparison_stats']=feature_stats
    q['case_classifications']=dict(Counter(c['classification'] for c in cases))
    q['case_root_causes']=dict(Counter(c['root_cause'] for c in cases))
    q['visual_scope']='Six sampled source frames per creation; not continuous biological ground truth.'
    write(DEST/(name+'_summary.json'),q);write(DEST/(name+'_cases.json'),cases)
    with (DEST/(name+'_cases.csv')).open('w') as f:
        fields=['case_id','old_app_id','candidate_ids','new_app_id','byte_track_id','new_id_creation_frame','frames_since_last_old_identity','trusted_identity_gaps_at_creation','reason_new_id_created','classification','root_cause','visual_review_note']
        w=csv.DictWriter(f,fields);w.writeheader()
        for c in cases:w.writerow({k:json.dumps(c[k]) if isinstance(c[k],(list,dict)) else c[k] for k in fields})
    return cases,q


def esc(s):return html.escape(str(s))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
def p(s):return '<p>'+s+'</p>'


primary,ps=enrich('primary');third,ts=enrich('third')
out=['<!doctype html><meta charset="utf-8"><title>Unresolved identity lifecycle investigation</title><style>body{font:16px/1.55 system-ui;max-width:1400px;margin:36px auto;padding:0 24px;color:#17212b}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{border:1px solid #d4dce3;padding:8px;text-align:left;vertical-align:top}th{background:#edf3f8}pre{white-space:pre-wrap;background:#f3f5f7;padding:18px;font-size:12px}img{max-width:100%}details{border:1px solid #d4dce3;margin:16px 0;padding:12px}a{color:#145ba3}.warn{background:#fff2d4;padding:15px}</style>']
out+=[ '<h1>Unresolved → new Application ID: evidence investigation</h1>',
 p('2026-10-08. Scope: accepted_primary (485 frames, 16.129 FPS) and accepted_third (490 frames, 49 FPS). No production tracking code, thresholds, overlap detection, retirement, or ambiguity behavior changed.'),
 '<h2>1. Root-cause finding</h2>',
 p('<strong>E. Combination: correct new-ID creation, a verified retirement boundary, and insufficient biological evidence.</strong> Two primary creations and three stress creations represent separate sperm from the candidate old identities. No same-sperm fragmentation is established by this investigation. Twenty primary and four stress creations remain UNCERTAIN. This is not evidence that fragmentation is absent.'),
 p('The retirement boundary is real: 15/22 primary creations and 2/7 stress creations follow retirement of all historical episode candidates. But those candidates are proximity hypotheses, not established predecessors. None of the 22 primary new ByteTrack IDs had an earlier accepted Application ID. Thus the first arrow “existing ID → this unresolved sperm” is unproven in most cases.'),
 p('The reported 12 → 22 counts are distinct new-ID creations. The underlying linked episodes are 13 → 30. Several episodes point to the same creation. The old audit labels a chain as NEW_ID_AFTER_UNRESOLVED when its last ByteTrack receives a new ID within one second; it does not establish the biological old ID. Old-ID fields therefore remain null and candidate-specific measurements are retained in maps.'),
 '<h2>2. All 22 primary creations</h2>',
 p('Frames are zero-indexed. “Old ID” lists candidate hypotheses, not confirmed predecessors. Gap lists trusted-anchor ages at the instant of creation (candidate:frames); the JSON separately retains gaps to the last accepted observation, which can be low confidence. B = DIFFERENT_SPERM_CORRECT_NEW_ID; U = UNCERTAIN. All distances are pixels and velocities pixels/frame.'),
 table(['Case','Old ID (candidates)','New ID','Frame','Trusted gap','Evidence','Classification','Root cause'],
 [[c['case_id'],','.join(map(str,c['candidate_ids'])),c['new_app_id'],c['new_id_creation_frame'],', '.join(k+':'+str(v) for k,v in c['trusted_identity_gaps_at_creation'].items()),VISUAL_NOTES[c['case_id']],c['classification'],c['root_cause']] for c in primary]),
 p('<a href="primary_cases.json">Complete primary records (JSON)</a> · <a href="primary_cases.csv">Primary CSV</a> · <a href="primary_creation_snapshots.json">Exact pre-creation state snapshots</a>'),
 '<h2>3. 49-FPS analysis</h2>',
 p('There are 95 unresolved candidate episodes: 35 with an original-candidate assignment outcome, 20 with another existing ID, 8 linked to 7 distinct new-ID creations, 26 expired without linked recovery, and 6 pending at end. The old audit’s “original recovery” is an assignment result, not ground-truth biological identity. These episode counts must not be confused with the 475 ambiguity events.'),
 p('The quality metric defines an ambiguity event as one unique (frame, detection_index) with an ambiguous diagnostic, excluding UNRESOLVED_CANDIDATE diagnostics. It is not a count of independent biological episodes. Following each event’s same ByteTrack to its first later accepted assignment anywhere in the remaining video gives the exhaustive partition below. ByteTrack changes can prevent linkage; those cases remain unlinked rather than being counted as failures.'),
 table(['475-event outcome','Count','Biological interpretation'],[
 ['Candidate original ID assigned',91,'Computational recovery; correctness not established'],
 ['Different existing ID assigned',120,'Not a new ID; requires identity review'],
 ['New ID created as first later same-ByteTrack assignment',0,'No directly linked creation under this definition'],
 ['No later same-ByteTrack assignment',264,'Includes track changes and end-of-video censoring'],
 ['Confirmed same-sperm fragmentation',0,'No confirmed ground truth for these 475 events'],
 ['Confirmed genuinely different sperm',0,'The separately reviewed seven creations are a different population'],
 ['Biologically uncertain',475,'Overlays/assignment records alone cannot certify 475 biological outcomes']]),
 p('The 475 events therefore do not demonstrate a regression. Increased conservative ambiguity and improved recoveries coexist. This investigation also does not certify every ambiguous decision as biologically correct.'),
 table(['Case','Candidate old IDs','New ID','Frame','Trusted gap','Classification','Evidence'],[[c['case_id'],c['candidate_ids'],c['new_app_id'],c['new_id_creation_frame'],c['trusted_identity_gaps_at_creation'],c['classification'],VISUAL_NOTES[c['case_id']]] for c in third]),
 p('<a href="third_cases.json">Stress creation records</a> · <a href="third_ambiguity_events.json">All 475 ambiguity records and their linked outcomes</a>'),
 '<h2>4. Exact code path and lifecycle finding</h2>',
 p('All functions below belong to SpermAnalysisPipeline in sperm_pipeline/sperm_pipeline/pipeline.py. The current source SHA-256 exactly matches the accepted-run provenance: <code>'+ps['current_pipeline_sha256']+'</code>.'),
 table(['Function / source line','Condition and transition','Implication'],[
 ['_link_inference_application_ids:2116','Expires candidate chains, marks merged detections, updates identity/group states; then continuation → reassociation → occluded re-ID → low support → deferral → allocation.','New ID is reached only for an unmatched, unblocked high-confidence observation.'],
 ['_update_occlusion_states:1173; retirement:1190','frame − last_high_conf_frame > ceil(3 × FPS) sets status/state TERMINATED.','49-frame limit in primary; 147-frame limit in stress. Low support does not refresh the trusted anchor.'],
 ['_update_overlap_group_states:1231','Retired members are removed; an empty group expires; confirmed surviving member can resolve the group.','Archived group membership is not permission to resurrect a terminated identity.'],
 ['_reidentify_occluded_tracks:1311','Only LOST/OCCLUDED/UNRESOLVED unreserved identities are eligible. Size, motion, trusted prediction, competing claims and group assignment gates apply.','Terminated identities are excluded before scoring. Null scores often mean rejection before a cost was calculated.'],
 ['_unresolved_candidate_trajectory_evidence:1601','Requires >=2 observations, a first-point spatial/size anchor to the candidate identity, and continuity from the most recent point.','This helper does not prove every intermediate chain observation belongs to one sperm.'],
 ['_expire_unresolved_candidates:1682','One second since last unresolved observation, not since chain start, archives the chain.','Chains with repeated observations can span more than one second. Archival evidence still exists; it is not globally deleted.'],
 ['_resolve_unresolved_candidate:1699','Accepted original candidate plus unique same-ByteTrack or trajectory-linked chain resolves it.','Seeing a candidate elsewhere later is not by itself proof that this chain recovered.'],
 ['_defer_unresolved_new_detection:1729','Eligible recent lost candidate, ratio <=3, min(predicted/anchor distance) <=max(60,3×bbox scale) defers creation; otherwise returns false.','A chain remaining in memory alone cannot defer creation when all eligible identities disappear.'],
 ['_link_inference_application_ids:2473–2569','False deferral → allocate_id → new identity → ID_CREATED/ID_ALLOCATED/ID_CONFIRMED → NEW_HIGH_ID.','The recorded new-ID reason can come from a different surviving rejected candidate, not the retired historical candidate.']]),
 p('The hypothesized expiration sequence is possible and observed computationally. Historical evidence is not simply lost: terminated identities and candidate archives remain in the registry. The restriction is eligibility plus the quality of the retained evidence. No primary creation has valid deferred trajectory evidence at allocation under the current helper. In stress case THIRD-06, a trajectory does pass that helper after A20 retires, but its chain is not a reliable biological history.'),
 p('THIRD-06: A20’s last trusted frame is 272. U64 starts at frame 273 on B157. Its sequence changes at frames 284→B5, 301→B165, 324→B25, 356→B184, 374→B5, and 408→B139. At frame 420, A20 is TERMINATED (gap 148), groups 117/123 are EXPIRED, and a new A45 is allocated to B139. The local trajectory prediction error is only 3.612 px. Nevertheless B139 previously carried A15 at frame 391, and other chain segments have accepted owners. Extending retirement based on this smooth endpoint could merge unrelated sperm.'),
 p('The unresolved-chain builder chooses the first spatially close compatible pending chain; it does not enforce ByteTrack ownership continuity across the entire chain. This is a limitation of evidence quality and a concrete reason to distrust automatic resurrection. It is not, on its own, a reproduced same-sperm fragmentation defect. No production change is justified from this counterexample.'),
 p('Prior accepted owners at the U64 segment changes are explicitly logged: B157→A20 at frame 272, B5→A11 at frame 283, B25→A21 at frame 289, and B139→A15 at frame 391. These are prior owners, not simultaneous accepted assignments to the unresolved detections. This evidence invalidates treating the chain as automatic proof of one sperm.'),
 p('Metric limitation: same_bytetrack_app_splits counts ownership changes only between accepted observations separated by at most two frames. It is not an all-time ownership invariant. Stress B139 moves from an accepted A15 observation at frame 391 to new A45 at 420; B169 moves from accepted A13 at 395 to new A46 at 422. Both are outside that metric’s gap limit and remain biologically uncertain. No primary new ByteTrack in these 22 cases has any earlier accepted owner.'),
 '<h2>5. Successful recovery versus new-ID creation</h2>',
 table(['Video / metric','Original-candidate assignment episodes','New-ID-linked episodes'],[
 ['Primary count','11','30 episodes → 22 creations'],['Primary median unresolved duration','2 frames / 0.124 s','15 frames / 0.930 s'],
 ['Primary duration range','1–39 frames','1–45 frames'],['Primary median historical candidate count','3','2'],
 ['Primary median confidence','0.822','0.840'],['Stress count','35','8 episodes → 7 creations'],
 ['Stress median unresolved duration','2 frames / 0.0408 s','5.5 frames / 0.1122 s'],
 ['Stress duration range','1–79 frames','1–147 frames'],['Stress median candidate count','3','1'],
 ['Stress median confidence','0.911','0.809']]),
 p('These distributions overlap. Neither confidence, duration nor candidate count reliably separates biological fragmentation from correct new IDs. The strongest measured distinction is lifecycle eligibility, which explains code behavior but does not label sperm identity. New-ID rows have no winning old-candidate score, so treating a missing score as zero would be invalid. Scores from continuation, reassociation and re-ID also use different cost definitions.'),
 table(['Trusted-anchor displacement','Successful assigned identity','All historical candidate/new-ID pairs'],[
 ['Primary range / median','4.4–87.7 px / 42.3 px (11)','96.1–656.6 px / 304.1 px (50 pairs)'],
 ['Stress range / median','0.2–241.9 px / 30.3 px (35)','37.2–240.8 px / 165.5 px (12 pairs)']]),
 p('Primary displacement separates these computational outcomes, but it does not generalize to the stress video. It is also selection-biased by existing recovery gates and compares known assigned identities with unproven candidate pairs. It cannot justify a new threshold or classify fragmentation. Velocity, direction, bbox ratio, confidence and lifecycle status for every pair are retained in the feature JSON files.'),
 p('Concrete comparisons: primary U28 recovers A17 at frame 105, while U59 leads to A34 at frame 215 after A17’s trusted age reaches 50. Primary U58 recovers A30 at frame 212; A30 remains confirmed on a separate sperm when U52 produces A35 at frame 215. Stress U1 recovers A26 at frame 6 while U10 produces separate A35 in the same frame. Primary U124 recovers A52 at frame 475, after separate A62 was created at frame 454.'),
 p('Candidate-by-candidate displacement, exact pre-assignment velocity, direction dot product, bbox ratio, confidence, overlap state, trusted gap, recovery limit and raw score evidence are in <a href="primary_feature_comparison.json">primary feature comparison</a> and <a href="third_feature_comparison.json">stress feature comparison</a>. New-ID comparisons retain every historical candidate instead of choosing the closest as presumed truth. Successful snapshots preserve the actual assigned identity and its pre-assignment state.'),
 '<h2>6. Recommendation and regression results</h2>',
 p('<strong>No tracking fix implemented.</strong> First obtain continuous frame-level predecessor annotations for the uncertain cases. If a reproducible same-sperm failure emerges, test bounded deferred recovery with chain ownership and competing-sperm evidence. Do not extend the window or reuse expired groups based solely on local trajectory smoothness. No safe threshold change is demonstrated here.'),
 p('The full existing regression suite passed: <strong>94 passed in 7.68s</strong>, with <code>PYTHONPATH=sperm_pipeline /home/user_garbha/sperm/bin/python3 -m pytest -q</code>. An initial invocation without PYTHONPATH failed during import collection; the correctly configured run passed. No behavior fix was implemented, so no new behavior regression tests were added. The new files are offline diagnostic/report runners, not production code.'),
 p('Primary saved-observation replay matches all 1,790 accepted assignments including detection indices. Stress matches all 10,391 accepted identity/geometry/type assignments after normalizing detection indices, and reproduces 475 ambiguity events. Six stress frames (3, 428–432) omit internal detection indices in the logs; compaction causes raw-index mismatch. Stress snapshots are qualified replay evidence, not a new full detector validation. Source-video samples were reviewed separately.'),
 '<h3>Previously verified engineering changes (historical before → accepted baseline)</h3>']
metrics=['application_ids_created','same_bytetrack_app_splits','possible_fragmentations','possible_id_switches','ambiguous_events','occlusion_recoveries']
for name,summary,cases in [('primary',ps,primary),('third',ts,third)]:
    before=read(BASE/'targeted_audit'/('before_'+name+'.json'))
    after=read(BASE/'targeted_audit'/('final_'+name+'.json'))
    rows=[[m,before['quality'][m],after['quality'][m],after['quality'][m]-before['quality'][m]] for m in metrics]
    rows.append(['overlap_groups',sum(before['group_states'].values()),sum(after['group_states'].values()),sum(after['group_states'].values())-sum(before['group_states'].values())])
    def count_creations(a):return len({e['following_same_byte_assignment'][0]['candidate_application_id'] for e in a['unresolved_episodes'] if e['outcome']=='NEW_ID_AFTER_UNRESOLVED'})
    rows.append(['unresolved → distinct new IDs',count_creations(before),count_creations(after),count_creations(after)-count_creations(before)])
    out.append('<h4>'+name+'</h4>'+table(['Metric','Before historical fixes','After historical fixes','Difference'],rows))
out += [p('For this investigation itself, before = the accepted baseline and after = unchanged tracking behavior: primary IDs 65→65, splits 0→0, flags 20→20, switches 0→0, groups 7→7, ambiguity 0→0, occlusion recoveries 15→15, unresolved creations 22→22. Stress IDs 49→49, splits 0→0, flags 11→11, switches 0→0, groups 145→145, ambiguity 475→475, occlusion recoveries 69→69, unresolved creations 7→7. Every difference is zero. Regression suite 94→94 (difference zero). These are unchanged-source/artifact metrics, not a new full detector rerun.'),
 '<h2>7. Final decision</h2>',
 p('<strong>NO TRACKING CODE CHANGE JUSTIFIED BY THE CURRENT EVIDENCE; BIOLOGICAL CORRECTNESS REMAINS PARTLY UNRESOLVED.</strong>'),
 p('None of the requested three unqualified final labels is supported: FIX REQUIRED or TARGETED DEFERRED RE-IDENTIFICATION FIX REQUIRED would assert a reproduced biological failure that is not established; NO CODE CHANGE — CURRENT BEHAVIOR IS CORRECT would certify 24 uncertain cases. This report does not declare the identity problem solved.'),
 '<h2>8. Per-case evidence and all nine lifecycle answers</h2>']
for name,cases in [('primary',primary),('third',third)]:
    for c in cases:
        out.append('<details><summary>'+esc(c['case_id']+' → A'+str(c['new_app_id'])+' — '+c['classification'])+'</summary>')
        out.append(p(esc(c['visual_review_note'])))
        out.append('<img loading="lazy" src="'+name+'_visual/'+c['case_id']+'.jpg" alt="Source frame evidence for '+c['case_id']+'">')
        out.append('<pre>'+esc(json.dumps(c['questions_and_answers'],indent=2))+'</pre></details>')
out.append(p('Reproduce in order: tests/investigate_unresolved_identity.py; PYTHONPATH=sperm_pipeline tests/replay_unresolved_lifecycle.py; tests/render_unresolved_lifecycle.py; tests/report_unresolved_lifecycle.py, all using /home/user_garbha/sperm/bin/python3. Original accepted artifacts are read-only. The report generation includes the explicitly documented sampled visual classifications above.'))
(DEST/'report.html').write_text('\n'.join(out))
write(DEST/'decision.json',{'root_cause_category':'E','production_code_changed':False,
    'decision':'NO_TRACKING_CODE_CHANGE_JUSTIFIED; BIOLOGICAL_CORRECTNESS_PARTLY_UNRESOLVED',
    'primary':ps['case_classifications'],'stress_creations':ts['case_classifications'],
    'tests':{'before':94,'after':94,'difference':0},
    'requested_final_labels_supported':False})
print('Report:',DEST/'report.html')
