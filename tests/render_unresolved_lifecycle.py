"""Render source-video evidence for every deduplicated unresolved creation."""
from collections import defaultdict
import cv2
import numpy as np
from investigate_unresolved_identity import BASE, DEST, read, write

for name in ['primary','third']:
    cases=read(DEST/(name+'_cases.json'))
    source=read(BASE/('accepted_'+name)/'replay_complete.json')['source']
    diagnostics=read(BASE/('accepted_'+name)/'meta/identity_assignment_diagnostics.json')
    by_frame=defaultdict(list)
    for row in diagnostics:
        if row.get('accepted'): by_frame[row['frame']].append(row)
    cap=cv2.VideoCapture(source)
    assert cap.isOpened(),source
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out=DEST/(name+'_visual');out.mkdir(exist_ok=True)
    for c in cases:
        frame=c['new_id_creation_frame'];start=c['start_frame'];end=c['unresolved_end_frame']
        candidates=c['per_candidate_evidence']
        closest=min(candidates,key=lambda r:r['distance_gap'] if r['distance_gap'] is not None else float('inf'))
        old_frame=closest['old_last_before_unresolved_frame']
        frames=[old_frame if old_frame is not None else start-1,start,(start+end)//2,end,frame,min(total-1,frame+8)]
        boxes=[c['new_bbox_after_unresolved']]+[r['old_bbox_before_unresolved'] for r in candidates if r['old_bbox_before_unresolved']]
        boxes+= [ob['bbox'] for ch in c['candidate_chains'] for ob in ch['observations']]
        lo=np.min(np.array(boxes)[:,:2],axis=0)-30; hi=np.max(np.array(boxes)[:,2:],axis=0)+30
        x1,y1=max(0,int(lo[0])),max(0,int(lo[1]));x2,y2=int(hi[0]),int(hi[1])
        panels=[]
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,f));ok,im=cap.read();assert ok
            for r in by_frame[f]:
                box=[round(v) for v in r['current_bbox']];app=r['candidate_application_id']
                color=(0,255,0) if app==c['new_app_id'] else (255,180,0) if app in c['candidate_ids'] else (150,150,150)
                cv2.rectangle(im,box[:2],box[2:],color,1)
                cv2.putText(im,f'A{app}/B{r.get("byte_track_id")}',(box[0],max(12,box[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            for chain in c['candidate_chains']:
                for ob in chain['observations']:
                    if ob['frame']==f:
                        box=[round(v) for v in ob['bbox']];cv2.rectangle(im,box[:2],box[2:],(0,255,255),2)
                        cv2.putText(im,f'U{chain["candidate_id"]}/B{ob["tracker_id"]}',(box[0],max(12,box[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
            crop=im[y1:y2,x1:x2];h,w=crop.shape[:2];scale=min(480/w,340/h)
            crop=cv2.resize(crop,(round(w*scale),round(h*scale)))
            panel=np.zeros((380,480,3),np.uint8);panel[35:35+crop.shape[0],:crop.shape[1]]=crop
            cv2.putText(panel,f'{c["case_id"]} f{f} new A{c["new_app_id"]}',(5,16),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.putText(panel,'Candidates '+str(c['candidate_ids']),(5,31),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
            panels.append(panel)
        sheet=np.concatenate([np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)])
        cv2.imwrite(str(out/(c['case_id']+'.jpg')),sheet)
        c['visual_evidence']={'source':source,'contact_sheet':str(out/(c['case_id']+'.jpg')),'frames':frames,'crop':[x1,y1,x2,y2],
                              'limitation':'Six sampled frames do not establish biological continuity through an occlusion.'}
    write(DEST/(name+'_cases.json'),cases)
    cap.release()
    print('rendered',name,len(cases),flush=True)
