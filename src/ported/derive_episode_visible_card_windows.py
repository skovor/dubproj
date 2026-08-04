"""Derive displayed-card intervals for already visually identified Episode Aigis cards.

Identity is read from EPISODE_AIGIS_PIXEL_CARDS_AUDITED.json. This program
only measures the on-screen card envelope around those known points; it never
uses an audio transcript to select a line.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2, numpy as np
from extract_subtitle_cards_from_longplay import card_mask, is_caption, distance

DATA=Path(r"E:\P3R_FMV_SUBTITLE_TIMING")
VIDEO=DATA/'EpisodeAigis_anime_bc0KWMeeAMc_1080p.mp4'
AUDIT=DATA/'EPISODE_AIGIS_PIXEL_CARDS_AUDITED.json'
FINAL=DATA/'EPISODE_AIGIS_USM_CARD_TIMING_FINAL.json'
OUT=DATA/'EPISODE_AIGIS_VISIBLE_CARD_WINDOWS.json'

def segments(cap, center, radius=3.0, hz=20.0):
    fps=cap.get(cv2.CAP_PROP_FPS) or 60.; step=max(1,round(fps/hz))
    cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,round((center-radius)*fps)))
    end=round((center+radius)*fps); state=None; result=[]; f=round(cap.get(cv2.CAP_PROP_POS_FRAMES))
    while f<=end:
        ok,frame=cap.read(); f+=1
        if not ok: break
        if f%step: continue
        t=f/fps; mask=card_mask(frame)
        if not is_caption(mask):
            if state: state['end_video_s']=t; result.append(state); state=None
            continue
        if state is None:
            state={'start_video_s':t,'end_video_s':t,'mask':mask}
        elif distance(mask,state['mask'])>.075:
            state['end_video_s']=t; result.append(state); state={'start_video_s':t,'end_video_s':t,'mask':mask}
        else: state['end_video_s']=t
    if state: result.append(state)
    return result

def main():
    d=json.loads(AUDIT.read_text(encoding='utf8'))
    accepted=[x for x in d['cards'] if x.get('verdict','').startswith('PIXEL_APPROVED')]
    # collapse repeated scans of the same held card; earliest sample is safest.
    chosen={}
    for x in accepted:
        k=(x['candidate_scene'],x['candidate_line_id'])
        if k not in chosen or x['video_time_s']<chosen[k]['video_time_s']: chosen[k]=x
    cap=cv2.VideoCapture(str(VIDEO)); rows=[]
    for key,x in sorted(chosen.items()):
        parts=segments(cap,float(x['video_time_s']))
        if not parts:
            rows.append({'scene':key[0],'line_id':key[1],'status':'WINDOW_NOT_FOUND','anchor_video_s':x['video_time_s']}); continue
        # pick the segment containing the visual point or closest interval.
        t=float(x['video_time_s'])
        p=min(parts,key=lambda y:0 if y['start_video_s']<=t<=y['end_video_s'] else min(abs(t-y['start_video_s']),abs(t-y['end_video_s'])))
        rows.append({'scene':key[0],'line_id':key[1],'status':'VISIBLE_WINDOW_READY','anchor_video_s':t,
                     'start_video_s':round(p['start_video_s'],3),'end_video_s':round(p['end_video_s'],3),
                     'duration_s':round(p['end_video_s']-p['start_video_s'],3),'identity_evidence':x['verdict']})
    cap.release(); OUT.write_text(json.dumps({'policy':'Identity locked by visible official card before window measurement; no ASR identity inference.','rows':rows},indent=2),encoding='utf8')
    print('ready',sum(x['status']=='VISIBLE_WINDOW_READY' for x in rows),'failed',sum(x['status']!='VISIBLE_WINDOW_READY' for x in rows),'total',len(rows))
if __name__=='__main__': main()
