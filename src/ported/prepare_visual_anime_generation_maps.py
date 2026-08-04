"""Convert the visual-card allowlist into producer-compatible scene maps."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(r"C:\Users\juand\Desktop\moddeutsch\p3r_text_pipeline")
INV=ROOT/'anime_all_20260725'/'anime_dialogue_inventory.json'
SRC=ROOT/'FMV_VISUAL_GATE_20260801'
OUT=ROOT/'P3R_ANIME_VISUAL_DUB_20260801'/'maps'
DELIVERY_OUT=ROOT/'P3R_ANIME_VISUAL_DUB_20260801'/'maps_delivery_aligned_v2'
OVERRIDES=OUT.parent/'NONVERBAL_AUDIO_OVERRIDES.json'

STRUCTURAL_KEYS={
    'id','speaker','source_text','target_text','start','end','force_clone',
    'mapping_validation','mapping_validation_reason','timing_source',
    'timing_review_required','window_contract','production_action',
}

# Keep the subtitle's full target text in the inventory while using a compact,
# semantically equivalent spoken delivery when the immutable visual window is
# too short for a literal German expansion.  This prevents OmniVoice from
# being forced into a late-word cutoff on the newly recovered L009 cue.
DELIVERY_TEXT_OVERRIDES={
    '100_090_M_L009': {
        'delivery_text':'Willkommen an der Gekkoukan-Oberschule. Ich hoffe, es gefällt dir!',
        'synthesis_text_override':'Willkommen an der Gekkoukan-Oberschule. Ich hoffe, es gefällt dir!',
        'delivery_note':'compact semantic delivery for 4.329 s visible card; subtitle target remains canonical',
    },
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    overrides={x['id']:x for x in json.loads(OVERRIDES.read_text(encoding='utf8'))} if OVERRIDES.is_file() else {}
    inventory={x['scene']:x for x in json.loads(INV.read_text(encoding='utf8'))}
    manifest=[]
    for p in sorted(SRC.glob('*_visible_card_map.json')):
        gate=json.loads(p.read_text(encoding='utf8')); inv=inventory[gate['scene']]
        lines=[]
        for row in gate['lines']:
            line={
              'id':row['id'],'speaker':row['speaker'],'source_text':row['source_text'],'target_text':row['target_text'],
              'start':row['window_start'],'end':row['window_end'],
              'force_clone':True,'mapping_validation':'HUMAN_CONFIRMED',
              'mapping_validation_reason':f"visible subtitle card: {row['identity_source']}",
              'timing_source':'VISIBLE_OFFICIAL_SUBTITLE_CARD_WINDOW',
              'timing_review_required':False,
              'window_contract':gate['window_contract'],
            }
            if row['id'] in overrides:
                line.update({'force_keep_original':True,'force_keep_reason':overrides[row['id']]['reason'],
                             'force_clone':False,'production_action':overrides[row['id']]['action']})
            if row['id'] in DELIVERY_TEXT_OVERRIDES:
                line.update(DELIVERY_TEXT_OVERRIDES[row['id']])
            lines.append(line)
        payload={'scene':gate['scene'],'kind':'ANIME_USM_EMBEDDED_MIX','source_stem':inv['source_dialog_ch5'],
                 'container_frames':inv['container_frames'],'sample_rate':inv['sample_rate'],
                 'expected_visual_ids':gate.get('expected_visual_ids', [x['id'] for x in lines]),
                 'coverage_status':gate.get('coverage_status','UNKNOWN'),
                 'notes':'Visual-card-only production map. No legacy ASR/order/energy mappings.',
                 'lines':lines}
        out=OUT/f"{gate['scene']}_map.json"
        out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf8')

        # The delivery-aligned directory is what the regular OmniVoice worker
        # consumes.  Keep its previously validated candidate/reference fields
        # (same-actress references, delivery_text/synthesis overrides, retry
        # policy) while replacing only the visual-card IDs and edges above.
        legacy_path=DELIVERY_OUT/f"{gate['scene']}_map.json"
        legacy_by_id={}
        if legacy_path.is_file():
            legacy=json.loads(legacy_path.read_text(encoding='utf8'))
            legacy_by_id={x['id']:x for x in legacy.get('lines',[])}
        delivery_lines=[]
        for line in lines:
            merged=dict(line)
            old=legacy_by_id.get(line['id'])
            if old:
                for key,value in old.items():
                    if key not in STRUCTURAL_KEYS:
                        merged[key]=value
            delivery_lines.append(merged)
        delivery_payload=dict(payload)
        delivery_payload['notes']=('Visual-card-complete delivery map. '
                                   'Visual windows are authoritative; legacy candidate/reference metadata retained.')
        delivery_payload['lines']=delivery_lines
        DELIVERY_OUT.mkdir(parents=True,exist_ok=True)
        legacy_path.write_text(json.dumps(delivery_payload,indent=2,ensure_ascii=False),encoding='utf8')
        manifest.append({'scene':gate['scene'],'map':str(out),'delivery_map':str(legacy_path),'lines':len(lines)})
    report={'profile':{'engine':'OmniVoice 0.2.1','steps':32,'guidance':2.0,'postprocess_output':False,'batch':2,
                       'initial_candidates':4,'conditional_retry_candidates':4},'scenes':manifest,'lines':sum(x['lines'] for x in manifest)}
    (OUT.parent/'MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'scenes':len(manifest),'lines':report['lines']},indent=2))
if __name__=='__main__':main()
