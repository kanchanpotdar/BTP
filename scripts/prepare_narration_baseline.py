"""Narration-only curation proposal. No denoising, splits, training or label edits."""
from pathlib import Path
import argparse,collections,hashlib,io,json,sys,unicodedata as ud
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_audit_2026-10-27/_analysis_deps'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_duplicate_flags(master, audio_hashes):
    out=master[['sample_id','transcript_missing','transcript_normalized']].merge(audio_hashes,on='sample_id',validate='one_to_one')
    usable=out[out.pcm_sha256.notna()].copy()
    count=usable.groupby('pcm_sha256').sample_id.size()
    label_counts=usable.loc[~usable.transcript_missing].groupby('pcm_sha256').transcript_normalized.nunique()
    # Prefer labeled records so a missing-label twin never suppresses the labeled example.
    canonical=usable.sort_values(['transcript_missing','sample_id']).drop_duplicates('pcm_sha256').set_index('pcm_sha256').sample_id
    out['exact_audio_group_size']=out.pcm_sha256.map(count).fillna(1).astype(int)
    out['exact_audio_label_conflict']=out.pcm_sha256.map(label_counts).fillna(0).gt(1)
    out['exact_audio_canonical_id']=out.pcm_sha256.map(canonical)
    out['exact_audio_redundant']=out.exact_audio_group_size.gt(1)&out.sample_id.ne(out.exact_audio_canonical_id)
    return out.drop(columns=['transcript_missing','transcript_normalized'])


def apply_policy(master, unresolved, duplicate_flags, low_level):
    flags=master.merge(duplicate_flags,on='sample_id',validate='one_to_one')
    flags['missing_label']=flags.transcript_missing
    flags['invalid_audio']=~flags.audio_ok
    flags['unresolved_symbol']=flags.sample_id.isin(unresolved)
    flags['low_absolute_level_review']=flags.sample_id.isin(low_level)
    flags['non_ok_speaker']=flags.speaker_status.ne('ok')
    flags['duration_outside_05_30']=~flags.duration_decoded.between(.5,30)
    flags['duration_outside_1_25']=~flags.duration_decoded.between(1,25)
    shared=~(flags.missing_label|flags.invalid_audio|flags.exact_audio_label_conflict|flags.exact_audio_redundant)
    flags['candidate_expanded_05_30']=shared&~flags.duration_outside_05_30
    flags['candidate_primary_05_30']=flags.candidate_expanded_05_30&~flags.non_ok_speaker&~flags.unresolved_symbol
    flags['candidate_common_1_25']=shared&~flags.duration_outside_1_25&~flags.non_ok_speaker&~flags.unresolved_symbol
    # Candidates are not reviewed references or final eligibility decisions.
    flags['review_status']='not_reviewed'
    flags['split_status']='unassigned'
    for col in ['candidate_primary_05_30','candidate_common_1_25']:
        selected=flags[flags[col]]
        assert not selected[['missing_label','invalid_audio','unresolved_symbol','exact_audio_label_conflict','exact_audio_redundant','non_ok_speaker']].any().any()
    return flags


def group_components(flags):
    ids=flags.sample_id.tolist();parent={sid:sid for sid in ids}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for key in ['speaker_group_id','recording_session_id','pcm_sha256']:
        for _,g in flags[flags[key].notna()].groupby(key,sort=True):
            members=g.sample_id.tolist()
            for sid in members[1:]:union(members[0],sid)
    flags['split_component_id']=flags.sample_id.map(lambda sid:'SC_'+find(sid))
    flags['component_train_only']=flags.groupby('split_component_id').group_train_only.transform('max')
    for key in ['speaker_group_id','recording_session_id','pcm_sha256']:
        assert flags[flags[key].notna()].groupby(key).split_component_id.nunique().max()==1
    return flags


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/narration_curation_20261028')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError('Refuse overwrite: '+str(out))
    out.mkdir(parents=True)
    source=ROOT/'manifests/master_A_v1_3.parquet';before=sha(source);master=pd.read_parquet(source)
    assert master.shape==(6718,57) and master.sample_id.is_unique
    unresolved=set(pd.read_csv(ROOT/'cleaning/unresolved_symbol_rows_A_v2.csv',keep_default_na=False).sample_id)
    measurements=pd.read_csv(ROOT/'reports/audio_signal_audit_2026-10-28/signal_metrics_A.csv')
    low_level=set(measurements.loc[measurements.rms_dbfs < -50,'sample_id'])
    lookup=dict(zip(master.audio_path,master.sample_id));records=[]
    for path in sorted((ROOT/'raw/A_narration/data').glob('*/*.parquet')):
        row_index=0
        for batch in pq.ParquetFile(path).iter_batches(columns=['audio'],batch_size=16):
            for a in batch.column('audio').to_pylist():
                locator='raw/A_narration/data/'+path.parent.name+'/'+path.name+'#row='+str(row_index);row_index+=1
                signal,rate=sf.read(io.BytesIO(a['bytes']),dtype='int16',always_2d=True)
                # Never group empty waveforms as ordinary duplicate speech.
                h=hashlib.sha256(f'{rate}:{signal.shape[1]}:PCM16:'.encode()+signal.astype('<i2',copy=False).tobytes()).hexdigest() if len(signal) else None
                records.append({'sample_id':lookup[locator],'pcm_sha256':h})
    hashes=pd.DataFrame(records);assert len(hashes)==len(master) and hashes.sample_id.is_unique
    dup=exact_duplicate_flags(master,hashes)
    flags=group_components(apply_policy(master,unresolved,dup,low_level))
    columns=['sample_id','audio_path','duration_decoded','transcript_missing','speaker_id','speaker_group_id',
             'recording_session_id','group_train_only','pcm_sha256','exact_audio_group_size','exact_audio_label_conflict',
             'exact_audio_canonical_id','exact_audio_redundant','missing_label','invalid_audio','unresolved_symbol',
             'low_absolute_level_review','non_ok_speaker','duration_outside_05_30','duration_outside_1_25',
             'candidate_expanded_05_30','candidate_primary_05_30','candidate_common_1_25','split_component_id',
             'component_train_only','review_status','split_status']
    flags[columns].to_parquet(out/'curation_flags_A.parquet',index=False)
    flags[columns].to_csv(out/'curation_flags_A.csv',index=False,encoding='utf-8-sig')
    duplicate_rows=flags[flags.exact_audio_group_size.gt(1)]
    duplicate_rows[['sample_id','pcm_sha256','exact_audio_group_size','exact_audio_label_conflict',
                    'exact_audio_canonical_id','transcript_normalized','audio_path']].to_csv(out/'exact_audio_duplicates.csv',index=False,encoding='utf-8-sig')
    for policy in ['candidate_expanded_05_30','candidate_primary_05_30','candidate_common_1_25']:
        flags.loc[flags[policy],['sample_id','audio_path','duration_decoded','transcript_normalized','speaker_group_id',
                    'recording_session_id','split_component_id','component_train_only','review_status','split_status']].to_parquet(out/(policy+'.parquet'),index=False)
    # Unicode inventory identifies conventions, without declaring mark meanings or inventing tone labels.
    inventory=[]
    for layer in ['transcript_raw','transcript_normalized']:
        text=flags.loc[~flags.transcript_missing,layer]
        count=collections.Counter(''.join(text));occ=collections.Counter(c for t in text for c in set(t))
        for char,n in sorted(count.items(),key=lambda x:ord(x[0])):
            inventory.append({'layer':layer,'character':char,'codepoint':f'U+{ord(char):04X}','unicode_name':ud.name(char,'UNNAMED'),
                              'category':ud.category(char),'occurrences':n,'transcripts':occ[char]})
    pd.DataFrame(inventory).to_csv(out/'ipa_inventory.csv',index=False,encoding='utf-8-sig')
    # Cohort waterfall has fixed sequential semantics. Other flags remain independently visible.
    waterfall=[];mask=pd.Series(True,index=flags.index)
    for name,condition in [('all_rows',pd.Series(True,index=flags.index)),('has_reference',~flags.missing_label),
            ('audio_integrity_ok',~flags.invalid_audio),('duration_05_30',~flags.duration_outside_05_30),
            ('no_conflicting_exact_audio_labels',~flags.exact_audio_label_conflict),('one_copy_per_exact_audio',~flags.exact_audio_redundant),
            ('speaker_status_ok',~flags.non_ok_speaker),('no_unresolved_symbols',~flags.unresolved_symbol)]:
        previous=int(mask.sum());mask &= condition
        waterfall.append({'step':name,'rows_remaining':int(mask.sum()),'hours_remaining':float(flags.loc[mask,'duration_decoded'].sum()/3600),'rows_removed_this_step':previous-int(mask.sum())})
    assert set(flags.loc[mask,'sample_id'])==set(flags.loc[flags.candidate_primary_05_30,'sample_id'])
    pd.DataFrame(waterfall).to_csv(out/'eligibility_waterfall.csv',index=False)
    # Small first listening pack: all symbol flags and anomalies, quiet examples and speaker coverage.
    chosen=collections.defaultdict(set)
    for sid in unresolved:chosen[sid].add('unresolved_symbol')
    for sid in flags.loc[flags.invalid_audio,'sample_id']:chosen[sid].add('audio_integrity_anomaly')
    for sid in measurements[measurements.frames.gt(0)].nsmallest(20,'rms_dbfs').sample_id:chosen[sid].add('low_level_review_not_noise_label')
    primary=flags[flags.candidate_primary_05_30]
    for _,g in primary.groupby('speaker_group_id',sort=True):
        # Stable deterministic representative; no test-set selection or assignment occurs.
        sid=min(g.sample_id,key=lambda x:hashlib.sha256(('20261028:'+x).encode()).hexdigest())
        chosen[sid].add('speaker_group_representative')
    queue=flags[flags.sample_id.isin(chosen)].copy();queue['selection_reason']=queue.sample_id.map(lambda x:';'.join(sorted(chosen[x])))
    queue['review_audio_path']=queue.sample_id.map(lambda x:'audio/'+x+'.wav')
    for col in ['audibility_review','noise_review','nyishi_language_review','reference_accuracy_review','boundary_review',
                'symbol_or_tone_review','reviewer','review_date','decision','review_notes']:queue[col]=''
    qcols=['sample_id','selection_reason','review_audio_path','audio_path','duration_decoded','speaker_group_id','recording_month',
           'transcript_raw','transcript_normalized','audibility_review','noise_review','nyishi_language_review',
           'reference_accuracy_review','boundary_review','symbol_or_tone_review','reviewer','review_date','decision','review_notes']
    queue[qcols].to_csv(out/'listening_review.csv',index=False,encoding='utf-8-sig')
    (out/'audio').mkdir();target=set(queue.sample_id)
    for path in sorted((ROOT/'raw/A_narration/data').glob('*/*.parquet')):
        requested=[(loc,sid) for loc,sid in lookup.items() if sid in target and loc.startswith('raw/A_narration/data/'+path.parent.name+'/'+path.name+'#')]
        if not requested:continue
        audio=pq.read_table(path,columns=['audio']).column('audio')
        for locator,sid in requested:(out/'audio'/(sid+'.wav')).write_bytes(audio[int(locator.rsplit('=',1)[1])].as_py()['bytes'])
    pools={}
    for col in ['candidate_expanded_05_30','candidate_primary_05_30','candidate_common_1_25']:
        subset=flags[flags[col]];pools[col]={'rows':len(subset),'hours':float(subset.duration_decoded.sum()/3600),
            'speaker_groups':int(subset.speaker_group_id.nunique()),'split_components':int(subset.split_component_id.nunique()),
            'components_allowed_as_heldout_candidates':int(subset.loc[~subset.component_train_only,'split_component_id'].nunique())}
    summary={'scope':'Dataset A Narration only','source_master_sha256':before,'script_sha256':sha(__file__),
        'policies_are_proposals':True,'references_linguistically_approved':False,'pools':pools,
        'duplicate_audio_groups':int(duplicate_rows.pcm_sha256.nunique()),'duplicate_audio_rows':len(duplicate_rows),
        'conflicting_label_audio_groups':int(flags.loc[flags.exact_audio_label_conflict,'pcm_sha256'].nunique()),
        'conflicting_label_audio_rows':int(flags.exact_audio_label_conflict.sum()),
        'review_pack_clips':len(queue),'low_level_rows_automatically_excluded':0,
        'tone_inventory_note':'Unicode names are descriptive; no automatic interpretation of accents as lexical tone.',
        'splits_created':False,'models_trained':False,'masters_modified':False,'waterfall':waterfall}
    assert sha(source)==before
    # All saved candidates reload and obey their flags; audio extracts retain the original bytes.
    reloaded=pd.read_parquet(out/'curation_flags_A.parquet');assert reloaded.shape==(6718,len(columns)) and reloaded.sample_id.is_unique
    for col in pools:
        selected=pd.read_parquet(out/(col+'.parquet'));assert set(selected.sample_id)==set(flags.loc[flags[col],'sample_id'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
