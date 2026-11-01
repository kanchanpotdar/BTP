"""Build private, frozen-train-only ~1h Phase 6B pilot bundle. No split mutation.

Run from BTP root with its Python 3.11 environment. Outputs stay under outputs/.
"""
from collections import defaultdict
from io import BytesIO
from pathlib import Path
import hashlib, json, math, zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'manifests/frozen_A_narration_phase3_5b_v1/train.parquet'
RAW=(ROOT/'raw/A_narration').resolve()
TARGET_SECONDS=3600.0; SEED=0
OUT=ROOT/'outputs/phase6b_pilot_1h_v1'
BUNDLE=OUT/'phase6b_pilot_train_bundle'; AUDIO=BUNDLE/'audio'
if OUT.exists(): raise FileExistsError(f'Refusing to overwrite versioned output: {OUT}')
train=pd.read_parquet(FROZEN)
assert len(train)==3629 and set(train.frozen_split)=={'train'}
assert train.sample_id.is_unique and train.component_id.nunique()==15
assert math.isclose(train.duration_exact.sum()/3600,10.851636892361112,abs_tol=1e-6)
assert train.transcript_phase2.notna().all() and train.duration_exact.between(.5,30,inclusive='left').all()
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):d.update(chunk)
 return d.hexdigest()
def order_key(row):return hashlib.sha256(f'{SEED}:{row.sample_id}'.encode()).hexdigest()
# Give each of 15 existing components a chance to contribute up to four minutes,
# then fill the remaining target by deterministic round-robin over unused clips.
groups={c:sorted(group.itertuples(index=False),key=order_key) for c,group in train.groupby('component_id')}
selected=[];seen=set();component_seconds=defaultdict(float)
for component,items in sorted(groups.items()):
 for row in items:
  if component_seconds[component]>=240:break
  selected.append(row);seen.add(row.sample_id);component_seconds[component]+=float(row.duration_exact)
remaining={c:[r for r in rows if r.sample_id not in seen] for c,rows in groups.items()}
while sum(float(r.duration_exact) for r in selected)<TARGET_SECONDS:
 progressed=False
 for component in sorted(remaining):
  if not remaining[component]:continue
  row=remaining[component].pop(0);selected.append(row);seen.add(row.sample_id)
  component_seconds[component]+=float(row.duration_exact);progressed=True
  if sum(float(r.duration_exact) for r in selected)>=TARGET_SECONDS:break
 if not progressed:raise RuntimeError('Frozen train pool exhausted before target.')
selected=sorted(selected,key=lambda r:r.sample_id)
selected_ids=[r.sample_id for r in selected]
assert len(selected_ids)==len(set(selected_ids)) and set(selected_ids).issubset(set(train.sample_id))
assert TARGET_SECONDS <= sum(float(r.duration_exact) for r in selected)<TARGET_SECONDS+30
# Private membership hash, independent of file layout.
membership_sha=hashlib.sha256(('\n'.join(selected_ids)+'\n').encode()).hexdigest()
alias={component:f'train_group_{i:02d}' for i,component in enumerate(sorted(groups),1)}
AUDIO.mkdir(parents=True)
source_plan=defaultdict(list)
for row in selected:
 text,idx=row.audio_path.rsplit('#row=',1);rel=Path(text)
 if rel.is_absolute() or '..' in rel.parts or not idx.isdigit():raise ValueError('Unsafe source locator')
 path=(ROOT/rel).resolve()
 if not path.is_relative_to(RAW) or not path.is_file():raise ValueError('Source outside raw Narration tree')
 source_plan[path].append((row,int(idx)))
def locate(parquet,index):
 start=0
 for group in range(parquet.num_row_groups):
  count=parquet.metadata.row_group(group).num_rows
  if start<=index<start+count:return group,index-start
  start+=count
 raise IndexError(index)
records=[];source_hashes={}
for path in sorted(source_plan):
 source_hashes[str(path.relative_to(ROOT))]=sha(path)
 parquet=pq.ParquetFile(path);by_group=defaultdict(list)
 for row,index in source_plan[path]:
  group,offset=locate(parquet,index);by_group[group].append((row,offset))
 for group,items in sorted(by_group.items()):
  column=parquet.read_row_group(group,columns=['audio']).column('audio')
  for row,offset in items:
   obj=column[offset].as_py();audio_bytes=obj.get('bytes') if isinstance(obj,dict) else None
   if not audio_bytes:raise ValueError('Missing source audio bytes')
   info=sf.info(BytesIO(audio_bytes))
   if info.format!='WAV' or info.subtype!='PCM_16':raise ValueError('Source audio format changed')
   samples,rate=sf.read(BytesIO(audio_bytes),dtype='int16',always_2d=True)
   if rate!=16000 or samples.shape!=(int(row.decoded_frames),1):raise ValueError('Source waveform mismatch')
   digest=hashlib.sha256(rate.to_bytes(4,'little')+samples.tobytes(order='C')).hexdigest()
   if digest!=row.exact_waveform_sha256:raise ValueError('Frozen waveform hash mismatch')
   dest=AUDIO/f'{row.sample_id}.wav';dest.write_bytes(audio_bytes)
   records.append({'sample_id':row.sample_id,'split':'train','component_id':alias[row.component_id],
    'audio_file':f'audio/{row.sample_id}.wav','reference_raw':row.transcript_raw,
    'reference':row.transcript_phase2,'duration_seconds':len(samples)/rate,
    'sample_rate':rate,'channels':1,'frames':len(samples),'audio_sha256':hashlib.sha256(audio_bytes).hexdigest(),
    'waveform_sha256':digest})
print('Extracted',len(records),'clips from',len(source_plan),'source shards')
records.sort(key=lambda r:r['sample_id'])
assert [r['sample_id'] for r in records]==selected_ids
manifest=pd.DataFrame(records);manifest.to_csv(BUNDLE/'train_manifest.csv',index=False)
summary={'bundle_type':'phase6b_pilot_train_v1','split_version':'phase3_5b_v1','scope':'frozen Narration train only',
 'clips':len(records),'hours':sum(r['duration_seconds'] for r in records)/3600,'components':len(set(r['component_id'] for r in records)),
 'train_membership_sha256':membership_sha,'frozen_train_manifest_sha256':sha(FROZEN),
 'manifest_sha256':sha(BUNDLE/'train_manifest.csv'),'contains_validation_or_test_audio':False,
 'seed':SEED,'target_seconds':TARGET_SECONDS,'sampling':'deterministic component-aware, full clips; up to 240s/component then round-robin',
 'reference_policy':'NFD, trim outer whitespace, collapse repeated ordinary spaces; raw retained'}
(BUNDLE/'bundle_record.json').write_text(json.dumps(summary,indent=2)+'\n')
files=sorted(p for p in BUNDLE.rglob('*') if p.is_file())
(BUNDLE/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(BUNDLE)}\n' for p in files))
with zipfile.ZipFile(OUT/'phase6b_pilot_train_bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for file in sorted(BUNDLE.rglob('*')):
  if file.is_file():z.write(file,arcname=f'{BUNDLE.name}/{file.relative_to(BUNDLE)}')
private_audit={'bundle':summary,'source_shard_sha256':source_hashes,
 'counts_by_anonymous_component':manifest.groupby('component_id').size().to_dict(),
 'hours_by_anonymous_component':(manifest.groupby('component_id').duration_seconds.sum()/3600).to_dict(),
 'month_counts':train.set_index('sample_id').loc[selected_ids].recording_month.value_counts().to_dict(),
 'gender_counts':train.set_index('sample_id').loc[selected_ids].gender.value_counts().to_dict(),
 'duration_min_max_seconds':[float(manifest.duration_seconds.min()),float(manifest.duration_seconds.max())]}
(OUT/'audit_private.json').write_text(json.dumps(private_audit,indent=2)+'\n')
print('Private bundle:',OUT/'phase6b_pilot_train_bundle.zip')
print('Aggregate:',{k:summary[k] for k in ['clips','hours','components','train_membership_sha256']})
