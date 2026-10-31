"""Compare independently executed stage artifacts without modifying either run."""
from pathlib import Path
import argparse
import json
import sys


def compare(first, second):
    # Existing project-local PyArrow; no installation or network fallback.
    project=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(project/'research_audit_2026-10-27/_analysis_deps'))
    import pandas as pd
    results=[]
    for stage in ('n','a_v1','a_v2'):
        a,b=Path(first)/stage,Path(second)/stage
        if not (a/'validation.json').exists() or not (b/'validation.json').exists():
            raise RuntimeError('Both runs must have independently validated '+stage)
        for path in sorted((a/'manifests').glob('*.parquet')):
            other=b/'manifests'/path.name
            left,right=pd.read_parquet(path),pd.read_parquet(other)
            pd.testing.assert_frame_equal(left,right,check_dtype=True)
            results.append(stage+'/manifests/'+path.name)
        left_files={p.name for p in (a/'cleaning').glob('*.csv')}
        right_files={p.name for p in (b/'cleaning').glob('*.csv')}
        if left_files != right_files:
            raise RuntimeError('Cleaning artifact coverage differs in '+stage)
        for name in sorted(left_files):
            left=pd.read_csv(a/'cleaning'/name,encoding='utf-8-sig',keep_default_na=False)
            right=pd.read_csv(b/'cleaning'/name,encoding='utf-8-sig',keep_default_na=False)
            # These describe the current recovery event rather than dataset semantics.
            ignore=['review_date'] if name.startswith('normalization_policy_') else []
            if name=='decision_log_A.csv':ignore=['date','recovery_run_id']
            pd.testing.assert_frame_equal(left.drop(columns=ignore,errors='ignore'),
                                          right.drop(columns=ignore,errors='ignore'),check_dtype=True)
            results.append(stage+'/cleaning/'+name)
    return {'deterministic_semantic_artifacts':True,'files_compared':len(results),'artifacts':results,
            'excluded_event_fields':['policy.review_date','decision_log.date','decision_log.recovery_run_id'],
            'note':'Input/code fingerprints and both fresh-process validation reports remain separately recorded.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('first',type=Path);p.add_argument('second',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();result=compare(args.first,args.second)
    text=json.dumps(result,indent=2)
    if args.output:args.output.write_text(text)
    print(text)


if __name__=='__main__':main()
