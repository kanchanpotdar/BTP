"""Run restored stages in isolated roots and publish only validated artifacts."""
import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from .integrity import copy_exact, digest, publish

STAGES = ('n', 'a_v1', 'a_v2')


def runtime(root):
    arrow = os.environ.get('NYISHI_PYARROW_PATH', str(root / 'research_audit_2026-10-27/_analysis_deps'))
    if Path(arrow).is_dir():
        sys.path.insert(0, arrow)
    versions = {}
    missing = []
    for name in ('numpy', 'pandas', 'pyarrow', 'soundfile'):
        try:
            module = importlib.import_module(name)
            versions[name] = {'version': module.__version__, 'path': module.__file__}
        except ImportError as exc:
            missing.append(name + ': ' + str(exc))
    if missing:
        raise RuntimeError('Missing dependencies; installation requires approval: ' + '; '.join(missing))
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': versions, 'arrow_path': arrow}


def code_fingerprint(root):
    paths = sorted((root / 'src/nyishi_asr').rglob('*.py')) + [root / 'run_preprocessing.py']
    hashes = {str(p.relative_to(root)): digest(p) for p in paths}
    import hashlib
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'code_revision': 'sha256:' + revision, 'git_revision': None,
            'git_note': 'No Git repository recovered; source-tree SHA-256 used.', 'files': hashes}


def verify_sources(root):
    """Offline pins and all raw-A locators, with no waveform transformations."""
    import pandas as pd
    import pyarrow.parquet as pq
    evidence_path = root / 'research_audit_2026-10-27/evidence/local_download_verification.json'
    evidence = json.loads(evidence_path.read_text())
    import hashlib
    for record in evidence['files']:
        path = root / 'raw' / ('A_narration' if record['dataset'] == 'A' else 'B_youtube') / record['path']
        if not path.exists() or path.stat().st_size != record['bytes']:
            raise RuntimeError('Source file absent/size changed: ' + str(path))
        h = hashlib.sha256() if record['hash_type'] == 'sha256' else hashlib.sha1()
        if record['hash_type'] != 'sha256':
            h.update(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != record['hash']:
            raise RuntimeError('Raw source hash changed: ' + str(path))
    master = pd.read_parquet(root / 'manifests/master_A.parquet')
    expected_schema = [x['name'] for x in json.loads((root / 'task1_recovery_audit/validation_evidence.json').read_text())['master']['ordered_schema']]
    if list(master.columns) != expected_schema or master.shape != (6718, 39) or not master.sample_id.is_unique:
        raise RuntimeError('Recovered master schema/identity differs')
    from .validation import check_csv
    check_csv(master, root / 'manifests/master_A.csv')
    for name in ('master_A.csv', 'master_A.parquet'):
        if digest(root / name) != digest(root / 'manifests' / name):
            raise RuntimeError('Original-master copy differs')
    keys = master.audio_path.str.extract(r'^(raw/A_narration/data/\w+/[^#]+)#row=(\d+)$')
    if keys.isna().any().any() or keys.duplicated().any():
        raise RuntimeError('Malformed/duplicate raw locators')
    for path in keys[0].unique():
        indices = keys.index[keys[0] == path]
        raw = pq.read_table(root / path, columns=['audio_id', 'sentence-IPA-transcription']).to_pandas()
        nums = keys.loc[indices, 1].astype(int).tolist()
        if any(n < 0 or n >= len(raw) for n in nums):
            raise RuntimeError('Raw row locator out of range')
        for raw_col, master_col in [('audio_id', 'original_audio_id'), ('sentence-IPA-transcription', 'transcript_raw')]:
            if raw.iloc[nums][raw_col].tolist() != master.loc[indices, master_col].tolist():
                raise RuntimeError('Raw metadata does not match: ' + path)
    return evidence


def prepare_root(root, temp, previous, provenance):
    for name in ('manifests', 'cleaning', 'metadata'):
        if previous:
            shutil.copytree(previous / name, temp / name)
        else:
            (temp / name).mkdir()
    (temp / 'raw').symlink_to(root / 'raw', target_is_directory=True)
    if not previous:
        for name in ('master_A.csv', 'master_A.parquet'):
            copy_exact(root / 'manifests' / name, temp / 'manifests' / name)
        (temp / 'metadata/provenance.json').write_text(json.dumps(provenance, indent=2))


def run(root, run_id, requested, no_publish):
    from .validation import validate
    env_info = runtime(root)
    source_evidence = verify_sources(root)
    # Validate identifiers before using them as directory names.
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('run-id must contain only letters, numbers, underscores and hyphens')
    run_dir = root / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / '.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError('Run already locked: ' + str(run_dir))
    try:
        stamp_path = run_dir / 'run.json'
        fingerprint = code_fingerprint(root)
        inputs = {name: digest(root / name) for name in ('master_A.csv', 'master_A.parquet',
                  'Nyishi_ASR_BTP_Recovery_2026-10-27.zip')}
        if stamp_path.exists():
            stamp = json.loads(stamp_path.read_text())
            if stamp['code']['code_revision'] != fingerprint['code_revision'] or stamp['input_hashes'] != inputs:
                raise RuntimeError('Run ID already belongs to different code/inputs; use a new run ID')
        else:
            stamp = {'run_id': run_id, 'recovery_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'runtime': env_info, 'code': fingerprint, 'input_hashes': inputs,
                     'approval_scope': 'Historical provisional-policy reproduction authorized by Task 2; no expert linguistic approval'}
            stamp_path.write_text(json.dumps(stamp, indent=2))
        revision = next(r['revision'] for r in source_evidence['files'] if r['dataset'] == 'A')
        provenance = {'A_narration': {'repo': 'Eries1506/nyishi-speed-tb-phase1', 'revision': revision,
                      'accessed_utc': stamp['recovery_utc'],
                      'accessed_utc_meaning': 'Current recovery read/access event, not original download time',
                      'original_download_timestamp': None,
                      'local_hash_evidence': 'research_audit_2026-10-27/evidence/local_download_verification.json',
                      'local_hash_evidence_sha256': digest(root / 'research_audit_2026-10-27/evidence/local_download_verification.json'),
                      'input_hashes': inputs, 'runtime': env_info, 'code_revision': fingerprint['code_revision'],
                      'recovery_run_id': run_id, 'speaker_groups_are_proposals': True}}
        stages = STAGES if requested == 'all' else (requested,)
        results = []
        for stage in stages:
            index = STAGES.index(stage)
            previous = run_dir / STAGES[index - 1] if index else None
            if previous and not (previous / 'validation.json').exists():
                raise RuntimeError('Prerequisite stage not validated: ' + str(previous))
            destination = run_dir / stage
            if destination.exists():
                raise RuntimeError('Stage already exists; retained unchanged. Use a new run ID for a clean reproduction: ' + str(destination))
            temp = Path(tempfile.mkdtemp(prefix='.' + stage + '-', dir=run_dir))
            prepare_root(root, temp, previous, provenance)
            child_env = os.environ.copy()
            child_env.update(BTP_ROOT=str(temp), NYISHI_RUN_ID=run_id, NYISHI_RECOVERY_UTC=stamp['recovery_utc'],
                             PYTHONDONTWRITEBYTECODE='1',
                             PYTHONPATH=os.pathsep.join([str(root / 'src'), env_info['arrow_path']]))
            log = run_dir / (stage + '.log')
            print('Running', stage, 'in a fresh Python process; log:', log, flush=True)
            with log.open('w') as stream:
                process = subprocess.run([sys.executable, '-m', 'nyishi_asr.execute_stage', stage, '--root', str(temp)],
                                         env=child_env, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                (temp / 'FAILED.txt').write_text('Recipe failed. See ' + str(log))
                raise RuntimeError(stage + ' failed; successful earlier outputs preserved. Inspect ' + str(log))
            try:
                info = validate(temp, stage)
            except Exception:
                (temp / 'FAILED.txt').write_text(traceback.format_exc())
                raise
            info.update(code_revision=fingerprint['code_revision'], runtime=env_info,
                        source_revision=revision, policy_version={'n': None, 'a_v1': 'A_v1', 'a_v2': 'A_v2'}[stage])
            (temp / 'validation.json').write_text(json.dumps(info, indent=2))
            (temp / 'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(temp)): digest(p)
                for folder in ('manifests', 'cleaning', 'metadata') for p in sorted((temp / folder).rglob('*')) if p.is_file()}, indent=2))
            temp.rename(destination)
            if not no_publish:
                info['published'] = publish(destination, root)
            (run_dir / (stage + '_result.json')).write_text(json.dumps(info, indent=2))
            results.append(info)
            print(json.dumps({k: v for k, v in info.items() if k not in ('runtime', 'published')}, indent=2), flush=True)
        (run_dir / 'summary.json').write_text(json.dumps(results, indent=2))
        return results
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--run-id', default='dataset_A_recovery_20261027')
    parser.add_argument('--stage', choices=['all', *STAGES], default='all')
    parser.add_argument('--no-publish', action='store_true')
    args = parser.parse_args()
    try:
        run(args.root.resolve(), args.run_id, args.stage, args.no_publish)
    except Exception as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
