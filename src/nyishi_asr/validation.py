"""Independent stage checks and full PCM decode, separate from historical header facts."""
from pathlib import Path
import csv
import io
import json
import re
import unicodedata as ud


def require(condition, message):
    if not bool(condition):
        raise RuntimeError('Validation failed: ' + message)


def check_unchanged(before, after, except_columns=()):
    require(before.sample_id.is_unique and after.sample_id.is_unique, 'unique sample IDs')
    require(set(before.sample_id) == set(after.sample_id), 'same sample ID coverage')
    a = before.set_index('sample_id').sort_index()
    b = after.set_index('sample_id').sort_index()
    for col in a.columns:
        if col not in except_columns:
            # Compare values and nulls, allowing historical Arrow string-storage representation.
            require(a[col].fillna('<NA>').astype(str).equals(b[col].fillna('<NA>').astype(str)),
                    'original column preserved: ' + col)


def check_csv(parquet, csv_path):
    import pandas as pd
    records = pd.read_csv(csv_path, encoding='utf-8-sig', keep_default_na=False, dtype=str)
    require(list(records.columns) == list(parquet.columns), 'CSV column order')
    require(len(records) == len(parquet), 'CSV row count')
    for col in parquet:
        expected = parquet[col].map(lambda x: '' if pd.isna(x) else str(x))
        if pd.api.types.is_numeric_dtype(parquet[col]) and not pd.api.types.is_bool_dtype(parquet[col]):
            require(all((s == '' and pd.isna(v)) or abs(float(s) - float(v)) < 1e-9
                        for s, v in zip(records[col], parquet[col])), 'CSV numeric values: ' + col)
        else:
            require(records[col].equals(expected.reset_index(drop=True)), 'CSV values: ' + col)


def decode_audio(root):
    """Actually read every embedded waveform with libsndfile; do not call this listening."""
    import pandas as pd
    import pyarrow.parquet as pq
    import soundfile as sf
    root = Path(root)
    master = pd.read_parquet(root / 'manifests/master_A.parquet')
    locations = dict(zip(master.audio_path, master.sample_id))
    decoded = []
    for path in sorted((root / 'raw/A_narration/data').glob('*/*.parquet')):
        base = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['audio']):
            for audio in batch.column('audio').to_pylist():
                locator = 'raw/A_narration/data/' + path.parent.name + '/' + path.name + '#row=' + str(base)
                sid = locations[locator]
                row = dict(sample_id=sid, full_decode_success=False, error=None,
                           decoded_frames=0, sample_rate=None, channels=None, duration_exact=0.0)
                try:
                    signal, rate = sf.read(io.BytesIO(audio['bytes']), dtype='float32', always_2d=True)
                    row.update(full_decode_success=True, decoded_frames=len(signal), sample_rate=rate,
                               channels=signal.shape[1], duration_exact=len(signal) / rate)
                except Exception as exc:
                    row['error'] = type(exc).__name__ + ': ' + str(exc)
                decoded.append(row)
                base += 1
    facts = pd.DataFrame(decoded)
    require(len(facts) == len(master) and facts.sample_id.is_unique, 'full decode covers every sample once')
    facts.to_parquet(root / 'manifests/audio_decode_facts_A.parquet', index=False)
    facts.to_csv(root / 'manifests/audio_decode_facts_A.csv', index=False, encoding='utf-8-sig')
    return facts


def validate(root, stage):
    import pandas as pd
    root = Path(root)
    base = pd.read_parquet(root / 'manifests/master_A.parquet')
    name = {'n': 'master_A_v1_1', 'a_v1': 'master_A_v1_2', 'a_v2': 'master_A_v1_3'}[stage]
    table = pd.read_parquet(root / 'manifests' / (name + '.parquet'))
    require(table.shape == (6718, 54 if stage == 'n' else 57), 'historical shape for ' + stage)
    require(list(table.sample_id) == list(base.sample_id), 'original stable sample order')
    check_unchanged(base, table, ('duration_basis', 'transcript_normalized'))
    check_csv(table, root / 'manifests' / (name + '.csv'))
    facts = pd.read_parquet(root / 'manifests/audio_facts_A.parquet')
    require(facts.sample_id.is_unique and set(facts.sample_id) == set(base.sample_id), 'header facts coverage')
    require(table.speaker_id.nunique() == 60, '60 public IDs')
    require(table.speaker_group_id.nunique() == 55, '55 proposed groups')
    require(table.loc[~table.group_train_only, 'speaker_group_id'].nunique() == 46, '46 non-train-only groups')
    require(table.groupby('speaker_id').speaker_group_id.nunique().max() == 1, 'public ID not split among groups')
    require(table.loc[table.train_only, 'group_train_only'].all(), 'train-only propagation')
    bad = table.loc[~table.audio_ok, ['sample_id', 'audio_issue']]
    expected = {'A_004972': 'length_differs_from_stated', 'A_004973': 'empty_audio',
                'A_006614': 'empty_audio', 'A_006615': 'empty_audio'}
    require(dict(zip(bad.sample_id, bad.audio_issue)) == expected, 'four exact audio anomalies')
    require(int(base.transcript_missing.sum()) == 1067, 'missing labels')
    require((table.transcript_missing == base.transcript_missing).all(), 'missing flags preserved')
    info = dict(stage=stage, master=name, rows=len(table), columns=len(table.columns),
                public_speaker_ids=60, proposed_speaker_groups=55, audio_issues=expected,
                validation_passed=True)
    if stage == 'n':
        require(table.transcript_normalized.isna().all(), 'v1.1 text still unnormalized')
        expected_filled = {'education_medium_after_12': 264, 'education_medium_upto_12': 361,
                           'life_speaker_id': 24, 'speaker2_place_of_recording': 11, 'speaker2_type_of_place': 11}
        require({c: int(table[c].notna().sum()) for c in expected_filled} == expected_filled,
                'five recovered metadata columns')
        decode = decode_audio(root)
        merged = table[['sample_id', 'duration_decoded']].merge(decode, on='sample_id', validate='one_to_one')
        require(merged.full_decode_success.all(), 'all WAVs fully readable (including empty files)')
        require((merged.duration_decoded - merged.duration_exact).abs().le(0.000501).all(),
                'full decode agrees with rounded header duration')
        require(set(merged.loc[merged.decoded_frames == 0, 'sample_id']) == set(expected) - {'A_004972'},
                'three empty decoded waveforms')
        info.update(full_decode_successes=int(decode.full_decode_success.sum()),
                    empty_waveforms=int((decode.decoded_frames == 0).sum()),
                    decoded_hours_exact=float(decode.duration_exact.sum() / 3600))
    else:
        from . import norm_a_v1, norm_a_v2
        version = 'A_v1' if stage == 'a_v1' else 'A_v2'
        old = pd.read_parquet(root / 'manifests/master_A_v1_1.parquet')
        check_unchanged(old, table, ('transcript_normalized',))
        text = table.loc[~table.transcript_missing]
        require(len(text) == 5651 and text.transcript_normalized.notna().all(), '5651 normalized labels')
        require(table.loc[table.transcript_missing, 'transcript_normalized'].isna().all(), 'missing labels remain null')
        require((text.normalization_policy_version == version).all(), 'explicit normalization policy version')
        changed = int((text.transcript_raw != text.transcript_normalized).sum())
        require(changed == (2487 if stage == 'a_v1' else 2495), 'historical changed row count')
        normalize = norm_a_v1.normalize_text if stage == 'a_v1' else norm_a_v2.normalize_text2
        for row in text.itertuples():
            require(ud.normalize('NFD', row.transcript_normalized) == row.transcript_normalized, 'NFD: ' + row.sample_id)
            require(normalize(row.transcript_normalized) == row.transcript_normalized, 'idempotence: ' + row.sample_id)
            require(normalize(row.transcript_raw) == row.transcript_normalized, 'raw reprocessing: ' + row.sample_id)
            require(len(row.transcript_raw.split()) == len(row.transcript_normalized.split()), 'word count: ' + row.sample_id)
            require(row.transcript_raw.count('?') == row.transcript_normalized.count('?'), 'question marks: ' + row.sample_id)
            before_marks = [c for c in row.transcript_raw if ud.category(c).startswith('M')]
            after_marks = [c for c in row.transcript_normalized if ud.category(c).startswith('M')]
            from collections import Counter
            require(not (Counter(before_marks) - Counter(after_marks)), 'original marks retained: ' + row.sample_id)
        policy = pd.read_csv(root / 'cleaning' / ('normalization_policy_' + version + '.csv'), keep_default_na=False)
        require(len(policy) == (18 if stage == 'a_v1' else 19) and policy.rule_id.is_unique, 'unique policy rules')
        require(policy.status.value_counts().to_dict() ==
                ({'Pending review': 7, 'Provisional': 5, 'Approved': 4, 'Rejected': 2} if stage == 'a_v1' else
                 {'Provisional': 7, 'Pending review': 6, 'Approved': 4, 'Rejected': 2}), 'policy statuses')
        review = pd.read_csv(root / 'cleaning' / ('unresolved_symbol_rows_' + version + '.csv'), keep_default_na=False)
        require(len(review) == (103 if stage == 'a_v1' else 52) and review.sample_id.is_unique, 'review queue count')
        report = pd.read_csv(root / 'cleaning' / ('normalization_change_report_' + version + '.csv'), keep_default_na=False)
        changes = pd.read_csv(root / 'cleaning' / ('normalization_row_changes_' + version + '.csv'), keep_default_na=False)
        counts = changes.loc[changes.kind == 'changed'].groupby('rule_id').n.sum().to_dict()
        for rid, count in {'NORM_003': 847, 'NORM_005': 20, 'NORM_006': 26,
                           'NORM_007': 28, 'NORM_008': 7440, 'NORM_009': 14}.items():
            require(int(counts[rid]) == count, 'occurrences: ' + rid)
        require((report.validation_passed == 'Yes').all(), 'saved historical validation report')
        require(int(report.iloc[-1].rows_affected) == changed, 'report distinct changed rows')
        require(int(report.iloc[-1].occurrences_changed) == sum(counts.values()), 'report occurrence total')
        symbols = len(set(''.join(text.transcript_normalized)))
        require(symbols == (98 if stage == 'a_v1' else 96), 'normalized symbol inventory')
        info.update(normalized_rows=5651, missing_rows=1067, changed_rows=changed,
                    unchanged_rows=5651 - changed, unresolved_rows=len(review),
                    distinct_symbols=symbols, rule_occurrences={k: int(v) for k, v in counts.items()})
        if stage == 'a_v2':
            v1 = pd.read_parquet(root / 'manifests/master_A_v1_2.parquet').set_index('sample_id')
            a = v1.loc[text.sample_id, 'transcript_normalized'].tolist()
            b = text.transcript_normalized.tolist()
            differences = sum(x != y for x, y in zip(a, b))
            require(differences == 51, '51 cross-version differences')
            require(all(re.sub(r'(?<=\S)\.(?=\S)', '', x).replace(',', '') == y for x, y in zip(a, b)),
                    'only internal dots/commas differ in this corpus')
            require(counts['NORM_010'] == 76 and counts['NORM_019'] == 21 and sum(counts.values()) == 8472,
                    'A_v2 punctuation counts and 8472 total changes')
            require(review.original_split.value_counts().to_dict() == {'train': 44, 'validation': 7, 'test': 1},
                    'A_v2 review split counts')
            info['cross_version_differences'] = differences
    with (root / 'cleaning/decision_log_A.csv').open(encoding='utf-8-sig', newline='') as f:
        logs = list(csv.DictReader(f))
    require(len({r['decision'] for r in logs}) == len(logs), 'decision log unique decisions')
    require(all(r['event_origin'] == 'historical_recipe_reproduced_now' and r['linguistic_approval'].startswith('not_established')
                for r in logs), 'honest current recovery log provenance')
    info['recovery_log_entries'] = len(logs)
    return info
