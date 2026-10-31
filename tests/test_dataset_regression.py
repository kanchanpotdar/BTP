"""Run against explicitly selected validated stage roots; never regenerate data here."""
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
STAGE_ROOT = os.environ.get('NYISHI_TEST_STAGE_ROOT')


@unittest.skipUnless(STAGE_ROOT, 'Set NYISHI_TEST_STAGE_ROOT to a validated run stage')
class DatasetRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT/'research_audit_2026-10-27/_analysis_deps'))
        import pandas as pd
        cls.pd = pd
        cls.path = Path(STAGE_ROOT)
        cls.v1 = pd.read_parquet(cls.path/'manifests/master_A.parquet')
        cls.v11 = pd.read_parquet(cls.path/'manifests/master_A_v1_1.parquet')

    def test_stable_identity_metadata_and_empty_raw_normalized(self):
        from nyishi_asr.validation import check_unchanged
        self.assertEqual(self.v1.shape, (6718,39))
        self.assertEqual(self.v11.shape, (6718,54))
        check_unchanged(self.v1,self.v11,('duration_basis',))
        self.assertTrue(self.v1.transcript_normalized.isna().all())
        self.assertTrue(self.v11.transcript_normalized.isna().all())

    def test_audio_header_and_actual_decode(self):
        header=self.pd.read_parquet(self.path/'manifests/audio_facts_A.parquet')
        decode=self.pd.read_parquet(self.path/'manifests/audio_decode_facts_A.parquet')
        self.assertEqual(len(header),6718)
        self.assertTrue(decode.full_decode_success.all())
        self.assertEqual(set(decode.loc[decode.decoded_frames==0,'sample_id']),
                         {'A_004973','A_006614','A_006615'})
        self.assertEqual(set(self.v11.loc[~self.v11.audio_ok,'sample_id']),
                         {'A_004972','A_004973','A_006614','A_006615'})

    def test_proposed_groups_do_not_change_public_ids(self):
        self.assertEqual(self.v11.speaker_id.nunique(),60)
        self.assertEqual(self.v11.speaker_group_id.nunique(),55)
        self.assertEqual(self.v11.groupby('speaker_id').speaker_group_id.nunique().max(),1)
        self.assertTrue(self.v11.loc[self.v11.train_only,'group_train_only'].all())

    def test_both_normalization_versions_if_present(self):
        p12=self.path/'manifests/master_A_v1_2.parquet'
        p13=self.path/'manifests/master_A_v1_3.parquet'
        if not (p12.exists() and p13.exists()):
            self.skipTest('Both policy outputs are not present at this stage')
        from nyishi_asr.validation import check_unchanged
        a=self.pd.read_parquet(p12);b=self.pd.read_parquet(p13)
        for table,expected in [(a,2487),(b,2495)]:
            self.assertEqual(table.shape,(6718,57))
            check_unchanged(self.v11,table,('transcript_normalized',))
            has=~table.transcript_missing
            self.assertEqual(int(has.sum()),5651)
            self.assertEqual(int((table.loc[has,'transcript_raw']!=table.loc[has,'transcript_normalized']).sum()),expected)
        has=~a.transcript_missing
        self.assertEqual(int((a.loc[has,'transcript_normalized']!=b.loc[has,'transcript_normalized']).sum()),51)
