"""Check curation choices without changing the recovered normalization policies."""
from pathlib import Path
import importlib.util,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_audit_2026-10-27/_analysis_deps'))
import pandas as pd
spec=importlib.util.spec_from_file_location('curation',ROOT/'scripts/prepare_narration_baseline.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def fixture():
    return pd.DataFrame(dict(sample_id=['A1','A2','A3'],transcript_missing=[False,False,True],
        transcript_normalized=['a','a',None],audio_ok=[True,True,True],speaker_status=['ok']*3,
        duration_decoded=[2.,2.,2.],speaker_group_id=['G1','G2','G3'],
        recording_session_id=['S1','S2','S3'],group_train_only=[False,False,False]))


class CurationTests(unittest.TestCase):
    def test_exact_duplicate_and_conflicting_labels(self):
        m=fixture();h=pd.DataFrame(dict(sample_id=['A1','A2','A3'],pcm_sha256=['h','h',None]))
        d=c.exact_duplicate_flags(m,h).set_index('sample_id')
        self.assertEqual(d.loc['A2','exact_audio_canonical_id'],'A1')
        self.assertTrue(d.loc['A2','exact_audio_redundant'])
        self.assertFalse(d.exact_audio_label_conflict.any())
        m.loc[1,'transcript_normalized']='b'
        d=c.exact_duplicate_flags(m,h).set_index('sample_id')
        self.assertTrue(d.loc[['A1','A2'],'exact_audio_label_conflict'].all())

    def test_missing_twin_does_not_suppress_labeled_record(self):
        m=fixture();m.loc[0,['transcript_missing','transcript_normalized']]=[True,None]
        h=pd.DataFrame(dict(sample_id=['A1','A2','A3'],pcm_sha256=['h','h',None]))
        d=c.exact_duplicate_flags(m,h).set_index('sample_id')
        self.assertEqual(d.loc['A2','exact_audio_canonical_id'],'A2')
        self.assertFalse(d.loc['A2','exact_audio_redundant'])

    def test_low_loudness_is_not_an_exclusion(self):
        m=fixture();h=pd.DataFrame(dict(sample_id=['A1','A2','A3'],pcm_sha256=['h1','h2',None]))
        f=c.apply_policy(m,set(),c.exact_duplicate_flags(m,h),{'A1'}).set_index('sample_id')
        self.assertTrue(f.loc['A1','low_absolute_level_review'])
        self.assertTrue(f.loc['A1','candidate_primary_05_30'])
        self.assertFalse(f.loc['A3','candidate_primary_05_30'])
        self.assertTrue((f.split_status=='unassigned').all())

    def test_symbol_review_and_uncertain_identity_block_primary_pool(self):
        m=fixture();m.loc[1,'speaker_status']='uncertain_identity'
        h=pd.DataFrame(dict(sample_id=['A1','A2','A3'],pcm_sha256=['h1','h2',None]))
        f=c.apply_policy(m,{'A1'},c.exact_duplicate_flags(m,h),set())
        self.assertFalse(f.candidate_primary_05_30.any())
        self.assertEqual(int(f.candidate_expanded_05_30.sum()),2)

    def test_session_connections_propagate_train_only(self):
        m=fixture();m.loc[1,'recording_session_id']='S1';m.loc[1,'group_train_only']=True
        h=pd.DataFrame(dict(sample_id=['A1','A2','A3'],pcm_sha256=['h1','h2',None]))
        f=c.group_components(c.apply_policy(m,set(),c.exact_duplicate_flags(m,h),set())).set_index('sample_id')
        self.assertEqual(f.loc['A1','split_component_id'],f.loc['A2','split_component_id'])
        self.assertTrue(f.loc['A1','component_train_only'])

    def test_keyed_results_do_not_depend_on_input_order(self):
        m=fixture();h=pd.DataFrame(dict(sample_id=['A1','A2','A3'],pcm_sha256=['h','h',None]))
        a=c.exact_duplicate_flags(m,h).set_index('sample_id').sort_index()
        b=c.exact_duplicate_flags(m.iloc[::-1],h.iloc[::-1]).set_index('sample_id').sort_index()
        pd.testing.assert_frame_equal(a,b)
