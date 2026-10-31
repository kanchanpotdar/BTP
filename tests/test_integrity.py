from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyishi_asr.integrity import copy_exact, publish


class IntegrityTests(unittest.TestCase):
    def test_copy_is_no_clobber(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d);(p/'a').write_bytes(b'original')
            copy_exact(p/'a', p/'b');copy_exact(p/'a', p/'b')
            (p/'a').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):copy_exact(p/'a', p/'b')
            self.assertEqual((p/'b').read_bytes(), b'original')

    def test_publication_checks_all_conflicts_first(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);stage=p/'stage';project=p/'project'
            for base in (stage,project):
                for folder in ('manifests','cleaning','metadata'):(base/folder).mkdir(parents=True)
            (stage/'manifests/new').write_text('new')
            (stage/'cleaning/conflict').write_text('different')
            (project/'cleaning/conflict').write_text('old')
            with self.assertRaises(RuntimeError):publish(stage,project)
            self.assertFalse((project/'manifests/new').exists())

    def test_only_append_log_is_published(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);stage=p/'stage';project=p/'project'
            for base in (stage,project):
                for folder in ('manifests','cleaning','metadata'):(base/folder).mkdir(parents=True)
            (project/'cleaning/decision_log_A.csv').write_text('header\nold\n')
            (stage/'cleaning/decision_log_A.csv').write_text('header\nold\nnew\n')
            publish(stage,project)
            self.assertEqual((project/'cleaning/decision_log_A.csv').read_text(),'header\nold\nnew\n')
            (stage/'cleaning/decision_log_A.csv').write_text('header\nreplacement\n')
            with self.assertRaises(RuntimeError):publish(stage,project)
