"""Source immutability and repaired recipe structure, independent of data execution."""
import ast
import hashlib
import json
from pathlib import Path
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class RecoveredSourcesTests(unittest.TestCase):
    def test_originals_and_copies_match_recovery_archive(self):
        with zipfile.ZipFile(ROOT/'Nyishi_ASR_BTP_Recovery_2026-10-27.zip') as archive:
            for name in archive.namelist():
                rel = Path(name).relative_to('Nyishi_ASR_Recovery')
                if rel.parts[0] == '01_notebooks':
                    self.assertEqual(archive.read(name), (ROOT/'notebooks/recovered'/rel.name).read_bytes())
                    if rel.name.endswith('_RECOVERED.ipynb'):
                        self.assertEqual(archive.read(name), (ROOT/rel.name).read_bytes())
                elif rel.parts[0] == '02_data':
                    self.assertEqual(archive.read(name), (ROOT/rel.name).read_bytes())
                    self.assertEqual(archive.read(name), (ROOT/'manifests'/rel.name).read_bytes())

    def test_repaired_v1_has_18_unique_literal_rule_ids(self):
        tree=ast.parse((ROOT/'src/nyishi_asr/recipes/a_v1.py').read_text())
        rule_list=next(n.value for n in tree.body if isinstance(n,ast.Assign) and
                       any(isinstance(t,ast.Name) and t.id=='R' for t in n.targets))
        ids=[n.elts[0].value for n in rule_list.elts]
        self.assertEqual(len(ids),18)
        self.assertEqual(len(set(ids)),18)
        self.assertEqual(ids,['NORM_'+str(i).zfill(3) for i in range(1,19)])

    def test_recipe_stages_and_paths_explicit(self):
        for stage in ('n','a_v1','a_v2'):
            source=(ROOT/'src/nyishi_asr/recipes'/f'{stage}.py').read_text()
            ast.parse(source)
            self.assertNotIn('C:\\Users\\',source)
            self.assertNotIn('from IPython',source)
        source=(ROOT/'src/nyishi_asr/recipes/a_v1.py').read_text()
        self.assertNotIn('CELL N8',source)
