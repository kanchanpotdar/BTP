"""Independent literal expectations for the recovered provisional rules."""
from pathlib import Path
import sys
import unittest
import unicodedata as ud
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyishi_asr import norm_a_v1 as a, norm_a_v2 as b


class NormalizationTests(unittest.TestCase):
    def test_literal_shared_cases(self):
        cases = [('  a:  b; ', 'aː b;'), ('a\u0301:', 'a\u0301ː'),
                 ('t[', 't\u032a'), ('x[', 'x['), ('english', 'english'),
                 ('gon', 'ɡon'), ('hoɡE', 'hoɡe'), ('d\u032a0', 'd\u032a0'),
                 ('həb?\u0301', 'həb?\u0301'), ('bed;aso', 'bed;aso'),
                 ('hoiskuː;oiŋliʃ', 'hoiskuː;oiŋliʃ')]
        for raw, expected in cases:
            for normalize in (a.normalize_text, b.normalize_text2):
                with self.subTest(raw=raw, version=normalize.__module__):
                    self.assertEqual(normalize(raw), ud.normalize('NFD', expected))

    def test_distinct_historical_policies(self):
        cases = [('ʉ:', 'ʉ:', 'ʉː'), ('t\u0301[', 't\u0301[', 't\u032a\u0301'),
                 ('english?', 'enɡlish?', 'english?'), ('hok.kə', 'hok.kə', 'hokkə'),
                 ('bag,a', 'baɡ,a', 'baɡa')]
        for raw, v1, v2 in cases:
            self.assertEqual(a.normalize_text(raw), ud.normalize('NFD', v1))
            self.assertEqual(b.normalize_text2(raw), ud.normalize('NFD', v2))

    def test_edge_punctuation_and_existing_dental(self):
        self.assertEqual(b.normalize_text2('. a. a?'), '. a. a?')
        self.assertEqual(b.normalize_text2('t\u032a['), 't\u032a[')
        self.assertEqual(b.normalize_text2('"english?"'), '"english?"')

    def test_idempotence_nfd_and_marks(self):
        for text in ['é g  t[', 'a\u0301: English?', 'd\u032a0', 'həb?\u0301', 't\u0301[']:
            for fn in (a.normalize_text, b.normalize_text2):
                new = fn(text)
                self.assertEqual(fn(new), new)
                self.assertEqual(ud.normalize('NFD', new), new)

    def test_constants_separate(self):
        self.assertNotIn('ʉ', a.VOWELS)
        self.assertIn('ʉ', b.VOWELS2)
        self.assertEqual(len(a.G_EXCEPTION_WORDS), 9)
        self.assertEqual(a.G_EXCEPTION_WORDS, b.G_EXCEPTION_WORDS2)
        self.assertEqual(len(a.PIPELINE), 9)
        self.assertEqual(len(b.PIPELINE2), 11)


if __name__ == '__main__':
    unittest.main()
