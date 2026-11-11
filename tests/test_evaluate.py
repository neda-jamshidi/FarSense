"""Run with: python -m unittest discover -s tests -v"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import csv

FILE = Path(__file__).resolve().parents[1]/'software/evaluation/evaluate.py'
spec = importlib.util.spec_from_file_location('farsense_evaluate', FILE)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

class EvaluationTests(unittest.TestCase):
    def test_gold_parsing(self):
        a = '{"multiple_choice_question_answer_predict": "B"}'
        self.assertEqual(app.normalise(app.extract_answer(a, 'multiple_choice', True), 'multiple_choice'), 'B')
        self.assertEqual(app.normalise(app.extract_answer('{"true_false_2_answer_predict": false}', 'true_false_2', True), 'true_false_2'), 'false')
    def test_persian_multiple_choice_letters(self):
        self.assertEqual(app.normalise('ب', 'multiple_choice'), 'B')
        self.assertEqual(app.normalise('الف', 'multiple_choice'), 'A')

    def test_true_false_2_prompt_typo(self):
        a = '{"true_false_1_answer_predict": true}'
        self.assertEqual(app.normalise(app.extract_answer(a, 'true_false_2'), 'true_false_2'), 'true')
        self.assertIsNone(app.extract_answer(a, 'true_false_2', reference=True))
    def test_last_json_response(self):
        text = 'An example: {"cloze_answer_predict":"placeholder"}\nActual: {"cloze_answer_predict":"تهران"}'
        self.assertEqual(app.extract_answer(text, 'cloze'), 'تهران')
    def test_summarise_and_invalid_output(self):
        gold = [
            {'task':'multiple_choice','assistant':'{"multiple_choice_question_answer_predict":"B"}'},
            {'task':'true_false_1','assistant':'{"true_false_1_answer_predict":true}'},
            {'task':'cloze','assistant':'{"cloze_answer_predict":"درخت سبز"}'},
        ]
        counts, invalid = app.validate(gold)
        self.assertFalse(invalid)
        self.assertEqual(sum(counts.values()), 3)
        rows, summary = app.evaluate(gold, {0:'B', 1:'nonsense', 2:'درخت سبز'})
        self.assertEqual([r['exact_match'] for r in rows], [1, 0, 1])
        self.assertEqual(rows[1]['parsed'], 0)
        self.assertEqual(rows[2]['token_f1'], 1.0)
        self.assertEqual(len(summary), 3)
    def test_partial_row_index_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pred.csv'
            p.write_text('row_index,prediction\n1,true\n', encoding='utf8')
            data, col = app.aligned_predictions(p, None, 3)
            self.assertEqual(data, {1:'true'})
            p.write_text('row_index,prediction\n1,true\n1,false\n', encoding='utf8')
            with self.assertRaises(ValueError): app.aligned_predictions(p, None, 3)

if __name__ == '__main__':
    unittest.main()
