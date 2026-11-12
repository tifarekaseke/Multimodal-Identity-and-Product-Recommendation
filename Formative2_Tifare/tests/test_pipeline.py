"""Run: python -m unittest discover -s tests -v"""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pipeline import normalize_id,prepare_data,FEATURES

class PipelineChecks(unittest.TestCase):
    def test_id_standardization(self):
        self.assertEqual(normalize_id('A178'),normalize_id(178))
        self.assertEqual(normalize_id('a00178'), '178')
    def test_merge_does_not_multiply_transactions(self):
        _,tx,merged,data,summary=prepare_data()
        self.assertEqual(len(merged),len(tx))
        self.assertEqual(data.customer_id.nunique(),summary['model_unique_customers'])
    def test_features_exclude_post_purchase_fields(self):
        self.assertNotIn('purchase_amount',FEATURES)
        self.assertNotIn('customer_rating',FEATURES)
        self.assertNotIn('customer_id',FEATURES)

if __name__=='__main__':unittest.main()
