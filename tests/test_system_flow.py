"""Integration checks for the face, voice, and product command-line flow.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Formative2_Tifare" / "src"))

import pandas as pd
from pipeline import FEATURES

import system_cli


class DecisionRule(unittest.TestCase):
    def test_same_person_with_a_passing_score_is_allowed(self):
        voice = {"approved": True, "score": 0.91, "predicted": "colombe", "claimed": "colombe", "threshold": 0.6}
        allowed, reason = system_cli.decide("colombe", voice)
        self.assertTrue(allowed)
        self.assertIn("colombe", reason)

    def test_another_members_voice_is_rejected_even_with_a_high_score(self):
        voice = {"approved": True, "score": 0.99, "predicted": "nelly", "claimed": "colombe", "threshold": 0.6}
        allowed, reason = system_cli.decide("colombe", voice)
        self.assertFalse(allowed)
        self.assertIn("nelly", reason)

    def test_low_score_for_the_same_person_is_rejected(self):
        voice = {"approved": False, "score": 0.2, "predicted": "colombe", "claimed": "colombe", "threshold": 0.6}
        allowed, _reason = system_cli.decide("colombe", voice)
        self.assertFalse(allowed)

    def test_unrecognized_face_closes_the_flow(self):
        voice = {"approved": True, "score": 0.99, "predicted": "colombe", "claimed": "colombe", "threshold": 0.6}
        allowed, _reason = system_cli.decide(None, voice)
        self.assertFalse(allowed)


class SavedModels(unittest.TestCase):
    def test_enrolled_face_and_voice_pass(self):
        name, confidence, _nearest, _threshold = system_cli.enrolled_face("colombe")
        voice = system_cli.enrolled_voice("colombe", claimed="colombe")
        self.assertEqual(name, "colombe")
        self.assertGreaterEqual(confidence, 0.7)
        self.assertTrue(voice["approved"])
        self.assertEqual(voice["predicted"], "colombe")

    def test_flat_histogram_is_rejected(self):
        name, confidence, _nearest, threshold = system_cli.score_face(system_cli.unauthorized_face_features())
        self.assertIsNone(name)
        self.assertLess(confidence, threshold)

    def test_other_member_and_unknown_voice_are_rejected(self):
        mismatch = system_cli.enrolled_voice("nelly", claimed="colombe")
        unknown = system_cli.score_voice(system_cli.unauthorized_voice_features(), "colombe")
        self.assertFalse(mismatch["approved"])
        self.assertEqual(mismatch["predicted"], "nelly")
        self.assertFalse(unknown["approved"])
        self.assertNotEqual(unknown["predicted"], "colombe")


class ProductGate(unittest.TestCase):
    def test_social_features_match_the_merged_training_row(self):
        frame, canonical = system_cli.customer_features("A151")
        saved = pd.read_csv(
            ROOT / "Formative2_Tifare" / "data" / "processed" / "model_dataset.csv",
            dtype={"customer_id": str},
        )
        row = saved[saved.customer_id == "151"].iloc[0]
        self.assertEqual(canonical, "151")
        self.assertEqual(list(frame.columns), FEATURES)
        for column in ("primary_platform", "primary_sentiment"):
            self.assertEqual(frame.iloc[0][column], row[column])
        for column in ("engagement_mean", "interest_mean", "social_activity_count", "distinct_platforms"):
            self.assertAlmostEqual(float(frame.iloc[0][column]), float(row[column]), places=6)

    def test_prediction_does_not_use_purchase_fields(self):
        frame, canonical = system_cli.customer_features("178")
        self.assertEqual(canonical, "178")
        self.assertNotIn("purchase_amount", frame.columns)
        self.assertNotIn("product_category", frame.columns)
        self.assertNotIn("customer_rating", frame.columns)

    def test_a178_and_178_return_the_same_category(self):
        first = system_cli.predict_product("A178")
        second = system_cli.predict_product("178")
        self.assertEqual(first["customer_id"], "178")
        self.assertEqual(first["category"], second["category"])
        self.assertIn(first["category"], {"Books", "Clothing", "Electronics", "Groceries", "Sports"})
        total = sum(probability for _label, probability in first["probabilities"])
        self.assertAlmostEqual(total, 1, places=5)

    def test_missing_customer_raises(self):
        with self.assertRaises(ValueError):
            system_cli.customer_features("999999")


class FullTransaction(unittest.TestCase):
    def test_authorized_flow_calls_the_product_model_once(self):
        calls = []

        def predict(customer_id):
            calls.append(customer_id)
            return system_cli.predict_product(customer_id)

        result = system_cli.scenario_authorized(customer_id="A178", predict=predict)
        self.assertTrue(result.allowed)
        self.assertEqual(calls, ["A178"])
        self.assertEqual(result.prediction["customer_id"], "178")
        self.assertIn("ACCESS GRANTED", result.transcript)
        self.assertIn("Predicted category:", result.transcript)

    def test_denied_paths_do_not_call_the_product_model(self):
        def fail_if_called(_customer_id):
            raise AssertionError("product model should stay closed")

        denied = [
            system_cli.scenario_unknown_face(predict=fail_if_called),
            system_cli.scenario_wrong_voice(predict=fail_if_called),
            system_cli.scenario_unknown_voice(predict=fail_if_called),
        ]
        for result in denied:
            self.assertFalse(result.allowed)
            self.assertIsNone(result.prediction)
            self.assertFalse(result.prediction_called)
            self.assertIn("ACCESS DENIED", result.transcript)
            self.assertNotIn("Predicted category:", result.transcript)

    def test_wrong_voice_is_checked_as_the_face_owner(self):
        result = system_cli.scenario_wrong_voice()
        self.assertEqual(result.face_name, "colombe")
        self.assertEqual(result.voice["claimed"], "colombe")
        self.assertEqual(result.voice["predicted"], "nelly")
        self.assertIn("nelly", result.reason)

    def test_unauthorized_files_are_rejected(self):
        try:
            import cv2
            import librosa
        except ImportError:
            self.skipTest("opencv and librosa are required for the file check")
        del cv2, librosa
        results = system_cli.scenario_live_unauthorized()
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertFalse(result.allowed)
            self.assertIsNone(result.prediction)
            self.assertFalse(result.prediction_called)
            self.assertIn("ACCESS DENIED", result.transcript)

    def test_command_line_prints_one_grant_and_three_denials(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = system_cli.main(["--scenario", "all", "--customer-id", "A178"])
        text = output.getvalue()
        self.assertEqual(status, 0)
        self.assertEqual(text.count("ACCESS GRANTED"), 1)
        self.assertEqual(text.count("ACCESS DENIED"), 3)
        self.assertEqual(text.count("Predicted category:"), 1)


if __name__ == "__main__":
    unittest.main()
