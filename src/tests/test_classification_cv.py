import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classification_cv import classifier_specs


class ClassificationCvTests(unittest.TestCase):
    def test_rf_one_vs_all_classifier_uses_legacy_estimator_count(self):
        classifiers = dict(classifier_specs(random_state=42))

        rf_pipeline = classifiers["RF"]
        self.assertIsInstance(rf_pipeline, Pipeline)
        self.assertIsInstance(rf_pipeline.steps[0][1], StandardScaler)
        self.assertIsInstance(rf_pipeline.steps[1][1], RandomForestClassifier)
        self.assertEqual(rf_pipeline.steps[1][1].n_estimators, 20)


if __name__ == "__main__":
    unittest.main()
