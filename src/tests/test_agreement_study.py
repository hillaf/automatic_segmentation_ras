import unittest

from submission_code.agreement_study import pairwise_agreement_rows


class AgreementStudyTests(unittest.TestCase):
    def test_pairwise_agreement_matches_sampled_table_values(self):
        rows = pairwise_agreement_rows()

        self.assertEqual(
            [
                (
                    row["pair"],
                    row["images"],
                    row["n_first"],
                    row["n_second"],
                    round(row["precision"], 3),
                    round(row["recall"], 3),
                    round(row["f1"], 3),
                    round(row["mean_iou"], 3),
                )
                for row in rows
            ],
            [
                ("A vs B", 76, 4702, 4520, 0.793, 0.825, 0.809, 0.594),
                ("C vs A", 76, 2436, 4702, 0.963, 0.499, 0.657, 0.652),
                ("C vs B", 76, 2436, 4520, 0.933, 0.503, 0.653, 0.635),
            ],
        )


if __name__ == "__main__":
    unittest.main()
