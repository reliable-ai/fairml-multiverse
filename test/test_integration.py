from pathlib import Path
import shutil
import unittest
import pandas as pd

import os

ROOT_DIR = Path(__file__).parent.parent
TEMP_DIR = ROOT_DIR / "test" / "temp"


def count_files(dir, glob):
    return len(list(dir.glob(glob)))


class TestMultiverseAnalysisIntegration(unittest.TestCase):
    """
    Test the multiverse analysis script end-to-end, running a small scale
    test multiverse analysis and comparing the final data to a stored snapshot.
    """

    def run_and_test_analysis(self, output_dir: Path, other_args: str):
        # Create and clear output directory
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run a test multiverse analysis
        os.system(
            f"python multiverse_analysis.py --output-dir {output_dir} {other_args}"
        )

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv.gz") == 1
        assert count_files(output_dir, "runs/1/data/*.csv") == 2
        assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 2
        assert count_files(output_dir, "counter.txt") == 1
        assert count_files(output_dir, "multiverse_grid.json") == 1

    def test_end_to_end(self):
        output_dir = TEMP_DIR / "e2e-org-seed"
        self.run_and_test_analysis(output_dir, "--mode test")

        # Compare final data to stored snapshot
        # Please note: any change to the analysis scripts will break this test
        # Either copy over the latest aggregated data file or disable this test altogether
        filename_df_agg = "agg_1_run_outputs.csv.gz"
        columns_to_ignore = ["execution_time"]
        df_snap = pd.read_csv(ROOT_DIR / "test" / "snapshots" / filename_df_agg).drop(
            columns=columns_to_ignore
        )
        df_new = pd.read_csv(output_dir / "runs/1/data/" / filename_df_agg).drop(
            columns=columns_to_ignore
        )
        pd.testing.assert_frame_equal(df_snap, df_new)

    def test_end_to_end_other_seed(self):
        output_dir = TEMP_DIR / "e2e-other-seed"
        self.run_and_test_analysis(output_dir, "--mode test --seed 42")

        # Compare final data to stored snapshot (note we expect this one to actually fail)
        filename_df_agg = "agg_1_run_outputs.csv.gz"
        columns_to_ignore = ["execution_time"]
        df_snap = pd.read_csv(ROOT_DIR / "test" / "snapshots" / filename_df_agg).drop(
            columns=columns_to_ignore
        )
        df_new = pd.read_csv(output_dir / "runs/1/data/" / filename_df_agg).drop(
            columns=columns_to_ignore
        )

        # Actually expect an error here
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(df_snap, df_new)


if __name__ == "__main__":
    unittest.main()
