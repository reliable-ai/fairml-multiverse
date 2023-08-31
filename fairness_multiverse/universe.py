from pathlib import Path
import time
from math import floor, ceil
import pandas as pd
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, precision_score
from fairlearn.metrics import (
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,
)
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
)
from .multiverse import generate_multiverse_grid


def list_wrap(value: Any):
    if isinstance(value, list):
        return value
    else:
        return [value]


# Modified version of pandas.cut, that also supports DataFrames instead of just Series
def cut_df(df, **kwargs):
    # Adapted from https://datascience.stackexchange.com/questions/75787/how-to-use-columntransformer-and-functiontransformer-to-apply-the-same-function
    if isinstance(df, pd.Series):
        return pd.cut(df, **kwargs)
    elif isinstance(df, pd.DataFrame):
        return df.apply(pd.cut, axis=0, **kwargs)
    else:
        raise "Unsupported type of data in cut_df"


def continuous_var_will_be_binned(configuration: str):
    return configuration.startswith(("quantiles", "bins"))


def preprocess_continuous(
    source_data: pd.DataFrame, column_name: str, configuration: str
) -> Tuple[Optional[ColumnTransformer], Optional[List[str]]]:
    if configuration == "none":
        # Skip transformation if "none" is specified
        return (None, None)
    elif configuration == "log":
        transformer = make_pipeline(
            # Calculate the log (+1 to gracefully handle 0)
            # Since negative values are undefined for log, we replace them with 0
            # (NAs cannot be handled by all algorithms)
            FunctionTransformer(lambda df: np.log1p(df.astype("float")).fillna(0))
        )
        binned_values = None
    elif continuous_var_will_be_binned(configuration=configuration):
        method, value = configuration.split("_")
        if method == "quantiles":
            n_bins = int(value)
            transformer = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
            )
        elif method == "bins":
            step = int(value)
            round_min = floor(source_data[column_name].min() / step) * step
            round_max = ceil(source_data[column_name].max() / step) * step
            bins = list(range(round_min, round_max, step)) + [round_max]

            print(
                f"Generated bins transformer for {column_name} with the following bins: {bins}"
            )

            transformer = FunctionTransformer(
                cut_df, kw_args={"bins": bins, "labels": False, "retbins": False}
            )
            n_bins = len(bins)
        else:
            raise Exception(
                "Unsupported method for preprocessing continuous variable: " + method
            )

        binned_values = list(range(n_bins))
    else:
        raise Exception(
            "Unsupported configuration for preprocessing continuous variable: "
            + configuration
        )

    column_transformer = ColumnTransformer(
        [(f"bin_{column_name}_{configuration}", transformer, [column_name])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    column_transformer.set_output(transform="pandas")
    return (column_transformer, binned_values)


def predict_w_threshold(probabilities: np.array, threshold: float):
    # Expect 2 classes
    assert probabilities.shape[1] == 2

    # Check whether probability for second column (class 1) is gr. or equal to threshold
    return probabilities[:, 1] >= threshold


def add_dict_to_df(df, dictionary, prefix=""):
    for key, value in dictionary.items():
        df[prefix + key] = value
    return df


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # Convert nested Pandas-Series to dict
        if isinstance(v, pd.Series):
            v = dict(v)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class UniverseAnalysis:
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "false positive rate": false_positive_rate,
        "false negative rate": false_negative_rate,
        "selection rate": selection_rate,
        "count": count,
    }

    fairness_metrics = {
        "equalized_odds_difference": equalized_odds_difference,
        "equalized_odds_ratio": equalized_odds_ratio,
        # "demographic_parity_difference": demographic_parity_difference,
        # "demographic_parity_ratio": demographic_parity_ratio,
    }
    ts_start = None
    ts_end = None

    def __init__(
        self,
        run_no: int,
        universe_id: str,
        universe: Dict,
        output_dir: Path = Path("./output"),
    ) -> None:
        self.ts_start = time.time()

        self.run_no = run_no
        self.universe_id = universe_id
        self.universe = universe

        self.output_dir = output_dir

    def get_execution_time(self):
        if self.ts_end is None:
            print("Stopping execution_time clock.")
            self.ts_end = time.time()
        return self.ts_end - self.ts_start

    def compute_metrics(
        self,
        sub_universe: Dict,
        y_pred_prob: pd.Series,
        y_test: pd.Series,
        x_test: pd.DataFrame,
    ):
        # Determine cutoff for predictions
        cutoff_type, cutoff_value = sub_universe["cutoff"].split("_")
        cutoff_value = float(cutoff_value)

        if cutoff_type == "raw":
            threshold = cutoff_value
        elif cutoff_type == "quantile":
            probabilities_true = y_pred_prob[:, 1]
            threshold = np.quantile(probabilities_true, cutoff_value)

        fairness_grouping = sub_universe["fairness_grouping"]
        if fairness_grouping == "majority-minority":
            fairness_group_column = "majmin"
        elif fairness_grouping == "race-all":
            fairness_group_column = "RAC1P"

        y_pred = predict_w_threshold(y_pred_prob, threshold)

        # Compute fairness metrics
        fairness_dict = {
            name: metric(
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=x_test[fairness_group_column],
            )
            for name, metric in self.fairness_metrics.items()
        }

        # Compute "normal" metrics (but split by fairness column)
        metric_frame = MetricFrame(
            metrics=self.metrics,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=x_test[fairness_group_column],
        )

        return (fairness_dict, metric_frame)

    def visit_sub_universe(self, sub_universe, y_pred_prob, y_test, x_test):
        final_output = pd.DataFrame(
            data={
                "run_no": self.run_no,
                "universe_id": self.universe_id,
                "universe_settings": json.dumps(sub_universe, sort_keys=True),
                "execution_time": self.get_execution_time(),
            },
            index=[self.universe_id],
        )

        # Compute metrics for majority-minority split
        fairness_dict, metric_frame = self.compute_metrics(
            sub_universe,
            y_pred_prob,
            y_test,
            x_test,
        )

        # Add main fairness metrics to final_output
        final_output = add_dict_to_df(final_output, fairness_dict, prefix="fair_main_")
        final_output = add_dict_to_df(
            final_output, dict(metric_frame.overall), prefix="perf_ovrl_"
        )

        # Add group metrics to final output
        final_output = add_dict_to_df(
            final_output, flatten_dict(metric_frame.by_group), prefix="perf_grp_"
        )

        return final_output

    def generate_sub_universes(self):
        # Wrap all non-lists in the universe to make them work with generate_multiverse_grid
        universe_all_lists = {k: list_wrap(v) for k, v in self.universe.items()}

        # Within-Universe variation
        return generate_multiverse_grid(universe_all_lists)

    def generate_final_output(self, y_pred_prob, y_test, x_test, save=True):
        # Within-Universe variation
        sub_universes = self.generate_sub_universes()

        final_outputs = list()
        for sub_universe in sub_universes:
            final_outputs.append(
                self.visit_sub_universe(
                    sub_universe=sub_universe,
                    y_pred_prob=y_pred_prob,
                    y_test=y_test,
                    x_test=x_test,
                ).reset_index(drop=True)
            )
        final_output = pd.concat(final_outputs)

        # Write the final output file
        if save:
            target_dir = self.output_dir / "runs" / str(self.run_no) / "data"
            # Make sure the directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"d_{str(self.run_no)}_{self.universe_id}.csv"
            final_output.to_csv(target_dir / filename, index=False)

        return final_output
