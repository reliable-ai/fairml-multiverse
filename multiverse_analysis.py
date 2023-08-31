from pathlib import Path

from fairness_multiverse.multiverse import MultiverseAnalysis

# Whether to only explore a small fraction of universes for testing
TEST_RUN = False
CONTINUE_RUN = True

multiverse_analysis = MultiverseAnalysis(
    dimensions={
        "scale": ["scale", "do-not-scale"],
        "encode_categorical": ["ordinal", "one-hot"],
        "stratify_split": [
            "none",
            "target",
            "protected-attribute",
            "both"
        ],
        "model": ["logreg", "rf", "gbm", "elasticnet"],
        "cutoff": [["raw_0.5", "quantile_0.1", "quantile_0.25"]],
        "fairness_grouping": [["majority-minority", "race-all"]],
        "preprocess_age": ["none", "bins_10", "quantiles_3", "quantiles_4"],
        "preprocess_income": ["none", "bins_10000", "quantiles_3", "quantiles_4"],
        "exclude_features": [
            "none",
            "race",
            "sex",
            "race-sex",
            # Also supported: immigration
        ],
        "exclude_subgroups": [
            "keep-all",
            "drop-smallest_race_1",
            "drop-smallest_race_2",
            "keep-largest_race_2",
            "drop-name_race_Some Other Race alone",
        ],
    },
    output_dir=Path("./output"),
    new_run=not CONTINUE_RUN
)

multiverse_grid = multiverse_analysis.generate_grid(save=True)
print(f"Generated N = {len(multiverse_grid)} universes")


print(f"~ Starting Run No. {multiverse_analysis.run_no} ~")

# Run the analysis for the first universe
if TEST_RUN:
    print("Small-Scale-Test Run")
    multiverse_analysis.visit_universe(multiverse_grid[0])
    multiverse_analysis.visit_universe(multiverse_grid[1])
elif CONTINUE_RUN:
    print("Continuing Previous Run")
    missing_universes = multiverse_analysis.check_missing_universes()[
        "missing_universes"
    ]

    # Run analysis only for missing universes
    multiverse_analysis.examine_multiverse(multiverse_grid=missing_universes)
else:
    print("Full Run")
    # Run analysis for all universes
    multiverse_analysis.examine_multiverse(multiverse_grid=multiverse_grid)

multiverse_analysis.aggregate_data(save=True)

multiverse_analysis.check_missing_universes()
