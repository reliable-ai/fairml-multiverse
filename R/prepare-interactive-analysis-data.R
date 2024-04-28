source("R/helpers.R")

# Read the data from the paper
vroom::vroom(
  file.path("paper", "data", "df_agg_prepared.csv.gz")
) %>%
  select(-universe_id) %>%
  # Shorten names and select only a subset of columns
  select(
    eq_odd_diff=fair_main_equalized_odds_difference,
    dem_par_diff=fair_main_demographic_parity_difference,
    accuracy=perf_ovrl_accuracy,
    balanced_accuracy=`perf_ovrl_balanced accuracy`,
    f1=perf_ovrl_f1,
    starts_with("sett_")
  ) %>%
  rename_with(~str_replace(., "sett_", "s_")) %>%
  # Reduce data size by rounding
  mutate(across(where(is.numeric), ~round(., 2))) %>%
  arrow::write_parquet(
    file.path("interactive-analysis", "docs", "data", "df_agg_prepared.parquet")
  )
