source("R/helpers.R")

# Extract a smaller subset of the full aggregated data for quicker loading
prepare_data_for_paper <- function(path_in, path_out) {
  stopifnot(
    length(path_in) == 1,
    length(path_out) == 1
  )

  df <- vroom::vroom(
    path_in,
    col_select = list(
      starts_with("sett_"),
      "universe_id",
      starts_with("fair_main_"),
      starts_with("perf_ovrl_")
    )
  )

  df %>%
    vroom::vroom_write(
      path_out
    )
}

# Main Data
analysis_id <- "42"
prepare_data_for_paper(
  file.path("output", "analyses", analysis_id, "df_agg_full.csv.gz"),
  file.path("paper", "data", "df_agg_prepared.csv.gz")
)
