library(tidyverse)

# Filter to include only variation on the the non-eval options
# We're using a "standard / conservative" evlauation strategy this way
filter_non_eval <- . %>%
  filter(
    (sett_eval_fairness_grouping == "separate") &
    (sett_eval_exclude_subgroups == "keep-in-eval") &
    (sett_eval_on_subset == "full")
  )
