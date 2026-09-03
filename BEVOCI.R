# =====================================================================
# STACK EXCHANGE METACOGNITION ANALYSIS
# BEVoCI REGRESSIONS, ROBUSTNESS CHECKS, TABLES, AND PLOTS
# =====================================================================

# ---------------------------------------------------------------------
# 1. Packages
# ---------------------------------------------------------------------
# If needed, install once with:
# install.packages(c("tidyverse", "lmtest", "sandwich"))

library(tidyverse)
library(lmtest)
library(sandwich)

# Clear environment and use the current project folder
rm(list = ls(all.names = TRUE))
setwd(".")

# ---------------------------------------------------------------------
# 2. Load and prepare data
# ---------------------------------------------------------------------
raw_data <- read_csv("stackexchange_enhanced_dataset.csv")

# Convert boolean columns to numeric (TRUE = 1, FALSE = 0)
raw_data$Has_Image <- as.numeric(raw_data$Has_Image)
raw_data$Is_Question_Format <- as.numeric(raw_data$Is_Question_Format)
raw_data$Has_Custom_Avatar <- as.numeric(raw_data$Has_Custom_Avatar)

# Missing-data check
cat("\n--- MISSING VALUES PER COLUMN ---\n")
missing_counts <- colSums(is.na(raw_data))
print(missing_counts[missing_counts > 0])

accept_rate_na_percent <- mean(is.na(raw_data$Accept_Rate)) * 100
cat("Percentage of missing Accept_Rate:", round(accept_rate_na_percent, 2), "%\n")

# Standardize metacognitive cues (IVs)
cue_variables_all <- c(
  "Has_Image", "User_Reputation", "Gold_Badges", "Silver_Badges",
  "Bronze_Badges", "Tag_Count", "Title_Word_Count", "Word_Count",
  "Code_Block_Count", "LaTeX_Comment_Count", "Link_Count"
)

raw_data <- raw_data %>%
  mutate(
    across(
      all_of(cue_variables_all),
      ~ as.numeric(scale(.x)),
      .names = "{.col}_iv_c"
    )
  )

# Standardize the two dependent measures
raw_data <- raw_data %>%
  mutate(
    across(
      c(Subjective_Score, Objective_Comment_Count),
      ~ as.numeric(scale(.x)),
      .names = "{.col}_dv_c"
    )
  )

# Align the direction of the two clarity measures.
# Higher Objective_Comment_Count means MORE clarification requests, hence LOWER clarity.
# Higher Subjective_Score means HIGHER perceived clarity/quality.
# Therefore, reverse the standardized comment-count measure so that higher values mean
# higher clarity for both dependent variables before comparing them in BEVoCI.
raw_data <- raw_data %>%
  mutate(Objective_Clarity_dv_c = -Objective_Comment_Count_dv_c)

# Post age is used later as a robustness-control variable.
raw_data <- raw_data %>%
  mutate(Post_Age_Days_control_c = as.numeric(scale(Post_Age_Days)))

# ---------------------------------------------------------------------
# 3. Correlation matrices and multicollinearity check
# ---------------------------------------------------------------------

# 3A. Before removing collinear variables
cues_all <- c(
  "Has_Image_iv_c", "User_Reputation_iv_c", "Gold_Badges_iv_c",
  "Silver_Badges_iv_c", "Bronze_Badges_iv_c", "Tag_Count_iv_c",
  "Title_Word_Count_iv_c", "Word_Count_iv_c", "Code_Block_Count_iv_c",
  "LaTeX_Comment_Count_iv_c", "Link_Count_iv_c"
)

corrs_all <- cor(raw_data[cues_all], use = "complete.obs")
cat("\n--- CORRELATION MATRIX (BEFORE BADGE/CODE/LINK REMOVAL) ---\n")
print(round(corrs_all, 2))

# 3B. After removing collinear variables
cues_filtered <- c(
  "Has_Image_iv_c", "User_Reputation_iv_c", "Tag_Count_iv_c",
  "Title_Word_Count_iv_c", "Word_Count_iv_c", "LaTeX_Comment_Count_iv_c"
)

corrs_filtered <- cor(raw_data[cues_filtered], use = "complete.obs")
cat("\n--- CORRELATION MATRIX (AFTER COLLINEAR REMOVAL) ---\n")
print(round(corrs_filtered, 2))

# ---------------------------------------------------------------------
# 4. Main regression models - Task 4
# ---------------------------------------------------------------------

# Model A: Objective clarity
model.objective <- lm(
  Objective_Clarity_dv_c ~
    Has_Image_iv_c +
    User_Reputation_iv_c +
    Tag_Count_iv_c +
    Title_Word_Count_iv_c +
    Word_Count_iv_c +
    LaTeX_Comment_Count_iv_c,
  data = raw_data
)

cat("\n--- OBJECTIVE CLARITY (REVERSE-CODED COMMENT COUNT) ---\n")
print(summary(model.objective))

# Model B: Subjective clarity
model.subjective <- lm(
  Subjective_Score_dv_c ~
    Has_Image_iv_c +
    User_Reputation_iv_c +
    Tag_Count_iv_c +
    Title_Word_Count_iv_c +
    Word_Count_iv_c +
    LaTeX_Comment_Count_iv_c,
  data = raw_data
)

cat("\n--- SUBJECTIVE MEASURE (UPVOTE SCORE) ---\n")
print(summary(model.subjective))

# Model C: BEVoCI interaction model
# Objective is explicitly set as the reference category.
raw_data_obj <- raw_data %>%
  mutate(measurev = Objective_Clarity_dv_c, measure = "objective")

raw_data_subj <- raw_data %>%
  mutate(measurev = Subjective_Score_dv_c, measure = "subjective")

raw_data_duplicated <- bind_rows(raw_data_obj, raw_data_subj) %>%
  mutate(measure = factor(measure, levels = c("objective", "subjective")))

model_measure_comparison <- lm(
  measurev ~ measure +
    Has_Image_iv_c +
    User_Reputation_iv_c +
    Tag_Count_iv_c +
    Title_Word_Count_iv_c +
    Word_Count_iv_c +
    LaTeX_Comment_Count_iv_c +
    measure * Has_Image_iv_c +
    measure * User_Reputation_iv_c +
    measure * Tag_Count_iv_c +
    measure * Title_Word_Count_iv_c +
    measure * Word_Count_iv_c +
    measure * LaTeX_Comment_Count_iv_c,
  data = raw_data_duplicated
)

cat("\n--- BIAS EXPOSURE (INTERACTION MODEL) ---\n")
print(summary(model_measure_comparison))

# ---------------------------------------------------------------------
# 5. Helper functions for reproducible result tables
# ---------------------------------------------------------------------

make_ranked_table <- function(model, interaction_only = FALSE) {
  coef_matrix <- coef(summary(model))

  result <- tibble(
    Term = rownames(coef_matrix),
    Standardized_Coefficient = coef_matrix[, "Estimate"],
    Absolute_Size = abs(coef_matrix[, "Estimate"]),
    p_value = coef_matrix[, "Pr(>|t|)"]
  ) %>%
    filter(Term != "(Intercept)")

  if (interaction_only) {
    result <- result %>% filter(str_detect(Term, ":"))
  }

  result %>%
    arrange(desc(Absolute_Size)) %>%
    mutate(Rank = row_number(), .before = 1)
}

coeftest_to_tibble <- function(test_object) {
  # lmtest::coeftest returns a matrix-like object with extra attributes.
  # Convert it to a plain matrix first so every output column has one value per term.
  x <- unclass(test_object)
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  tibble(
    Term = rownames(x),
    Estimate = as.numeric(x[, 1]),
    Robust_SE = as.numeric(x[, 2]),
    t_value = as.numeric(x[, 3]),
    p_value = as.numeric(x[, 4])
  )
}

compare_model_coefficients <- function(base_model, controlled_model, terms) {
  base_summary <- coef(summary(base_model))
  controlled_summary <- coef(summary(controlled_model))

  tibble(
    Term = terms,
    Base_Coefficient = base_summary[terms, "Estimate"],
    Base_p_value = base_summary[terms, "Pr(>|t|)"],
    Controlled_Coefficient = controlled_summary[terms, "Estimate"],
    Controlled_p_value = controlled_summary[terms, "Pr(>|t|)"],
    Coefficient_Change = Controlled_Coefficient - Base_Coefficient,
    Base_Significant_05 = Base_p_value < 0.05,
    Controlled_Significant_05 = Controlled_p_value < 0.05
  )
}

# Main regression tables
objective_table <- make_ranked_table(model.objective)
subjective_table <- make_ranked_table(model.subjective)
interaction_table <- make_ranked_table(model_measure_comparison, interaction_only = TRUE)

cat("\n--- RANKED OBJECTIVE REGRESSION COEFFICIENTS ---\n")
print(objective_table)
cat("\n--- RANKED SUBJECTIVE REGRESSION COEFFICIENTS ---\n")
print(subjective_table)
cat("\n--- RANKED BEVOCI INTERACTION COEFFICIENTS ---\n")
print(interaction_table)

# ---------------------------------------------------------------------
# 6. Regression assumption checks and HC3 robust standard errors
# ---------------------------------------------------------------------

cat("\n=====================================================================\n")
cat("REGRESSION ASSUMPTION CHECKS AND HC3 ROBUST STANDARD ERRORS\n")
cat("=====================================================================\n")

# 6A. Breusch-Pagan test for heteroskedasticity
bp_objective <- bptest(model.objective)
bp_subjective <- bptest(model.subjective)

cat("\n--- BREUSCH-PAGAN: OBJECTIVE MODEL ---\n")
print(bp_objective)
cat("\n--- BREUSCH-PAGAN: SUBJECTIVE MODEL ---\n")
print(bp_subjective)

# 6B. Shapiro-Wilk test for normality of residuals
# R's shapiro.test accepts up to 5000 observations; this dataset has exactly 5000.
shapiro_objective <- shapiro.test(residuals(model.objective))
shapiro_subjective <- shapiro.test(residuals(model.subjective))

cat("\n--- SHAPIRO-WILK: OBJECTIVE MODEL RESIDUALS ---\n")
print(shapiro_objective)
cat("\n--- SHAPIRO-WILK: SUBJECTIVE MODEL RESIDUALS ---\n")
print(shapiro_subjective)

# 6C. HC3 robust standard errors
hc3_objective_test <- coeftest(
  model.objective,
  vcov. = vcovHC(model.objective, type = "HC3")
)

hc3_subjective_test <- coeftest(
  model.subjective,
  vcov. = vcovHC(model.subjective, type = "HC3")
)

cat("\n--- OBJECTIVE MODEL WITH HC3 ROBUST STANDARD ERRORS ---\n")
print(hc3_objective_test)
cat("\n--- SUBJECTIVE MODEL WITH HC3 ROBUST STANDARD ERRORS ---\n")
print(hc3_subjective_test)

hc3_objective_table <- coeftest_to_tibble(hc3_objective_test)
hc3_subjective_table <- coeftest_to_tibble(hc3_subjective_test)

# ---------------------------------------------------------------------
# 7. Multiple-comparison diagnostic across the 18 cue-related tests
# ---------------------------------------------------------------------
# The report does not use these corrected p-values as the primary inference.
# They are calculated here so the statement about multiple comparisons is reproducible.

objective_p <- objective_table %>%
  transmute(Model = "Objective", Term, p_value)

subjective_p <- subjective_table %>%
  transmute(Model = "Subjective", Term, p_value)

interaction_p <- interaction_table %>%
  transmute(Model = "BEVoCI interaction", Term, p_value)

multiple_comparison_table <- bind_rows(objective_p, subjective_p, interaction_p) %>%
  mutate(
    p_FDR_BH = p.adjust(p_value, method = "BH"),
    p_Bonferroni = p.adjust(p_value, method = "bonferroni")
  )

cat("\n--- MULTIPLE-COMPARISON DIAGNOSTIC (18 TESTS) ---\n")
print(multiple_comparison_table)

# ---------------------------------------------------------------------
# 8. Robustness check: add Post_Age_Days as a control variable
# ---------------------------------------------------------------------
# Motivation: Post_Age_Days was dominant in the automatic model, especially
# for the subjective measure. We therefore test whether cue coefficients remain
# stable after controlling for the amount of time the post has been exposed.

model.objective.post_age <- lm(
  Objective_Clarity_dv_c ~
    Has_Image_iv_c +
    User_Reputation_iv_c +
    Tag_Count_iv_c +
    Title_Word_Count_iv_c +
    Word_Count_iv_c +
    LaTeX_Comment_Count_iv_c +
    Post_Age_Days_control_c,
  data = raw_data
)

model.subjective.post_age <- lm(
  Subjective_Score_dv_c ~
    Has_Image_iv_c +
    User_Reputation_iv_c +
    Tag_Count_iv_c +
    Title_Word_Count_iv_c +
    Word_Count_iv_c +
    LaTeX_Comment_Count_iv_c +
    Post_Age_Days_control_c,
  data = raw_data
)

cat("\n=====================================================================\n")
cat("ROBUSTNESS CHECK: POST AGE CONTROL\n")
cat("=====================================================================\n")

cat("\n--- ROBUSTNESS: OBJECTIVE MODEL + POST AGE CONTROL ---\n")
print(summary(model.objective.post_age))

cat("\n--- ROBUSTNESS: SUBJECTIVE MODEL + POST AGE CONTROL ---\n")
print(summary(model.subjective.post_age))

# Compare R-squared before and after adding Post_Age_Days
r2_comparison <- tibble(
  Measure = c("Objective", "Subjective"),
  R2_Base = c(
    summary(model.objective)$r.squared,
    summary(model.subjective)$r.squared
  ),
  R2_With_Post_Age = c(
    summary(model.objective.post_age)$r.squared,
    summary(model.subjective.post_age)$r.squared
  )
) %>%
  mutate(Delta_R2 = R2_With_Post_Age - R2_Base)

cat("\n--- ROBUSTNESS: R-SQUARED COMPARISON ---\n")
print(r2_comparison)

# Compare the six cue coefficients before and after controlling for post age
cue_terms <- c(
  "Has_Image_iv_c",
  "User_Reputation_iv_c",
  "Tag_Count_iv_c",
  "Title_Word_Count_iv_c",
  "Word_Count_iv_c",
  "LaTeX_Comment_Count_iv_c"
)

objective_post_age_comparison <- compare_model_coefficients(
  model.objective,
  model.objective.post_age,
  cue_terms
)

subjective_post_age_comparison <- compare_model_coefficients(
  model.subjective,
  model.subjective.post_age,
  cue_terms
)

cat("\n--- OBJECTIVE COEFFICIENTS: BASE VS POST AGE CONTROL ---\n")
print(objective_post_age_comparison)

cat("\n--- SUBJECTIVE COEFFICIENTS: BASE VS POST AGE CONTROL ---\n")
print(subjective_post_age_comparison)

# HC3 robust standard errors for the post-age-controlled models as an additional check
hc3_objective_post_age_test <- coeftest(
  model.objective.post_age,
  vcov. = vcovHC(model.objective.post_age, type = "HC3")
)

hc3_subjective_post_age_test <- coeftest(
  model.subjective.post_age,
  vcov. = vcovHC(model.subjective.post_age, type = "HC3")
)

cat("\n--- OBJECTIVE + POST AGE WITH HC3 ROBUST STANDARD ERRORS ---\n")
print(hc3_objective_post_age_test)

cat("\n--- SUBJECTIVE + POST AGE WITH HC3 ROBUST STANDARD ERRORS ---\n")
print(hc3_subjective_post_age_test)

# ---------------------------------------------------------------------
# 9. Robustness check for the BEVoCI interaction model
# ---------------------------------------------------------------------
# Post age is allowed to have a different association with each measure through
# measure * Post_Age_Days_control_c. This avoids forcing one common post-age effect
# on objective and subjective clarity.

model_measure_comparison.post_age <- lm(
  measurev ~ measure +
    Has_Image_iv_c +
    User_Reputation_iv_c +
    Tag_Count_iv_c +
    Title_Word_Count_iv_c +
    Word_Count_iv_c +
    LaTeX_Comment_Count_iv_c +
    Post_Age_Days_control_c +
    measure * Has_Image_iv_c +
    measure * User_Reputation_iv_c +
    measure * Tag_Count_iv_c +
    measure * Title_Word_Count_iv_c +
    measure * Word_Count_iv_c +
    measure * LaTeX_Comment_Count_iv_c +
    measure * Post_Age_Days_control_c,
  data = raw_data_duplicated
)

cat("\n--- ROBUSTNESS: BIAS EXPOSURE MODEL + POST AGE CONTROL ---\n")
print(summary(model_measure_comparison.post_age))

interaction_post_age_table <- make_ranked_table(
  model_measure_comparison.post_age,
  interaction_only = TRUE
) %>%
  filter(!str_detect(Term, "Post_Age_Days_control_c")) %>%
  arrange(desc(Absolute_Size)) %>%
  mutate(Rank = row_number(), .before = 1)

cat("\n--- RANKED BEVOCI INTERACTIONS WITH POST AGE CONTROL ---\n")
print(interaction_post_age_table)

# ---------------------------------------------------------------------
# 10. Save all numerical results
# ---------------------------------------------------------------------

if (!dir.exists("results")) {
  dir.create("results")
}

write_csv(objective_table, "results/regression_objective.csv")
write_csv(subjective_table, "results/regression_subjective.csv")
write_csv(interaction_table, "results/regression_bevoci_interactions.csv")

write_csv(hc3_objective_table, "results/hc3_objective.csv")
write_csv(hc3_subjective_table, "results/hc3_subjective.csv")

write_csv(multiple_comparison_table, "results/multiple_comparison_diagnostic.csv")

write_csv(r2_comparison, "results/robustness_post_age_r2.csv")
write_csv(objective_post_age_comparison, "results/robustness_post_age_objective_coefficients.csv")
write_csv(subjective_post_age_comparison, "results/robustness_post_age_subjective_coefficients.csv")
write_csv(interaction_post_age_table, "results/robustness_bevoci_interactions_post_age.csv")

# Assumption-test summary
assumption_tests <- tibble(
  Model = c(
    "Objective - Breusch-Pagan",
    "Subjective - Breusch-Pagan",
    "Objective - Shapiro-Wilk",
    "Subjective - Shapiro-Wilk"
  ),
  Statistic = c(
    unname(bp_objective$statistic),
    unname(bp_subjective$statistic),
    unname(shapiro_objective$statistic),
    unname(shapiro_subjective$statistic)
  ),
  p_value = c(
    bp_objective$p.value,
    bp_subjective$p.value,
    shapiro_objective$p.value,
    shapiro_subjective$p.value
  )
)

write_csv(assumption_tests, "results/regression_assumption_tests.csv")

# ---------------------------------------------------------------------
# 11. Graph generation with aligned clarity direction and 95% CIs
# ---------------------------------------------------------------------

color_objective <- "aquamarine4"
color_subjective <- "darkorchid4"

# Shared theme: readable labels and a vertical legend on the right.
plot_theme <- theme_minimal(base_size = 15) +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 14),
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 13),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    plot.margin = margin(12, 20, 12, 12)
  )

# ---------------------------
# Graph 1: Has Image
# ---------------------------
image_summary <- raw_data %>%
  group_by(Has_Image) %>%
  summarise(
    Obj_Mean = mean(Objective_Clarity_dv_c, na.rm = TRUE),
    Obj_CI = (sd(Objective_Clarity_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(Obj_Mean, Sub_Mean),
    names_to = "Measure_Type",
    values_to = "Mean_Z_Score"
  ) %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Has_Image = ifelse(Has_Image == 1, "Yes (Image Included)", "No (Text Only)"),
    Measure = ifelse(
      Measure_Type == "Obj_Mean",
      "Objective Clarity (-Comment Count)",
      "Subjective (Upvote Score)"
    )
  )

plot_image <- ggplot(image_summary, aes(x = Has_Image, y = Mean_Z_Score, fill = Measure)) +
  geom_col(position = "dodge", color = "black", width = 0.6) +
  geom_errorbar(
    aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI),
    position = position_dodge(0.6),
    width = 0.2,
    color = "black"
  ) +
  scale_fill_manual(values = c(color_objective, color_subjective)) +
  plot_theme +
  labs(
    title = "Effect of Image Inclusion on Actual vs. Perceived Clarity",
    x = "Presence of Image in Question",
    y = "Standardized Mean Clarity Score (Z-Score)",
    fill = "Measure Type"
  )

# ---------------------------
# Graph 2: Word Count
# ---------------------------
raw_data <- raw_data %>%
  mutate(
    Word_Count_Bin = ntile(Word_Count, 4),
    Word_Count_Category = case_when(
      Word_Count_Bin == 1 ~ "Q1 (Shortest)",
      Word_Count_Bin == 2 ~ "Q2",
      Word_Count_Bin == 3 ~ "Q3",
      Word_Count_Bin == 4 ~ "Q4 (Longest)"
    )
  )

word_summary <- raw_data %>%
  group_by(Word_Count_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Clarity_dv_c, na.rm = TRUE),
    Obj_CI = (sd(Objective_Clarity_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(Obj_Mean, Sub_Mean),
    names_to = "Measure_Type",
    values_to = "Mean_Z_Score"
  ) %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(
      Measure_Type == "Obj_Mean",
      "Objective Clarity (-Comment Count)",
      "Subjective (Upvote Score)"
    )
  )

word_summary$Word_Count_Category <- factor(
  word_summary$Word_Count_Category,
  levels = c("Q1 (Shortest)", "Q2", "Q3", "Q4 (Longest)")
)

plot_word_count <- ggplot(
  word_summary,
  aes(x = Word_Count_Category, y = Mean_Z_Score, group = Measure, color = Measure)
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI),
    width = 0.15,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  plot_theme +
  labs(
    title = "Effect of Word Count on Actual vs. Perceived Clarity",
    x = "Word Count Quartiles",
    y = "Standardized Mean Clarity Score (Z-Score)",
    color = "Measure Type"
  )

# ---------------------------
# Graph 3: User Reputation
# ---------------------------
raw_data <- raw_data %>%
  mutate(
    Reputation_Bin = ntile(User_Reputation, 4),
    Reputation_Category = case_when(
      Reputation_Bin == 1 ~ "Q1 (Lowest Rep)",
      Reputation_Bin == 2 ~ "Q2",
      Reputation_Bin == 3 ~ "Q3",
      Reputation_Bin == 4 ~ "Q4 (Highest Rep)"
    )
  )

rep_summary <- raw_data %>%
  group_by(Reputation_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Clarity_dv_c, na.rm = TRUE),
    Obj_CI = (sd(Objective_Clarity_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(Obj_Mean, Sub_Mean),
    names_to = "Measure_Type",
    values_to = "Mean_Z_Score"
  ) %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(
      Measure_Type == "Obj_Mean",
      "Objective Clarity (-Comment Count)",
      "Subjective (Upvote Score)"
    )
  )

rep_summary$Reputation_Category <- factor(
  rep_summary$Reputation_Category,
  levels = c("Q1 (Lowest Rep)", "Q2", "Q3", "Q4 (Highest Rep)")
)

plot_reputation <- ggplot(
  rep_summary,
  aes(x = Reputation_Category, y = Mean_Z_Score, group = Measure, color = Measure)
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI),
    width = 0.15,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  plot_theme +
  labs(
    title = "Effect of User Reputation on Actual vs. Perceived Clarity",
    x = "User Reputation Quartiles",
    y = "Standardized Mean Clarity Score (Z-Score)",
    color = "Measure Type"
  )

# ---------------------------
# Graph 4: LaTeX Comment Count
# ---------------------------
raw_data <- raw_data %>%
  mutate(
    LaTeX_Category = case_when(
      LaTeX_Comment_Count == 0 ~ "0",
      LaTeX_Comment_Count == 1 ~ "1",
      LaTeX_Comment_Count == 2 ~ "2",
      TRUE ~ "3+"
    )
  )

latex_summary <- raw_data %>%
  group_by(LaTeX_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Clarity_dv_c, na.rm = TRUE),
    Obj_CI = (sd(Objective_Clarity_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(Obj_Mean, Sub_Mean),
    names_to = "Measure_Type",
    values_to = "Mean_Z_Score"
  ) %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(
      Measure_Type == "Obj_Mean",
      "Objective Clarity (-Comment Count)",
      "Subjective (Upvote Score)"
    )
  )

latex_summary$LaTeX_Category <- factor(
  latex_summary$LaTeX_Category,
  levels = c("0", "1", "2", "3+")
)

plot_latex <- ggplot(
  latex_summary,
  aes(x = LaTeX_Category, y = Mean_Z_Score, group = Measure, color = Measure)
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI),
    width = 0.15,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  plot_theme +
  labs(
    title = "Effect of LaTeX Comment Usage on Actual vs. Perceived Clarity",
    x = "Number of LaTeX Comments in Question",
    y = "Standardized Mean Clarity Score (Z-Score)",
    color = "Measure Type"
  )

# ---------------------------
# Graph 5: Title Word Count
# ---------------------------
raw_data <- raw_data %>%
  mutate(
    Title_Word_Count_Bin = ntile(Title_Word_Count, 4),
    Title_Word_Count_Category = case_when(
      Title_Word_Count_Bin == 1 ~ "Q1 (Shortest)",
      Title_Word_Count_Bin == 2 ~ "Q2",
      Title_Word_Count_Bin == 3 ~ "Q3",
      Title_Word_Count_Bin == 4 ~ "Q4 (Longest)"
    )
  )

title_summary <- raw_data %>%
  group_by(Title_Word_Count_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Clarity_dv_c, na.rm = TRUE),
    Obj_CI = (sd(Objective_Clarity_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(Obj_Mean, Sub_Mean),
    names_to = "Measure_Type",
    values_to = "Mean_Z_Score"
  ) %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(
      Measure_Type == "Obj_Mean",
      "Objective Clarity (-Comment Count)",
      "Subjective (Upvote Score)"
    )
  )

title_summary$Title_Word_Count_Category <- factor(
  title_summary$Title_Word_Count_Category,
  levels = c("Q1 (Shortest)", "Q2", "Q3", "Q4 (Longest)")
)

plot_title <- ggplot(
  title_summary,
  aes(x = Title_Word_Count_Category, y = Mean_Z_Score, group = Measure, color = Measure)
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI),
    width = 0.15,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  plot_theme +
  labs(
    title = "Effect of Title Length on Actual vs. Perceived Clarity",
    x = "Title Word Count Quartiles",
    y = "Standardized Mean Clarity Score (Z-Score)",
    color = "Measure Type"
  )

# ---------------------------
# Graph 6: Tag Count
# ---------------------------
tag_summary <- raw_data %>%
  group_by(Tag_Count) %>%
  summarise(
    Obj_Mean = mean(Objective_Clarity_dv_c, na.rm = TRUE),
    Obj_CI = (sd(Objective_Clarity_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(Obj_Mean, Sub_Mean),
    names_to = "Measure_Type",
    values_to = "Mean_Z_Score"
  ) %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(
      Measure_Type == "Obj_Mean",
      "Objective Clarity (-Comment Count)",
      "Subjective (Upvote Score)"
    )
  )

tag_summary$Tag_Count <- as.factor(tag_summary$Tag_Count)

plot_tag <- ggplot(
  tag_summary,
  aes(x = Tag_Count, y = Mean_Z_Score, group = Measure, color = Measure)
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI),
    width = 0.15,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  plot_theme +
  labs(
    title = "Effect of Tag Count on Actual vs. Perceived Clarity",
    x = "Number of Tags",
    y = "Standardized Mean Clarity Score (Z-Score)",
    color = "Measure Type"
  )

# ---------------------------------------------------------------------
# 12. Save all plots
# ---------------------------------------------------------------------

if (!dir.exists("plots")) {
  dir.create("plots")
}

ggsave("plots/plot_image_bias.png", plot_image, width = 10.5, height = 5.5, dpi = 300, bg = "white")
ggsave("plots/plot_word_count_bias.png", plot_word_count, width = 10.5, height = 5.5, dpi = 300, bg = "white")
ggsave("plots/plot_reputation_bias.png", plot_reputation, width = 10.5, height = 5.5, dpi = 300, bg = "white")
ggsave("plots/plot_latex_bias.png", plot_latex, width = 10.5, height = 5.5, dpi = 300, bg = "white")
ggsave("plots/plot_title_bias.png", plot_title, width = 10.5, height = 5.5, dpi = 300, bg = "white")
ggsave("plots/plot_tag_bias.png", plot_tag, width = 10.5, height = 5.5, dpi = 300, bg = "white")

cat("\n=====================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("=====================================================================\n")
cat("Main regression tables, robustness results, and assumption checks were saved in 'results'.\n")
cat("All 6 corrected plots were saved in 'plots'.\n")
