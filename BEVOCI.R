# =====================================================================
# STACK EXCHANGE METACOGNITION ANALYSIS (BEVOCI MODEL & PLOTS)
# =====================================================================

# 1. Load Packages
# install.packages("tidyverse")
library(tidyverse)

# Clear environment & set directory
rm(list=ls(all.names=TRUE))
setwd(".")

# 2. Load and Prepare Data
raw_data <- read_csv("stackexchange_enhanced_dataset.csv")

# Convert boolean columns to numeric (TRUE=1, FALSE=0)
raw_data$Has_Image <- as.numeric(raw_data$Has_Image)
raw_data$Is_Question_Format <- as.numeric(raw_data$Is_Question_Format)
raw_data$Has_Custom_Avatar <- as.numeric(raw_data$Has_Custom_Avatar)

# --- MISSING DATA CHECK ---
print("--- MISSING VALUES PER COLUMN ---")
missing_counts <- colSums(is.na(raw_data))
print(missing_counts[missing_counts > 0])

accept_rate_na_percent <- sum(is.na(raw_data$Accept_Rate)) / nrow(raw_data) * 100
print(paste("Percentage of missing Accept_Rate:", round(accept_rate_na_percent, 2), "%"))

# Standardize Cues (IVs)
raw_data <- raw_data %>%
  mutate(across(c(Has_Image, User_Reputation, Gold_Badges, Silver_Badges, Bronze_Badges,
                  Tag_Count, Title_Word_Count, Word_Count, Code_Block_Count,
                  LaTeX_Comment_Count, Link_Count),
                scale,
                .names="{.col}_iv_c"))

# Standardize Measures (DVs)
raw_data <- raw_data %>%
  mutate(across(c(Subjective_Score, Objective_Comment_Count),
                scale,
                .names="{.col}_dv_c"))

# 3. Correlation Matrices (Check for Multicollinearity)

# 3A. Matrix BEFORE removing collinear variables
cues_all <- c("Has_Image_iv_c", "User_Reputation_iv_c", "Gold_Badges_iv_c",
              "Silver_Badges_iv_c", "Bronze_Badges_iv_c", "Tag_Count_iv_c",
              "Title_Word_Count_iv_c", "Word_Count_iv_c", "Code_Block_Count_iv_c",
              "LaTeX_Comment_Count_iv_c", "Link_Count_iv_c")
corrs_all <- cor(raw_data[cues_all], use = "complete.obs")
print("--- CORRELATION MATRIX (BEFORE BADGE/CODE/LINK REMOVAL) ---")
print(round(corrs_all, 2))

# 3B. Matrix AFTER removing collinear variables
cues_filtered <- c("Has_Image_iv_c", "User_Reputation_iv_c", "Tag_Count_iv_c",
                   "Title_Word_Count_iv_c", "Word_Count_iv_c", "LaTeX_Comment_Count_iv_c")
corrs_filtered <- cor(raw_data[cues_filtered], use = "complete.obs")
print("--- CORRELATION MATRIX (AFTER COLLINEAR REMOVAL) ---")
print(round(corrs_filtered, 2))


# =====================================================================
# REGRESSION MODELS (TASK 4)
# =====================================================================

# Model A: Objective Measure (Actual Clarity / Friction)
model.objective <- lm(Objective_Comment_Count_dv_c ~ Has_Image_iv_c +
                        User_Reputation_iv_c +
                        Tag_Count_iv_c + Title_Word_Count_iv_c +
                        Word_Count_iv_c + LaTeX_Comment_Count_iv_c,
                      data = raw_data)

print("--- OBJECTIVE MEASURE (COMMENT COUNT) ---")
summary(model.objective)


# Model B: Subjective Measure (Perceived Clarity / Upvotes)
model.subjective <- lm(Subjective_Score_dv_c ~ Has_Image_iv_c +
                         User_Reputation_iv_c +
                         Tag_Count_iv_c + Title_Word_Count_iv_c +
                         Word_Count_iv_c + LaTeX_Comment_Count_iv_c,
                       data = raw_data)

print("--- SUBJECTIVE MEASURE (UPVOTE SCORE) ---")
summary(model.subjective)


# Model C: Interaction Model (Bias Exposure)
raw_data_obj  <- raw_data %>% mutate(measurev = Objective_Comment_Count_dv_c, measure = "objective")
raw_data_subj <- raw_data %>% mutate(measurev = Subjective_Score_dv_c, measure = "subjective")
raw_data_duplicated <- bind_rows(raw_data_obj, raw_data_subj)

model_measure_comparison <- lm(measurev ~ measure +
                                 Has_Image_iv_c + User_Reputation_iv_c +
                                 Tag_Count_iv_c + Title_Word_Count_iv_c + Word_Count_iv_c +
                                 LaTeX_Comment_Count_iv_c +
                                 measure*Has_Image_iv_c +
                                 measure*User_Reputation_iv_c +
                                 measure*Tag_Count_iv_c +
                                 measure*Title_Word_Count_iv_c +
                                 measure*Word_Count_iv_c +
                                 measure*LaTeX_Comment_Count_iv_c,
                               data = raw_data_duplicated)

print("--- BIAS EXPOSURE (INTERACTION MODEL) ---")
summary(model_measure_comparison)


# =====================================================================
# GRAPH GENERATION WITH 95% CONFIDENCE INTERVAL ERROR BARS
# =====================================================================

# Define consistent colors for the plots
color_objective <- "aquamarine4"
color_subjective <- "darkorchid4"

# Common theme configuration for larger fonts
custom_theme <- theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 13)
  )

# ---------------------------------------------------------
# GRAPH 1: BAR CHART FOR 'HAS IMAGE'
# ---------------------------------------------------------
image_summary <- raw_data %>%
  group_by(Has_Image) %>%
  summarise(
    Obj_Mean = mean(Objective_Comment_Count_dv_c, na.rm = TRUE),
    Obj_CI   = (sd(Objective_Comment_Count_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI   = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96
  ) %>%
  pivot_longer(cols = c(Obj_Mean, Sub_Mean), names_to = "Measure_Type", values_to = "Mean_Z_Score") %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Has_Image = ifelse(Has_Image == 1, "Yes (Image Included)", "No (Text Only)"),
    Measure = ifelse(Measure_Type == "Obj_Mean", "Objective", "Subjective")
  )

plot_image <- ggplot(image_summary, aes(x = Has_Image, y = Mean_Z_Score, fill = Measure)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", width = 0.6) +
  geom_errorbar(aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI), position = position_dodge(0.6), width = 0.2, color = "black") +
  scale_fill_manual(values = c(color_objective, color_subjective)) +
  labs(title = "Has Image", x = "Presence of Image in Question", y = "Standardized Mean Score (Z-Score)", fill = "Measure Type") +
  custom_theme

# ---------------------------------------------------------
# GRAPH 2: LINE GRAPH FOR 'WORD COUNT' (Binned by Quartiles)
# ---------------------------------------------------------
raw_data <- raw_data %>%
  mutate(Word_Count_Bin = ntile(Word_Count, 4),
         Word_Count_Category = case_when(
           Word_Count_Bin == 1 ~ "Q1 (Shortest)",
           Word_Count_Bin == 2 ~ "Q2",
           Word_Count_Bin == 3 ~ "Q3",
           Word_Count_Bin == 4 ~ "Q4 (Longest)"
         ))

word_summary <- raw_data %>%
  group_by(Word_Count_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Comment_Count_dv_c, na.rm = TRUE),
    Obj_CI   = (sd(Objective_Comment_Count_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI   = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96
  ) %>%
  pivot_longer(cols = c(Obj_Mean, Sub_Mean), names_to = "Measure_Type", values_to = "Mean_Z_Score") %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(Measure_Type == "Obj_Mean", "Objective", "Subjective")
  )

word_summary$Word_Count_Category <- factor(word_summary$Word_Count_Category, levels = c("Q1 (Shortest)", "Q2", "Q3", "Q4 (Longest)"))

plot_word_count <- ggplot(word_summary, aes(x = Word_Count_Category, y = Mean_Z_Score, group = Measure, color = Measure)) +
  geom_line(size = 1.2) + geom_point(size = 3) +
  geom_errorbar(aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI), width = 0.15, show.legend = FALSE) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  labs(title = "Word Count", x = "Word Count Quartiles", y = "Standardized Mean Score (Z-Score)", color = "Measure Type") +
  custom_theme

# ---------------------------------------------------------
# GRAPH 3: LINE GRAPH FOR 'USER REPUTATION' (Binned by Quartiles)
# ---------------------------------------------------------
raw_data <- raw_data %>%
  mutate(Reputation_Bin = ntile(User_Reputation, 4),
         Reputation_Category = case_when(
           Reputation_Bin == 1 ~ "Q1 (Lowest Rep)",
           Reputation_Bin == 2 ~ "Q2",
           Reputation_Bin == 3 ~ "Q3",
           Reputation_Bin == 4 ~ "Q4 (Highest Rep)"
         ))

rep_summary <- raw_data %>%
  group_by(Reputation_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Comment_Count_dv_c, na.rm = TRUE),
    Obj_CI   = (sd(Objective_Comment_Count_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI   = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96
  ) %>%
  pivot_longer(cols = c(Obj_Mean, Sub_Mean), names_to = "Measure_Type", values_to = "Mean_Z_Score") %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(Measure_Type == "Obj_Mean", "Objective", "Subjective")
  )

rep_summary$Reputation_Category <- factor(rep_summary$Reputation_Category, levels = c("Q1 (Lowest Rep)", "Q2", "Q3", "Q4 (Highest Rep)"))

plot_reputation <- ggplot(rep_summary, aes(x = Reputation_Category, y = Mean_Z_Score, group = Measure, color = Measure)) +
  geom_line(size = 1.2) + geom_point(size = 3) +
  geom_errorbar(aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI), width = 0.15, show.legend = FALSE) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  labs(title = "User Reputation", x = "User Reputation Quartiles", y = "Standardized Mean Score (Z-Score)", color = "Measure Type") +
  custom_theme

# ---------------------------------------------------------
# GRAPH 4: LINE GRAPH FOR 'LATEX COMMENT COUNT'
# ---------------------------------------------------------
raw_data <- raw_data %>%
  mutate(LaTeX_Category = case_when(
    LaTeX_Comment_Count == 0 ~ "0",
    LaTeX_Comment_Count == 1 ~ "1",
    LaTeX_Comment_Count == 2 ~ "2",
    TRUE ~ "3+"
  ))

latex_summary <- raw_data %>%
  group_by(LaTeX_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Comment_Count_dv_c, na.rm = TRUE),
    Obj_CI   = (sd(Objective_Comment_Count_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI   = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96
  ) %>%
  pivot_longer(cols = c(Obj_Mean, Sub_Mean), names_to = "Measure_Type", values_to = "Mean_Z_Score") %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(Measure_Type == "Obj_Mean", "Objective", "Subjective")
  )

latex_summary$LaTeX_Category <- factor(latex_summary$LaTeX_Category, levels = c("0", "1", "2", "3+"))

plot_latex <- ggplot(latex_summary, aes(x = LaTeX_Category, y = Mean_Z_Score, group = Measure, color = Measure)) +
  geom_line(size = 1.2) + geom_point(size = 3) +
  geom_errorbar(aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI), width = 0.15, show.legend = FALSE) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  labs(title = "LaTeX Comment Count", x = "Number of LaTeX Comments in Question", y = "Standardized Mean Score (Z-Score)", color = "Measure Type") +
  custom_theme

# ---------------------------------------------------------
# GRAPH 5: LINE GRAPH FOR 'TITLE WORD COUNT' (Binned by Quartiles)
# ---------------------------------------------------------
raw_data <- raw_data %>%
  mutate(Title_Word_Count_Bin = ntile(Title_Word_Count, 4),
         Title_Word_Count_Category = case_when(
           Title_Word_Count_Bin == 1 ~ "Q1 (Shortest)",
           Title_Word_Count_Bin == 2 ~ "Q2",
           Title_Word_Count_Bin == 3 ~ "Q3",
           Title_Word_Count_Bin == 4 ~ "Q4 (Longest)"
         ))

title_summary <- raw_data %>%
  group_by(Title_Word_Count_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Comment_Count_dv_c, na.rm = TRUE),
    Obj_CI   = (sd(Objective_Comment_Count_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI   = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96
  ) %>%
  pivot_longer(cols = c(Obj_Mean, Sub_Mean), names_to = "Measure_Type", values_to = "Mean_Z_Score") %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(Measure_Type == "Obj_Mean", "Objective", "Subjective")
  )

title_summary$Title_Word_Count_Category <- factor(title_summary$Title_Word_Count_Category, levels = c("Q1 (Shortest)", "Q2", "Q3", "Q4 (Longest)"))

plot_title <- ggplot(title_summary, aes(x = Title_Word_Count_Category, y = Mean_Z_Score, group = Measure, color = Measure)) +
  geom_line(size = 1.2) + geom_point(size = 3) +
  geom_errorbar(aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI), width = 0.15, show.legend = FALSE) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  labs(title = "Title Word Count", x = "Title Word Count Quartiles", y = "Standardized Mean Score (Z-Score)", color = "Measure Type") +
  custom_theme

# ---------------------------------------------------------
# GRAPH 6: LINE GRAPH FOR 'TAG COUNT' (Exact Counts 0 to 5)
# ---------------------------------------------------------
tag_summary <- raw_data %>%
  group_by(Tag_Count) %>%
  summarise(
    Obj_Mean = mean(Objective_Comment_Count_dv_c, na.rm = TRUE),
    Obj_CI   = (sd(Objective_Comment_Count_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96,
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_CI   = (sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())) * 1.96
  ) %>%
  pivot_longer(cols = c(Obj_Mean, Sub_Mean), names_to = "Measure_Type", values_to = "Mean_Z_Score") %>%
  mutate(
    CI = ifelse(Measure_Type == "Obj_Mean", Obj_CI, Sub_CI),
    Measure = ifelse(Measure_Type == "Obj_Mean", "Objective", "Subjective")
  )

tag_summary$Tag_Count <- as.factor(tag_summary$Tag_Count)

plot_tag <- ggplot(tag_summary, aes(x = Tag_Count, y = Mean_Z_Score, group = Measure, color = Measure)) +
  geom_line(size = 1.2) + geom_point(size = 3) +
  geom_errorbar(aes(ymin = Mean_Z_Score - CI, ymax = Mean_Z_Score + CI), width = 0.15, show.legend = FALSE) +
  scale_color_manual(values = c(color_objective, color_subjective)) +
  labs(title = "Tag Count", x = "Number of Tags", y = "Standardized Mean Score (Z-Score)", color = "Measure Type") +
  custom_theme

# ---------------------------------------------------------
# SAVE ALL PLOTS INTO "plots" FOLDER
# ---------------------------------------------------------
# Create the 'plots' directory if it doesn't already exist
if (!dir.exists("plots")) {
  dir.create("plots")
}

ggsave("plots/plot_image_bias.png", plot = plot_image, width = 7, height = 5, dpi = 300)
ggsave("plots/plot_word_count_bias.png", plot = plot_word_count, width = 7, height = 5, dpi = 300)
ggsave("plots/plot_reputation_bias.png", plot = plot_reputation, width = 7, height = 5, dpi = 300)
ggsave("plots/plot_latex_bias.png", plot = plot_latex, width = 7, height = 5, dpi = 300)
ggsave("plots/plot_title_bias.png", plot = plot_title, width = 7, height = 5, dpi = 300)
ggsave("plots/plot_tag_bias.png", plot = plot_tag, width = 7, height = 5, dpi = 300)

print("All 6 colored plots with updated labels and typography successfully saved as PNG files in the 'plots' folder!")