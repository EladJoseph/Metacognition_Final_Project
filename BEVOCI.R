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
# This will print a list of all columns and the exact number of NAs in each
missing_counts <- colSums(is.na(raw_data))
print(missing_counts[missing_counts > 0]) # Only print columns that actually have missing data

# Check what percentage of the dataset is missing the Accept_Rate
accept_rate_na_percent <- sum(is.na(raw_data$Accept_Rate)) / nrow(raw_data) * 100
print(paste("Percentage of missing Accept_Rate:", round(accept_rate_na_percent, 2), "%"))

# Standardize Cues (IVs)
# We standardize the badges here ONLY so we can prove their multicollinearity in the matrix.
# They will be excluded from the actual regression models.
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
# Removing badges, Code_Block_Count, and Link_Count from the cue list to display the clean matrix used for regressions
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
# GRAPH GENERATION WITH ERROR BARS (TASK 4 PLOTS)
# =====================================================================

# GRAPH 1: BAR CHART FOR 'HAS IMAGE' (With Error Bars)
image_summary <- raw_data %>%
  group_by(Has_Image) %>%
  summarise(
    Obj_Mean = mean(Objective_Comment_Count_dv_c, na.rm = TRUE),
    Obj_SE   = sd(Objective_Comment_Count_dv_c, na.rm = TRUE) / sqrt(n()),
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_SE   = sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())
  ) %>%
  pivot_longer(cols = c(Obj_Mean, Sub_Mean),
               names_to = "Measure_Type",
               values_to = "Mean_Z_Score") %>%
  mutate(
    SE = ifelse(Measure_Type == "Obj_Mean", Obj_SE, Sub_SE),
    Has_Image = ifelse(Has_Image == 1, "Yes (Image Included)", "No (Text Only)"),
    Measure = ifelse(Measure_Type == "Obj_Mean", "Objective (Comment Count)", "Subjective (Upvote Score)")
  )

plot_image <- ggplot(image_summary, aes(x = Has_Image, y = Mean_Z_Score, fill = Measure)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", width = 0.6) +
  geom_errorbar(aes(ymin = Mean_Z_Score - SE, ymax = Mean_Z_Score + SE),
                position = position_dodge(0.6), width = 0.2, color = "black") +
  scale_fill_manual(values = c("gray60", "gray90")) +
  theme_minimal() +
  labs(title = "Effect of Including an Image on Actual vs. Perceived Clarity",
       x = "Presence of Image in Question",
       y = "Standardized Mean Score (Z-Score)",
       fill = "Measure Type") +
  theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"))


# GRAPH 2: LINE GRAPH FOR 'LINK COUNT' (With Error Bars)
raw_data <- raw_data %>%
  mutate(Link_Category = case_when(
    Link_Count == 0 ~ "0 Links",
    Link_Count == 1 ~ "1 Link",
    Link_Count == 2 ~ "2 Links",
    TRUE ~ "3+ Links"
  ))

link_summary <- raw_data %>%
  group_by(Link_Category) %>%
  summarise(
    Obj_Mean = mean(Objective_Comment_Count_dv_c, na.rm = TRUE),
    Obj_SE   = sd(Objective_Comment_Count_dv_c, na.rm = TRUE) / sqrt(n()),
    Sub_Mean = mean(Subjective_Score_dv_c, na.rm = TRUE),
    Sub_SE   = sd(Subjective_Score_dv_c, na.rm = TRUE) / sqrt(n())
  ) %>%
  pivot_longer(cols = c(Obj_Mean, Sub_Mean),
               names_to = "Measure_Type",
               values_to = "Mean_Z_Score") %>%
  mutate(
    SE = ifelse(Measure_Type == "Obj_Mean", Obj_SE, Sub_SE),
    Measure = ifelse(Measure_Type == "Obj_Mean", "Objective (Comment Count)", "Subjective (Upvote Score)")
  )

link_summary$Link_Category <- factor(link_summary$Link_Category,
                                     levels = c("0 Links", "1 Link", "2 Links", "3+ Links"))

plot_links <- ggplot(link_summary, aes(x = Link_Category, y = Mean_Z_Score, group = Measure, linetype = Measure)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Mean_Z_Score - SE, ymax = Mean_Z_Score + SE), width = 0.15) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  theme_minimal() +
  labs(title = "Effect of Link Count on Actual vs. Perceived Clarity",
       x = "Number of Links in Question",
       y = "Standardized Mean Score (Z-Score)",
       linetype = "Measure Type") +
  theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"))

# Save the plots to your project directory for PyCharm
ggsave("plot_image_bias.png", plot = plot_image, width = 7, height = 5, dpi = 300)
ggsave("plot_links_utilization.png", plot = plot_links, width = 7, height = 5, dpi = 300)

print("Plots with error bars successfully saved as PNG files!")