# =====================================================================
# STACK EXCHANGE METACONTINITION ANALYSIS (BEVOCI MODEL & PLOTS)
# =====================================================================

# 1. Load Packages
# install.packages("tidyverse")
library(tidyverse)

# Clear environment & set directory
rm(list=ls(all.names=TRUE))
setwd(".")

# 2. Load and Prepare Data
raw_data <- read_csv("stackexchange_enhanced_dataset.csv")

# Convert boolean Has_Image to numeric (TRUE=1, FALSE=0)
raw_data$Has_Image <- as.numeric(raw_data$Has_Image)

# Standardize Cues (IVs)
raw_data <- raw_data %>%
  mutate(across(c(Has_Image, Word_Count, Code_Block_Count, Link_Count, Title_Word_Count),
                scale,
                .names="{.col}_iv_c"))

# Standardize Measures (DVs)
raw_data <- raw_data %>%
  mutate(across(c(Subjective_Score, Objective_Comment_Count),
                scale,
                .names="{.col}_dv_c"))

# 3. Correlation Matrix (Check for Multicollinearity)
cues <- c("Has_Image_iv_c", "Word_Count_iv_c", "Code_Block_Count_iv_c",
          "Link_Count_iv_c", "Title_Word_Count_iv_c")
corrs <- cor(raw_data[cues], use = "complete.obs")
print("--- CORRELATION MATRIX ---")
round(corrs, 2)


# =====================================================================
# REGRESSION MODELS (TASK 4)
# =====================================================================

# Model A: Objective Measure (Actual Clarity / Friction)
model.objective <- lm(Objective_Comment_Count_dv_c ~ Has_Image_iv_c +
                        Word_Count_iv_c + Code_Block_Count_iv_c +
                        Link_Count_iv_c + Title_Word_Count_iv_c,
                      data = raw_data)

print("--- OBJECTIVE MEASURE (COMMENT COUNT) ---")
summary(model.objective)


# Model B: Subjective Measure (Perceived Clarity / Upvotes)
model.subjective <- lm(Subjective_Score_dv_c ~ Has_Image_iv_c +
                         Word_Count_iv_c + Code_Block_Count_iv_c +
                         Link_Count_iv_c + Title_Word_Count_iv_c,
                       data = raw_data)

print("--- SUBJECTIVE MEASURE (UPVOTE SCORE) ---")
summary(model.subjective)


# Model C: Interaction Model (Bias Exposure - TASK 5)
raw_data_obj  <- raw_data %>% mutate(measurev = Objective_Comment_Count_dv_c, measure = "objective")
raw_data_subj <- raw_data %>% mutate(measurev = Subjective_Score_dv_c, measure = "subjective")
raw_data_duplicated <- bind_rows(raw_data_obj, raw_data_subj)

model_measure_comparison <- lm(measurev ~ measure +
                                 Has_Image_iv_c + Word_Count_iv_c +
                                 Code_Block_Count_iv_c + Link_Count_iv_c + Title_Word_Count_iv_c +
                                 measure*Has_Image_iv_c +
                                 measure*Word_Count_iv_c +
                                 measure*Code_Block_Count_iv_c +
                                 measure*Link_Count_iv_c +
                                 measure*Title_Word_Count_iv_c,
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