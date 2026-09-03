# Metacognition Final Project - Stack Exchange Clarity Analysis

## Overview

This project examines problem clarity in questions posted to the LaTeX community on Stack Exchange.

We use two proxy measures of clarity:

- **Objective measure:** `Objective_Comment_Count` - the number of comments on a question. Comments often contain requests for clarification, so a higher value may indicate greater clarification friction.
- **Subjective measure:** `Subjective_Score` - the question score based on upvotes and downvotes, used as a proxy for perceived clarity or quality.

The project combines machine-learning models with regression and BEVoCI-style analyses to examine both predictive performance and possible metacognitive cue effects.

## Data

The analysis uses `stackexchange_enhanced_dataset.csv`, containing 5,000 Stack Exchange questions and their associated metadata.

The dataset includes:

- user information and reputation
- question text characteristics
- tags
- images and links
- LaTeX/code-related features
- question score and comment count
- post age and additional response-related variables

Some variables collected after publication were excluded from the baseline models when they could create target leakage.

## Main analyses

### 1. Baseline machine-learning models

Random Forest and XGBoost regressors are trained separately for the objective and subjective measures.

Hyperparameters are selected using Grid Search with 5-fold cross-validation on the training set. Final model performance is evaluated on a held-out test set using:

- R-squared
- MAE
- RMSE
- feature importance

### 2. Metacognitive cue selection

Potential cues were selected based on course material and relevant literature. Correlated variables were screened to reduce multicollinearity before regression analysis.

The final regression cue set includes:

- `Has_Image`
- `User_Reputation`
- `Tag_Count`
- `Title_Word_Count`
- `Word_Count`
- `LaTeX_Comment_Count`

### 3. Regression and BEVoCI analysis

`BEVOCI.R` runs:

1. an objective regression predicting standardized comment count
2. a subjective regression predicting standardized question score
3. an interaction model comparing cue effects across the two measures
4. descriptive plots with 95% confidence intervals

### 4. Robustness and sensitivity checks

The R analysis also includes two robustness checks.

#### Regression diagnostics and HC3 standard errors

The script tests regression assumptions using:

- Breusch-Pagan tests for heteroskedasticity
- Shapiro-Wilk tests for residual normality
- HC3 heteroskedasticity-robust standard errors

The HC3 analysis is used to check whether statistical significance is sensitive to heteroskedasticity.

#### Post-age sensitivity analysis

`Post_Age_Days` is added as a control variable because post age was highly important in the machine-learning models, especially for the subjective score.

This analysis checks whether associations attributed to the metacognitive cues remain similar after accounting for the amount of time a post has been exposed to users and able to accumulate votes or comments.

A robustness version of the BEVoCI interaction model is also estimated with post age included as a control.

### 5. Model comparison

The project compares the regression approach with the automatic machine-learning models and evaluates whether adding statistically significant metacognitive cues improves predictive performance.

Where applicable, paired bootstrap analyses are used to assess whether changes in model performance are larger than expected from test-set sampling variability.

## R requirements

The R analysis requires:

```r
install.packages(c("tidyverse", "lmtest", "sandwich"))
```

The main R packages are:

- `tidyverse`
- `lmtest`
- `sandwich`

## Python requirements

The machine-learning analyses use standard Python data-science packages, including:

- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- matplotlib

Install missing packages with `pip` as needed.

## Running the R analysis

Place `BEVOCI.R` and `stackexchange_enhanced_dataset.csv` in the same working directory, then run:

```bash
Rscript BEVOCI.R
```

On Windows, if R is installed but `Rscript` is not on the system PATH, run it using the full path, for example:

```powershell
& "C:\Program Files\R\R-4.3.2\bin\Rscript.exe" BEVOCI.R
```

The script prints regression results and robustness checks to the console and saves the six BEVoCI plots in the `plots` folder.

## Reproducibility notes

- The same dataset should be used for all analyses.
- Baseline and extended ML models use the same train/test split for valid comparison.
- Variables created after question publication are excluded when their inclusion would create target leakage.
- `Post_Age_Days` is treated as an exposure-time control in the robustness analysis, not as a metacognitive cue.

## Repository contents

Key files include:

- `stackexchange_enhanced_dataset.csv` - analysis dataset
- `BEVOCI.R` - regression, BEVoCI, plots, diagnostics, and robustness analyses
- Python analysis scripts - machine-learning training, model comparison, and statistical evaluation
- `plots/` - figures generated by the R analysis
- final project report - written summary of the methodology and results

## Authors

Maya Rosenstein, Elad Yosef, and Thomas Ehrlich  
Course: Metacognition, 00960694
