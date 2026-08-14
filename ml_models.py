import pandas as pd
import numpy as np
import os
import joblib
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Define Features
base_features = [
    "User_Type",
    "Has_Custom_Avatar",
    "Accept_Rate",
    "Post_Age_Days",
    "Is_Question_Format",
]

# Cues found significant in BEVOCI Model A (Actual Clarity / Comments)
objective_cues = [
    "Word_Count",
    "Tag_Count",
    "Has_Image",
]

# Cues found significant in BEVOCI Model B (Perceived Quality / Upvotes)
subjective_cues = [
    "Word_Count",
    "User_Reputation",
    "LaTeX_Comment_Count",
    "Tag_Count",
    "Has_Image",
]

# Combine all unique features for a single preprocessing pass
all_cues = list(set(objective_cues + subjective_cues))
all_features = list(set(base_features + all_cues))

# --- DYNAMIC CONFIGURATION ---
# Create both directories so the script can save base and metacognitive models simultaneously
DIR_BASE = "saved_models_base"
DIR_META = "saved_models_metacognitive"
os.makedirs(DIR_BASE, exist_ok=True)
os.makedirs(DIR_META, exist_ok=True)

# Load the Data
df = pd.read_csv("stackexchange_enhanced_dataset.csv")

target_objective = "Objective_Comment_Count"
target_subjective = "Subjective_Score"

# Isolate features and targets using the master feature list
X = df[all_features].copy()
y_obj = df[target_objective]
y_subj = df[target_subjective]

# Convert boolean columns to floats (0.0 / 1.0) so XGBoost does not crash on them
for col in X.columns:
    if X[col].dtype == 'bool' or X[col].dtype == 'boolean':
        X[col] = X[col].astype(float)

# Missing Values Report
print("\n--- MISSING VALUES REPORT ---")
missing_counts = X.isnull().sum()
missing_percentages = (missing_counts / len(X)) * 100
missing_data = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage (%)': missing_percentages
})

missing_data = missing_data[missing_data['Missing Count'] > 0]

if not missing_data.empty:
    print(missing_data.round(2).to_string())
else:
    print("No missing values found!")

# --- MISSING VALUES AND CATEGORICAL CASTING ---
print("\n--- PREPROCESSING DATA ---")
numeric_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(exclude=['number']).columns

# Impute Numeric Columns (Median Strategy)
num_imputer = SimpleImputer(strategy='median')
X_num = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

# Impute and Cast Categorical Columns
if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

    # Cast directly to pandas 'category' datatype (No encoding)
    X_cat[categorical_cols] = X_cat[categorical_cols].astype('category')

    X_imputed = pd.concat([X_num, X_cat], axis=1)
    print(f"Categorical features cast to native 'category' dtype: {list(categorical_cols)}")
else:
    X_imputed = X_num

print(f"Total features preprocessed: {X_imputed.shape[1]}")


# Helper function to print text feature importances
def print_feature_importances(model, feature_names):
    print("\n--- Feature Importances ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")


# Helper function to plot and save feature importances (Now accepts model_dir)
def plot_and_save_feature_importances(model, feature_names, target_title, plot_filename, model_dir):
    importances = model.feature_importances_
    # Sort features in ascending order for a clean horizontal bar plot (most important at top)
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances: {target_title}", fontsize=14, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices], align='center', color='steelblue', edgecolor='black')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(model_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()  # Close the plot to free up memory
    print(f"Saved feature importance plot to '{plot_path}'")


# --- CUSTOM CHECKPOINTING GRID SEARCH --- (Now accepts model_dir)
def robust_grid_search(estimator, param_grid, X_train, y_train, model_prefix, model_dir):
    best_score = -np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        # Include the exact column names in the hash to prevent loading old models when feature lists change
        param_str = str(params) + str(list(X_train.columns))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()

        model_filename = os.path.join(model_dir, f"{model_prefix}_{param_hash}.pkl")
        score_filename = os.path.join(model_dir, f"{model_prefix}_{param_hash}_score.pkl")

        if os.path.exists(model_filename) and os.path.exists(score_filename):
            # Load cached model
            score = joblib.load(score_filename)
            model = joblib.load(model_filename)
        else:
            # Train model from scratch
            model = clone(estimator)
            model.set_params(**params)

            # Cross-validation (neg_MSE score)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
            score = scores.mean()

            # Fit on full training split
            model.fit(X_train, y_train)

            # Save checkpoint
            joblib.dump(model, model_filename)
            joblib.dump(score, score_filename)
            print(f"Trained & Saved {model_prefix} | Params: {params} | neg_MSE: {score:.4f}")

        # Track overall winner
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params


# Define the Evaluation Pipeline (Now accepts model_dir)
def tune_and_evaluate(X_data, y_data, target_name, target_prefix, model_dir):
    print(f"\n{'=' * 60}")
    print(f"PIPELINE FOR: {target_name}")
    print(f"{'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # --- Processing Random Forest ---
    print(f"\n--- Processing Random Forest for {target_name} ---")

    # SAFEGUARD: RF cannot process 'category' types natively. We extract the category codes just for RF.
    X_train_rf = X_train.copy()
    X_test_rf = X_test.copy()
    cat_cols = X_train_rf.select_dtypes(include=['category']).columns
    for col in cat_cols:
        X_train_rf[col] = X_train_rf[col].cat.codes
        X_test_rf[col] = X_test_rf[col].cat.codes

    rf = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 3, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    best_rf, best_rf_params = robust_grid_search(rf, rf_param_grid, X_train_rf, y_train, f"{target_prefix}_RF", model_dir)
    rf_predictions = best_rf.predict(X_test_rf)

    print(f"\nBest RF Parameters: {best_rf_params}")
    print(f"RF R-squared (R2): {r2_score(y_test, rf_predictions):.4f}")
    print(f"RF Mean Absolute Error (MAE): {mean_absolute_error(y_test, rf_predictions):.4f}")
    print(f"RF Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, rf_predictions)):.4f}")

    # --- Processing XGBoost ---
    print(f"\n--- Processing XGBoost for {target_name} ---")

    # NATIVE CATEGORY SUPPORT: Pass the unencoded data directly, but configure the tree method.
    xgb = XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        enable_categorical=True,
        tree_method='hist'
    )
    xgb_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 5, 7]
    }

    # XGBoost uses the original, unencoded X_train and X_test
    best_xgb, best_xgb_params = robust_grid_search(xgb, xgb_param_grid, X_train, y_train, f"{target_prefix}_XGB", model_dir)
    xgb_predictions = best_xgb.predict(X_test)

    print(f"\nBest XGB Parameters: {best_xgb_params}")
    print(f"XGB R-squared (R2): {r2_score(y_test, xgb_predictions):.4f}")
    print(f"XGB Mean Absolute Error (MAE): {mean_absolute_error(y_test, xgb_predictions):.4f}")
    print(f"XGB Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, xgb_predictions)):.4f}")

    # --- Pick the Winner ---
    if r2_score(y_test, rf_predictions) > r2_score(y_test, xgb_predictions):
        winner_name = "Random Forest"
        winner = best_rf
    else:
        winner_name = "XGBoost"
        winner = best_xgb

    print(f"\nOVERALL WINNER: {winner_name}")

    # Print console importances
    print_feature_importances(winner, X_data.columns)

    # Plot & Save Feature Importances
    plot_filename = f"feature_importance_{target_prefix}.png"
    plot_and_save_feature_importances(winner, X_data.columns, f"{target_name} ({winner_name})", plot_filename, model_dir)

    # Save the absolute best model object
    final_winner_path = os.path.join(model_dir, f"WINNER_{target_prefix}.pkl")
    joblib.dump(winner, final_winner_path)
    print(f"Saved the overall winning model to '{final_winner_path}'")

    return winner


# --- DATA SPLITTING FOR THE DUAL PIPELINE ---
# Split the master preprocessed dataframe into the specific baseline and metacognitive feature sets
X_base = X_imputed[base_features]
X_obj_meta = X_imputed[base_features + objective_cues]
X_subj_meta = X_imputed[base_features + subjective_cues]

# --- PHASE 1: BASELINE MODELS (BEFORE BEVOCI) ---
print("\n" + "*" * 60)
print("PHASE 1: BASELINE MODELS (NO METACOGNITIVE CUES)")
print("*" * 60)
base_model_objective = tune_and_evaluate(X_base, y_obj, "OBJECTIVE BASELINE (Comment Count)", "Base_Obj", DIR_BASE)
base_model_subjective = tune_and_evaluate(X_base, y_subj, "SUBJECTIVE BASELINE (Upvote Score)", "Base_Subj", DIR_BASE)

# --- PHASE 2: METACOGNITIVE MODELS (AFTER BEVOCI) ---
print("\n" + "*" * 60)
print("PHASE 2: METACOGNITIVE MODELS (WITH CUES)")
print("*" * 60)
meta_model_objective = tune_and_evaluate(X_obj_meta, y_obj, "OBJECTIVE METACOGNITIVE (Comment Count)", "Meta_Obj", DIR_META)
meta_model_subjective = tune_and_evaluate(X_subj_meta, y_subj, "SUBJECTIVE METACOGNITIVE (Upvote Score)", "Meta_Subj", DIR_META)