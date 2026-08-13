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

# Define Features First
base_features = [
    "User_Reputation",
    "Gold_Badges",
    "Silver_Badges",
    "Bronze_Badges",
    "Has_Custom_Avatar",
    "Accept_Rate"
]

# Uncomment features here to include them
bevoci_cues_to_include = [
    # "Has_Image",
    # "Word_Count",
    # "Code_Block_Count",
    # "Link_Count",
    # "Title_Word_Count",
    # "LaTeX_Comment_Count",
    # "Tag_Count",
    # "Is_Question_Format"
]

features = base_features + bevoci_cues_to_include

# --- DYNAMIC CONFIGURATION ---
# Set the folder name based on whether metacognitive cues are being used
if len(bevoci_cues_to_include) > 0:
    MODEL_DIR = "saved_models_metacognitive"
else:
    MODEL_DIR = "saved_models_base"

os.makedirs(MODEL_DIR, exist_ok=True)  # Creates the folder if it doesn't exist

# Load the Data
df = pd.read_csv("stackexchange_enhanced_dataset.csv")

target_objective = "Objective_Comment_Count"
target_subjective = "Subjective_Score"

X = df[features]
y_obj = df[target_objective]
y_subj = df[target_subjective]

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

# Handle Missing Values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# Helper function to print text feature importances
def print_feature_importances(model, feature_names):
    print("\n--- Feature Importances ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")


# Helper function to plot and save feature importances
def plot_and_save_feature_importances(model, feature_names, target_title, plot_filename):
    importances = model.feature_importances_
    # Sort features in ascending order for a clean horizontal bar plot (most important at top)
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances: {target_title}", fontsize=14, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices], align='center', color='steelblue', edgecolor='black')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(MODEL_DIR, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()  # Close the plot to free up memory
    print(f"Saved feature importance plot to '{plot_path}'")


# --- CUSTOM CHECKPOINTING GRID SEARCH ---]
def robust_grid_search(estimator, param_grid, X_train, y_train, model_prefix):
    best_score = -np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        # Create a unique, safe filename based on this exact combination of parameters
        param_str = str(params)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()

        model_filename = os.path.join(MODEL_DIR, f"{model_prefix}_{param_hash}.pkl")
        score_filename = os.path.join(MODEL_DIR, f"{model_prefix}_{param_hash}_score.pkl")

        if os.path.exists(model_filename) and os.path.exists(score_filename):
            # Load cached model
            score = joblib.load(score_filename)
            model = joblib.load(model_filename)
        else:
            # Train model from scratch
            model = clone(estimator)
            model.set_params(**params)

            # 5-fold cross-validation (neg_MSE score)
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


# Define the Evaluation Pipeline
def tune_and_evaluate(X_data, y_data, target_name, target_prefix):
    print(f"\n{'=' * 60}")
    print(f"PIPELINE FOR: {target_name}")
    print(f"{'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # --- Model 1: Random Forest ---
    print(f"\n--- Processing Random Forest for {target_name} ---")
    rf = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 3, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    best_rf, best_rf_params = robust_grid_search(rf, rf_param_grid, X_train, y_train, f"{target_prefix}_RF")
    rf_predictions = best_rf.predict(X_test)

    print(f"\nBest RF Parameters: {best_rf_params}")
    print(f"RF R-squared (R2): {r2_score(y_test, rf_predictions):.4f}")
    print(f"RF Mean Absolute Error (MAE): {mean_absolute_error(y_test, rf_predictions):.4f}")
    print(f"RF Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, rf_predictions)):.4f}")

    # --- Model 2: XGBoost ---
    print(f"\n--- Processing XGBoost for {target_name} ---")
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    xgb_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 5, 7]
    }

    best_xgb, best_xgb_params = robust_grid_search(xgb, xgb_param_grid, X_train, y_train, f"{target_prefix}_XGB")
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
    plot_and_save_feature_importances(winner, X_data.columns, f"{target_name} ({winner_name})", plot_filename)

    # Save the absolute best model object
    final_winner_path = os.path.join(MODEL_DIR, f"WINNER_{target_prefix}.pkl")
    joblib.dump(winner, final_winner_path)
    print(f"Saved the overall winning model to '{final_winner_path}'")

    return winner


# 6. Execute Pipeline
best_model_objective = tune_and_evaluate(X_imputed, y_obj, "OBJECTIVE (Comment Count)", "Obj")
best_model_subjective = tune_and_evaluate(X_imputed, y_subj, "SUBJECTIVE (Upvote Score)", "Subj")