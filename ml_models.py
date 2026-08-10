import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# 1. Load the Data
df = pd.read_csv("stackexchange_enhanced_dataset.csv")

# 2. Define Features (Independent Variables) and Targets (Dependent Variables)
features = [
    "Has_Image", "Word_Count", "Code_Block_Count", "Link_Count",
    "Title_Word_Count", "LaTeX_Comment_Count", "Tag_Count",
    "Is_Question_Format", "User_Reputation", "Gold_Badges",
    "Silver_Badges", "Bronze_Badges", "Has_Custom_Avatar", "Accept_Rate"
]

target_objective = "Objective_Comment_Count"
target_subjective = "Subjective_Score"

X = df[features]
y_obj = df[target_objective]
y_subj = df[target_subjective]

# 3. Missing Values Report
print("\n--- MISSING VALUES REPORT ---")
missing_counts = X.isnull().sum()
missing_percentages = (missing_counts / len(X)) * 100
missing_data = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage (%)': missing_percentages
})

# Filter to only show columns that actually have missing data
missing_data = missing_data[missing_data['Missing Count'] > 0]

if not missing_data.empty:
    print(missing_data.round(2).to_string())
else:
    print("No missing values found!")

# 4. Handle Missing Values
# We impute missing values with the median so the models don't crash.
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# Helper function to print feature importances
def print_feature_importances(model, feature_names):
    print("\n--- Feature Importances ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")


# 5. Define the Tuning and Evaluation Function
def tune_and_evaluate(X_data, y_data, target_name, filename):
    print(f"\n{'=' * 50}")
    print(f"PIPELINE FOR: {target_name}")
    print(f"{'=' * 50}")

    # Check if model already exists on disk
    if os.path.exists(filename):
        print(f"Found saved model! Loading '{filename}' from disk...")
        best_model = joblib.load(filename)
        print_feature_importances(best_model, X_data.columns)
        return best_model

    print("No saved model found. Beginning tuning process...")
    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # --- Model 1: Random Forest Regressor ---
    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 3, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Note: n_jobs=1 bypasses the Windows multiprocessing bug
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
    rf_grid.fit(X_train, y_train)

    best_rf = rf_grid.best_estimator_
    rf_predictions = best_rf.predict(X_test)

    print(f"Best RF Parameters: {rf_grid.best_params_}")
    print(f"RF R-squared (R2): {r2_score(y_test, rf_predictions):.4f}")
    print(f"RF Mean Absolute Error (MAE): {mean_absolute_error(y_test, rf_predictions):.4f}")
    print(f"RF Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, rf_predictions)):.4f}")

    # --- Model 2: Gradient Boosting Regressor ---
    print("\n--- Training Gradient Boosting ---")
    gb = GradientBoostingRegressor(random_state=42)
    gb_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 5, 7]
    }

    # Note: n_jobs=1 bypasses the Windows multiprocessing bug
    gb_grid = GridSearchCV(gb, gb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
    gb_grid.fit(X_train, y_train)

    best_gb = gb_grid.best_estimator_
    gb_predictions = best_gb.predict(X_test)

    print(f"Best GB Parameters: {gb_grid.best_params_}")
    print(f"GB R-squared (R2): {r2_score(y_test, gb_predictions):.4f}")
    print(f"GB Mean Absolute Error (MAE): {mean_absolute_error(y_test, gb_predictions):.4f}")
    print(f"GB Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, gb_predictions)):.4f}")

    # Pick the winner based on R2 score
    if r2_score(y_test, rf_predictions) > r2_score(y_test, gb_predictions):
        print("\nWinner: Random Forest")
        winner = best_rf
    else:
        print("\nWinner: Gradient Boosting")
        winner = best_gb

    # Print feature importances for the winning model
    print_feature_importances(winner, X_data.columns)

    # Save the winning model to disk
    joblib.dump(winner, filename)
    print(f"\nSaved winning model to '{filename}'")

    return winner


# 6. Execute Pipeline for Both Targets
best_model_objective = tune_and_evaluate(X_imputed, y_obj, "OBJECTIVE (Comment Count)", "best_objective_model.pkl")
best_model_subjective = tune_and_evaluate(X_imputed, y_subj, "SUBJECTIVE (Upvote Score)", "best_subjective_model.pkl")