import os
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 3000

DATA_FILE = "stackexchange_enhanced_dataset.csv"
MODEL_DIR = "saved_models_base"
OBJ_MODEL_FILE = os.path.join(MODEL_DIR, "WINNER_Base_Obj.pkl")
SUBJ_MODEL_FILE = os.path.join(MODEL_DIR, "WINNER_Base_Subj.pkl")

BASE_FEATURES = [
    "User_Type",
    "Has_Custom_Avatar",
    "Accept_Rate",
    "Post_Age_Days",
    "Is_Question_Format",
]

REGRESSION_FEATURES = [
    "Has_Image",
    "User_Reputation",
    "Tag_Count",
    "Title_Word_Count",
    "Word_Count",
    "LaTeX_Comment_Count",
]

TARGET_OBJECTIVE = "Objective_Comment_Count"
TARGET_SUBJECTIVE = "Subjective_Score"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def empirical_two_sided_p(delta_samples):
    """
    Two-sided empirical bootstrap p-value with a +1 correction.
    If all 3000 bootstrap differences have the same sign, p = 2/3001 < .001.
    """
    delta_samples = np.asarray(delta_samples)
    p_left = (np.sum(delta_samples <= 0) + 1) / (len(delta_samples) + 1)
    p_right = (np.sum(delta_samples >= 0) + 1) / (len(delta_samples) + 1)
    return min(1.0, 2 * min(p_left, p_right))


def paired_bootstrap(y_true, pred_ml, pred_reg, n_bootstrap=3000, seed=42):
    """
    Paired bootstrap on the same test-set observations.
    The same resampled indices are applied to y_true, ML predictions,
    and regression predictions in every bootstrap repetition.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    delta_r2 = np.empty(n_bootstrap)
    delta_mae = np.empty(n_bootstrap)
    delta_rmse = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        y_b = y_true[idx]
        ml_b = pred_ml[idx]
        reg_b = pred_reg[idx]

        delta_r2[b] = r2_score(y_b, ml_b) - r2_score(y_b, reg_b)
        delta_mae[b] = mean_absolute_error(y_b, ml_b) - mean_absolute_error(y_b, reg_b)
        delta_rmse[b] = rmse(y_b, ml_b) - rmse(y_b, reg_b)

    return {
        "R2": delta_r2,
        "MAE": delta_mae,
        "RMSE": delta_rmse,
    }


def summarize_bootstrap(measure, y_true, pred_ml, pred_reg, samples):
    point = {
        "R2": r2_score(y_true, pred_ml) - r2_score(y_true, pred_reg),
        "MAE": mean_absolute_error(y_true, pred_ml) - mean_absolute_error(y_true, pred_reg),
        "RMSE": rmse(y_true, pred_ml) - rmse(y_true, pred_reg),
    }

    rows = []
    for metric in ["R2", "MAE", "RMSE"]:
        low, high = np.percentile(samples[metric], [2.5, 97.5])
        rows.append({
            "Measure": measure,
            "Metric": metric,
            "Delta_ML_minus_Regression": point[metric],
            "CI_2.5%": low,
            "CI_97.5%": high,
            "Empirical_p_two_sided": empirical_two_sided_p(samples[metric]),
        })
    return rows


def main():
    df = pd.read_csv(DATA_FILE)

    # ------------------------------------------------------------------
    # Reconstruct the baseline ML input exactly as in ml_models.py.
    # The saved XGBoost models expect these five baseline features.
    # ------------------------------------------------------------------
    X_ml = df[BASE_FEATURES].copy()

    for col in X_ml.columns:
        if X_ml[col].dtype == "bool" or str(X_ml[col].dtype) == "boolean":
            X_ml[col] = X_ml[col].astype(float)

    numeric_cols = X_ml.select_dtypes(include=["number"]).columns
    categorical_cols = X_ml.select_dtypes(exclude=["number"]).columns

    num_imputer = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(
        num_imputer.fit_transform(X_ml[numeric_cols]),
        columns=numeric_cols
    )

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat = pd.DataFrame(
            cat_imputer.fit_transform(X_ml[categorical_cols]),
            columns=categorical_cols
        )
        X_cat[categorical_cols] = X_cat[categorical_cols].astype("category")
        X_ml_ready = pd.concat([X_num, X_cat], axis=1)
    else:
        X_ml_ready = X_num

    X_ml_ready = X_ml_ready[BASE_FEATURES]

    # Same test observations used in Question 1.
    all_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_ml_test = X_ml_ready.iloc[test_idx]

    y_obj = df[TARGET_OBJECTIVE].to_numpy()
    y_subj = df[TARGET_SUBJECTIVE].to_numpy()

    y_obj_train, y_obj_test = y_obj[train_idx], y_obj[test_idx]
    y_subj_train, y_subj_test = y_subj[train_idx], y_subj[test_idx]

    # ------------------------------------------------------------------
    # Load the original saved baseline ML winners.
    # ------------------------------------------------------------------
    ml_obj_model = joblib.load(OBJ_MODEL_FILE)
    ml_subj_model = joblib.load(SUBJ_MODEL_FILE)

    pred_ml_obj = ml_obj_model.predict(X_ml_test)
    pred_ml_subj = ml_subj_model.predict(X_ml_test)

    # ------------------------------------------------------------------
    # Rebuild the Question 5 linear regressions.
    # Standardization is fit on the training set only.
    # ------------------------------------------------------------------
    X_reg = df[REGRESSION_FEATURES].copy()
    for col in X_reg.columns:
        if X_reg[col].dtype == "bool" or str(X_reg[col].dtype) == "boolean":
            X_reg[col] = X_reg[col].astype(float)

    scaler = StandardScaler()
    X_reg_train = scaler.fit_transform(X_reg.iloc[train_idx])
    X_reg_test = scaler.transform(X_reg.iloc[test_idx])

    reg_obj = LinearRegression().fit(X_reg_train, y_obj_train)
    reg_subj = LinearRegression().fit(X_reg_train, y_subj_train)

    pred_reg_obj = reg_obj.predict(X_reg_test)
    pred_reg_subj = reg_subj.predict(X_reg_test)

    # ------------------------------------------------------------------
    # Table 7 reproduction check.
    # ------------------------------------------------------------------
    print("\n=== TABLE 7 REPRODUCTION ===")
    print(
        f"Objective ML         R2={r2_score(y_obj_test, pred_ml_obj):.4f} "
        f"MAE={mean_absolute_error(y_obj_test, pred_ml_obj):.4f} "
        f"RMSE={rmse(y_obj_test, pred_ml_obj):.4f}"
    )
    print(
        f"Objective Regression R2={r2_score(y_obj_test, pred_reg_obj):.4f} "
        f"MAE={mean_absolute_error(y_obj_test, pred_reg_obj):.4f} "
        f"RMSE={rmse(y_obj_test, pred_reg_obj):.4f}"
    )
    print(
        f"Subjective ML         R2={r2_score(y_subj_test, pred_ml_subj):.4f} "
        f"MAE={mean_absolute_error(y_subj_test, pred_ml_subj):.4f} "
        f"RMSE={rmse(y_subj_test, pred_ml_subj):.4f}"
    )
    print(
        f"Subjective Regression R2={r2_score(y_subj_test, pred_reg_subj):.4f} "
        f"MAE={mean_absolute_error(y_subj_test, pred_reg_subj):.4f} "
        f"RMSE={rmse(y_subj_test, pred_reg_subj):.4f}"
    )

    # ------------------------------------------------------------------
    # Table 8 paired bootstrap.
    # Important: RandomState(42), not default_rng(42).
    # This reproduces the reported confidence intervals.
    # ------------------------------------------------------------------
    obj_boot = paired_bootstrap(
        y_obj_test, pred_ml_obj, pred_reg_obj,
        n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE
    )
    subj_boot = paired_bootstrap(
        y_subj_test, pred_ml_subj, pred_reg_subj,
        n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE
    )

    results = []
    results.extend(
        summarize_bootstrap(
            "Objective", y_obj_test, pred_ml_obj, pred_reg_obj, obj_boot
        )
    )
    results.extend(
        summarize_bootstrap(
            "Subjective", y_subj_test, pred_ml_subj, pred_reg_subj, subj_boot
        )
    )

    results_df = pd.DataFrame(results)

    print("\n=== TABLE 8 PAIRED BOOTSTRAP REPRODUCTION ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    results_df.to_csv("question5_bootstrap_results.csv", index=False)
    print("\nSaved: question5_bootstrap_results.csv")


if __name__ == "__main__":
    main()
