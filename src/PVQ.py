import json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import numpy as np


PVQ_JSON_PATH = Path("../datasets/PVQ_result_beta=1.0.json")
PVQ_XLSX_PATH = Path("../datasets/PVQ_answers.xlsx")

PVQ_VALUES = {
    "Universalism":   [3, 8, 19, 23, 29, 40],
    "Benevolence":    [12, 18, 27, 33],
    "Tradition":      [9, 20, 25, 38],
    "Conformity":     [7, 16, 28, 36],
    "Security":       [5, 14, 21, 31, 35],
    "Power":          [2, 17, 39],
    "Achievement":    [4, 13, 24, 32],
    "Hedonism":       [10, 26, 37],
    "Stimulation":    [6, 15, 30],
    "Self-direction": [1, 11, 22, 34],
}
VALUE_ORDER = list(PVQ_VALUES.keys())


def load_json_scores(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def to_int_key_dict(d):
    # JSON keys -> int
    return {int(k): int(v) for k, v in d.items()}

def compute_centered_value_scores(item_scores: dict):
    """
    item_scores: {1..40 -> 1..6}
    returns:
      overall_mean,
      mean_by_value (raw means),
      centered_by_value (mean - overall_mean)
    """
    overall_mean = sum(item_scores[i] for i in range(1, 41)) / 40.0
    mean_by_value = {}
    centered_by_value = {}
    for v, items in PVQ_VALUES.items():
        mv = sum(item_scores[i] for i in items) / float(len(items))
        mean_by_value[v] = mv
        centered_by_value[v] = mv - overall_mean
    return overall_mean, mean_by_value, centered_by_value

def read_gold_from_xlsx(user: str):
    """
    Reads C42:C51 (10 rows) from the user's sheet.
    Returns a dict {value_name: gold_value}
    """
    df = pd.read_excel(PVQ_XLSX_PATH, sheet_name=user, header=None, engine="openpyxl")

    gold_series = df.iloc[41:51, 2].reset_index(drop=True)

    if len(gold_series) != 10:
        raise ValueError(f"Gold slice for {user} did not yield 10 rows. Got {len(gold_series)}")

    # Convert to float
    gold_vals = []
    for x in gold_series.tolist():
        if pd.isna(x):
            raise ValueError(f"Gold label has NaN for {user} in C42:C51")
        gold_vals.append(float(x))

    return {VALUE_ORDER[i]: gold_vals[i] for i in range(10)}

def squared_errors(pred: dict, gold: dict):
    """
    pred/gold: {value_name: float}
    returns:
      se_by_value (squared error),
      mse (mean squared error over 10 values)
    """
    se = {}
    for v in VALUE_ORDER:
        se[v] = (float(pred[v]) - float(gold[v])) ** 2
    mse = sum(se.values()) / len(se)
    return se, mse

data = load_json_scores(PVQ_JSON_PATH)

rows = []
mse_rows = []
per_value_se_rows = []

for user, user_block in data.items():
    gold = read_gold_from_xlsx(user)

    model_items = to_int_key_dict(user_block["model"])
    ref_items   = to_int_key_dict(user_block["ref_model"])

    model_overall, model_mean, model_centered = compute_centered_value_scores(model_items)
    ref_overall,   ref_mean,   ref_centered   = compute_centered_value_scores(ref_items)

    for v in VALUE_ORDER:
        rows.append({
            "user": user,
            "value": v,
            "model_centered": model_centered[v],
            "ref_centered": ref_centered[v],
            "gold": gold[v],
        })

    # MSE
    model_se, model_mse = squared_errors(model_centered, gold)
    ref_se,   ref_mse   = squared_errors(ref_centered, gold)

    # Spearman
    gold_vec = np.array([gold[v] for v in VALUE_ORDER], dtype=float)
    model_vec = np.array([model_centered[v] for v in VALUE_ORDER], dtype=float)
    ref_vec   = np.array([ref_centered[v] for v in VALUE_ORDER], dtype=float)

    # spearmanr returns (rho, pvalue)
    model_rho, model_p = spearmanr(model_vec, gold_vec)
    ref_rho, ref_p     = spearmanr(ref_vec, gold_vec)

    mse_rows.append({
        "user": user,
        "model_MSE": model_mse,
        "ref_model_MSE": ref_mse,
        "model_overall_mean": model_overall,
        "ref_overall_mean": ref_overall,
        "model_spearman_rho": float(model_rho),
        "model_spearman_p": float(model_p),
        "ref_spearman_rho": float(ref_rho),
        "ref_spearman_p": float(ref_p),
    })

    per_value_se_rows.append({
        "user": user,
        **{f"model_SE_{v}": model_se[v] for v in VALUE_ORDER},
        **{f"ref_SE_{v}": ref_se[v] for v in VALUE_ORDER},
    })

df_scores = pd.DataFrame(rows)

df_wide_model = df_scores.pivot(index="user", columns="value", values="model_centered")
df_wide_ref   = df_scores.pivot(index="user", columns="value", values="ref_centered")
df_wide_gold  = df_scores.pivot(index="user", columns="value", values="gold")

df_wide_model = df_wide_model.add_prefix("model_")
df_wide_ref   = df_wide_ref.add_prefix("ref_")
df_wide_gold  = df_wide_gold.add_prefix("gold_")

df_table = pd.concat([df_wide_model, df_wide_ref, df_wide_gold], axis=1).reset_index()

df_mse = pd.DataFrame(mse_rows)
df_se_wide = pd.DataFrame(per_value_se_rows)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 200)

print("=== Centered value scores table (wide) ===")
print(df_table)

print("\n=== MSE summary ===")
print(df_mse)

print("\n=== Per-value Squared Errors (wide) ===")
print(df_se_wide)

df_table.to_csv("../datasets/PVQ_value_scores_centered_beta=1.0.csv", index=False)
df_mse.to_csv("../datasets/PVQ_mse_summary_beta=1.0.csv", index=False)
df_se_wide.to_csv("../datasets/PVQ_squared_errors_by_value_beta=1.0.csv", index=False)