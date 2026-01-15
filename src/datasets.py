import json
import pandas as pd
import numpy as np
from pathlib import Path


# Config

# need to add user1 later
users = ["user0", "user2", "user3", "user4"] 

questions_path = Path("../datasets/PRISM-questions.json")
answers_xlsx_path = Path("../datasets/PRISM-answers.xlsx")
out_dir = Path("../datasets")


# Load PRISM questions
with open(questions_path, "r", encoding="utf-8") as f:
    PRISM_Q = json.load(f)

print(f"Loaded {len(PRISM_Q)} PRISM questions from {questions_path}.")

num_questions = len(PRISM_Q)

# sanity check: required keys
required_keys = {"query", "completion_A", "completion_B"}
for i, item in enumerate(PRISM_Q[:3]):
    missing = required_keys - set(item.keys())
    if missing:
        raise KeyError(f"PRISM-questions.json item {i} missing keys: {missing}")


# Helper
def normalize_choice(x) -> str | None:
    """Normalize cell value to 'A'/'B' or None."""
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    # allow common variants
    if s in {"A", "1", "0", "LEFT"}:
        return "A"
    if s in {"B", "2", "RIGHT"}:
        return "B"
    return None

# Build per-user json

for user in users:
    # Read sheet: header=None means raw cells; column B is index 1
    df_user = pd.read_excel(answers_xlsx_path, sheet_name=user, header=None)

    # Convert to numpy and extract column B
    arr = df_user.to_numpy()
    if arr.shape[1] < 2:
        raise ValueError(f"Sheet '{user}' has only {arr.shape[1]} columns; need at least 2 (B column).")

    choices_raw = arr[:, 1]  # B column
    choices = [normalize_choice(x) for x in choices_raw]

    print(f"Loaded {len(df_user)} rows for {user} from sheet '{user}'.")

    # Determine how many rows we can safely align
    n = min(len(choices), num_questions)
    if len(choices) != num_questions:
        print(f"[WARN] {user}: answers={len(choices)} vs questions={num_questions}. Using first {n} aligned rows.")

    out = []
    skipped = 0

    for i in range(n):
        ch = choices[i]
        if ch not in {"A", "B"}:
            skipped += 1
            continue

        q = PRISM_Q[i]["query"]
        a = PRISM_Q[i]["completion_A"]
        b = PRISM_Q[i]["completion_B"]

        if ch == "A":
            chosen, reject = a, b
        else:
            chosen, reject = b, a

        out.append({
            "query": q,
            "chosen": chosen,
            "reject": reject
        })

    print(f"{user}: built {len(out)} preference pairs (skipped {skipped} invalid/blank choices).")

    # Save per-user JSON
    out_path = out_dir / f"{user}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")

print("Done.")
