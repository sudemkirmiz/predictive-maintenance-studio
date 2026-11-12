"""Stage 1 - Dataset exploration for the AI4I 2020 Predictive Maintenance dataset.

Run:  python explore_dataset.py
Reads data/ai4i2020.csv (read-only) and saves one chart to reports/figures/.
No model is trained here.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save the figure to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/ai4i2020.csv")
FIGURE_PATH = Path("reports/figures/target_distribution.png")

TARGET = "Machine failure"
ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_TYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # leakage: never use as features
FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    # The file starts with a UTF-8 BOM; "utf-8-sig" removes it from the first column name.
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    section("1. Column names")
    print(list(df.columns))

    # Make sure the column groups above match the real file.
    expected = set(ID_COLUMNS + FEATURES + [TARGET] + FAILURE_TYPE_COLUMNS)
    assert expected == set(df.columns), f"Column mismatch: {expected ^ set(df.columns)}"

    section("2. Shape (rows, columns)")
    print(df.shape)

    section("3. First five rows")
    print(df.head())

    section("4. Data types")
    print(df.dtypes)

    section("5. Missing values per column")
    print(df.isna().sum())
    print("Total missing:", int(df.isna().sum().sum()))

    section("6. Duplicate rows")
    print("Fully duplicated rows:", int(df.duplicated().sum()))
    print("Duplicated rows ignoring identifiers:", int(df.drop(columns=ID_COLUMNS).duplicated().sum()))
    print("Unique UDI values:", df["UDI"].nunique())
    print("Unique Product ID values:", df["Product ID"].nunique())

    section("7. Column roles")
    print("Identifier columns :", ID_COLUMNS)
    print("Feature columns    :", FEATURES)
    print("Target column      :", TARGET)
    print("Failure-type labels:", FAILURE_TYPE_COLUMNS, "(excluded - leakage)")

    section("8. Target distribution")
    counts = df[TARGET].value_counts().sort_index()
    proportions = df[TARGET].value_counts(normalize=True).sort_index()
    summary = pd.DataFrame({"count": counts, "proportion": proportions.round(4)})
    summary.index = summary.index.map({0: "0 = no failure", 1: "1 = failure"})
    print(summary)
    print(f"Imbalance ratio (no failure : failure) = {counts[0] / counts[1]:.1f} : 1")

    section("9. Numerical statistics (features)")
    print(df[FEATURES].describe().round(2).T)

    section("10. Categorical feature 'Type'")
    print(df["Type"].value_counts())
    print("\nFailure rate by Type:")
    print(df.groupby("Type")[TARGET].mean().round(4))

    section("11. Leakage check: failure-type labels vs. target")
    print("Failure-type counts:")
    print(df[FAILURE_TYPE_COLUMNS].sum())
    any_type = df[FAILURE_TYPE_COLUMNS].max(axis=1)
    print("\nCrosstab: any failure type (rows) vs. Machine failure (columns)")
    print(pd.crosstab(any_type, df[TARGET]))
    print("\nIs 'Product ID' first letter always equal to 'Type'?",
          bool((df["Product ID"].str[0] == df["Type"]).all()))

    # One simple chart: class counts of the target.
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["No failure (0)", "Failure (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for bar, n, p in zip(bars, counts.values, proportions.values):
        ax.annotate(f"{n} ({p:.1%})", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Target distribution: Machine failure")
    ax.set_ylabel("Number of records")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=120)
    print(f"\nChart saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
