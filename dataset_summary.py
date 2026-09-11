import pandas as pd
import numpy as np


DATASET = "branch_features.csv"


def main():

    df = pd.read_csv(DATASET)

    print("====================")
    print("Dataset shape")
    print("====================")
    print(df.shape)


    print("\n====================")
    print("Cases")
    print("====================")

    print(
        "Number of cases:",
        df["case_id"].nunique()
    )


    print("\n====================")
    print("Branches per case")
    print("====================")

    branch_count = (
        df.groupby("case_id")
        .size()
    )

    print(branch_count.describe())


    print("\n====================")
    print("Missing values")
    print("====================")

    print(
        df.isnull().sum()
    )


    print("\n====================")
    print("Feature statistics")
    print("====================")

    feature_cols = [
        c for c in df.columns
        if c.startswith("feature_")
    ]

    print(
        df[feature_cols]
        .describe()
    )


    print("\n====================")
    print("Zero variance features")
    print("====================")

    for col in feature_cols:

        if df[col].std() == 0:
            print(
                "Constant feature:",
                col
            )


if __name__ == "__main__":
    main()