import pandas as pd
import numpy as np
from pathlib import Path
import random


# ==========================
# Configuration
# ==========================

INPUT_FILE = "branch_features.csv"

OUTPUT_FILE = "training_dataset_feature_corruption.csv"

CORRUPTION_RATIO = 0.2

RANDOM_SEED = 42


# ==========================
# Feature corruption
# ==========================

def corrupt_features(df):

    """
    Generate corrupted airway branch features.

    label:
        1 -> original branch
        0 -> corrupted branch
    """

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


    corrupted_df = df.copy()


    # default:
    # all branches are normal

    corrupted_df["label"] = 1


    feature_cols = [
        f"feature_{i}"
        for i in range(20)
    ]


    total_samples = len(df)


    corruption_num = int(
        total_samples * CORRUPTION_RATIO
    )


    corrupted_indices = random.sample(
        range(total_samples),
        corruption_num
    )


    for idx in corrupted_indices:

        corruption_type = random.choice(
            [
                "length",
                "volume",
                "children",
                "geometry"
            ]
        )


        if corruption_type == "length":

            # feature_4:
            # branch length

            corrupted_df.loc[
                idx,
                "feature_4"
            ] *= 0.2



        elif corruption_type == "volume":

            # feature_13:
            # branch volume

            corrupted_df.loc[
                idx,
                "feature_13"
            ] *= 0.2



        elif corruption_type == "children":

            # feature_11:
            # children number

            corrupted_df.loc[
                idx,
                "feature_11"
            ] = 0



        elif corruption_type == "geometry":

            # modify branch direction

            # feature 1-3:
            # delta xyz

            noise = np.random.normal(
                0,
                0.5,
                3
            )


            for i in range(3):

                corrupted_df.loc[
                    idx,
                    f"feature_{i+1}"
                ] += noise[i]


        corrupted_df.loc[
            idx,
            "label"
        ] = 0


    return corrupted_df



def main():


    print("Loading dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        "Original dataset:",
        df.shape
    )


    result = corrupt_features(
        df
    )


    print(
        "\nLabel distribution:"
    )

    print(
        result["label"].value_counts()
    )


    result.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )



if __name__ == "__main__":

    main()