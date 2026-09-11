import os
from pathlib import Path
import numpy as np
import pandas as pd


# AirMorph dataset path
DATA_ROOT = Path("../sample_data/ATM22")

# Output
OUTPUT_FILE = Path("branch_features.csv")


def build_dataset():

    records = []

    case_dirs = sorted(DATA_ROOT.iterdir())

    for case_dir in case_dirs:

        if not case_dir.is_dir():
            continue

        case_id = case_dir.name

        feature_file = case_dir / f"{case_id}_airway_feature_cls.npy"

        if not feature_file.exists():
            print(f"Skip {case_id}: feature missing")
            continue

        print(f"Loading {case_id}")

        features = np.load(
            feature_file,
            allow_pickle=True
        )


        # features:
        # shape = (number_of_branches, 20)

        for branch_id, feature in enumerate(features):

            row = {
                "case_id": case_id,
                "branch_id": branch_id
            }

            # add 20 AirMorph features
            for i, value in enumerate(feature):
                row[f"feature_{i}"] = float(value)

            records.append(row)


    df = pd.DataFrame(records)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("====================")
    print("Dataset created")
    print(df.shape)
    print("====================")


if __name__ == "__main__":
    build_dataset()