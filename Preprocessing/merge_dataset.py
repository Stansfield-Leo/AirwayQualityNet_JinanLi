import pandas as pd


FEATURE_FILE = "branch_features.csv"
LABEL_FILE = "branch_labels.csv"
OUTPUT_FILE = "training_dataset.csv"


def main():

    print("Loading feature dataset...")
    features = pd.read_csv(FEATURE_FILE)

    print("Loading labels...")
    labels = pd.read_csv(LABEL_FILE)


    print("Feature shape:")
    print(features.shape)

    print("Label shape:")
    print(labels.shape)


    # Merge by case and branch identity
    dataset = features.merge(
        labels,
        on=[
            "case_id",
            "branch_id"
        ],
        how="inner"
    )


    print("====================")
    print("Merged dataset")
    print(dataset.shape)
    print("====================")


    print("\nLabel distribution:")
    print(
        dataset["label"].value_counts()
    )


    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print(
        f"Saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()