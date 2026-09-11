import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



# =====================================================
# Configuration
# =====================================================

DATASET = "training_dataset_feature_corruption.csv"

RANDOM_SEED = 42

EPOCHS = 300

LR = 1e-3



# =====================================================
# Experiment Settings
# =====================================================


# Baseline:
# []

# Remove important features:
# [4]
# [13]
# [4,13,7]

REMOVE_FEATURES = []


# Top-K experiment

USE_TOP_K = False

TOP_FEATURES = {
    3:[
        4,
        13,
        7
    ],

    5:[
        4,
        13,
        7,
        5,
        6
    ],

    10:[
        4,
        13,
        7,
        5,
        6,
        11,
        9,
        10,
        15,
        8
    ]
}


TOP_K = 5



# =====================================================
# MLP
# =====================================================


class AirwayMLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                64
            ),

            nn.ReLU(),


            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),


            nn.Linear(
                32,
                1
            )

        )


    def forward(self,x):

        return self.network(x)



# =====================================================
# Dataset
# =====================================================


def load_dataset(feature_indices):


    df = pd.read_csv(
        DATASET
    )


    cases = df["case_id"].unique()


    train_val_cases, test_cases = train_test_split(
        cases,
        test_size=0.2,
        random_state=RANDOM_SEED
    )


    train_cases, val_cases = train_test_split(
        train_val_cases,
        test_size=0.125,
        random_state=RANDOM_SEED
    )



    train_df = df[
        df.case_id.isin(train_cases)
    ]


    val_df = df[
        df.case_id.isin(val_cases)
    ]


    test_df = df[
        df.case_id.isin(test_cases)
    ]



    feature_cols = [
        f"feature_{i}"
        for i in feature_indices
    ]



    scaler = StandardScaler()



    X_train = scaler.fit_transform(
        train_df[feature_cols]
    )


    X_val = scaler.transform(
        val_df[feature_cols]
    )


    X_test = scaler.transform(
        test_df[feature_cols]
    )



    y_train = train_df.label.values

    y_val = val_df.label.values

    y_test = test_df.label.values



    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )



# =====================================================
# Train One Experiment
# =====================================================


def run_experiment(feature_indices):


    print("\n==============================")

    print(
        "Using features:",
        feature_indices
    )

    print("==============================")



    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test

    ) = load_dataset(
        feature_indices
    )



    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )


    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )


    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    ).reshape(-1,1)



    model = AirwayMLP(
        len(feature_indices)
    )



    criterion = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )



    for epoch in range(EPOCHS):


        model.train()


        optimizer.zero_grad()


        output = model(
            X_train
        )


        loss = criterion(
            output,
            y_train
        )


        loss.backward()

        optimizer.step()



    model.eval()


    with torch.no_grad():

        probs = torch.sigmoid(
            model(
                X_test
            )
        ).numpy().flatten()



    preds = (
        probs > 0.5
    ).astype(int)



    auc = roc_auc_score(
        y_test,
        probs
    )


    f1 = f1_score(
        y_test,
        preds
    )


    acc = accuracy_score(
        y_test,
        preds
    )


    print(
        "Accuracy:",
        acc
    )

    print(
        "F1:",
        f1
    )

    print(
        "AUC:",
        auc
    )


    print(
        confusion_matrix(
            y_test,
            preds
        )
    )



    return auc



# =====================================================
# Main
# =====================================================


def main():



    all_features = list(
        range(20)
    )



    results = {}



    # ----------------------------
    # Baseline
    # ----------------------------

    results["Full-20"] = run_experiment(
        all_features
    )



    # ----------------------------
    # Ablation
    # ----------------------------


    ablations = [

        [4],

        [13],

        [4,13,7]

    ]


    for remove in ablations:


        features = [

            i

            for i in all_features

            if i not in remove

        ]


        results[
            "Remove_"+str(remove)
        ] = run_experiment(
            features
        )



    # ----------------------------
    # Top-K
    # ----------------------------


    for k,features in TOP_FEATURES.items():

        results[
            f"Top-{k}"
        ] = run_experiment(
            features
        )



    print("\n\n==============================")

    print("Summary")

    print("==============================")


    for k,v in results.items():

        print(
            k,
            ":",
            v
        )



if __name__ == "__main__":

    main()