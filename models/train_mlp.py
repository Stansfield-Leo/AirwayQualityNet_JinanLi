import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


# ============================
# Configuration
# ============================

DATASET = "training_dataset_feature_corruption.csv"

MODEL_SAVE = "best_airway_mlp.pth"

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

EPOCHS = 500

LR = 1e-3



# ============================
# MLP model
# ============================

class AirwayMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(20, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)

        )


    def forward(self, x):

        return self.network(x)



# ============================
# Dataset split
# ============================

def load_dataset():

    df = pd.read_csv(DATASET)


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
        df["case_id"].isin(train_cases)
    ]

    val_df = df[
        df["case_id"].isin(val_cases)
    ]

    test_df = df[
        df["case_id"].isin(test_cases)
    ]


    print("====================")
    print("Train label distribution")
    print(train_df["label"].value_counts())

    print("====================")
    print("Validation label distribution")
    print(val_df["label"].value_counts())

    print("====================")
    print("Test label distribution")
    print(test_df["label"].value_counts())


    feature_cols = [
        f"feature_{i}"
        for i in range(20)
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


    y_train = train_df["label"].values

    y_val = val_df["label"].values

    y_test = test_df["label"].values


    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )



# ============================
# Plot Loss Curve
# ============================

def save_loss_curve(
        train_losses,
        val_losses
):

    plt.figure(figsize=(8,5))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()


    plt.savefig(
        RESULT_DIR / "loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# ============================
# ROC Curve
# ============================

def save_roc_curve(
        y_test,
        probs
):

    fpr, tpr, _ = roc_curve(
        y_test,
        probs
    )


    auc = roc_auc_score(
        y_test,
        probs
    )


    plt.figure(figsize=(6,6))


    plt.plot(
        fpr,
        tpr,
        label=f"AUC={auc:.3f}"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )


    plt.legend()


    plt.savefig(
        RESULT_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# ============================
# Confusion Matrix
# ============================

def save_confusion_matrix(
        y_test,
        probs,
        threshold=0.3
):

    preds = (
        probs > threshold
    ).astype(int)


    cm = confusion_matrix(
        y_test,
        preds
    )


    plt.figure(
        figsize=(5,5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )


    plt.xlabel(
        "Prediction"
    )

    plt.ylabel(
        "Ground Truth"
    )

    plt.title(
        f"Confusion Matrix Threshold={threshold}"
    )


    plt.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# ============================
# Feature Importance
# ============================

def feature_importance(
        model,
        X_test,
        y_test,
        baseline_auc
):

    feature_names = [
        f"feature_{i}"
        for i in range(20)
    ]


    importance = []


    for i in range(20):

        X_perm = X_test.copy()


        np.random.shuffle(
            X_perm[:, i]
        )


        X_tensor = torch.tensor(
            X_perm,
            dtype=torch.float32
        )


        with torch.no_grad():

            prob = torch.sigmoid(
                model(X_tensor)
            ).numpy().flatten()


        auc = roc_auc_score(
            y_test,
            prob
        )


        importance.append(
            baseline_auc - auc
        )


    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance
        }
    )


    df = df.sort_values(
        by="importance",
        ascending=False
    )


    df.to_csv(
        RESULT_DIR / "feature_importance.csv",
        index=False
    )


    print("\nFeature Importance")

    print(df)



# ============================
# Training
# ============================

def train():


    torch.manual_seed(
        RANDOM_SEED
    )


    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test

    ) = load_dataset()



    num_normal = np.sum(
        y_train == 1
    )

    num_corrupted = np.sum(
        y_train == 0
    )


    weight_corrupted = (
        num_normal /
        num_corrupted
    )


    print("====================")

    print(
        "Corrupted class weight:",
        weight_corrupted
    )

    print("====================")



    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )


    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )


    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )



    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    ).reshape(-1,1)


    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32
    ).reshape(-1,1)



    sample_weights = torch.ones_like(
        y_train_tensor
    )


    sample_weights[
        y_train_tensor == 0
    ] = weight_corrupted



    model = AirwayMLP()



    criterion = nn.BCEWithLogitsLoss(
        weight=sample_weights
    )


    val_criterion = nn.BCEWithLogitsLoss()



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )


    best_val_loss = float("inf")


    train_losses = []

    val_losses = []



    for epoch in range(EPOCHS):


        model.train()


        optimizer.zero_grad()


        output = model(
            X_train_tensor
        )


        loss = criterion(
            output,
            y_train_tensor
        )


        loss.backward()


        optimizer.step()



        train_losses.append(
            loss.item()
        )



        model.eval()


        with torch.no_grad():

            val_output = model(
                X_val_tensor
            )


            val_loss = val_criterion(
                val_output,
                y_val_tensor
            )


        val_losses.append(
            val_loss.item()
        )



        if val_loss.item() < best_val_loss:


            best_val_loss = val_loss.item()


            torch.save(
                model.state_dict(),
                MODEL_SAVE
            )



        if (epoch+1)%50==0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss={loss.item():.4f} | "
                f"Val Loss={val_loss.item():.4f}"
            )



    print(
        "\nBest validation loss:",
        best_val_loss
    )


    print(
        "Saved model:",
        MODEL_SAVE
    )



    save_loss_curve(
        train_losses,
        val_losses
    )



    model.load_state_dict(
        torch.load(
            MODEL_SAVE,
            weights_only=True
        )
    )

    model.eval()



    with torch.no_grad():

        probs = torch.sigmoid(
            model(
                X_test_tensor
            )
        ).numpy().flatten()



    print(
        "\n========== Test Evaluation =========="
    )


    for threshold in [0.3,0.5,0.7]:

        preds = (
            probs > threshold
        ).astype(int)


        print(
            f"\nThreshold={threshold}"
        )


        print(
            "Accuracy:",
            accuracy_score(
                y_test,
                preds
            )
        )

        print(
            "Precision:",
            precision_score(
                y_test,
                preds
            )
        )

        print(
            "Recall:",
            recall_score(
                y_test,
                preds
            )
        )

        print(
            "F1:",
            f1_score(
                y_test,
                preds
            )
        )


        print(
            confusion_matrix(
                y_test,
                preds
            )
        )



    auc = roc_auc_score(
        y_test,
        probs
    )


    print(
        "\nROC-AUC:",
        auc
    )



    save_roc_curve(
        y_test,
        probs
    )


    save_confusion_matrix(
        y_test,
        probs
    )


    feature_importance(
        model,
        X_test,
        y_test,
        auc
    )



if __name__ == "__main__":

    train()