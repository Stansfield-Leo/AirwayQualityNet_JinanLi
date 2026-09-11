import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import pearsonr


# =====================================================
# Configuration
# =====================================================

DATASET = "branch_features_with_severity.csv"

MODEL_SAVE = "best_airway_severity_mlp.pth"

RANDOM_SEED = 42

EPOCHS = 500

LR = 1e-3



# =====================================================
# MLP Regression Model
# =====================================================

class AirwaySeverityMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                20,
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
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        return self.network(x)



# =====================================================
# Dataset
# =====================================================

def load_dataset():


    df = pd.read_csv(
        DATASET
    )


    print("====================")
    print("Severity distribution")
    print(
        df["severity"].describe()
    )
    print("====================")



    cases = df["case_id"].unique()



    train_cases, test_cases = train_test_split(
        cases,
        test_size=0.2,
        random_state=RANDOM_SEED
    )


    train_cases, val_cases = train_test_split(
        train_cases,
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



    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))



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



    y_train = train_df["severity"].values

    y_val = val_df["severity"].values

    y_test = test_df["severity"].values



    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )



# =====================================================
# Training
# =====================================================

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



    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )


    X_val = torch.tensor(
        X_val,
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


    y_val = torch.tensor(
        y_val,
        dtype=torch.float32
    ).reshape(-1,1)



    model = AirwaySeverityMLP()



    criterion = nn.MSELoss()



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )



    best_val_loss = float("inf")



    for epoch in range(EPOCHS):


        model.train()


        optimizer.zero_grad()


        pred = model(
            X_train
        )


        loss = criterion(
            pred,
            y_train
        )


        loss.backward()

        optimizer.step()



        model.eval()


        with torch.no_grad():

            val_pred = model(
                X_val
            )


            val_loss = criterion(
                val_pred,
                y_val
            )



        if val_loss.item() < best_val_loss:


            best_val_loss = val_loss.item()


            torch.save(
                model.state_dict(),
                MODEL_SAVE
            )



        if (epoch+1) % 50 == 0:


            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss={loss.item():.5f} | "
                f"Val Loss={val_loss.item():.5f}"
            )



    print("====================")

    print(
        "Best validation loss:",
        best_val_loss
    )

    print(
        "Saved model:",
        MODEL_SAVE
    )



    # =====================================================
    # Evaluation
    # =====================================================


    model.load_state_dict(
        torch.load(
            MODEL_SAVE,
            weights_only=True
        )
    )


    model.eval()


    with torch.no_grad():

        predictions = model(
            X_test
        ).numpy().flatten()



    y_test = y_test.flatten()



    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    pearson = pearsonr(
        y_test,
        predictions
    )[0]



    print("\n========== Regression Evaluation ==========")


    print(
        "MAE:",
        mae
    )


    print(
        "RMSE:",
        rmse
    )


    print(
        "R2:",
        r2
    )


    print(
        "Pearson:",
        pearson
    )



    result = pd.DataFrame({

        "ground_truth_severity": y_test,

        "predicted_severity": predictions

    })


    result.to_csv(
        "severity_prediction_results.csv",
        index=False
    )


    print(
        "Saved: severity_prediction_results.csv"
    )



if __name__ == "__main__":

    train()