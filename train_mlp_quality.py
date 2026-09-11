import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from scipy.stats import pearsonr



# =====================================================
# Configuration
# =====================================================

DATASET = "branch_features_quality_v2.csv"

MODEL_SAVE = "best_airway_quality_mlp_v2.pth"


RANDOM_SEED = 42

EPOCHS = 500

LR = 1e-3



# =====================================================
# Model
# =====================================================

class AirwayQualityMLP(nn.Module):

    def __init__(self):

        super().__init__()


        # shared encoder

        self.encoder = nn.Sequential(

            nn.Linear(20,64),

            nn.ReLU(),

            nn.Linear(64,32),

            nn.ReLU()

        )



        # ==========================
        # Regression heads
        # ==========================


        self.length_damage_head = nn.Sequential(

            nn.Linear(32,1),

            nn.Sigmoid()

        )


        self.volume_damage_head = nn.Sequential(

            nn.Linear(32,1),

            nn.Sigmoid()

        )



        # ==========================
        # Classification heads
        # ==========================


        self.length_failure_head = nn.Linear(
            32,
            1
        )


        self.volume_failure_head = nn.Linear(
            32,
            1
        )


        self.branch_failure_head = nn.Linear(
            32,
            1
        )


        self.geometry_failure_head = nn.Linear(
            32,
            1
        )



    def forward(self,x):

        feature = self.encoder(x)


        length_damage = self.length_damage_head(
            feature
        )


        volume_damage = self.volume_damage_head(
            feature
        )


        length_failure = self.length_failure_head(
            feature
        )


        volume_failure = self.volume_failure_head(
            feature
        )


        branch_failure = self.branch_failure_head(
            feature
        )


        geometry_failure = self.geometry_failure_head(
            feature
        )


        return (

            length_damage,

            volume_damage,

            length_failure,

            volume_failure,

            branch_failure,

            geometry_failure

        )



# =====================================================
# Dataset
# =====================================================

def load_dataset():

    df = pd.read_csv(
        DATASET
    )


    cases = df.case_id.unique()



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



    target_cols = [

        "length_damage",

        "volume_damage",

        "length_failure",

        "volume_failure",

        "branch_failure",

        "geometry_failure"

    ]



    y_train = train_df[target_cols].values

    y_val = val_df[target_cols].values

    y_test = test_df[target_cols].values



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

    )


    y_val = torch.tensor(

        y_val,

        dtype=torch.float32

    )



    model = AirwayQualityMLP()



    mse_loss = nn.MSELoss()


    bce_loss = nn.BCEWithLogitsLoss()



    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LR

    )



    best_val_loss = float("inf")



    for epoch in range(EPOCHS):


        model.train()


        optimizer.zero_grad()



        (

            length_damage,

            volume_damage,

            length_failure,

            volume_failure,

            branch_failure,

            geometry_failure

        ) = model(X_train)



        # regression

        loss_length_damage = mse_loss(

            length_damage,

            y_train[:,0:1]

        )


        loss_volume_damage = mse_loss(

            volume_damage,

            y_train[:,1:2]

        )



        # classification

        loss_length_failure = bce_loss(

            length_failure,

            y_train[:,2:3]

        )


        loss_volume_failure = bce_loss(

            volume_failure,

            y_train[:,3:4]

        )


        loss_branch_failure = bce_loss(

            branch_failure,

            y_train[:,4:5]

        )


        loss_geometry_failure = bce_loss(

            geometry_failure,

            y_train[:,5:6]

        )



        loss = (

            0.25 * loss_length_damage

            +

            0.25 * loss_volume_damage

            +

            0.10 * loss_length_failure

            +

            0.10 * loss_volume_failure

            +

            0.15 * loss_branch_failure

            +

            0.15 * loss_geometry_failure

        )



        loss.backward()

        optimizer.step()



        # validation

        model.eval()


        with torch.no_grad():


            val_output = model(

                X_val

            )


            val_loss = (

                0.25*mse_loss(

                    val_output[0],

                    y_val[:,0:1]

                )

                +

                0.25*mse_loss(

                    val_output[1],

                    y_val[:,1:2]

                )

                +

                0.0*bce_loss(

                    val_output[2],

                    y_val[:,2:3]

                )

                +

                0.10*bce_loss(

                    val_output[3],

                    y_val[:,3:4]

                )

                +

                0.15*bce_loss(

                    val_output[4],

                    y_val[:,4:5]

                )

                +

                0.15*bce_loss(

                    val_output[5],

                    y_val[:,5:6]

                )

            )



        if val_loss.item() < best_val_loss:


            best_val_loss = val_loss.item()


            torch.save(

                model.state_dict(),

                MODEL_SAVE

            )



        if (epoch+1)%50==0:


            print(

                f"Epoch {epoch+1}/{EPOCHS} "

                f"Train Loss={loss.item():.5f} "

                f"Val Loss={val_loss.item():.5f}"

            )



    print(

        "\nBest validation loss:",

        best_val_loss

    )



    print(

        "Saved:",

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

        pred = model(

            X_test

        )



    # regression

    length_pred = pred[0].numpy().flatten()

    volume_pred = pred[1].numpy().flatten()



    print("\n========== Length Damage Regression ==========")


    print(

        "MAE:",

        mean_absolute_error(

            y_test[:,0],

            length_pred

        )

    )


    print(

        "R2:",

        r2_score(

            y_test[:,0],

            length_pred

        )

    )


    print(

        "Pearson:",

        pearsonr(

            y_test[:,0],

            length_pred

        )[0]

    )



    print("\n========== Volume Damage Regression ==========")


    print(

        "MAE:",

        mean_absolute_error(

            y_test[:,1],

            volume_pred

        )

    )


    print(

        "R2:",

        r2_score(

            y_test[:,1],

            volume_pred

        )

    )


    print(

        "Pearson:",

        pearsonr(

            y_test[:,1],

            volume_pred

        )[0]

    )



    # classification

    names = [

        "Length Failure",

        "Volume Failure",

        "Branch Failure",

        "Geometry Failure"

    ]



    for i,name in enumerate(names):


        probs = torch.sigmoid(

            pred[i+2]

        ).numpy().flatten()



        target = y_test[:,i+2]



        print(

            f"\n========== {name} =========="

        )


        print(

            "AUC:",

            roc_auc_score(

                target,

                probs

            )

        )



        for threshold in [

            0.3,

            0.5,

            0.7

        ]:


            pred_label = (

                probs > threshold

            ).astype(int)



            print(

                f"\nThreshold={threshold}"

            )


            print(

                "F1:",

                f1_score(

                    target,

                    pred_label,

                    zero_division=0

                )

            )


            print(

                "Precision:",

                precision_score(

                    target,

                    pred_label,

                    zero_division=0

                )

            )


            print(

                "Recall:",

                recall_score(

                    target,

                    pred_label,

                    zero_division=0

                )

            )


            print(

                confusion_matrix(

                    target,

                    pred_label

                )

            )



if __name__=="__main__":

    train()