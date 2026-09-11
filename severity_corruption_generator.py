import pandas as pd
import numpy as np


# =====================================================
# Configuration
# =====================================================

INPUT = "branch_features.csv"

OUTPUT = "branch_features_quality_v2.csv"

RANDOM_SEED = 42

CORRUPTION_RATIO = 0.5

FAILURE_THRESHOLD = 0.5


np.random.seed(RANDOM_SEED)



# =====================================================
# Corruption Functions
# =====================================================


# ==========================
# Continuous damage
# ==========================


def corrupt_length(row, damage):

    original = row["feature_4"]

    row["feature_4"] = (
        original * (1-damage)
    )


    row["length_damage"] = damage


    if damage >= FAILURE_THRESHOLD:

        row["length_failure"] = 1


    return row




def corrupt_volume(row, damage):

    original = row["feature_13"]


    row["feature_13"] = (
        original * (1-damage)
    )


    row["volume_damage"] = damage


    if damage >= FAILURE_THRESHOLD:

        row["volume_failure"] = 1


    return row




# ==========================
# Discrete topology damage
# ==========================


def corrupt_branch(row):


    # branch-related features

    branch_features = [

        "feature_7",

        "feature_8",

        "feature_9",

        "feature_11",

        "feature_15"

    ]



    # randomly select damaged features

    num_features = np.random.randint(

        1,

        4

    )


    affected_features = np.random.choice(

        branch_features,

        size=num_features,

        replace=False

    )



    # apply random degradation

    for feature in affected_features:


        damage = np.random.uniform(

            0.2,

            0.8

        )


        row[feature] *= (

            1-damage

        )



    row["branch_failure"] = 1


    return row




# ==========================
# Geometry damage
# ==========================


def corrupt_geometry(row):


    geometry_features = [

        "feature_1",

        "feature_2",

        "feature_3",

        "feature_5",

        "feature_6",

        "feature_7"

    ]



    # correlated deformation strength

    noise_scale = np.random.uniform(

        0.1,

        0.5

    )



    for feature in geometry_features:


        row[feature] += np.random.normal(

            0,

            noise_scale

        )



    row["geometry_failure"] = 1


    return row




# =====================================================
# Main
# =====================================================


def main():


    df = pd.read_csv(

        INPUT

    )


    output_rows = []



    for _, row in df.iterrows():


        row = row.copy()



        # initialize labels


        row["length_damage"] = 0.0

        row["volume_damage"] = 0.0



        row["length_failure"] = 0

        row["volume_failure"] = 0


        row["branch_failure"] = 0

        row["geometry_failure"] = 0



        row["corruption_type"] = "normal"



        if np.random.rand() < CORRUPTION_RATIO:



            corruption_type = np.random.choice(

                [

                    "length",

                    "volume",

                    "branch",

                    "geometry"

                ]

            )



            if corruption_type == "length":


                damage = np.random.uniform(

                    0.1,

                    0.9

                )


                row = corrupt_length(

                    row,

                    damage

                )



            elif corruption_type == "volume":


                damage = np.random.uniform(

                    0.1,

                    0.9

                )


                row = corrupt_volume(

                    row,

                    damage

                )



            elif corruption_type == "branch":


                row = corrupt_branch(

                    row

                )



            elif corruption_type == "geometry":


                row = corrupt_geometry(

                    row

                )



            row["corruption_type"] = corruption_type



        output_rows.append(row)



    output = pd.DataFrame(

        output_rows

    )


    output.to_csv(

        OUTPUT,

        index=False

    )



    print(

        "Saved:",

        OUTPUT

    )



    print("\nCorruption distribution")


    print(

        output["corruption_type"].value_counts()

    )



    print("\nLabel statistics")


    print(

        output[

            [

                "length_damage",

                "volume_damage",

                "length_failure",

                "volume_failure",

                "branch_failure",

                "geometry_failure"

            ]

        ].describe()

    )



if __name__ == "__main__":

    main()