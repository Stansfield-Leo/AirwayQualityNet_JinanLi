# AirwayQualityNet

## Branch-level Airway Structural Quality Assessment Based on AirMorph Features

---

## Overview

This repository contains the implementation of **AirwayQualityNet**, a downstream airway structural quality assessment framework built on top of AirMorph-generated branch-level airway representations.

The project investigates whether anatomical and structural features extracted from AirMorph can be used to evaluate airway branch integrity and estimate structural degradation severity.

Instead of modifying the original AirMorph segmentation framework, this work focuses on downstream analysis of AirMorph outputs.

The current framework contains:

1. AirMorph branch-level feature extraction.
2. Synthetic airway structural corruption generation.
3. Multi-task MLP-based airway quality assessment.

The current objective is to evaluate whether AirMorph branch representations contain sufficient information for identifying structural degradation and provide a foundation for future topology-aware airway reconstruction.

---

# Main Contributions

The current repository provides:

- Construction of branch-level airway feature datasets from AirMorph outputs.

- Development of synthetic corruption strategies for simulating airway structural degradation.

- Design of a multi-task MLP model for airway quality assessment.

- Prediction of continuous degradation severity and binary structural failure states.

- Establishment of a baseline framework for future graph-based airway topology repair.


---

# Processing Workflow


```
AirMorph Output

        |
        v

Branch-level Feature Extraction

        |
        v

Airway Branch Feature Dataset

        |
        v

Synthetic Structural Corruption

        |
        v

Severity-aware Quality Labels

        |
        v

Multi-task MLP Quality Assessment

        |
        v

Structural Integrity Prediction

        |
        v

Future Graph-based Airway Repair
```


---

# Dataset Construction

## AirMorph Branch Feature Representation


The current framework uses branch-level feature representations extracted from AirMorph airway outputs.

Each airway branch is represented as a 20-dimensional feature vector.

The dataset construction pipeline:

```
ATM22 Cases

    |
    v

AirMorph airway feature files

    |
    v

Branch-level feature extraction

    |
    v

branch_features.csv
```


The dataset construction script extracts:

- Case identifier.
- Branch identifier.
- 20 AirMorph branch-level features.


Example:

```
case_id
branch_id
feature_0
feature_1
...
feature_19
```


---

# Synthetic Airway Corruption Generation


To evaluate structural quality assessment, synthetic airway degradation is introduced.

The corruption module generates controlled airway structural damage.

Current corruption types:

## 1. Length Damage

Continuous degradation.

The branch length feature is gradually reduced.

Target:

```
length_damage
```

A failure label is assigned when:

```
damage >= 0.5
```


---

## 2. Volume Damage

Continuous degradation.

The branch volume feature is gradually reduced.

Target:

```
volume_damage
```

A failure label is assigned when:

```
damage >= 0.5
```


---

## 3. Branch Structural Damage

Discrete topology-related degradation.

The corruption modifies branch-related structural features.

Target:

```
branch_failure
```


---

## 4. Geometry Damage

Spatial feature perturbation.

The corruption modifies geometric characteristics.

Target:

```
geometry_failure
```


---

Generated dataset:

```
branch_features_quality_v2.csv
```


Current dataset statistics:

```
Cases:
40

Airway branches:
7016

Feature dimension:
20
```


Corruption distribution:

```
normal       3519

length        926

volume        879

geometry      868

branch        824
```


---

# Model Architecture


The current baseline model is a multi-task MLP.

Input:

```
20-dimensional AirMorph branch feature vector
```


Architecture:


```
AirMorph Features

        |
        v

Linear(20,64)

        |
       ReLU

        |
        v

Linear(64,32)

        |
       ReLU

        |
        +-----------------------+
        |                       |
        v                       v

Regression Heads        Classification Heads

Length Damage           Length Failure

Volume Damage           Volume Failure

                         Branch Failure

                         Geometry Failure
```


The model jointly learns:

### Regression Tasks

- Length degradation estimation.
- Volume degradation estimation.


### Classification Tasks

- Length failure prediction.
- Volume failure prediction.
- Branch structural failure prediction.
- Geometry failure prediction.


---

# Experimental Setup


## Dataset

```
Dataset:
ATM22

Cases:
40

Branches:
7016

Features:
20
```


## Training

Environment:

```
Python 3.10

PyTorch

NumPy

Pandas

Scikit-learn

SciPy
```


Example:

```bash
conda create -n airwayatlas python=3.10

conda activate airwayatlas

pip install torch numpy pandas scipy scikit-learn
```


---

# Usage


## Step 1

Build AirMorph Branch Feature Dataset


```bash
python build_dataset.py
```


Output:

```
branch_features.csv
```


---

## Step 2

Generate Structural Corruption Dataset


```bash
python severity_corruption_generator.py
```


Output:

```
branch_features_quality_v2.csv
```


---

## Step 3

Train AirwayQualityNet


```bash
python train_mlp_quality.py
```


Output:

```
best_airway_quality_mlp_v2.pth
```


---

# Experimental Results


The current experiment evaluates whether AirMorph branch features can predict simulated airway structural degradation.


## Regression Results


### Length Damage Prediction


```
MAE:

0.0381


R2:

0.841


Pearson:

0.920
```


---

### Volume Damage Prediction


```
MAE:

0.0722


R2:

0.556


Pearson:

0.749
```


---

# Classification Results


## Length Failure


ROC-AUC:

```
0.999
```


At threshold 0.3:

```
F1:

0.926


Precision:

0.943


Recall:

0.909
```


---

## Volume Failure


ROC-AUC:

```
0.951
```


At threshold 0.3:

```
F1:

0.678


Precision:

0.732


Recall:

0.631
```


---

## Branch Failure


ROC-AUC:

```
0.884
```


At threshold 0.5:

```
F1:

0.663


Precision:

0.886


Recall:

0.530
```


---

## Geometry Failure


ROC-AUC:

```
0.985
```


At threshold 0.5:

```
F1:

0.901


Precision:

0.958


Recall:

0.850
```


---

# Current Findings


The baseline experiment demonstrates that AirMorph branch-level features contain meaningful structural information.

The model can successfully estimate continuous degradation severity for branch length and volume changes.

The results also show that different structural degradation types have different levels of predictability:

- Length degradation is highly predictable.
- Geometry degradation can be detected reliably.
- Volume degradation remains more challenging.
- Branch topology degradation requires further graph-based modelling.


---

# Limitations


Current limitations include:


## Synthetic Corruption

The current degradation labels are generated through simulated corruption rather than expert annotation.

Therefore, the model evaluates structural consistency detection instead of clinical abnormality detection.


## Branch-level Representation

The current MLP treats airway branches independently.

However, airway anatomy is naturally represented as a graph structure.

Parent-child relationships and connectivity information are not explicitly modelled.


## No Automatic Repair

The current implementation focuses on quality assessment.

Automatic airway topology reconstruction has not yet been implemented.


---

# Future Work


Future directions include:


## Graph-based Airway Representation

Replace independent branch modelling with airway graph representation.


Potential methods:

- Graph Neural Networks.
- Graph Attention Networks.
- Transformer-based graph models.


---

## Topology-aware Structural Repair


Future framework:


```
Incomplete airway graph

        |

        v

Graph Neural Network

        |

        v

Missing branch prediction

        |

        v

Graph completion

        |

        v

Recovered airway structure
```


---

# Repository Structure


```
AirwayQualityNet/

|

|-- build_dataset.py

|-- severity_corruption_generator.py

|-- train_mlp_quality.py


|

|-- branch_features.csv

|-- branch_features_quality_v2.csv


|

|-- best_airway_quality_mlp_v2.pth


|

|-- results/

```


---

# Data Availability


Medical imaging datasets and original AirMorph outputs are not included.

Users should provide authorised airway datasets and generate AirMorph representations before running this framework.

No patient information is distributed.


---

# Citation


If this repository is used in academic work:


```
Li, J.

AirwayQualityNet:
Branch-level Airway Structural Quality Assessment Based on AirMorph Features.

MSc Final Year Project.

University College London.

2026.
```


---

# Acknowledgements


This project uses airway branch representations generated from AirMorph.

The authors acknowledge the AirMorph project for providing the upstream airway analysis framework.


AirMorph repository:

https://github.com/EndoluminalSurgicalVision-IMR/AirMorph


---

# License


This repository contains code developed for academic research purposes.

This repository does not include:

- AirMorph source code.
- Medical imaging datasets.
- Patient information.
- External pretrained models.


Users are responsible for obtaining appropriate permissions for all third-party resources.
