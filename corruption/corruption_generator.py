import numpy as np
import pandas as pd
from pathlib import Path
import random


# ==============================
# Configuration
# ==============================

DATA_ROOT = Path("../sample_data/ATM22")

OUTPUT_DIR = Path("./corrupted_data")

CORRUPTION_RATIO = 0.1   # 10% branches corrupted


# ==============================
# Graph corruption
# ==============================

def load_graph(graph_file):

    graph = np.load(
        graph_file,
        allow_pickle=True
    )

    return graph



def remove_random_edges(graph, ratio=0.1):
    """
    Randomly remove edges from airway graph.

    Returns:
        corrupted graph
        damaged nodes
    """

    graph = graph.copy()


    # graph format:
    # edge shape usually:
    # (2, N)

    if graph.ndim != 2:
        raise ValueError(
            f"Unexpected graph shape: {graph.shape}"
        )


    if graph.shape[0] != 2:
        raise ValueError(
            "Expected edge list format (2,N)"
        )


    edge_num = graph.shape[1]


    remove_num = max(
        1,
        int(edge_num * ratio)
    )


    remove_indices = random.sample(
        range(edge_num),
        remove_num
    )


    damaged_edges = graph[:, remove_indices]


    damaged_nodes = set(
        damaged_edges.flatten().tolist()
    )


    mask = np.ones(
        edge_num,
        dtype=bool
    )

    mask[remove_indices] = False


    corrupted_graph = graph[:, mask]


    return corrupted_graph, damaged_nodes



# ==============================
# Process dataset
# ==============================

def process_case(case_dir):

    case_id = case_dir.name


    graph_file = (
        case_dir /
        f"{case_id}_airway_graph.npy"
    )


    if not graph_file.exists():

        return None


    graph = load_graph(graph_file)


    corrupted_graph, damaged_nodes = (
        remove_random_edges(
            graph,
            CORRUPTION_RATIO
        )
    )


    case_output = OUTPUT_DIR / case_id

    case_output.mkdir(
        parents=True,
        exist_ok=True
    )


    np.save(
        case_output /
        "corrupted_graph.npy",
        corrupted_graph
    )


    labels = []


    max_node = int(
        np.max(graph)
    )


    for node_id in range(max_node + 1):

        label = 0 if node_id in damaged_nodes else 1

        labels.append(
            {
                "case_id": case_id,
                "branch_id": node_id,
                "label": label
            }
        )


    return labels



def main():

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )


    all_labels = []


    cases = sorted(
        DATA_ROOT.iterdir()
    )


    for case in cases:

        if not case.is_dir():
            continue


        result = process_case(case)


        if result:

            all_labels.extend(result)

            print(
                f"Processed {case.name}"
            )


    df = pd.DataFrame(
        all_labels
    )


    df.to_csv(
        "branch_labels.csv",
        index=False
    )


    print("====================")
    print("Finished")
    print(df["label"].value_counts())
    print("====================")



if __name__ == "__main__":

    main()