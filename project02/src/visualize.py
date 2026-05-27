import matplotlib.pyplot as plt
from pathlib import Path

# =====================================
# SAVE TRAINING GRAPHS
# =====================================

def plot_graphs(history):

    BASE_DIR = Path(__file__).resolve().parent.parent

    GRAPH_DIR = BASE_DIR / "graphs"

    GRAPH_DIR.mkdir(exist_ok=True)

    # =====================================
    # ACCURACY GRAPH
    # =====================================

    plt.figure(figsize=(10, 5))

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend([
        "Train Accuracy",
        "Validation Accuracy"
    ])

    plt.grid(True)

    accuracy_path = GRAPH_DIR / "accuracy_graph.png"

    plt.savefig(accuracy_path)

    plt.close()

    # =====================================
    # LOSS GRAPH
    # =====================================

    plt.figure(figsize=(10, 5))

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend([
        "Train Loss",
        "Validation Loss"
    ])

    plt.grid(True)

    loss_path = GRAPH_DIR / "loss_graph.png"

    plt.savefig(loss_path)

    plt.close()

    print("\nGraphs saved successfully!")