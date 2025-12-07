import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test

def visualize_processed_image(idx, X_origin, X_processed):
    feature_vector = X_processed[idx]
    vector_len = feature_vector.shape[0]
    
    img_origin = X_origin[idx]
    
    plt.figure(figsize=(10, 4))
    
    # --- Function 1: Nomalization (784 features) ---
    if vector_len == 784:
        img_reconstructed = feature_vector.reshape(28, 28)
        
        plt.subplot(1, 2, 1); plt.imshow(img_origin, cmap='gray'); plt.title("Original Image (28x28)")
        plt.subplot(1, 2, 2); plt.imshow(img_reconstructed, cmap='gray'); plt.title("Nomalization (28x28)")

    # --- Function 2: Edge Nomalization (1568 features - With Sobel/Canny) ---
    elif vector_len == 1568:
        img_reconstructed = feature_vector.reshape(28, 28, 2)
        
        ch_edges    = img_reconstructed[:, :, 1]
        
        plt.subplot(1, 2, 1); plt.imshow(img_origin, cmap='gray'); plt.title("Original Image (28x28)")
        plt.subplot(1, 2, 2); plt.imshow(ch_edges, cmap='gray'); plt.title("Edge Nomalization (28x28)")

    # --- Function 3: Block Averaging (Size 196) ---
    elif vector_len == 196:
        img_reconstructed = feature_vector.reshape(14, 14)
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_origin, cmap='gray')
        plt.title("Original Image (28x28)")
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_reconstructed, cmap='gray')
        plt.title("Block Averaging (14x14)")
        
    else:
        print(f"Size {vector_len} is not from any designed functions.")
        
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.imshow(conf_matrix, cmap="Blues")
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)

    ax.set_xticks(np.arange(conf_matrix.shape[1]))
    ax.set_yticks(np.arange(conf_matrix.shape[0]))

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, conf_matrix[i, j],
                    ha="center", va="center",
                    color="black", fontsize=10)

    fig.colorbar(cax)
    plt.tight_layout()
    plt.show()

def plot_confusion_insights(conf_matrix):
    num_classes = conf_matrix.shape[0]
    plt.figure(figsize=(16, 24))

    for i in range(num_classes):
        plt.subplot(5, 2, i + 1)

        row = conf_matrix[i]

        sorted_idx = np.argsort(row)[::-1]

        top_idx = sorted_idx[:3]
        top_vals = row[top_idx]

        other_val = np.sum(row) - np.sum(top_vals)

        x_labels = [str(c) for c in top_idx] + ["Others"]
        y_vals = list(top_vals) + [other_val]

        bars = plt.bar(x_labels, y_vals)

        for bar, val in zip(bars, y_vals):
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(row)*0.02,
                     str(val),
                     ha="center",
                     fontsize=10)

        plt.title(f"Class {i} – Prediction Distribution", fontsize=14)
        plt.xlabel("Predicted Class")
        plt.ylabel("Count")
        plt.ylim(0, max(row) * 1.25)

    plt.tight_layout()
    plt.show()


def plot_class_metrics(metrics, title="Per-Class Metrics"):
    classes = np.arange(len(metrics["precision"]))

    plt.figure(figsize=(12, 6))
    bar_width = 0.25

    plt.bar(classes - bar_width, metrics["precision"],
            width=bar_width, label="Precision")
    plt.bar(classes, metrics["recall"],
            width=bar_width, label="Recall")
    plt.bar(classes + bar_width, metrics["f1"],
            width=bar_width, label="F1-Score")

    plt.xticks(classes)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=15)
    plt.ylim(0, 1)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
