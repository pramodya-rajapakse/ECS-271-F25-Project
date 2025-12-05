import os
import json
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = "/home/jobe/ECS271/ECS-271-F25-Project/evaluation_results.json"  # path to your saved results
OUTPUT_DIR = "figures"
COLORS = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']

def load_results(json_path):
    with open(json_path, "r") as f:
        results = json.load(f)
    return results

def add_legend(fig, model_names):
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=COLORS[i]) for i in range(len(model_names))],
        model_names,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(model_names),
        title="Model Architecture"
    )

def generate_plots_from_json(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_names = list(results.keys())
    num_models = len(model_names)
    num_classes = 5

    # Average Accuracy
    overall_accuracies = [results[m]["overall_accuracy"] for m in model_names]
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    bars = plt.bar(model_names, overall_accuracies, color=COLORS[:num_models])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    plt.ylabel("Accuracy")
    plt.title("Average Accuracy")
    add_legend(fig, model_names)

    # Manually adjust layout to leave space for legend
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(OUTPUT_DIR, "avg_accuracy.png"))

    # ------------------- Average Loss -------------------
    average_losses = [results[m]["avg_loss"] for m in model_names]
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    bars = plt.bar(model_names, average_losses, color=COLORS[:num_models])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    plt.ylabel("Loss")
    plt.title("Average Loss")
    add_legend(fig, model_names)

    # Manually adjust layout to leave space for legend
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(OUTPUT_DIR, "avg_loss.png"))


    # Per-Class Accuracy
    class_data = {i: [results[m]["class_accuracy"][str(i)] for m in model_names] for i in range(num_classes)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    bar_width = 0.25
    index = np.arange(1)

    for i in range(num_classes):
        ax = axes[i]
        accuracies = class_data[i]
        for j in range(num_models):
            ax.bar(index + j * bar_width, accuracies[j], bar_width, color=COLORS[j], label=model_names[j])
            yval = accuracies[j]
            ax.text(index[0] + j * bar_width, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(f"Class {i} Accuracy", fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xticks(index + bar_width * (num_models - 1) / 2)
        ax.set_xticklabels([""])

    if num_classes < len(axes):
        fig.delaxes(axes[num_classes])

    fig.suptitle("Per-Class Accuracy For Each Model (Grades 0 to 4)", fontsize=16, y=1.02)
    add_legend(fig, model_names)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, "class_accuracy_grid.png"))

    # Precision, Recall, F1 per class
    metrics = ["precision", "recall", "f1"]
    metric_data = {metric: {i: [results[m]["per_class_metrics"][metric][str(i)] for m in model_names] for i in range(num_classes)} for metric in metrics}
    fig, axes = plt.subplots(len(metrics), num_classes, figsize=(20, 12), sharey=True)
    index = np.arange(1)

    for row_idx, metric_name in enumerate(metrics):
        for col_idx in range(num_classes):
            ax = axes[row_idx, col_idx]
            scores = metric_data[metric_name][col_idx]
            for j in range(num_models):
                ax.bar(index + j * bar_width, scores[j], bar_width, color=COLORS[j])
                yval = scores[j]
                ax.text(index[0] + j * bar_width, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(index + bar_width * (num_models - 1) / 2)
            ax.set_xticklabels([""])
            if col_idx == 0:
                ax.set_ylabel(metric_name.capitalize(), fontsize=14, weight="bold")
            if row_idx == 0:
                ax.set_title(f"Class {col_idx}", fontsize=14)

    add_legend(fig, model_names)
    fig.suptitle("Per-Class Performance Comparison (Precision, Recall, F1)", fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, "class_metrics_grid.png"))

if __name__ == "__main__":
    results = load_results(JSON_PATH)
    generate_plots_from_json(results)
