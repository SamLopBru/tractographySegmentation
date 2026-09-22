import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import sys
import os
import torchmetrics
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.helpers import _get_test_loader, _parse_test_args


# def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
#     f1 = f1_score(y_true, y_pred, average='macro')
#     acc = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='macro')
#     recall = recall_score(y_true, y_pred, average='macro')
#     return f1, acc, precision, recall

@torch.no_grad()
def test_loop(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, num_classes: int, output_path: str):
    model.eval()
    os.makedirs(output_path, exist_ok=True)

    test_metrics = torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro'),
            "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro'),
            "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro'),
            "aucroc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
        },
        prefix="test_"
    ).to(device)

    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, prefix="test_").to(device)

    for streamlines, lengths, labels in dataloader:

        streamlines = streamlines.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(streamlines, lengths)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        test_metrics["accuracy"].update(predictions, labels)
        test_metrics["f1"].update(predictions, labels)
        test_metrics["precision"].update(predictions, labels)
        test_metrics["recall"].update(predictions, labels)
        test_metrics["aucroc"].update(probs, labels)

    results = test_metrics.compute()

    results_serializable = {k: v.item() for k, v in results.items()}
    json_path = os.path.join(output_path, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=4)

    df = pd.DataFrame([results_serializable])
    csv_path = os.path.join(output_path, "test_metrics.csv")
    df.to_csv(csv_path, index=False)

    scalar_names = ["accuracy", "f1", "precision", "recall"]

    fig, axes = plt.subplots(nrows=1, ncols=len(scalar_names), figsize=(4 * len(scalar_names), 4))
    for ax, name in zip(axes, scalar_names):
        test_metrics[name].plot(val=results[f"test_{name}"], ax=ax)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "test_metrics.png"), bbox_inches="tight")
    plt.close(fig)

    fig_cm, ax_cm = confmat.plot()
    fig_cm.savefig(os.path.join(output_path, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig_cm)
    return results

def main():

    args = _parse_test_args()

    test_loader = _get_test_loader(test_dir=args.testLoader_path,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sampling_percentage_test=args.sampling_percentage_test,
                                min_streamlines=args.min_streamlines,
                                max_streamlines=args.max_streamlines,
                                seed=args.seed,
                                verbose=True)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.model_path)
    model.to(device)

    # Run the test loop
    results = test_loop(model, test_loader, device, num_classes=args.num_classes, output_path=args.output_path)

    for name, value in results.items():
        print(f"{name}: {value.item():.4f}")

if __name__ == "__main__":
    main()
