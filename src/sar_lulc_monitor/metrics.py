import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class SegmentationMetrics:
    """Accumulate confusion matrix and compute segmentation metrics.

    Rows: ground truth
    Columns: predictions
    """

    def __init__(
            self,
            num_classes: int,
            ignore_index: int = 255,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

    def update(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
    ) -> None:
        """Update confusion matrix with predicted class IDs."""
        predictions = predictions.reshape(-1).long()
        targets = targets.reshape(-1).long()

        valid = (
                (targets != self.ignore_index)
                & (targets >= 0)
                & (targets < self.num_classes)
        )

        predictions = predictions[valid]
        targets = targets[valid]

        if predictions.numel() == 0:
            return

        if (
                (predictions < 0).any()
                or (predictions >= self.num_classes).any()
        ):
            raise ValueError("Predicted class ID out of range")

        indices = (
                targets * self.num_classes + predictions
        )

        counts = torch.bincount(
            indices,
            minlength=self.num_classes ** 2,
        )

        self.confusion_matrix += counts.reshape(
            self.num_classes,
            self.num_classes,
        ).cpu()

    def compute(self) -> dict:
        """Compute OA, mIoU, and per-class IoU/F1."""
        cm = self.confusion_matrix.numpy().astype(np.float64)

        tp = np.diag(cm)
        actual = cm.sum(axis=1)
        predicted = cm.sum(axis=0)

        fp = predicted - tp
        fn = actual - tp

        union = tp + fp + fn
        f1_denom = 2 * tp + fp + fn

        iou = np.divide(
            tp,
            union,
            out=np.full(self.num_classes, np.nan),
            where=union > 0,
        )

        f1 = np.divide(
            2 * tp,
            f1_denom,
            out=np.full(self.num_classes, np.nan),
            where=f1_denom > 0,
        )

        total = cm.sum()

        oa = (
            float(tp.sum() / total)
            if total > 0 else float("nan")
        )

        miou = (
            float(np.nanmean(iou))
            if np.isfinite(iou).any()
            else float("nan")
        )

        return {
            "oa": oa,
            "miou": miou,
            "per_class_iou": iou,
            "per_class_f1": f1,
            "confusion_matrix": cm.astype(np.int64),
        }


@torch.inference_mode()
def evaluate_segmentation(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        num_classes: int = 8,
        ignore_index: int = 255,
) -> dict:
    """Evaluate segmentation metrics over an entire dataset."""
    model.eval()

    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        metrics.update(predictions, masks)

    return metrics.compute()
