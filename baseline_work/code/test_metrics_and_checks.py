import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score

def compute_per_image_iou_f1(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """
    Computes per-image pixel IoU and F1.
    Strict empty mask policy:
      - If both GT and Pred are completely empty: returns (1.0, 1.0)
      - If one is empty and the other is non-empty: returns (0.0, 0.0)
      - Otherwise: standard IoU = TP / (TP + FP + FN), F1 = 2*TP / (2*TP + FP + FN)
    """
    assert pred_bin.shape == gt_bin.shape, f"Shape mismatch: {pred_bin.shape} vs {gt_bin.shape}"
    gt_empty = (gt_bin.sum() == 0)
    pred_empty = (pred_bin.sum() == 0)
    
    if gt_empty and pred_empty:
        return 1.0, 1.0
    if gt_empty or pred_empty:
        return 0.0, 0.0
        
    tp = float((pred_bin * gt_bin).sum())
    fp = float((pred_bin * (1 - gt_bin)).sum())
    fn = float(((1 - pred_bin) * gt_bin).sum())
    
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    return float(iou), float(f1)

def classify_edit_tier(area_fraction: float) -> str:
    """
    Tier assignment convention:
      - '<2%': 0.0 <= area_fraction < 0.02
      - '2-10%': 0.02 <= area_fraction <= 0.10
      - '>10%': area_fraction > 0.10
    """
    if area_fraction < 0.02:
        return "<2%"
    elif area_fraction <= 0.10:
        return "2-10%"
    else:
        return ">10%"

def compute_classification_metrics(y_true, y_scores, threshold=0.5, score_direction="higher_is_fake"):
    """
    y_true: array-like of 0 (real) and 1 (fake/edited)
    y_scores: continuous scores
    score_direction: 'higher_is_fake' or 'lower_is_fake'
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    
    if score_direction == "lower_is_fake":
        # If lower score indicates fake, decision is y_scores <= threshold
        y_pred = (y_scores <= threshold).astype(int)
        auc_scores = -y_scores  # invert for ROC-AUC so higher indicates fake
    else:
        y_pred = (y_scores >= threshold).astype(int)
        auc_scores = y_scores
        
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    acc = (tp + tn) / max(1, (tp + fp + tn + fn))
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    prec = tp / max(1e-8, (tp + fp))
    rec = tp / max(1e-8, (tp + fn))
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))
    fpr = fp / max(1e-8, (fp + tn))
    
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, auc_scores))
    else:
        roc_auc = np.nan
        
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "FPR": fpr,
        "ROC_AUC": roc_auc,
        "Score_Direction": score_direction,
        "Threshold": threshold
    }

class TestBaselineMetricsAndIntegrity(unittest.TestCase):

    def test_iou_perfect_match(self):
        gt = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        pred = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        iou, f1 = compute_per_image_iou_f1(pred, gt)
        self.assertAlmostEqual(iou, 1.0, places=5)
        self.assertAlmostEqual(f1, 1.0, places=5)

    def test_iou_disjoint_masks(self):
        gt = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        pred = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        iou, f1 = compute_per_image_iou_f1(pred, gt)
        self.assertAlmostEqual(iou, 0.0, places=5)
        self.assertAlmostEqual(f1, 0.0, places=5)

    def test_iou_known_2x2(self):
        # GT has 2 pixels (top row)
        gt = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        # Pred has 2 pixels (left column)
        pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        # Intersection = 1 (top-left)
        # Union = 3 (top-left, top-right, bottom-left)
        # IoU = 1/3 ~ 0.333333
        # F1 = 2*1 / (2*1 + 1 + 1) = 2/4 = 0.5
        iou, f1 = compute_per_image_iou_f1(pred, gt)
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=5)
        self.assertAlmostEqual(f1, 0.5, places=5)

    def test_iou_empty_policies(self):
        empty_gt = np.zeros((4, 4), dtype=np.uint8)
        empty_pred = np.zeros((4, 4), dtype=np.uint8)
        non_empty = np.zeros((4, 4), dtype=np.uint8)
        non_empty[0, 0] = 1

        # Both empty -> IoU=1, F1=1
        iou, f1 = compute_per_image_iou_f1(empty_pred, empty_gt)
        self.assertAlmostEqual(iou, 1.0, places=5)
        self.assertAlmostEqual(f1, 1.0, places=5)

        # GT empty, Pred non-empty (false alarm) -> IoU=0, F1=0
        iou, f1 = compute_per_image_iou_f1(non_empty, empty_gt)
        self.assertAlmostEqual(iou, 0.0, places=5)
        self.assertAlmostEqual(f1, 0.0, places=5)

        # GT non-empty, Pred empty (miss) -> IoU=0, F1=0
        iou, f1 = compute_per_image_iou_f1(empty_pred, non_empty)
        self.assertAlmostEqual(iou, 0.0, places=5)
        self.assertAlmostEqual(f1, 0.0, places=5)

    def test_confusion_matrix_and_metrics(self):
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.8, 0.2, 0.9] # with threshold 0.5: preds are [0, 1, 0, 1]
        m = compute_classification_metrics(y_true, y_scores, threshold=0.5, score_direction="higher_is_fake")
        self.assertEqual(m["TP"], 1)
        self.assertEqual(m["FP"], 1)
        self.assertEqual(m["TN"], 1)
        self.assertEqual(m["FN"], 1)
        self.assertAlmostEqual(m["Accuracy"], 0.5, places=5)
        self.assertAlmostEqual(m["Balanced_Accuracy"], 0.5, places=5)
        self.assertAlmostEqual(m["Precision"], 0.5, places=5)
        self.assertAlmostEqual(m["Recall"], 0.5, places=5)
        self.assertAlmostEqual(m["F1"], 0.5, places=5)
        self.assertAlmostEqual(m["FPR"], 0.5, places=5)

    def test_edit_tier_bins(self):
        self.assertEqual(classify_edit_tier(0.015), "<2%")
        self.assertEqual(classify_edit_tier(0.019999), "<2%")
        self.assertEqual(classify_edit_tier(0.02), "2-10%")
        self.assertEqual(classify_edit_tier(0.05), "2-10%")
        self.assertEqual(classify_edit_tier(0.10), "2-10%")
        self.assertEqual(classify_edit_tier(0.10001), ">10%")
        self.assertEqual(classify_edit_tier(0.45), ">10%")

    def test_split_leakage_detector(self):
        train_hashes = {"hashA": "train_01", "hashB": "train_02"}
        test_hashes = {"hashB": "test_01", "hashC": "test_02"}
        overlap = set(train_hashes.keys()).intersection(set(test_hashes.keys()))
        self.assertEqual(len(overlap), 1)
        self.assertIn("hashB", overlap)

if __name__ == "__main__":
    unittest.main()

