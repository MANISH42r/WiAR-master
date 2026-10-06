import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

def evaluate_model(model, test_dataset, device='cpu', batch_size=16):
    """
    Evaluates a trained model on test_dataset.
    Computes accuracy, precision, recall, f1, inference time, and confusion matrix.
    """
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []

    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    inference_time = time.time() - start_time
    time_per_sample_ms = (inference_time / len(test_dataset)) * 1000.0

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(16)))

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_macro': f1,
        'f1_weighted': f1_weighted,
        'inference_time_sec': inference_time,
        'inference_time_ms_per_sample': time_per_sample_ms,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }

    return metrics
