import torch
import numpy as np


def model_evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        for inputs, y in test_loader:
            inputs, y = inputs.to(device), y.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        mae = np.mean(np.abs(all_predictions - all_targets))
        mse = np.mean((all_predictions - all_targets) ** 2)
    return all_predictions, all_targets, mae, mse
