import numpy as np
import torch
from dataloader import create_dataloaders, load_data
from evaluation import model_evaluate
from models.LSTM import LSTMModel
from train import model_train
from utils import draw_loss, draw_mse_and_mae, draw_truth_and_prediction
from utils import draw_loss, draw_truth_and_prediction

def test_lstm(n_in=90, n_out=90,  model_name='lstm', hidden_size=256, num_layers=1, num_epochs=100,  lr=0.001, batch_size=64):
    exp_name = f"{model_name}-{n_in}-{n_out}-{hidden_size}-{num_layers}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_X, test_y = load_data('data/test.csv', n_in, n_out)
    test_loader = create_dataloaders(test_X, test_y, 1)

    train_X, train_y = load_data('data/train.csv',  n_in, n_out)
    train_loader = create_dataloaders(train_X, train_y, batch_size)
    input_size = train_X.shape[1]
    output_size = train_y.shape[1]
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, eval_losses = model_train(model, criterion, optimizer,
                                            num_epochs, train_loader, test_loader, device, exp_name)
    draw_loss(train_losses, eval_losses, exp_name)
    model.eval()
    all_predictions, all_targets, mae, mse = model_evaluate(
        model, train_loader, device)

    draw_truth_and_prediction(all_predictions, all_targets, exp_name)
    return mse, mae


def hyperparameter_search():
    """
    Perform a hyperparameter search for the LSTM model.

    Results:
    hidden_size=128, num_layers=1 => MSE: 0.0045, MAE: 0.0522
    hidden_size=128, num_layers=2 => MSE: 0.0048, MAE: 0.0539
    hidden_size=256, num_layers=1 => MSE: 0.0035, MAE: 0.0459
    hidden_size=256, num_layers=2 => MSE: 0.0041, MAE: 0.0492
    hidden_size=512, num_layers=1 => MSE: 0.0027, MAE: 0.0403*
    hidden_size=512, num_layers=2 => MSE: 0.0041, MAE: 0.0495
    hidden_size=864, num_layers=1 => MSE: 0.0028, MAE: 0.0416
    hidden_size=864, num_layers=2 => MSE: 0.0033, MAE: 0.0440
    hidden_size=1024, num_layers=1 => MSE: 0.0028, MAE: 0.0410
    hidden_size=1024, num_layers=2 => MSE: 0.0038, MAE: 0.0470
    """
    res_map = {}
    for hs in [128, 256, 512, 864, 1024]:
        for nl in [1, 2]:
            print(f"Testing LSTM with hidden_size={hs}, num_layers={nl}")
            mse, mae = test_lstm(hidden_size=hs, num_layers=nl)
            res_map[(hs, nl)] = (mse, mae)
    print("Results:")
    for (hs, nl), (mse, mae) in res_map.items():
        print(f"hidden_size={hs}, num_layers={nl} => MSE: {mse:.4f}, MAE: {mae:.4f}")

def test_task_1_90():
    mse_list, mae_list = [], []
    for _ in range(5):
        mse, mae = test_lstm(n_in=90, n_out=90, model_name='lstm', hidden_size=512, num_layers=1, num_epochs=100, lr=0.001, batch_size=64)
        mse_list.append(mse)
        mae_list.append(mae)
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    print(f"Task 1 (90 days) results: MSE = {mse_mean:.4f} ± {mse_std:.4f}, MAE = {mae_mean:.4f} ± {mae_std:.4f}")

def test_task_1_365():
    mse_list, mae_list = [], []
    for _ in range(5):
        mse, mae = test_lstm(n_in=90, n_out=365, model_name='lstm', hidden_size=512, num_layers=1, num_epochs=100, lr=0.001, batch_size=64)
        mse_list.append(mse)
        mae_list.append(mae)
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    print(f"Task 1 (365 days) results: MSE = {mse_mean:.4f} ± {mse_std:.4f}, MAE = {mae_mean:.4f} ± {mae_std:.4f}")


if __name__ == "__main__":
    hyperparameter_search()
