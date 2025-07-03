import numpy as np
import torch
from dataloader import create_dataloaders, load_data
from evaluation import model_evaluate
from models.CTSAN import CTSANModel
from models.LSTM import LSTMModel
from models.Transformer import TransformerModel
from train import model_train
from utils import draw_loss, draw_truth_and_prediction


def test_lstm(n_in=90, n_out=90,  model_name='lstm', hidden_size=256, num_layers=1, num_epochs=100,  lr=0.001, batch_size=64):
    exp_name = f"{model_name}-{n_in}-{n_out}-{hidden_size}-{num_layers}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_X, train_y = load_data('data/train.csv',  n_in, n_out)
    train_loader = create_dataloaders(train_X, train_y, batch_size)
    test_X, test_y = load_data('data/test.csv', n_in, n_out)
    test_loader = create_dataloaders(test_X, test_y, batch_size)
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
        model, test_loader, device)

    draw_truth_and_prediction(all_predictions, all_targets, exp_name)
    return mse, mae

def test_transformer(n_in=90, n_out=90,  model_name='transformer', hidden_size=256, num_layers=1, d_model=128, nhead=4, num_epochs=100,  lr=0.001, batch_size=64):
    exp_name = f"{model_name}-{n_in}-{n_out}-{hidden_size}-{num_layers}-{d_model}-{nhead}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_X, train_y = load_data('data/train.csv',  n_in, n_out)
    train_loader = create_dataloaders(train_X, train_y, batch_size)
    test_X, test_y = load_data('data/test.csv', n_in, n_out)
    test_loader = create_dataloaders(test_X, test_y, batch_size)
    input_size = train_X.shape[1]
    output_size = train_y.shape[1]
    model = TransformerModel(input_size, d_model, nhead, num_layers, output_size)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, eval_losses = model_train(model, criterion, optimizer,
                                            num_epochs, train_loader, test_loader, device, exp_name)
    draw_loss(train_losses, eval_losses, exp_name)
    model.eval()
    all_predictions, all_targets, mae, mse = model_evaluate(
        model, test_loader, device)

    draw_truth_and_prediction(all_predictions, all_targets, exp_name)
    return mse, mae

def test_CTSAN(n_in=90, n_out=90,  model_name='CTSAN', hidden_size=256, num_layers=1, d_model=128, nhead=4, num_epochs=100,  lr=0.001, batch_size=64):
    exp_name = f"{model_name}-{n_in}-{n_out}-{hidden_size}-{num_layers}-{d_model}-{nhead}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_X, train_y = load_data('data/train.csv',  n_in, n_out, reshape=False)
    train_loader = create_dataloaders(train_X, train_y, batch_size)
    test_X, test_y = load_data('data/test.csv', n_in, n_out, reshape=False)
    test_loader = create_dataloaders(test_X, test_y, batch_size)
    input_size = train_X.shape[-1]
    output_size = train_y.shape[-1]
    model = CTSANModel(input_size, d_model, nhead, num_layers, output_size)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, eval_losses = model_train(model, criterion, optimizer,
                                            num_epochs, train_loader, test_loader, device, exp_name)
    draw_loss(train_losses, eval_losses, exp_name)
    model.eval()
    all_predictions, all_targets, mae, mse = model_evaluate(
        model, test_loader, device)

    draw_truth_and_prediction(all_predictions, all_targets, exp_name)
    return mse, mae

def hyperparameter_search_lstm():
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

def hyperparameter_search_transformer():
    """
    Perform a hyperparameter search for the Transformer model.

    best results:
    hidden_size=256, num_layers=1, d_model=128, nhead=2 => MSE: 0.0067, MAE: 0.0643
    hidden_size=256, num_layers=1, d_model=128, nhead=4 => MSE: 0.0064, MAE: 0.0626*
    hidden_size=256, num_layers=1, d_model=128, nhead=8 => MSE: 0.0067, MAE: 0.0639
    hidden_size=512, num_layers=1, d_model=128, nhead=2 => MSE: 0.0068, MAE: 0.0648
    hidden_size=512, num_layers=1, d_model=128, nhead=4 => MSE: 0.0067, MAE: 0.0640
    hidden_size=512, num_layers=1, d_model=128, nhead=8 => MSE: 0.0069, MAE: 0.0650
    hidden_size=1024, num_layers=1, d_model=128, nhead=2 => MSE: 0.0068, MAE: 0.0646
    hidden_size=1024, num_layers=1, d_model=128, nhead=4 => MSE: 0.0067, MAE: 0.0641
    hidden_size=1024, num_layers=1, d_model=128, nhead=8 => MSE: 0.0067, MAE: 0.0641
    """
    res_map = {}
    for hs in [256, 512, 1024]:
        for nl in [1, 2, 4]:
            for dm in [32, 64, 128]:
                for nh in [2, 4, 8]:
                    print(f"Testing Transformer with hidden_size={hs}, num_layers={nl}, d_model={dm}, nhead={nh}")
                    mse, mae = test_transformer(hidden_size=hs, num_layers=nl, d_model=dm,  nhead=nh)
                    res_map[(hs, nl, dm, nh)] = (mse, mae)
    print("Results:")
    for (hs, nl, dm, nh), (mse, mae) in res_map.items():
        print(f"hidden_size={hs}, num_layers={nl}, d_model={dm}, nhead={nh} => MSE: {mse:.4f}, MAE: {mae:.4f}")

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

def test_task_2_90():
    mse_list, mae_list = [], []
    for _ in range(1):
        mse, mae = test_transformer(n_in=90, n_out=90, model_name='transformer')
        mse_list.append(mse)
        mae_list.append(mae)
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    print(f"Task 2 (90 days) results: MSE = {mse_mean:.4f} ± {mse_std:.4f}, MAE = {mae_mean:.4f} ± {mae_std:.4f}")

def test_task_2_365():
    mse_list, mae_list = [], []
    for _ in range(5):
        mse, mae = test_transformer(n_in=90, n_out=365, model_name='transformer')
        mse_list.append(mse)
        mae_list.append(mae)
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    print(f"Task 2 (365 days) results: MSE = {mse_mean:.4f} ± {mse_std:.4f}, MAE = {mae_mean:.4f} ± {mae_std:.4f}")

def test_task_3_90():
    mse_list, mae_list = [], []
    for _ in range(5):
        mse, mae = test_CTSAN(n_in=90, n_out=90, model_name='CTSAN')
        mse_list.append(mse)
        mae_list.append(mae)
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    print(f"Task 3 (90 days) results: MSE = {mse_mean:.4f} ± {mse_std:.4f}, MAE = {mae_mean:.4f} ± {mae_std:.4f}")

def test_task_3_365():
    mse_list, mae_list = [], []
    for _ in range(5):
        mse, mae = test_CTSAN(n_in=90, n_out=365, model_name='CTSAN')
        mse_list.append(mse)
        mae_list.append(mae)
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    print(f"Task 3 (365 days) results: MSE = {mse_mean:.4f} ± {mse_std:.4f}, MAE = {mae_mean:.4f} ± {mae_std:.4f}")

if __name__ == "__main__":
    test_task_1_90()
    test_task_1_365()
    test_task_2_90()
    test_task_2_365()
    test_task_3_90()
    test_task_3_365()