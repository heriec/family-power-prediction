import argparse
import torch
import torch.nn as nn

from dataloader import create_dataloaders, load_data
from models.LSTM import LSTMModel
from models.Transformer import TransformerModel


from train import model_train
from evaluation import model_evaluate
from utils import draw_loss, draw_mse_and_mae, draw_truth_and_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',   '--model',             type=str,
                        default='lstm',         help="Which model to use")
    parser.add_argument('-l',   '--load',              type=str,
                        default='',             help="Path to the model you want to load")
    parser.add_argument('-t',   '--train',             action="store_true",
                        default=False, help="Whether to train the model")
    parser.add_argument('--nin', type=int, default=90,
                        help="Number of input time steps")
    parser.add_argument('--nout', type=int, default=90,
                        help="Number of output time steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 512
    num_layers = 1
    d_model = 1
    nhead = 1
    num_epochs = 100
    lr = 0.001
    batch_size = 64
    n_in = args.nin
    n_out = args.nout
    model_name = f"{args.model}-{n_in}-{n_out}"
    test_X, test_y = load_data('data/test.csv', n_in, n_out)
    test_loader = create_dataloaders(test_X, test_y, 1)
    if args.train:
        train_X, train_y = load_data('data/train.csv',  n_in, n_out)
        train_loader = create_dataloaders(train_X, train_y, batch_size)
        input_size = train_X.shape[1]
        output_size = train_y.shape[1]
        if args.model == 'lstm':
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        elif args.model == 'Transformer':
            model = TransformerModel(
                input_size, d_model, nhead, num_layers, output_size)
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses, eval_losses = model_train(model, criterion, optimizer,
                                                num_epochs, train_loader, test_loader, device, model_name)
        draw_loss(train_losses, eval_losses, model_name)
    elif args.load != '':
        input_size = test_X.shape[1]
        output_size = test_y.shape[1]
        if args.model == 'lstm':
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        elif args.model == 'Transformer':
            model = TransformerModel(
                input_size, d_model, nhead, num_layers, output_size)
        model.load_state_dict(torch.load(args.load, weights_only=True))
        model.to(device)
        model.eval()
        all_predictions, all_targets, mae, mse = model_evaluate(model, test_loader, device)

        draw_truth_and_prediction(all_predictions, all_targets, model_name)
        draw_mse_and_mae(mse, mae, model_name)
