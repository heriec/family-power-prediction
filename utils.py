
import matplotlib.pyplot as plt

def draw_loss(train_losses, eval_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(eval_losses, label='Eval Loss', color='orange')
    plt.title(f'Loss Curve for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{model_name}-loss.png')