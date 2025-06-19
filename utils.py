
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


def draw_truth_and_prediction(y_pred, y_true, model_name):
    num_samples = 6
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].plot(y_true[i], label='Ground Truth', linewidth=2)
        axes[i].plot(y_pred[i], label='Prediction',
                     linestyle='--', linewidth=2)
        axes[i].set_title(f'Sample {i}')
        axes[i].set_xlabel('Day')
        axes[i].set_ylabel('Power')
        axes[i].legend()
        axes[i].grid(True)

    plt.suptitle('Prediction vs Ground Truth (90-Day Forecast)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'results/{model_name}-truth-prediction.png')
