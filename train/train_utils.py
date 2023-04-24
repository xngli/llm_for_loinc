import matplotlib.pyplot as plt


def plot_history(history, output_name):
    # plot and save training curve
    history_dict = history.history
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epochs = range(1, len(loss) + 1)

    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(output_name)
