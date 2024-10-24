import matplotlib.pyplot as plt

def plotgraphs(epoch, train_acc, train_loss, val_acc, val_loss):
    plt.close()
    plt.figure(figsize=(10,6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_acc, label="Train Accuracy", color="blue")
    plt.plot(range(1, epoch + 2), train_acc, "x", color = "blue")
    plt.plot(range(1, epoch + 2), val_acc, label="Validation Accuracy", color="orange")
    plt.plot(range(1, epoch + 2), val_acc, "x", color = "orange")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Per Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_loss, label="Train Loss", color="blue")
    plt.plot(range(1, epoch + 2), train_loss, "x", color = "blue")
    plt.plot(range(1, epoch + 2), val_loss, label="Validation Loss", color="orange")
    plt.plot(range(1, epoch + 2), val_loss, "x", color = "orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()