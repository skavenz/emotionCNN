import torch
from cnn import CNN
from training import trainmodel
from evaluation import valmodel
from data_loader import load_data
import os
from metrics_display import plotgraphs

current_dir = os.getcwd()
model_path = os.path.join(current_dir, "trainedModel.pth")

def load_model(usePretrained, batch_size, epoch_count, base_lr, weight_decay):
    print("\nInitiating Train Phase")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    if usePretrained:
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    else:
        train_loader, test_loader = load_data(batch_size)
        model = CNN().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

        for epoch_index in range(epoch_count):
            print("Epoch", epoch_index + 1)
            train_acc, train_loss = trainmodel(model, train_loader, criterion, optimizer, device, train_acc, train_loss)
            val_acc, val_loss = valmodel(model, test_loader, criterion, device, val_acc, val_loss)
            print("Displaying Graph. Close to continue training.\n")
            plotgraphs(epoch_index, train_acc, train_loss, val_acc, val_loss)

        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model