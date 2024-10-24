import torch

def valmodel(model, test_loader, criterion, device, val_acc, val_loss):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    for data in test_loader:
        image, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = model(image)
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct / len(labels)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_acc = (running_accuracy / len(test_loader)) * 100
    avg_loss = running_loss / len(test_loader)
    val_acc.append(avg_acc)
    val_loss.append(avg_loss)
    return val_acc, val_loss
