import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


class Model:
    def __init__(
        self, learning_rate, momentum, train_loader, validation_loader, test_loader
    ):
        self.__model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.__train_loader = train_loader
        self.__valid_loader = validation_loader
        self.__test_loader = test_loader
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimiser = optim.SGD(
            self.__model.parameters(), lr=learning_rate, momentum=momentum
        )
        self.__model.fc = nn.Linear(self.__model.fc.in_features, 2)
        self.__scheduler = lr_scheduler.StepLR(
            self.__optimiser, step_size=7, gamma=0.1
        )  # Decay learning rate

    def train(self, epochs):
        accuracies = []
        losses = []
        val_accuracies = []
        val_losses = []
        total_step = len(self.__train_loader)

        # Setup for early stopping
        last_loss = np.Inf
        patience = 5
        trigger_times = 0

        for epoch in range(epochs):
            self.__model.train()
            running_loss = 0
            correct = 0
            total = 0
            for data, labels in self.__train_loader:
                self.__optimiser.zero_grad()

                outputs = self.__model(data)
                loss = self.__criterion(outputs, labels)
                loss.backward()
                self.__optimiser.step()

                running_loss += loss.item()
                correct += self.__guess(outputs, labels)
                total += labels.size(0)

            # Check for early stopping
            current_loss, val_accuracy = self.__validation()
            val_accuracies.append(val_accuracy)
            val_losses.append(current_loss)
            if current_loss > last_loss:
                trigger_times += 1
                if trigger_times == patience:
                    print("Early stopping")
                    accuracies.append(100 * correct / total)
                    losses.append(running_loss / total_step)
                    return accuracies, losses, val_accuracies, val_losses
            last_loss = current_loss

            self.__scheduler.step()

            accuracies.append(100 * correct / total)
            losses.append(running_loss / total_step)
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / total_step:.4f}"
            )
        return accuracies, losses, val_accuracies, val_losses

    def __validation(self):
        with torch.no_grad():
            self.__model.eval()
            correct = 0
            total = 0
            total_loss = 0
            for data, labels in self.__valid_loader:
                outputs = self.__model(data)
                loss = self.__criterion(outputs, labels)
                batch_loss = loss.item()

                correct += self.__guess(outputs, labels)
                total += labels.size(0)
                total_loss += batch_loss
            accuracy = 100 * correct / total
            print(f"Validation accuracy: {accuracy:.4f}")
            return total_loss / len(self.__valid_loader), accuracy

    def save(self):
        torch.save(self.__model, "model.pt")

    def load(self):
        self.__model = torch.load("model.pt")

    def evaluate(self):
        self.__model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for data, label in self.__test_loader:
                output = self.__model(data)
                _, pred = torch.max(output, dim=1)
                y_true.append(label.item())
                y_pred.append(pred.item())
            return np.array(y_true), np.array(y_pred)

    def __guess(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.sum(preds == labels).item()
