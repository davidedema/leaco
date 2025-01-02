import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
from model import NeuralNet

DATASET_NAME = "/home/student/shared/orc_project/double_pendulum/datasets/dataset_N15_2000.csv"
SAVE_PATH = "/home/student/shared/orc_project/double_pendulum/models/model.pt"

def model_train(model, X_train, y_train, X_val, y_val):
    # Loss function and optimizer
    loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 350  # Number of epochs to run
    batch_size = 8  # Size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Metrics tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Hold the best model
    best_acc = -np.inf
    best_weights = None

    for epoch in range(n_epochs):
        # Training loop
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # Take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                
                # Forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                epoch_loss += loss.item() * len(X_batch)
                correct += (y_pred.round() == y_batch).float().sum().item()
                total += len(y_batch)

                # Print progress
                bar.set_postfix(loss=float(loss), acc=float(correct / total))
        
        # Record training metrics
        train_loss = epoch_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation loop
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = loss_fn(y_pred_val, y_val).item()
            val_acc = (y_pred_val.round() == y_val).float().mean().item()
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    # Restore the best model
    model.load_state_dict(best_weights)

    # Plot metrics
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return best_acc


# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        y_pred = model(X_test).round()
    return y_pred

data = pd.read_csv(DATASET_NAME)

X = data.iloc[2:, :4]    # features   
y = data.iloc[2:, 4]      # target

# Encoding the target
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model
model = NeuralNet()
model_train(model, X_train, y_train, X_test, y_test)

# Evaluate the model on test set
y_pred = evaluate_model(model, X_test, y_test)

# Convert predictions and true labels to numpy arrays for compatibility with sklearn
y_pred_np = y_pred.numpy().astype(int).flatten()
y_test_np = y_test.numpy().astype(int).flatten()

# Compute the confusion matrix
cm = confusion_matrix(y_test_np, y_pred_np)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Save the model
torch.save(model.state_dict(), SAVE_PATH)
