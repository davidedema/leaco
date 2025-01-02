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
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 350   # number of epochs to run
    batch_size = 8  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf  
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch) 
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
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
