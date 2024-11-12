import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 1st layer: input (from input dimension to 64 neurons) 64 is arbitrary!
        self.fc2 = nn.Linear(64, 32)  # number of other layers is arbitrary!
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)  # last layer is output: 1 prediction (churn, or not)
        self.relu = nn.ReLU()  # activation functions used
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # activation function of the 1st layer
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.output(x))
        return x


def preprocess(df: DataFrame):
    one_hot_country = pd.get_dummies(df.country, prefix='country')
    one_hot_gender = pd.get_dummies(df.gender, prefix='gender')
    df = df.drop(["country", "gender"], axis=1)
    df = pd.concat([df, one_hot_country, one_hot_gender], axis=1)
    y = df["churn"]
    X = df.drop("churn", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def pytorch_preprocess(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(
        1)  # technicality: add a dimension to y_train to match the output of the NN
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True)  # WE ACTUALLY FEED DATA TO THE NN 32 SAMPLES ("ROWS") AT ATIME
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return X_test_tensor, train_loader, test_loader


def train(num_epochs, model, train_loader, optimizer, criterion, test_loader):
    global inputs, best_loss, trigger_times
    for epoch in range(num_epochs):
        model.train()  # train with batch
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # reset gradients
            outputs = model(inputs)  # predict
            loss = criterion(outputs, labels)  # compute loss based on ground truth
            loss.backward()  # BACKPROPAGATION: compute gradients
            optimizer.step()  # Adam
            running_loss += loss.item()  # track loss

        epoch_loss = running_loss / len(train_loader)

        model.eval()  # NO TRAINING NOW: validate model on TEST data!
        val_loss = 0.0
        with torch.no_grad():  # we now evaluate the model built after the 1st training epoch, WE DON'T LEARN!
            for inputs, labels in test_loader:  # TEST DATA!!
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:  # track best model
            best_loss = val_loss
            best_model = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1

        if trigger_times >= patience:
            print('Early stopping!')
            model.load_state_dict(best_model)
            break


def test(model, X_test_tensor):
    model.eval()  # NO TRAINING
    y_pred_list = []
    with torch.no_grad():
        for inputs in X_test_tensor:
            y_pred = model(inputs)
            y_pred = torch.round(y_pred)  # Convert to binary (0 or 1)
            y_pred_list.append(y_pred.item())
    y_pred_list = [int(pred) for pred in y_pred_list]
    return y_pred_list


if __name__ == "__main__":

    df = pd.read_csv("../../data/bank_customer_churn.csv", index_col="customer_id")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess(df)

    # Preprocess specific to PyTorch ("tensors")
    X_test_tensor, train_loader, test_loader = pytorch_preprocess(X_train, X_test, y_train, y_test)

    input_dim = X_train.shape[1]
    model = ANNModel(input_dim)  # prediction model is NN with defined architecture

    criterion = nn.BCELoss()  # Binary Cross Entropy loss (whatever, it is a measure of prediction error)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # gradient descent optimisation: Adam (check slides!)

    best_loss = np.inf
    patience = 10  # early stopping: if after 10 epochs the loss doesn't change substantially, stop training
    trigger_times = 0
    num_epochs = 100  # how many times to feed the NN "batch_size" data points for training

    train(num_epochs, model, train_loader, optimizer, criterion, test_loader)

    y_pred_list = test(model, X_test_tensor)

    accuracy = accuracy_score(y_test, y_pred_list) * 100
    print(f'Accuracy: {accuracy:.4f}')

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_list))
