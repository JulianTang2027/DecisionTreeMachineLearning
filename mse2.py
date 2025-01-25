import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# post preliminary NN 
# added more hidden layers
class L1RegularizedNN(torch.nn.Module):
    def __init__(self, input_features, k):
        super(L1RegularizedNN, self).__init__()
        # create the pass through
        self.hidden_layers = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            # make k in each hidden layer dynamic 
            nn.Linear(in_features=k, out_features=k//2),
            nn.BatchNorm1d(k//2),
            nn.ReLU()
        )
        # doing MSE so only 1 output layer
        self.output_layer = nn.Linear(in_features=k//2, out_features=1)
        
        # create weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # randomized and minimal to prevent one weight being overly strong from the start
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# new train model incorporates batch_size, l1 regulizer
def train_model(model, X_train, y_train, epochs=300, batch_size=1, 
                learning_rate=0.001, l1_lambda=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # normalize training data to cull outlier values
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    
    # using batches to calculate loss
    # Anything too big will generalize too much
    # anything too small will result in noise and be far too intensive
    n_batches = len(X_train) // batch_size
    
    # track the loss
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # randomize training data order
        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train_norm = y_train_norm[indices]
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train_norm[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            
            mse_loss = criterion(outputs, batch_y)
            
            
            l1_penalty = 0
            for param in model.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            
            
            loss = mse_loss + l1_lambda * l1_penalty
            
            # perform gradient descent
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / n_batches
        train_losses.append(avg_train_loss)
        
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'L1 Penalty: {l1_penalty.item():.4f}\n')
    
    return train_losses, y_mean, y_std

def plot_predictions_vs_actual(predictions, actual_values, title="Prediction vs Actual"):
    plt.figure(figsize=(10, 8))
    
    plt.scatter(actual_values, predictions, alpha=0.5, color='blue', label='Predictions')
    
    min_val = min(min(actual_values), min(predictions))
    max_val = max(max(actual_values), max(predictions))
    
    # just make the "true" a line with slope of 1 so anything on the line represents accurate labels
    diagonal = np.linspace(min_val, max_val, 100)
    
    plt.plot(diagonal, diagonal, 'r--', label='True Prediction')
    
    plt.xlabel('actual values')
    plt.ylabel('predicated values')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    
    plt.tight_layout()
    plt.show()


# train_data = pd.read_csv("training_set.csv")
# valid_data = pd.read_csv("valid_set.csv")

# # data is waaaayyyyy too big to do. Only using smaller sets for hyperparameter tesitng
# train_labels = train_data.iloc[:, -1][0:1000]
# train_features = train_data.iloc[:, :-1][0:1000]
# valid_labels = valid_data.iloc[:, -1][0:1000]
# valid_features = valid_data.iloc[:, :-1][0:1000]

# X_train_tensor = torch.tensor(train_features.values, dtype=torch.float32)
# y_train_tensor = torch.tensor(train_labels.values, dtype=torch.float32).reshape(-1, 1)
# X_val_tensor = torch.tensor(valid_features.values, dtype=torch.float32)
# y_val_tensor = torch.tensor(valid_labels.values, dtype=torch.float32).reshape(-1, 1)


# input_features = X_train_tensor.shape[1]
# k = 6

# model = L1RegularizedNN(input_features=input_features, k=k)
# train_losses, y_mean, y_std = train_model(
#     model, 
#     X_train_tensor, 
#     y_train_tensor,
#     epochs=200,
#     batch_size=8,
#     learning_rate=0.0001,
#     l1_lambda=0.00001
# )

# # Evaluate on validation set
# model.eval()
# with torch.no_grad():
#     # normalize the training data
#     val_predictions = model(X_val_tensor)
#     # unnormalize for the output
#     val_predictions = val_predictions * y_std + y_mean  

# plot_predictions_vs_actual(
#     val_predictions.numpy().flatten(), 
#     y_val_tensor.numpy().flatten(), 
#     title="model predictions vs actual Values (test set)"
# )