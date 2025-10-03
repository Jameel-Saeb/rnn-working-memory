# Legacy Decoder Implementation
# 
# This is the original decoder implementation used in the project.
# For the improved, well-documented version, see src/decoder.py
#
# This file is kept for reproducibility of original results.

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from Generate_td import generate_trainingdata, model

class Decoder(nn.Module):
    """
    Simple feedforward decoder for predicting past inputs from current RNN states.
    
    Architecture:
    - Input layer: RNN state vector (typically 1000 neurons)
    - Hidden layer: ReLU activated (64 units by default)  
    - Output layer: Softmax over input classes (typically 20 classes)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # input_dim: model['N'] in RNN, number of neurons in network state
        # hidden_dim: number of hidden units in decoder network
        # output_dim: number of possible input classes (model['nIn'])
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Simple two-layer feedforward network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through decoder network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        return self.network(x)

    
def train_decoder(model, delay, num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train a decoder to predict input identity at time (t-delay) from network state at time t.
    
    Parameters:
    - model: Dictionary containing model parameters
    - delay: The delay (X) for which to predict past inputs
    - num_epochs: Number of training epochs
    - batch_size: Batch size for training
    - learning_rate: Learning rate for optimizer
    
    Returns:
    - decoder: Trained decoder model
    - test_accuracy: Accuracy on test set
    """
    # Create dataset
    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.LongTensor(Y_train)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    # Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    hidden_dim = 64  
    decoder = Decoder(model['N'], hidden_dim, model['nIn'])
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    epoch_losses = []
    for epoch in range(num_epochs):
        decoder.train()
        running_loss = 0.0

        for states, past_inputs in train_loader:
            states, past_inputs = states.to(device), past_inputs.to(device)

            optimizer.zero_grad()

            outputs = decoder(states)
            loss = criterion(outputs, past_inputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs},t Loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

    decoder.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for states, past_inputs in test_loader:
            states, past_inputs = states.to(device), past_inputs.to(device)
            outputs = decoder(states)
            _, predicted = torch.max(outputs.data, 1)
            total += past_inputs.size(0)
            correct += (predicted == past_inputs).sum().item()
    
    test_accuracy = correct / total
    print(f"Test Accuracy for delay={delay}: {test_accuracy:.4f}")
    
    return decoder, test_accuracy, epoch_losses

loss_curves = {}
test_accuracies = {}

for delay in [3, 5, 10, 15, 20]:
    X_train, Y_train = generate_trainingdata(model, 3000, delay)
    decoder, test_accuracy, losses = train_decoder(model, delay, num_epochs=10, batch_size=32, learning_rate=0.001)
    loss_curves[delay] = losses
    test_accuracies[delay] = test_accuracy
    print(f"Test Accuracy for delay={delay}: {test_accuracy:.4f}")

plt.figure(figsize=(10, 6))
for delay, losses in loss_curves.items():
    plt.plot(range(1, len(losses) + 1), losses, label=f'Delay = {delay}')

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves for Different Delays')
plt.legend()
plt.grid(True)
plt.show()

print("Test Accuracies:", test_accuracies)