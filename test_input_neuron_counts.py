# Legacy Test Script for Input Neuron Count Experiments  
#
# This script tests how the number of input neurons affects decoder performance.
# For improved experiment management, see experiments/ directory.
#
# This file is kept for reproducibility of original results.

import math
import matplotlib.pyplot as plt
from Generate_td import generate_trainingdata, model
from Decoder import Decoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_and_test(model, delay, num_epochs=10):
    # Generate training data
    X, Y = generate_trainingdata(model, sequence_length=200, delay=delay)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    decoder = Decoder(input_dim=model['N'], hidden_dim=64, output_dim=model['nIn'])
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        decoder.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = decoder(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    decoder.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = decoder(batch_x)
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    return accuracy

# Try different input sizes
input_sizes = [5, 10, 15, 20]
delay = 10

print("Testing decoder performance across different input neuron counts:")
for n in input_sizes:
    model['nIn'] = n
    # Update input weight matrix after changing nIn
    randMatIn = np.random.normal(0, 1, size=(model['N'], model['nIn']))
    spMatIn = np.random.uniform(0, 1, size=(model['N'], model['nIn'])) <= model['spIn']
    model['Jin'] = np.multiply(randMatIn, spMatIn) * model['gIn'] / math.sqrt(model['nIn'] * model['spIn'])
    print(f"\nInput size: {n}")
    acc = train_and_test(model, delay=delay)
    print(f"Test accuracy: {acc:.4f}")
input_sizes = [5, 10, 15, 20]
accuracies = [0.75, 0.82, 0.88, 0.95]  # Replace with your results

plt.plot(input_sizes, accuracies, marker='o')
plt.xlabel("Number of Input Neurons")
plt.ylabel("Test Accuracy")
plt.title("Decoder Accuracy vs. Input Size")
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()
