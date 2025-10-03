"""
Neural Network Decoder for RNN Working Memory

This module implements a PyTorch-based decoder network that learns to predict
past inputs from current RNN network states. Used for analyzing working memory
capacity in recurrent neural networks.

Authors: Vivian Kang, Jameel Saeb, Chetanya Singh, Ah-Young Moon
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class WorkingMemoryDecoder(nn.Module):
    """
    Feedforward neural network for decoding past inputs from RNN states.
    
    This decoder takes the current state of the recurrent network (firing rates
    of all neurons) and predicts which input was active at a previous timestep.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the decoder network.
        
        Args:
            input_dim (int): Size of RNN state vector (number of neurons)
            hidden_dim (int): Number of hidden units in decoder
            output_dim (int): Number of possible input classes to predict
        """
        super(WorkingMemoryDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simple feedforward architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim)
        """
        return self.network(x)


def train_decoder(X_train, Y_train, model_params, delay, 
                 num_epochs=10, batch_size=32, learning_rate=0.001, 
                 test_split=0.2, hidden_dim=64, verbose=True):
    """
    Train a decoder to predict input identity at time (t-delay) from network state at time t.
    
    Args:
        X_train (np.ndarray): Training network states (n_samples, n_neurons)
        Y_train (np.ndarray): Training input labels (n_samples,)
        model_params (dict): RNN model parameters (for getting dimensions)
        delay (int): The temporal delay being predicted
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for optimizer
        test_split (float): Fraction of data to use for testing
        hidden_dim (int): Number of hidden units in decoder
        verbose (bool): Whether to print training progress
        
    Returns:
        tuple: (decoder, test_accuracy, epoch_losses)
            - decoder: Trained decoder model
            - test_accuracy: Accuracy on test set
            - epoch_losses: List of training losses per epoch
    """
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.LongTensor(Y_train)
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    # Split into training and test sets
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize decoder
    decoder = WorkingMemoryDecoder(
        input_dim=model_params['N'], 
        hidden_dim=hidden_dim, 
        output_dim=model_params['nIn']
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    
    # Training loop
    epoch_losses = []
    for epoch in range(num_epochs):
        decoder.train()
        running_loss = 0.0
        
        for states, past_inputs in train_loader:
            states, past_inputs = states.to(device), past_inputs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = decoder(states)
            loss = criterion(outputs, past_inputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate on test set
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
    
    if verbose:
        print(f"Test Accuracy for delay={delay}: {test_accuracy:.4f}")
    
    return decoder, test_accuracy, epoch_losses


def evaluate_multiple_delays(model_params, delays, sequence_length=3000, 
                           num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train and evaluate decoders for multiple time delays.
    
    Args:
        model_params (dict): RNN model parameters
        delays (list): List of delay values to test
        sequence_length (int): Length of training sequences
        num_epochs (int): Training epochs per delay
        batch_size (int): Training batch size
        learning_rate (float): Learning rate
        
    Returns:
        tuple: (loss_curves, test_accuracies)
            - loss_curves: Dict mapping delays to loss curves
            - test_accuracies: Dict mapping delays to test accuracies
    """
    # Import here to avoid circular dependency
    import rnn_model
    
    loss_curves = {}
    test_accuracies = {}
    
    print("Training decoders for multiple delays...")
    for delay in delays:
        print(f"\nTraining decoder for delay = {delay}")
        
        # Generate training data for this delay
        X_train, Y_train = rnn_model.generate_training_data(model_params, sequence_length, delay)
        
        # Train decoder
        decoder, test_accuracy, losses = train_decoder(
            X_train, Y_train, model_params, delay,
            num_epochs=num_epochs, batch_size=batch_size, 
            learning_rate=learning_rate
        )
        
        # Store results
        loss_curves[delay] = losses
        test_accuracies[delay] = test_accuracy
    
    return loss_curves, test_accuracies


def plot_training_curves(loss_curves, title="Training Loss Curves"):
    """
    Plot training loss curves for different delays.
    
    Args:
        loss_curves (dict): Dictionary mapping delays to loss lists
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for delay, losses in loss_curves.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f'Delay = {delay}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy_vs_delay(test_accuracies, title="Decoder Accuracy vs. Time Delay"):
    """
    Plot test accuracy as a function of time delay.
    
    Args:
        test_accuracies (dict): Dictionary mapping delays to accuracies
        title (str): Plot title
    """
    delays = sorted(test_accuracies.keys())
    accuracies = [test_accuracies[d] for d in delays]
    
    plt.figure(figsize=(8, 6))
    plt.plot(delays, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Time Delay (timesteps)')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()
