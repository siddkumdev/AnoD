import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Import your custom modules
from src import config
from src.dataset import TelemetryDataset
from src.model import TelemetryAutoencoder


def train_model():
    # 1. Setup Data Paths
    data_path = os.path.join(os.path.dirname(__file__), "data", "mock_healthy_telemetry.csv")         
    # 2. Hyperparameters (The dials you can tune)
    EPOCHS = 50          # How many times to read the whole CSV
    BATCH_SIZE = 64       # How many rows to look at before updating the math
    LEARNING_RATE = 0.0001 # How fast the model learns
    
    # 3. Initialize Dataset and DataLoader
    print("Loading dataset...")
    dataset = TelemetryDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Initialize the Model, Loss Function, and Optimizer
    print("Initializing Autoencoder...")
    model = TelemetryAutoencoder()
    
    # Mean Squared Error: Perfect for autoencoders measuring reconstruction accuracy
    criterion = nn.MSELoss() 
    # Adam Optimizer: The industry standard for fast, reliable training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    # 5. The Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    model.train() # Put the model in training mode
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        for batch in dataloader:
            # Step A: Clear old gradients
            optimizer.zero_grad()
            
            # Step B: Forward pass (compress and reconstruct)
            reconstructed = model(batch)
            
            # Step C: Calculate how wrong the reconstruction was
            loss = criterion(reconstructed, batch)
            
            # Step D: Backward pass (calculate the math updates)
            loss.backward()
            
            # Step E: Apply the math updates
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print progress at the end of each epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Average Loss: {avg_loss:.6f}")
        
    # 6. Save the trained weights
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"🎉 Training Complete! Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()