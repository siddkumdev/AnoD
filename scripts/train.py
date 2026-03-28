import os
import sys

# Add the root 'ANOMD' directory to the system path so we can import from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import your custom modules
from src import config
from src.dataset import TelemetryDataset
from src.model import TelemetryAutoencoder

def train_model():
    # 1. Setup Data Paths -> UPDATED TO V2
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_data_v2.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Please run generate_mock_data.py first.")
        return

    # 2. Hyperparameters
    EPOCHS = 150          # Increased from 50
    BATCH_SIZE = 64       
    LEARNING_RATE = 0.001 # Increased by a factor of 10
    
    # 3. Initialize Dataset and DataLoader
    print(f"Loading dataset from {data_path}...")
    dataset = TelemetryDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Initialize the Model, Loss Function, and Optimizer
    print(f"Initializing Autoencoder for {len(config.EXPECTED_METRICS)} features...")
    model = TelemetryAutoencoder()
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    # 5. The Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    model.train() 
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        # Print progress every 5 epochs to keep the terminal clean
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Average Loss: {avg_loss:.6f}")
            
    # 6. Save the trained weights
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"\n🎉 Training Complete! Model saved to {config.MODEL_SAVE_PATH}")

    # =================================================================
    # 7. NEW: THRESHOLD RECALIBRATION WIZARD
    # =================================================================
    print("\n" + "="*50)
    print("🧠 THRESHOLD RECALIBRATION WIZARD")
    print("="*50)
    print("Calculating the reconstruction error for normal cluster behavior...")
    
    model.eval()
    all_losses = []
    
    # Run data through one by one without shuffling to get individual loss scores
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in eval_loader:
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            all_losses.append(loss.item())
            
    # Calculate exactly where the "normal" data stops
    max_loss = np.max(all_losses)
    p99_loss = np.percentile(all_losses, 99)
    
    print(f"Highest Normal Loss:    {max_loss:.6f}")
    print(f"99th Percentile Loss:   {p99_loss:.6f}")
    print(f"\n🎯 ACTION REQUIRED:")
    print(f"Open src/config.py and update ANOMALY_THRESHOLD to ~ {p99_loss:.6f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    train_model()