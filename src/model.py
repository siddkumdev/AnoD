import torch
import torch.nn as nn
from src import config

class TelemetryAutoencoder(nn.Module):
    def __init__(self):
        super(TelemetryAutoencoder, self).__init__()
        
        # Dynamically size the input based on the API Contract in config.py.
        input_dim = len(config.EXPECTED_METRICS)
        
        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(True),
            nn.Linear(24, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),     # A wider Bottleneck
            nn.ReLU(True)
        )
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 24),
            nn.ReLU(True),
            nn.Linear(24, input_dim),
            nn.Sigmoid() # ONLY keep this if your input data is scaled 0 to 1!
        )

    def forward(self, x):
        """Passes the data through the network."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def predict_anomaly(self, original_tensor):
        """
        Helper method for the live API. 
        Calculates global reconstruction error AND per-service Root Cause Analysis (RCA).
        """
        self.eval() 
        with torch.no_grad():
            reconstructed_tensor = self(original_tensor)
            
            # Calculate the raw errors for every single feature separately
            criterion = nn.MSELoss(reduction='none')
            feature_errors = criterion(reconstructed_tensor, original_tensor)
            
            # 1. Calculate the GLOBAL Score (Average of all features combined)
            global_error = feature_errors.mean(dim=1)
            is_anomaly = global_error > config.ANOMALY_THRESHOLD
            
            # 2. Calculate PER-SERVICE Scores (Root Cause Analysis)
            service_scores = {}
            
            # Iterate through our 5 active services
            for service in config.MONITORED_SERVICES:
                safe_svc = service.replace("-", "_")
                
                # Find exactly which columns belong to this specific service
                service_indices = [
                    i for i, metric in enumerate(config.EXPECTED_METRICS) 
                    if metric.startswith(safe_svc)
                ]
                
                if service_indices:
                    # Slice the tensor to grab just this service's columns
                    svc_error_tensor = feature_errors[:, service_indices]
                    # Calculate the average MSE for just this service
                    svc_mse = svc_error_tensor.mean().item()
                    
                    # Store the score in our dictionary
                    service_scores[service] = svc_mse

            # Return the global data AND the new detailed dictionary
            return is_anomaly, global_error, service_scores