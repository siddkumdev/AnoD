from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import uvicorn
import os

# Import your custom logic
from src import config
from src.model import TelemetryAutoencoder
from src.preprocessor import enforce_feature_contract

app = FastAPI(title="SRE Anomaly Detection AI")

# ==========================================
# 1. LOAD THE TRAINED BRAIN
# ==========================================
print("Loading PyTorch model...")
model = TelemetryAutoencoder()
if os.path.exists(config.MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval() # Set to evaluation mode (no training)
    print("✅ Model loaded successfully!")
else:
    print(f"⚠️ WARNING: No trained model found at {config.MODEL_SAVE_PATH}. Run train.py first.")

# ==========================================
# 2. DEFINE THE INCOMING DATA FORMAT
# ==========================================
# This tells FastAPI exactly what JSON structure to expect from Role 2/3
class TelemetryPayload(BaseModel):
    timestamp: str
    pod_id: str
    metrics: dict  # e.g., {"cart_api_cpu_usage": 85.0, ...}

# ==========================================
# 3. THE PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
async def predict_anomaly(payload: TelemetryPayload):
    try:
        # 1. Convert the incoming JSON dictionary into a Pandas DataFrame (1 row)
        # We wrap it in a list so Pandas knows it's a single record
        df = pd.DataFrame([payload.metrics])
        
        # 2. Pass it through your shield (handles missing columns/chaos)
        clean_df = enforce_feature_contract(df)
        
        # 3. Convert to PyTorch Tensor
        input_tensor = torch.tensor(clean_df.values, dtype=torch.float32)
        
        # 4. Ask the model for a verdict
        is_anomaly, error_score = model.predict_anomaly(input_tensor)
        
        # Extract the boolean and float values from the PyTorch tensors
        anomaly_detected = bool(is_anomaly.item())
        confidence = float(error_score.item())
        
        # 5. Build the final response for Role 3
        response = {
            "timestamp": payload.timestamp,
            "anomaly_detected": anomaly_detected,
            "confidence_score": round(confidence, 4),
            "attributed_service": payload.pod_id,
            "action_required": anomaly_detected # A helpful flag for Role 3
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server if executed directly
if __name__ == "__main__":
    print("🚀 Starting API server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 