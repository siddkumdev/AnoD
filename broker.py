import os
import sys
import torch
from fastapi import FastAPI, Request
import uvicorn

# --- PATH SETUP FOR IMPORTS ---
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from src import config
from src.model import TelemetryAutoencoder
from src.preprocessor import enforce_feature_contract

app = FastAPI()

# This dictionary holds the live state in memory
latest_state = {}

# ==========================================
# 🧠 LOAD THE ML MODEL INTO MEMORY ONCE
# ==========================================
print("Loading PyTorch Anomaly Model...")
MODEL_PATH = os.path.join(ROOT_DIR, "data", "saved_models", "anomaly_model_v2.pth")
model = TelemetryAutoencoder()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval() # Set to evaluation mode
print("Model ready to score telemetry streams!")

@app.post("/predict")
async def receive_telemetry(request: Request):
    global latest_state
    payload = await request.json()
    metrics = payload.get("metrics", {})
    
    # ==========================================
    # 🧠 REAL AI INFERENCE
    # ==========================================
    try:
        # 1. Scale and Convert to Tensor
        input_tensor = enforce_feature_contract(metrics).unsqueeze(0)
        
        # 2. Run the Autoencoder prediction (NOW UNPACKING 3 VALUES)
        is_anomaly_tensor, error_tensor, service_scores = model.predict_anomaly(input_tensor)
        
        # 3. Extract standard Python values
        is_anomaly = bool(is_anomaly_tensor.item())
        mse_loss = float(error_tensor.item()) # This is your global loss score
        
    except Exception as e:
        print(f"Error processing telemetry: {e}")
        is_anomaly = False
        mse_loss = 0.0
        service_scores = {}
    
    # Store the metrics and the AI scores so the dashboard can fetch them
    latest_state = metrics
    latest_state["ml_score"] = mse_loss
    latest_state["service_scores"] = service_scores # <-- RCA Scores saved to memory!
    
    # Respond to the simulator
    return {
        "anomaly_detected": is_anomaly, 
        "confidence_score": mse_loss,
        "service_scores": service_scores
    }

@app.get("/")
def send_to_dashboard():
    return latest_state

if __name__ == "__main__":
    print("🚀 Broker running on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)