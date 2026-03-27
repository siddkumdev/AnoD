import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timezone

API_URL = "http://localhost:8000/predict"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mock_anomaly_telemetry.csv")

def fire_test_request(metrics_dict, test_name):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pod_id": "payment_api",
        "metrics": metrics_dict
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()
        
        is_anomaly = data.get("anomaly_detected")
        score = data.get("confidence_score")
        
        if is_anomaly:
            print(f"🚨 {test_name}: ANOMALY DETECTED! (Score: {score})")
        else:
            print(f"✅ {test_name}: System Healthy. (Score: {score})")
            
    except Exception as e:
        print(f"❌ Connection failed: Is 'python api.py' running?")

if __name__ == "__main__":
    print("🚀 Firing instant AI verification test...\n")
    
    df = pd.read_csv(CSV_PATH)
    
    # Grab row 100 (Guaranteed to be healthy normal data)
    healthy_data = df.iloc[100].to_dict()
    
    # Grab row 1150 (Guaranteed to be right in the middle of the CPU spike)
    spiked_data = df.iloc[1150].to_dict()
    
    fire_test_request(healthy_data, "TEST 1 (Healthy Row)")
    fire_test_request(spiked_data, "TEST 2 (Spiked Row) ")